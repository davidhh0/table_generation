# Utilities/Help
import re
import time

import dateutil.parser as parser
import huggingface_hub
from ast import literal_eval
import  stealth_requests as  requests
from bs4 import BeautifulSoup

from datetime import datetime, timedelta
from pandas import read_html
from io import StringIO
import json
import diskcache
import anthropic
import os
from google.genai import types, Client
from google.genai import Client
from google.genai.types import Tool, GenerateContentConfig
from google.genai.errors import ServerError
import git
import dotenv

# .env file is expected to have the following variables:
dotenv.load_dotenv()
working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

client = Client(api_key=os.environ.get("gemini_api_key"))
claude_client = anthropic.Anthropic(
    api_key=os.environ.get("claude_api")
)

# --- Batch / collect mode ---------------------------------------------------
# When COLLECT_MODE is True, get_llm_response records cache-missing prompts into
# COLLECTED (model -> set of stripped prompt strings) and returns None instead of
# making a live API call. A batch warmer (llm_generation/batch.py) then fills the
# per-model diskcache so a subsequent normal pass is all cache hits.
COLLECT_MODE = False
COLLECTED = {}



def get_revid(page_id=None, by='pageids', starting=datetime(2013, 11, 1)):
    user_agent = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }

    page_title_response = requests.get(
        f"https://en.wikipedia.org/w/api.php?action=query&prop=info&{by}={page_id}&inprop=url&format=json",
        headers=user_agent,
    ).json()
    page_id = next(iter(page_title_response['query']['pages']))
    if 'missing' in page_title_response['query']['pages'][page_id.__str__()]:
        return None
    page_url: str = page_title_response['query']['pages'][page_id.__str__()]['editurl']
    page_title = re.search(r"title=(.+)&", page_url).group(1)
    page_history = page_url.replace("&action=edit", "")
    page_revs = f"https://en.wikipedia.org/w/api.php?format=json&action=query&titles={page_title}&prop=revisions&rvprop=ids|timestamp|size&rvlimit=500"
    rev_id = None
    keep_search = True
    _continue_token = ""
    while keep_search:
        if 'None' in _continue_token:
            rev_id = None
            break
        page_edits = requests.get(
            page_revs + _continue_token,
            headers=user_agent,
        ).json()
        revisions = page_edits['query']['pages'][page_id.__str__()]['revisions']
        if parser.parse(revisions[0]['timestamp']).timestamp() < starting.timestamp():
            rev_id = revisions[0]['revid']
            break
        for edit_idx in range(len(revisions) - 1):
            if (
                    parser.parse(revisions[edit_idx]['timestamp']).timestamp()
                    >= starting.timestamp()
                    >= parser.parse(revisions[edit_idx + 1]['timestamp']).timestamp()
            ):
                keep_search = False
                rev_id = revisions[edit_idx + 1]['revid']
                break
        _continue_token = (
            f"&rvcontinue={page_edits.get('continue', {}).get('rvcontinue')}"
        )
    if rev_id is None:
        return_value = {
            'url': None,
            'e_title': page_title,
            'title': page_title_response['query']['pages'][page_id.__str__()]['title'],
            'page_id': page_id,
        }
    else:
        return_value = {
            'url': page_history + f"&oldid={rev_id}",
            'e_title': page_title,
            'title': page_title_response['query']['pages'][page_id.__str__()]['title'],
            'page_id': page_id,
        }
    return return_value


def get_articles_to_parse(conf, ts):
    import string
    import random
    import git
    from collections import OrderedDict
    from wikiparser import WikiTableParser
    fetched_tbls = 0
    working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    wiki_obj = WikiTableParser()
    basic_url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=insource:%22wikitable%22%20intitle:{TITLE}*&format=json&srlimit=500&sroffset={OFFSET}"
    tbl_metadata_db = diskcache.Cache(f'{working_dir}/local_dbs/tbl_metadata.db')
    for _ in range(1000):
        title = random.choice(string.ascii_uppercase) + random.choice(
            string.ascii_lowercase
        )
        offset_rand = random.randint(0, 9500)
        response = requests.get(
            basic_url.format(TITLE=title, OFFSET=offset_rand)
        ).json()
        for j in [k for k in response["query"]["search"] if k['title'].isascii()]:
            try:
                df, idx, msg, dates, paragraph = wiki_obj.run(
                    # f'https://en.wikipedia.org/?curid={j["pageid"]}', j['title']
                    "https://en.wikipedia.org/wiki/2018_Korean_Tour", "2018_Korean_Tour"
                )
                if df is not None:
                    is_tbl_consistent, page_id, article_name = is_consistent(
                        # f'https://en.wikipedia.org/wiki/{j["title"]}', idx
                        "https://en.wikipedia.org/wiki/2018_Korean_Tour", idx
                    )
                    if is_tbl_consistent == 'match':
                        article_metadata = page_details(
                            # f'https://en.wikipedia.org/wiki/{j["title"]}',
                            f'https://en.wikipedia.org/wiki/2018_Korean_Tour',
                            page_id,
                        )
                        if 'minimum_popularity' in conf and conf.get('minimum_popularity'):
                            if conf.get('minimum_popularity') > article_metadata['popularity']:
                                print(f"Dropping article: {j['title']} as it doesn't have enough popularity")
                        article_title = get_revid(page_id)
                        url = f'https://en.wikipedia.org/wiki/{article_title["e_title"]}'
                        # Writing the DataFrame to CSV as it passed all checks - rules and consistency
                        df.to_csv(f'tbls/{page_id}_{idx}.csv', index=False)
                        tbl_details = {
                            'url': url,
                            'page_id': page_id,
                            'article_name': article_name,
                            'table_idx': idx,
                            'dates': dates,
                            'paragraph': paragraph,
                            'columns': OrderedDict(
                                **{col: str(df[col].dtype) for col in df.columns}
                            ),
                            'shape': df.shape,
                            'article_metadata': article_metadata,
                            'insert_ts': ts.__str__()
                        }
                        # Storing the table details in local DB

                        tbl_metadata_db.set(
                            url,
                            json.dumps(tbl_details),
                        )
                        fetched_tbls += 1
                        print(f"Fetched {fetched_tbls}/{conf['run_tables_to_fetch']}")
                        if fetched_tbls == conf['run_tables_to_fetch']:
                            return

            except:
                continue


def is_consistent(url, tbl_idx, years_ago=1):
    page_id = url.split('/')[-1]
    previous_id = get_revid(
        page_id,
        by='titles',
        starting=datetime.today() - timedelta(days=years_ago * 365),
    )
    if previous_id is None or previous_id['url'] is None:
        print('No old page for:', url)
        return 'noOldPage', None, None
    try:
        html_current = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        })
        page_content = BeautifulSoup(html_current.text, 'html.parser')
        current_tbls = page_content.select('table[class*=wikitable]')[tbl_idx]
        current_df = read_html(StringIO(current_tbls.__str__()))[0]

        html_old = requests.get(previous_id['url'], headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        })
        page_content = BeautifulSoup(html_old.text, 'html.parser')
        old_tbls = page_content.select('table[class*=wikitable]')[tbl_idx]
        old_df = read_html(StringIO(old_tbls.__str__()))[0]
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return 'Error_parsing', None, None
    if current_df is None or old_df is None:
        print('No new table for:', url)
        return 'Error_parsing', None, None
    if current_df.columns.tolist() != old_df.columns.tolist():
        return 'misMatchHeaders', None, None
    for idx, (old, new) in enumerate(zip(old_df.to_numpy(), current_df.to_numpy())):
        if not (old.astype(str) == new.astype(str)).all():
            if False in list(old == new):
                try:
                    print(
                        url,
                        'misMatchValues-'
                        + old_df.columns[list(old == new).index(False)]
                        + '-'
                        + str(idx),
                    )
                    print(
                        f"Mismatch in {url} at {old_df.columns[list(old == new).index(False)]} {idx}"
                    )
                    return (
                        'misMatchValues-'
                        + old_df.columns[list(old == new).index(False)]
                        + '-'
                        + str(idx),
                        None,
                        None,
                    )
                except Exception as e:
                    print(f"Error in mismatch values for {url}: {e}")
                    return 'Error_mismatch_values', None, None

    print(f'Match in {url}')
    return 'match', previous_id['page_id'], previous_id['title']


def page_details(url, page_id):
    import statistics
    from datetime import datetime, timedelta

    page_title = url.split('/')[-1]
    user_agent = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    page_edits = requests.get(
        f"https://en.wikipedia.org/w/api.php?format=json&action=query&titles={page_title}&prop=revisions&rvprop=ids|timestamp|size&rvlimit=max",
        headers=user_agent,
    ).json()
    first_changed = min(
        page_edits['query']['pages'][page_id]['revisions'],
        key=lambda x: x['timestamp'],
    )['timestamp'][:10]
    num_of_changes = len(page_edits['query']['pages'][page_id]['revisions'])
    today = datetime.now().date().strftime('%Y%m%d')
    two_years_ago = (datetime.now() - timedelta(days=730)).date().strftime('%Y%m%d')
    popularity = requests.get(
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{page_title}/monthly/{two_years_ago}/{today}",
        headers=user_agent,
    ).json()
    popularity = int(statistics.mean([value['views'] for value in popularity['items']]))
    return_obj = json.dumps(
        {
            'first_changed': first_changed,
            'num_of_changes': num_of_changes,
            'popularity': popularity,
        }
    )
    return return_obj


def get_sample(_df, try_cast):
    from collections import Counter

    return_str = ""
    table_size = _df.shape[0]
    _sample_data = _df.sample(n=min(table_size // 3, 5), random_state=1)
    for col in _df.columns.tolist():
        values = _df[col].apply(try_cast)
        data = _sample_data[col].apply(try_cast).tolist()
        try:
            max_value = values.max()
            min_value = values.min()
            nunique = values.nunique()
        except TypeError as e:
            common_type = Counter([type(k) for k in values]).most_common(1)[0][0]
            values = [k for k in values if isinstance(k, common_type)]
            try:
                max_value = max(values)
                min_value = min(values)
                nunique = len(set(values))
            except TypeError as e:
                max_value = "N/A"
                min_value = "N/A"
                nunique = len(set(values))

        return_str += f"`{col}` - max value: {max_value}, min value: {min_value}, number of distinct values: {nunique}, random sample data: {data}. \n"
    return return_str


def chatgpt(MODEL, prompt_string, retry_count=3):
    from requests.exceptions import ChunkedEncodingError
    from openai import OpenAI
    if retry_count < 1:
        raise "ChatGPT API request failed after multiple retries."

    client = OpenAI(api_key=os.environ["openai_api_key"], timeout=30)
    from openai._exceptions import APITimeoutError, APIConnectionError, RateLimitError, BadRequestError
    params = {
        'model': MODEL,
        'messages': [
            {
                "role": "user",
                "content": prompt_string,
            },
        ],
        # "temperature": 0.0,
    }
    try:
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    except (APITimeoutError, RateLimitError, ChunkedEncodingError) as e:
        return None
    except APIConnectionError as e:
        return None
    except BadRequestError:
        print("ChatGPT API request failed")
        time.sleep(3)
        return chatgpt(MODEL, prompt_string, retry_count - 1)


def gemini(MODEL, prompt_string, context=False, retry_count=3):
    """Get response from Gemini API with retry logic and proper error handling."""

    for attempt in range(retry_count):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt_string,
                config=GenerateContentConfig(
                    temperature=0.0,
                    # max_output_tokens=512,  # Limit response length to control costs
                ),
            )
            print('Received response from Gemini')
            return response.text
        except ServerError as e:
            if 'Spikes in demand' in e.message:
                print(f"Gemini API server error: {e}")
                time.sleep(15)  # Wait before retrying
                continue
            print(f'Unexpected error calling Gemini: {e}')
            return None
    return None

def general_hf(MODEL, prompt_string, retry_count=4):
    from huggingface_hub import InferenceClient
    hf_client = InferenceClient(
        model=MODEL,
        token=os.environ.get("hf_api")
    )

    try:
        completion = hf_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_string
                        }
                    ]
                }
            ],
        )
    except huggingface_hub.errors.HfHubHTTPError as e:
        if retry_count == 0:
            raise e
        print(f"Hugging Face API error: {e}")
        return general_hf(MODEL, prompt_string, retry_count=retry_count-1)
    return completion.choices[0].message.content

def claude(MODEL, prompt_string, context=False):
    prompt_string = prompt_string.strip()
    with claude_client.messages.stream(
            max_tokens=1024,
            messages=[
                {"role": "user",
                 "content": prompt_string}
            ],
            model=MODEL,
    ) as stream:
        response = stream.get_final_message()
    if len(response.content) == 0:
        return None
    return response.content[0].text



def _get_llm_response(
        prompt_string, MODEL, use_cache=True, cache=None, context=False
):
    prompt_string = prompt_string.strip()
    if cache is None:
        import git
        working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        cache = diskcache.Cache(f'{working_dir}/local_dbs/cache/llm_cache.db')
    context_key = "Context" if context else "NoContext"
    prompt_cache_key = f"{context_key}_{MODEL}_{prompt_string}"

    if use_cache and prompt_cache_key in cache and cache[prompt_cache_key] != '':
        return cache[prompt_cache_key]
    if use_cache and f"{context_key}_{MODEL}_\n{prompt_string}" in cache and cache[f"{context_key}_{MODEL}_\n{prompt_string}"] != '':
        return cache[f"{context_key}_{MODEL}_\n{prompt_string}"]
    if 'gpt' in MODEL:
        response_str = chatgpt(MODEL, prompt_string)
    if 'gemini' in MODEL:
        response_str = gemini(MODEL, prompt_string, context)
    if 'llama' in MODEL.lower():
        response_str = general_hf(MODEL, prompt_string)
    if 'claude' in MODEL.lower():
        response_str = claude(MODEL, prompt_string, context)
    if response_str is None:
        return None
    prompt_response = response_str.strip()
    if use_cache:
        cache[prompt_cache_key] = prompt_response
    return prompt_response


def get_llm_response(
        model,
        prompt_string,
        answer=None,
):

    cache = diskcache.Cache(f'{working_dir}/local_dbs/cache/llm_cache/{model}.db')
    prompt_string = prompt_string.strip()
    if prompt_string in cache and cache[prompt_string] != '':
        print(f"Cache hit! ({model})")
        return cache[prompt_string]
    if prompt_string in cache and cache[prompt_string] == '':
        return None
    if answer is not None and "Select one from the following options: " in prompt_string:
        prompt_prefix = prompt_string.split("Select one from the following options: ")
        cache_similar = [k for k in cache if k.startswith(prompt_prefix[0]) and "Select one from the following options: " in k]
        if isinstance(answer, list):
            for i in cache_similar:
                # cache_answers = literal_eval(i.split("Select one from the following options: ")[-1].splitlines()[0])
                answer_exists_in_cache = [k for k in answer if str(k) in i.split("Select one from the following options: ")[-1].splitlines()[0]]
                if i.startswith(prompt_prefix[0]) and answer_exists_in_cache:
                    print(f"Cache hit! ({model})")
                    return cache[i]
        else:
            for i in cache_similar:
                if str(answer) in i.split("Select one from the following options: ")[-1].splitlines()[0]:
                    print(f"Cache hit! ({model})")
                    return cache[i]



    print(f"Cache miss... ({model})")
    if COLLECT_MODE:
        COLLECTED.setdefault(model, set()).add(prompt_string)
        return None
    if 'gpt' in model:
        response_str = chatgpt(model, prompt_string)
    elif 'gemini' in model:
        response_str = gemini(model, prompt_string)
    elif 'claude' in model.lower():
        response_str = claude(model, prompt_string)
    else:
        response_str = general_hf(model, prompt_string)
    if response_str is None:
        return None
    prompt_response = response_str.strip()
    cache[prompt_string] = prompt_response
    return prompt_response


