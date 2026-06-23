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


# --- Output-token / reasoning caps -----------------------------------------
# Answers in this benchmark are "exact value only" (a single value) or a short
# comma-separated list, yet reasoning models (gpt-5*, o-series) will otherwise
# burn hundreds-thousands of *reasoning* tokens per trivial answer at the top
# output price. We cap max_completion_tokens (which, for reasoning models,
# counts reasoning + visible output together) and set a low reasoning_effort.
# Both are read from config.yaml so they can be tuned without code edits.
_CONFIG_CACHE = None


def _config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        import yaml
        with open(f"{working_dir}/config.yaml", "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f) or {}
    return _CONFIG_CACHE


def _is_openai_reasoning_model(model):
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith(("o1", "o3", "o4"))



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


def chatgpt(MODEL, prompt_string, retry_count=3, max_out_tokens=None):
    from requests.exceptions import ChunkedEncodingError
    from openai import OpenAI
    if retry_count < 1:
        raise "ChatGPT API request failed after multiple retries."

    client = OpenAI(api_key=os.environ["openai_api_key"], timeout=30)
    from openai._exceptions import APITimeoutError, APIConnectionError, RateLimitError, BadRequestError
    conf = _config()
    if max_out_tokens is None:
        max_out_tokens = conf.get("max_completion_tokens", 512)
    params = {
        'model': MODEL,
        'messages': [
            {
                "role": "user",
                "content": prompt_string,
            },
        ],
        # Cap total output (reasoning + visible) so trivial answers can't run up
        # a huge reasoning-token bill. Safe because answers are a single value or
        # a short list; pair with low reasoning_effort below.
        'max_completion_tokens': max_out_tokens,
        # "temperature": 0.0,  # reasoning models reject non-default temperature
    }
    # reasoning_effort is only valid for reasoning models (gpt-5*, o-series);
    # sending it to e.g. gpt-4.1 raises BadRequestError.
    if _is_openai_reasoning_model(MODEL):
        params['reasoning_effort'] = conf.get("openai_reasoning_effort", "minimal")
    try:
        response = client.chat.completions.create(**params)
        choice = response.choices[0]
        content = choice.message.content
        # A reasoning model that exhausts max_completion_tokens during reasoning
        # returns finish_reason="length" with empty content. Surface it loudly and
        # return None (a cache miss) instead of caching an empty answer.
        if not content or choice.finish_reason == "length":
            print(f"[chatgpt] empty/truncated response from {MODEL} "
                  f"(finish_reason={choice.finish_reason}, "
                  f"max_completion_tokens={max_out_tokens}) -- raise the ceiling")
            return None
        return content
    except (APITimeoutError, RateLimitError, ChunkedEncodingError) as e:
        return None
    except APIConnectionError as e:
        return None
    except BadRequestError:
        print("ChatGPT API request failed")
        time.sleep(3)
        return chatgpt(MODEL, prompt_string, retry_count - 1, max_out_tokens=max_out_tokens)


def gemini(MODEL, prompt_string, context=False, retry_count=3):
    """Get response from Gemini API with retry logic and proper error handling."""

    # Mirror the batch path (llm_generation/batch.py): cap visible output via the
    # shared max_completion_tokens, and optionally set a thinking budget. Gemini's
    # max_output_tokens caps visible output only (thinking has its own budget), so a
    # generous cap never truncates the answer.
    conf = _config()
    cfg_kwargs = {"temperature": 0.0}
    cap = conf.get("max_completion_tokens")
    if cap:
        cfg_kwargs["max_output_tokens"] = cap
    _budget = conf.get("gemini_thinking_budget")
    if _budget is not None:
        from google.genai.types import ThinkingConfig
        cfg_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=int(_budget))
    for attempt in range(retry_count):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt_string,
                config=GenerateContentConfig(**cfg_kwargs),
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
    # Mirror the batch path (llm_generation/batch.py): cap visible output via the
    # shared max_completion_tokens, and optionally enable extended thinking (off by
    # default; when on, its budget is added on top because Anthropic requires
    # max_tokens > thinking budget_tokens).
    conf = _config()
    answer_cap = conf.get("max_completion_tokens", 1024)
    params = {
        "max_tokens": answer_cap,
        "messages": [{"role": "user", "content": prompt_string}],
        "model": MODEL,
    }
    if conf.get("claude_extended_thinking", False):
        budget = int(conf.get("claude_thinking_budget", 4096))
        params["max_tokens"] = budget + answer_cap
        params["thinking"] = {"type": "enabled", "budget_tokens": budget}
    with claude_client.messages.stream(**params) as stream:
        response = stream.get_final_message()
    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
    if not text:
        return None
    return text



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


def _describe_prompt(prompt_string):
    """Best-effort one-line label for a prompt, derived from its content, so the
    cache hit/miss logs say which question they refer to. Returns:
    'Test=CB/MC/OB | Type=Count/Max/Min/LR/Single | Constraint=... | Table=...'."""
    p = prompt_string
    # 1. Test (CB / MC / OB)
    if p.startswith("Given a factual table titled"):
        test = "OB"
    elif "Select one from the following options" in p or "Select all that apply" in p:
        test = "MC"
    else:
        test = "CB"
    # 2. Test type
    if "Count the number of rows" in p:
        ttype = "Count"
    elif "with the highest" in p:
        ttype = "Max"
    elif "with the lowest" in p:
        ttype = "Min"
    elif "List all values in" in p:
        ttype = "LR"
    elif "Identify the value of the column" in p:
        ttype = "Single"
    else:
        ttype = "?"
    # 3. Constraint / break-down
    if "does not equal" in p:
        constraint = "Not Equals"
    elif "is either" in p:
        constraint = "In"
    elif "is greater than" in p:
        constraint = "Greater Than"
    elif "is less than" in p:
        constraint = "Less Than"
    elif "is between" in p:
        constraint = "Between"
    elif "equals" in p:
        constraint = "Equals"
    else:
        constraint = "-"
    # 4. Table name. Greedy up to the structural delimiter so titles containing
    # quotes/apostrophes aren't truncated: CB/MC end with "' with a primary key",
    # OB ends with "':" before the CSV. Non-greedy fallback if neither is present.
    m = (re.search(r"titled '(.*)' with a primary key", p)
         or re.search(r"titled '(.*)':", p)
         or re.search(r"titled '(.*?)'", p))
    table = m.group(1) if m else "?"
    return f"Test={test} | Type={ttype} | Constraint={constraint} | Table={table!r}"


def get_llm_response(
        model,
        prompt_string,
        answer=None,
        max_out_tokens=None,
):

    cache = diskcache.Cache(f'{working_dir}/local_dbs/cache/llm_cache/{model}.db')
    prompt_string = prompt_string.strip()
    if prompt_string in cache and cache[prompt_string] != '':
        print(f"Cache hit! ({model}) | {_describe_prompt(prompt_string)}")
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
                    print(f"Cache hit! ({model}) | {_describe_prompt(prompt_string)}")
                    return cache[i]
        else:
            for i in cache_similar:
                if str(answer) in i.split("Select one from the following options: ")[-1].splitlines()[0]:
                    print(f"Cache hit! ({model}) | {_describe_prompt(prompt_string)}")
                    return cache[i]



    print(f"Cache miss... ({model}) | {_describe_prompt(prompt_string)}")
    if COLLECT_MODE:
        COLLECTED.setdefault(model, set()).add(prompt_string)
        return None
    if 'gpt' in model:
        response_str = chatgpt(model, prompt_string, max_out_tokens=max_out_tokens)
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


