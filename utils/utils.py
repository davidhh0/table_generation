import re
import dateutil.parser as parser
import requests
from bs4 import BeautifulSoup

from datetime import datetime, timedelta
from pandas import DataFrame, read_html
from io import StringIO
import json
import diskcache
import os



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
            keep_search = False
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


def get_articles_to_parse():
    import string
    import requests
    import random
    import git
    from collections import OrderedDict
    from wikiparser import WikiTableParser
    working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    wiki_obj = WikiTableParser()
    basic_url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=insource:%22wikitable%22%20intitle:{TITLE}*&format=json&srlimit=500&sroffset={OFFSET}"
    tbl_metadata_db = diskcache.Cache(f'{working_dir}/local_dbs/tbl_metadata.db')
    for _ in range(1000):
        title = random.choice(string.ascii_uppercase) + random.choice(
            string.ascii_lowercase
        )
        response = requests.get(
            basic_url.format(TITLE=title, OFFSET=random.randint(0, 9500))
        ).json()
        for j in [k for k in response["query"]["search"] if k['title'].isascii()]:
            try:
                df, idx, msg, dates, paragraph = wiki_obj.run(
                    f'https://en.wikipedia.org/?curid={j["pageid"]}', j['title']
                )
                if df is not None:
                    is_tbl_consistent, page_id, article_name = is_consistent(
                        f'https://en.wikipedia.org/wiki/{j["title"]}', idx
                    )
                    if is_tbl_consistent == 'match':
                        article_metadata = page_details(
                            f'https://en.wikipedia.org/wiki/{j["title"]}',
                            str(j["pageid"]),
                        )
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
                        }
                        # Storing the table details in local DB

                        tbl_metadata_db.set(
                            url,
                            json.dumps(tbl_details),
                        )

            except:
                continue
    b = 5


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
        html_current = requests.get(url)
        page_content = BeautifulSoup(html_current.text, 'html.parser')
        current_tbls = page_content.select('table[class*=wikitable]')[tbl_idx]
        current_df = read_html(StringIO(current_tbls.__str__()))[0]

        html_old = requests.get(previous_id['url'])
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
    import requests
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


def get_llm_response(
    prompt_string, use_cache=True, MODEL='gpt-4.1-2025-04-14', cache=None
):
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ["openai_api_key"], timeout=30)
    from openai._exceptions import APITimeoutError, APIConnectionError, RateLimitError
    from requests.exceptions import ChunkedEncodingError

    if cache is None:
        import git
        working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        cache = diskcache.Cache(f'{working_dir}/local_dbs/cache/llm_cache.db')
    prompt_cache_key = f"{MODEL}_{prompt_string}"
    if use_cache and prompt_cache_key in cache:
        return cache[prompt_cache_key]
    params = {
        'model': MODEL,
        'messages': [
            {
                "role": "user",
                "content": prompt_string,
            },
        ],
        "temperature": 0.0,
        "stream": True,
    }
    try:
        response_str = ""
        response = client.chat.completions.create(**params)
        for chunk in response:
            if chunk.choices[0].delta.content is None:
                break
            if hasattr(chunk,'choices') and len(chunk.choices) > 0:
                chunk_content = chunk.choices[0].delta.content
                response_str += chunk_content
    except (APITimeoutError, RateLimitError, ChunkedEncodingError) as e:
        cache[prompt_cache_key] = None
        return None
    except APIConnectionError as e:
        return None
    prompt_response = response_str.strip()
    if use_cache:
        cache[prompt_cache_key] = prompt_response
    return prompt_response
