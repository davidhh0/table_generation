import json

import redis

from scraped_tbls.utils.utils import page_details, get_revid
from scraped_tbls.wikiparser import WikiTableParser
from scraped_tbls.consistency import is_consistent
from collections import OrderedDict
from pandas import DataFrame
r_failed_and_passed = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
r_is_consistent = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
r_tbl_details = redis.Redis(host='localhost', port=6379, db=8, decode_responses=True)
r_all_tables = redis.Redis(host='localhost', port=6379, db=5, decode_responses=True)
import diskcache

cache = diskcache.Cache('bigger_tbls_cache')

def worker(url_list):
    import sys
    parser_obj = WikiTableParser()
    for url in url_list:

        df, idx, msg, dates, paragraph = parser_obj.run("https://en.wikipedia.org/?curid=" + url)
        info = get_revid(url)
        _page_details = page_details("https://en.wikipedia.org/wiki/" + info['e_title'], str(url))
        is_tbl_consistent, page_id, article_name = is_consistent("https://en.wikipedia.org/wiki/" + info['e_title'], idx)
        r_is_consistent.set(f'{url}@{idx}', is_tbl_consistent)
        if is_tbl_consistent == 'match':
            df.to_csv(f'tbls/{page_id}_{idx}.csv', index=False)
            tbl_details = {
                'url': url,
                'page_id': page_id,
                'article_name': article_name,
                'table_idx': idx,
                'dates': dates,
                'paragraph': paragraph,
                'columns': OrderedDict(**{col: str(df[col].dtype) for col in df.columns}),
                'shape': df.shape,
            }
            r_tbl_details.set(f'{url}', json.dumps(tbl_details))


if __name__ == "__main__":
    from numpy import array_split
    import multiprocessing

    all_urls = cache.iterkeys()
    jobs = []  # list of jobs
    worker(all_urls)
