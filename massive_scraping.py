import json

import redis
from wikiparser import WikiTableParser
from consistency import is_consistent
from collections import OrderedDict
from pandas import DataFrame
r_failed_and_passed = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
r_is_consistent = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
r_tbl_details = redis.Redis(host='localhost', port=6379, db=3, decode_responses=True)
r_all_tables = redis.Redis(host='localhost', port=6379, db=5, decode_responses=True)


def worker(url_list):
    import sys
    parser_obj = WikiTableParser()
    for url in url_list:
        idx = None
        value:str = r_failed_and_passed.get(url)
        if value is None or 1:
            try:
                df, idx, msg, dates, paragraph = parser_obj.run(url)
            except:
                df, idx, msg, dates, paragraph = None, None, 'Error parsing', None, None
            if df is None:
                r_failed_and_passed.set(f'{url}', msg)
                continue
            else:
                value = f'PASSED_{df.shape[0]}x{df.shape[1]}@{idx}'
                r_failed_and_passed.set(url, f'PASSED_{df.shape[0]}x{df.shape[1]}@{idx}')
        if not value.startswith('PASSED'):
            continue
        is_tbl_consistent, page_id, article_name = is_consistent(url, idx)
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

    all_urls = r_tbl_details.keys()
    jobs = []  # list of jobs
    worker(all_urls)
    jobs_num = 10 # number of workers
    list_divided = array_split(all_urls, jobs_num)
    for url_chunk in list_divided:
        # Declare a new process and pass arguments to it
        p1 = multiprocessing.Process(target=worker, args=(url_chunk,))
        jobs.append(p1)
        p1.start()  # starting workers
    for job in jobs:
        job.join()