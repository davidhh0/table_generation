from wikiparser import WikiTableParser
from personal.utils import get_revid
import json
import os
import pandas as pd
start_indx = 64
parser = WikiTableParser()
with open('cfg.json') as f:
    tbls = json.load(f)
for idx, tbl in tbls.items():
    if int(idx) < start_indx:
        continue
    evgenii = pd.read_csv(f'../benchmark/tables/{tbl["file"]}')
    # evgenii.to_csv(f'scraped_tbls/{idx}_evgenii.csv', index=False)
    # david = pd.read_csv(f'scraped_tbls/{idx}.csv')
    article_id, tbl_idx = tbl['wikiId'].split('-')
    old_page = get_revid(article_id)
    if old_page is None or  old_page['url'] is None:
        print(f"Article {article_id} not found, skipping.")
        continue
    df = parser.run(old_page['url'], tbl_idx=int(tbl_idx))
    print(str(tbl['columns'])+ " [" + str((tbl['numDataRows'],tbl['numCols'])) + "]")
    print()

    if df is None or df[0] is None:
        print(f"Table {idx} not found or empty, skipping.")
        continue
    if not df[0].columns.tolist() == evgenii.columns.tolist() :
        b=5
    match = True
    david_list = [k for k in df[0].to_numpy().tolist()]
    evgenii_list = [k for k in evgenii.to_numpy().tolist()]
    # for i, (old, new) in enumerate(zip(evgenii.to_numpy(), david.to_numpy())):
    #     if not (old.astype(str) == new.astype(str)).all():
    #         match = False
    #         a=5
    print(f'{idx} - {match}')
    # article_id, tbl_idx = tbl['wikiId'].split('-')
    # old_page = get_revid(article_id)
    # if old_page is None or  old_page['url'] is None:
    #     print(f"Article {article_id} not found, skipping.")
    #     continue
    # df = parse_single_page(old_page['url'], redis_insertion=False, tbl_idx=int(tbl_idx))
    # print(str(tbl['columns'])+ " [" + str((tbl['numDataRows'],tbl['numCols'])) + "]")
    # print()
    # if df is not None and df[0] is not None:
    #     df[0].to_csv(f'scraped_tbls/{idx}.csv', index=False)
    b=5


b=5