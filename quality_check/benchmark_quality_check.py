from wikiparser import WikiTableParser
from utils.utils import get_revid
import json
import pandas as pd
import os
start_indx = 0
success = 0
fail = 0
parser = WikiTableParser()
with open('benchmark_tbls.json') as f:
    tbls = json.load(f)
for idx, tbl in tbls.items():
    if int(idx) < start_indx:
        continue
    print(f"Working on index: {idx}")
    if not os.path.isfile(f'tables/{tbl["file"]}'):
        continue
    evgenii = pd.read_csv(f'tables/{tbl["file"]}')
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
        fail += 1
        continue
    success += 1
    print(f"Success: {success}, Failed: {fail} \n")




# Success: 78, Failed: 12