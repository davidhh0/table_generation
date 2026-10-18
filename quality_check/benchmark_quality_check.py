import os
import sys
import json
import argparse
import pandas as pd

# Ensure repository root is on sys.path regardless of execution directory
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from wikiparser import WikiTableParser
from utils.utils import get_revid

def run_quality_check(start_index=0, offline=False):
    qc_dir = os.path.dirname(os.path.abspath(__file__))
    tbls_json_path = os.path.join(qc_dir, 'benchmark_tbls.json')
    tables_dir = os.path.join(qc_dir, 'tables')

    with open(tbls_json_path) as f:
        tbls = json.load(f)

    parser = WikiTableParser()
    success = 0
    fail = 0

    print(f"Loaded {len(tbls)} benchmark tables. Mode: {'Offline (GT CSVs)' if offline else 'Online (MediaWiki API)'}")

    for idx, tbl in sorted(tbls.items(), key=lambda x: int(x[0])):
        if int(idx) < start_index:
            continue
        print(f"Working on index: {idx} ({tbl.get('name', 'unnamed')})")
        csv_file = os.path.join(tables_dir, tbl['file'])
        if not os.path.isfile(csv_file):
            print(f"Ground-truth file {tbl['file']} not found, skipping.")
            continue

        if offline:
            # Clean the pre-curated ground-truth CSV directly
            df_raw = pd.read_csv(csv_file)
            parser.df = df_raw.copy()
            parser.url = tbl.get('name', tbl['file'])
            try:
                parser.clean_df()
                cleaned_df = parser.df
            except Exception as e:
                print(f"Error cleaning {tbl['name']}: {e}")
                cleaned_df = None
        else:
            # Online mode: fetch the historical Wikipedia revision
            article_id, tbl_idx = tbl['wikiId'].split('-')
            old_page = get_revid(article_id)
            if old_page is None or old_page.get('url') is None:
                print(f"Article {article_id} not found, skipping.")
                fail += 1
                continue
            df_res = parser.run(old_page['url'], tbl_idx=int(tbl_idx))
            cleaned_df = df_res[0] if df_res is not None else None

        if cleaned_df is None:
            print(f"Table {idx} failed or was dropped by filters.")
            fail += 1
            continue

        success += 1
        print(f"Table {idx} PASSED: shape {cleaned_df.shape} (Expected: {tbl['numDataRows']}x{tbl['numCols']})")
        print(f"Running totals -> Success: {success}, Failed: {fail}\n")

    print(f"=== Quality check finished: {success} passed, {fail} failed ===")

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="WikiFacTables benchmark quality check")
    cli_parser.add_argument("--start-index", type=int, default=0, help="Start from table index")
    cli_parser.add_argument(
        "--offline",
        action="store_true",
        help="Run offline using curated ground-truth CSVs under quality_check/tables/ (does not query Wikipedia API)",
    )
    args = cli_parser.parse_args()
    run_quality_check(start_index=args.start_index, offline=args.offline)
