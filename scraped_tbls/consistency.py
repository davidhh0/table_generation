from datetime import datetime

import requests
from bs4 import BeautifulSoup
from personal.utils import get_revid
from datetime import datetime, timedelta
from pandas import DataFrame, read_html
from io import StringIO

def is_consistent(url, tbl_idx):
    page_id = url.split('/')[-1]
    previous_id = get_revid(page_id, by='titles', starting=datetime.today() - timedelta(days=365))
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
                    print(url, 'misMatchValues-' + old_df.columns[list(old == new).index(False)] + '-' + str(idx))
                    print(f"Mismatch in {url} at {old_df.columns[list(old == new).index(False)]} {idx}")
                    return 'misMatchValues-' + old_df.columns[list(old == new).index(False)] + '-' + str(idx), None, None
                except Exception as e:
                    print(f"Error in mismatch values for {url}: {e}")
                    return 'Error_mismatch_values', None, None

    print(f'Match in {url}')
    return 'match', previous_id['page_id'], previous_id['title']

