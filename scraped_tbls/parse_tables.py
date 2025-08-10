import multiprocessing
import threading
from datetime import datetime

import math
import requests
import yaml
from bs4 import BeautifulSoup
from io import StringIO
import redis
import json
import re
from pandas import read_html, MultiIndex, DataFrame, isna, to_numeric, read_csv
from numpy import array_split
from collections import Counter
from price_parser import Price

unicode_dict = {'\xa0': ' '}
LINK_CHAR = '∟'
STYLE_CHAR = '∫'
DROP_COLUMNS = [
    "Notes",
    "Rank",
    "Reference",
    "Ref.",
    "References",
    "Notes",
    "Note",
    "Remarks",
    "Remarks/Notes",
    "Remarks/Note",
    "Remarks/Ref.",
    "Remarks/Ref",
    "Ref.",
    "Ref",
    "Reference/Note",
    "No.",
]
NULL_CHARS = ["—", '-', 'none', '–']
OK_CHARS = ["–", "—", "§", LINK_CHAR, STYLE_CHAR]

r_drop_table = redis.Redis(host='localhost', port=6379, db=0)
r_drop_row = redis.Redis(host='localhost', port=6379, db=1)
r_drop_column = redis.Redis(host='localhost', port=6379, db=2)
r_transform_column = redis.Redis(host='localhost', port=6379, db=3)
r_transform_row = redis.Redis(host='localhost', port=6379, db=4)
r_all_tables = redis.Redis(host='localhost', port=6379, db=5)
r_failed_and_passed = redis.Redis(host='localhost', port=6379, db=7)


class CustomFormatter:
    BLUE = '\033[94m'
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    ENDC = '\033[0m'
    format = " %(levelname)s - %(message)s"

    def debug(self, msg):
        print('\033[94m' + msg + '\033[0m')

    def info(self, msg):
        print('\x1b[33;20m' + msg + '\033[0m')

    def error(self, msg):
        print('\x1b[31;20m' + msg + '\033[0m')

    def ok(self, msg):
        print('\033[92m' + msg + '\033[0m')


def get_type(obj):
    if isinstance(obj, str):
        return 'str'
    if isna(obj):
        return 'nan'
    return 'num'


logger = CustomFormatter()

known_formats = [
    "%d %b",  # 7 May
    "%d %B",  # 7 May (full month)
    "%d/%m/%Y",  # 07/05/2025
    "%Y-%m-%d",  # 2025-05-07
    "%b %d, %Y",  # May 7, 2025
    "%d-%m-%Y",  # 07-05-2025
    "%m/%d/%Y",  # 05/07/2025
    "%d %b %Y",  # 7 May 2025
    "%b %Y",  # May 2025
    "%B %Y",  # June 2025
    "%B %d, %Y",  # June 7, 2025
    "%B %d",  # June 7
    "%b %d",  # June 7
    "%d %b %Y",  # 7 May 2025
    "%d %B %Y",  # 7 May 2025
]


def detect_date_format(date_str):
    for fmt in known_formats:
        try:
            datetime.strptime(date_str, fmt)
            return fmt
        except ValueError:
            continue
    return None


# Example formats to check


class WikiTableParser:
    def __init__(self):
        with open('config.yaml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
            self.url = None
            self.df = None

    def drop_irrelevant_columns(self):
        columns_to_drop = self.cfg['drop_columns_blacklist']
        drop_column = [any(j == k for j in columns_to_drop) or k == '' for k in self.df.columns if
                       not isinstance(k, (int, float))]
        drop_column_idx = [self.df.columns[j] for j in range(len(drop_column)) if drop_column[j]]
        if any(drop_column) and drop_column.index(True) in [0, len(self.df.columns) - 1]:
            self.df.drop(drop_column_idx, axis=1, inplace=True)
            logger.debug(
                f"Dropping column {drop_column_idx} as it is irrelevant ({self.url})"
            )

    def different_style(self, _df):
        for col in _df.columns:
            if _df[col].dtype == 'object':
                if any(
                        [
                            re.search(f'{STYLE_CHAR}(.+){STYLE_CHAR}', k)
                            for k in df[col].tolist()
                            if k and not isinstance(k, (float, int))
                        ]
                ):
                    _df[col] = _df[col].apply(clean_style_string)
                    logger.debug(f"Transforming rows in {self.url} as it has different style")
                _df[col] = df[col].apply(
                    lambda x: x.replace(f'{LINK_CHAR}', '') if isinstance(x, str) else x
                )
                _df[col] = df[col].apply(
                    lambda x: (x.replace(f'{STYLE_CHAR}', '') if isinstance(x, str) else x)
                )

    def run(self, url):
        def get_previous_paragraph(tbl):
            previous_string = tbl.find_previous(name='p').get_text(strip=True)
            return previous_string

        def get_table_from_html(tbls):
            return sorted(tbls, key=lambda tbl: tbl['df'].count(axis=1).sum(), reverse=True)

        page = requests.get(url)
        page_content = BeautifulSoup(page.text, 'html.parser')
        tbls = page_content.select('table[class*=wikitable]')
        for tbl in tbls:
            for a_class in tbl.find_all(['a', 'abbr']):
                list_of_strings = list(a_class.strings)
                if len(list_of_strings) == 0:
                    continue
                first_string = list_of_strings[0]
                if len(list_of_strings) == 1:
                    first_string.replace_with(f'{LINK_CHAR}{first_string.text}{LINK_CHAR}')
                    continue
                last_string = list_of_strings[-1]
                first_string.replace_with(f'{LINK_CHAR}{first_string.text}')
                last_string.replace_with(f'{last_string.text}{LINK_CHAR}')
            for style in tbl.find_all('span', style=True):
                style.replace_with(f'{STYLE_CHAR}{style.text}{STYLE_CHAR}')
        try:
            tbls = [
                {
                    'df': read_html(StringIO(k.__str__()))[0],
                    'paragraph': get_previous_paragraph(k),
                    'raw_html': k,
                }
                for k in tbls
            ]
        except:
            return None
        fetched_tbls = get_table_from_html([k for k in tbls])
        min_rows, min_cols = self.cfg['minimum_table_size']['rows'], self.cfg['minimum_table_size']['columns']
        for idx, fetched_tbl in enumerate(fetched_tbls):
            if fetched_tbl['df'].shape[0] < min_rows or fetched_tbl['df'].shape[1] < min_cols:
                logger.error(f"Dropping {url} as it has less than {min_rows} rows or {min_cols} columns")
                continue
            if isinstance(fetched_tbl['df'].columns, MultiIndex):
                logger.error(f"Dropping {url} as it has MultiIndex columns")
                continue
            if fetched_tbl['df'].columns.tolist() == list(range(len(fetched_tbl['df'].columns))):
                logger.error(f"Dropping {url} as it has empty header column names")
                continue

            if any([re.search('\w+\.\d', k) for k in fetched_tbl['df'].columns.tolist()]):
                logger.error(f"Dropping {url} as it has column names with numbers")

                continue
            self.df = fetched_tbl['df']
            self.url = url
            self.clean_df()
            if self.df is not None:
                logger.ok(
                    f'Passed: {url} ({[f"{k}:{self.df[k].dtype}" for k in self.df.columns]}) [{fetched_tbl["df"].shape}] ({idx})')
            else:
                logger.error(f"Failed {url} ({fetched_tbl['df'].columns}) [{fetched_tbl['df'].shape}]")
            return self.df, idx
        return None

    def apply_rules_columns(self):
        pass

    def apply_rules_rows(self):
        # Repeating header columns
        header_columns = self.df.columns.tolist()
        rows_to_delete = [
            k[0] for k in enumerate(self.df.to_numpy()) if all(k[1] == header_columns)
        ]
        if rows_to_delete:
            self.df.drop(rows_to_delete, inplace=True)
            logger.debug(
                f"Dropping row {rows_to_delete} as it is repeating header columns ({self.url})"
            )
        # Row as divider or all NaN
        idx_to_delete = [
            k[0] for k in enumerate(self.df.to_numpy()) if all(k[1][0] == j for j in k[1])
        ]
        if idx_to_delete:
            self.df.drop(idx_to_delete, inplace=True)
            logger.debug(
                f"Dropping row {idx_to_delete} as Row as divider or all NaN ({self.url})"
            )

    def apply_cell_rules(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].apply(clean_string)

        if any(
                [
                    re.search("[\(\[].*?[\)\]]", k)  # removes text in brackets
                    for k in self.df[col].tolist()
                    if k and not isinstance(k, (float, int))
                ]
        ):
            self.df[col] = df[col].apply(
                lambda x: (
                    re.sub("[\(\[].*?[\)\]]", '', x).strip()
                    if isinstance(x, str)
                    else x
                )
            )
            logger.debug(f"Removing data inside brackets {self.url} ({col} column)")
        numbers_counter = Counter([get_type(k) for k in self.df[col].apply(clean_string).tolist()[1:-1]])
        counter_len = sum(numbers_counter.values())
        if ((numbers_counter['num'] != counter_len) and (numbers_counter['str'] != counter_len) and
                (numbers_counter['num'] > numbers_counter['str'] and numbers_counter['str'] > 2)
        ):
            logger.error(f"Dropping {self.url} as it has mismatch numbers and strings ({col} column)")
            return None

    def clean_df(self):
        # Check if the DataFrame has empty column headers
        if any([isna(k) or 'Unnamed:' in k for k in self.df.columns.tolist() if not isinstance(k, (int, float))]):
            logger.error(f"Dropping {self.url} as it has empty column header")
            return None
        self.drop_irrelevant_columns()
        self.apply_rules_rows()
        self.apply_rules_columns()
        self.apply_cell_rules()

        return self.df


def clean_style_string(string):
    if isinstance(string, (float, int)) or string is None:
        return string
    style_string = re.search(f"{STYLE_CHAR}.+{STYLE_CHAR}", string).group()
    if style_string == string:
        return string
    return re.sub(f"{STYLE_CHAR}.+{STYLE_CHAR}", '', string)


def try_cast(obj):
    import dateutil.parser as parser

    if isinstance(obj, float) and isna(obj):
        return None

    if str(obj) in NULL_CHARS:
        return None

    try:
        return int(obj.__str__().replace(',', '').replace(' ', ''))
    except ValueError:
        pass

    try:
        return float(obj.__str__().replace(',', '').replace(' ', ''))
    except ValueError:
        pass

    # if len(obj.__str__()) < 10:
    #     return obj.__str__()
    if not any(sep in obj.__str__() for sep in [' ', '/', '-', '–', '—']):
        return obj.__str__()
    if ':' in obj.__str__() or any([k.isalpha() for k in obj]):
        return obj.__str__()
    try:
        date_obj = parser.parse(obj, default=datetime(9999, 1, 1))
        date_format = detect_date_format(obj)
        if date_format is None:
            return obj.__str__()
            b = 5
            raise "Error in detecting date format"
        if date_obj.year == 9999:
            return date_obj.strftime(date_format)
        return date_obj.strftime(date_format)  # parser.parse(obj)
    except (parser.ParserError, OverflowError):
        pass

    return obj.__str__()


def is_ascii(s):
    if s is None:
        return True
    if isinstance(s, int):
        return True
    if isinstance(s, float):
        return True
    return any(ord(c) < 128 or c in OK_CHARS for c in s)


def clean_string(string):
    if isinstance(string, str):
        for unicode, value in unicode_dict.items():
            string = string.replace(unicode, value)
        return try_cast(string.strip(f' {LINK_CHAR}'))
    return string


def clean_df(df, url, redis_insertion, idx):
    if any([isna(k) or 'Unnamed:' in k for k in df.columns.tolist() if not isinstance(k, (int, float))]):
        logger.error(f"Dropping {url} as it has empty column header")
        if redis_insertion:
            r_drop_table.set(f'{url}@{idx}', "Empty column header")
        return None
    apply_rules_rows(df, url, redis_insertion, idx)
    drop_irrelevant_columns(df, url, redis_insertion, idx)
    # apply_rules_columns(df, url, redis_insertion, idx)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(clean_string)
        if any(
                [
                    re.search("[\(\[].*?[\)\]]", k)  # removes text in brackets
                    for k in df[col].tolist()
                    if k and not isinstance(k, (float, int))
                ]
        ):
            df[col] = df[col].apply(
                lambda x: (
                    re.sub("[\(\[].*?[\)\]]", '', x).strip()
                    if isinstance(x, str)
                    else x
                )
            )
            logger.debug(f"Removing data inside brackets {url} ({col} column)")
            if redis_insertion:
                r_transform_row.lpush(
                    f'{url}@{idx}', *[json.dumps({'col': col, 'reason': 'Data in brackets'})]
                )
        numbers_counter = Counter([get_type(k) for k in df[col].apply(clean_string).tolist()[1:-1]])
        counter_len = sum(numbers_counter.values())
        if ((numbers_counter['num'] != counter_len) and (numbers_counter['str'] != counter_len) and
                (numbers_counter['num'] > numbers_counter['str'] and numbers_counter['str'] > 2)
        ):
            logger.error(f"Dropping {url} as it has mismatch numbers and strings ({col} column)")
            if redis_insertion:
                r_drop_table.set(f'{url}@{idx}', "Mismatch numbers and strings")
            return None
    # Don't remove brackets from column names, as it maybe be informative
    for col in df.columns:
        if re.search("[\[].*?[\]]", clean_string(col).__str__()):
            if redis_insertion:
                r_transform_column.lpush(
                    f'{url}@{idx}',
                    *[json.dumps({'col': col, 'reason': 'Data in brackets'})],
                )
            logger.debug(f"Removing data inside brackets {url} ({col} column)")
            df.rename(columns={col: re.sub("[\[].*?[\]]", '', clean_string(col).__str__()).strip()}, inplace=True)
    if any([isinstance(df[k], DataFrame) for k in df.columns]):
        # Same column name
        logger.error(f"Dropping {url} as it has Same column names")
        if redis_insertion:
            r_drop_table.set(f'{url}@{idx}', "Same column names")
        return None
    original_columns = df.columns.tolist()
    apply_rules_rows(df, url, redis_insertion, idx)
    if apply_rules_columns(df, url, redis_insertion, idx, original_columns) is None:
        return None
    if df.shape[0] < 8 or df.shape[1] < 2:
        logger.error(f"Dropping {url} as it has less than 8 rows or 2 columns")
        if redis_insertion:
            r_drop_table.set(f'{url}@{idx}', "Table has less than 8 rows or 2 columns")
        return None
    apply_rules_rows(df, url, redis_insertion, idx)
    for col in df.columns:
        if df[col].dtype == 'object':
            if any(
                    [
                        re.search(f'{STYLE_CHAR}(.+){STYLE_CHAR}', k)
                        for k in df[col].tolist()
                        if k and not isinstance(k, (float, int))
                    ]
            ):
                df[col] = df[col].apply(clean_style_string)
                if redis_insertion:
                    r_transform_row.lpush(
                        f'{url}@{idx}',
                        *[json.dumps({'col': col, 'reason': 'Different style'})],
                    )
                logger.debug(f"Transforming rows in {url} as it has different style")
            df[col] = df[col].apply(
                lambda x: x.replace(f'{LINK_CHAR}', '') if isinstance(x, str) else x
            )
            df[col] = df[col].apply(
                lambda x: (x.replace(f'{STYLE_CHAR}', '') if isinstance(x, str) else x)
            )
    df = read_csv(StringIO(df.to_csv().strip(',')))
    df = aggr_row(df, url, redis_insertion, idx)
    return df


def aggr_row(df, url, redis_insertion, idx):
    df_without_last_row = read_csv(StringIO(df[:-1].to_csv().strip(',')))
    for col in [df.columns[0], df.columns[-1]]:
        if (
                df[col].dtype != df_without_last_row[col].dtype
                and df[col].iloc[-1] not in NULL_CHARS
        ):
            try:
                logger.debug(
                    f'Last row data type: {df_without_last_row[col].dtype} - {df[col].dtype}'
                )
                if redis_insertion:
                    r_drop_row.lpush(
                        f'{url}@{idx}',
                        *[
                            json.dumps(
                                {
                                    'row': -1,
                                    'reason': 'Last row different data type',
                                }
                            )
                        ],
                    )
                df = df_without_last_row.copy(deep=True)
            except:
                pass

    df_without_first_row = read_csv(StringIO(df[1:].to_csv().strip(',')))
    for col in [df.columns[0], df.columns[-1]]:
        if (
                df[col].dtype != df_without_first_row[col].dtype
                and df[col].iloc[0] not in NULL_CHARS
        ):
            try:
                logger.debug(
                    f'First row data type: {df_without_first_row[col].dtype} - {df[col].dtype}'
                )
                if redis_insertion:
                    r_drop_row.lpush(
                        f'{url}@{idx}',
                        *[
                            json.dumps(
                                {
                                    'row': 1,
                                    'reason': 'First row different data type',
                                }
                            )
                        ],
                    )
                df = df_without_first_row.copy(deep=True)
            except:
                pass
    return df


def apply_rules_rows(df, url, redis_insertion, idx):
    # Repeating header columns
    header_columns = df.columns.tolist()
    rows_to_delete = [
        k[0] for k in enumerate(df.to_numpy()) if all(k[1] == header_columns)
    ]
    if rows_to_delete:
        df.drop(rows_to_delete, inplace=True)
        if redis_insertion:
            r_drop_row.lpush(
                f'{url}@{idx}',
                *[
                    json.dumps(
                        {
                            'row': rows_to_delete,
                            'reason': 'Repeating header columns',
                        }
                    )
                ],
            )
        logger.debug(
            f"Dropping row {rows_to_delete} as it is repeating header columns ({url})"
        )
    idx_to_delete = [
        k[0] for k in enumerate(df.to_numpy()) if all(k[1][0] == j for j in k[1])
    ]
    if idx_to_delete:
        df.drop(idx_to_delete, inplace=True)
        if redis_insertion:
            r_drop_row.lpush(
                f'{url}@{idx}',
                *[
                    json.dumps(
                        {
                            'row': idx_to_delete,
                            'reason': 'Row as divider or all NaN',
                        }
                    )
                ],
            )
        logger.debug(
            f"Dropping row {idx_to_delete} as Row as divider or all NaN ({url})"
        )


def apply_rules_columns(df, url, redis_insertion, idx, original_columns=None):
    # # Data in [] or ()
    # for col in df.columns:
    #     if re.search(r'\[(.+)]', col):
    #         new_column_name = re.sub(r'\[(.+)]', '', col)
    #         df.rename(columns={col: new_column_name}, inplace=True)
    #         if redis_insertion:
    #             r_transform_column.lpush(
    #                 url, *[json.dumps({'col': col, 'reason': 'Data in []'})]
    #             )
    #         logger.debug(
    #             f"Renamed column {col} to {new_column_name} as it has data in [] ({url})"
    #         )
    # for col in df.columns:
    #     if re.search(f"[(\[]{LINK_CHAR}.+{LINK_CHAR}[])]", col):
    #         new_column_name = re.sub(f"{LINK_CHAR}.+{LINK_CHAR}", '', col)
    #         df.rename(columns={col: new_column_name}, inplace=True)
    #         if redis_insertion:
    #             r_transform_column.lpush(
    #                 url,
    #                 *[json.dumps({'col': col, 'reason': 'Link in parenthesis'})],
    #             )
    #         logger.debug(
    #             f"Renamed column {col} to {new_column_name} as it has Link in parenthesis ({url})"
    #         )

    # Non-ASCII characters
    for idx, col in enumerate([{'col_name': k, 'values': df[k].tolist()} for k in df.columns]):

        counter = Counter([is_ascii(k) for k in col['values']])
        if counter[False] > counter[True]:
            df.drop(col['col_name'], axis=1, inplace=True)
            if redis_insertion:
                r_drop_column.lpush(
                    f'{url}@{idx}',
                    *[
                        json.dumps(
                            {
                                'col': col['col_name'],
                                'reason': 'Non-ASCII characters',
                            }
                        )
                    ],
                )
            logger.debug(
                f"Dropping column {col['col_name']} as it has non-ascii characters ({url})"
            )
            if original_columns.index(col['col_name']) not in [0, len(original_columns) - 1]:
                logger.error(f"Dropping {url} as it has non-ascii characters")
                if redis_insertion:
                    r_drop_table.set(f'{url}@{idx}', "Non-ASCII characters")
                return None

    # Irrelevant columns such as Rank, Notes, Reference...
    # drop_column = [any(j in k for j in DROP_COLUMNS) or k == '' for k in df.columns if not isinstance(k, (int,float) )]
    drop_column = [any(j == k for j in DROP_COLUMNS) or k == '' for k in df.columns if not isinstance(k, (int, float))]
    drop_column_idx = [df.columns[j] for j in range(len(drop_column)) if drop_column[j]]
    if any(drop_column) and drop_column.index(True) in [0, len(df.columns) - 1]:
        df.drop(drop_column_idx, axis=1, inplace=True)
        logger.debug(
            f"Dropping column {drop_column_idx} as it is irrelevant ({url})"
        )
        if redis_insertion:
            r_drop_column.lpush(
                f'{url}@{idx}',
                *[
                    json.dumps(
                        {
                            'col': ','.join(drop_column_idx),
                            'reason': 'Irrelevant column',
                        }
                    )
                ],
            )

    # Long text
    for col in df.columns:
        mean = df[col].map(lambda x: len(str(x))).mean()
        _max = df[col].map(lambda x: len(str(x))).max()
        _min = df[col].map(lambda x: len(str(x))).min()
        if _max > 110:
            df.drop(col, axis=1, inplace=True)
            if redis_insertion:
                r_drop_column.lpush(
                    f'{url}@{idx}', *[json.dumps({'col': col, 'reason': 'Long text'})]
                )
            logger.debug(f"Dropping column {col} as it has long text ({url}) ({col})")
            if original_columns.index(col) not in [0, len(original_columns) - 1]:
                logger.error(f"Dropping {url} as it has long text")
                if redis_insertion:
                    r_drop_table.set(f'{url}@{idx}', "Long text")
                return None

    return df


def parse_single_page(url, article_name=None, redis_insertion=True, tbl_idx=None):
    def get_previous_paragraph(tbl):
        previous_string = tbl.find_previous(name='p').get_text(strip=True)
        return previous_string

    def get_table_from_html(tbls):
        return sorted(tbls, key=lambda tbl: tbl['df'].count(axis=1).sum(), reverse=True)

    page = requests.get(url)
    page_content = BeautifulSoup(page.text, 'html.parser')
    tbls = page_content.select('table[class*=wikitable]')
    b = 5
    for tbl in tbls:
        continue
        for a_class in tbl.find_all(['a', 'abbr']):
            list_of_strings = list(a_class.strings)
            if len(list_of_strings) == 0:
                continue
            first_string = list_of_strings[0]
            if len(list_of_strings) == 1:
                first_string.replace_with(f'{LINK_CHAR}{first_string.text}{LINK_CHAR}')
                continue
            last_string = list_of_strings[-1]
            first_string.replace_with(f'{LINK_CHAR}{first_string.text}')
            last_string.replace_with(f'{last_string.text}{LINK_CHAR}')
        for style in tbl.find_all('span', style=True):
            style.replace_with(f'{STYLE_CHAR}{style.text}{STYLE_CHAR}')
    try:
        tbls = [
            {
                'df': read_html(StringIO(k.__str__()))[0],
                'paragraph': get_previous_paragraph(k),
                'raw_html': k,
            }
            for k in tbls
        ]
    except:
        return None

    b = 5
    if tbl_idx is not None:
        fetched_tbls = [tbls[tbl_idx - 1]]
    else:
        fetched_tbls = get_table_from_html([k for k in tbls])
    for idx, fetched_tbl in enumerate(fetched_tbls):
        if fetched_tbl['df'].shape[0] < 8 or fetched_tbl['df'].shape[1] < 2:
            logger.error(f"Dropping {url} as it has less than 8 rows or 2 columns")
            if redis_insertion:
                r_drop_table.set(f'{url}@{idx}', "Table has less than 8 rows or 2 columns")
            continue
        if isinstance(fetched_tbl['df'].columns, MultiIndex):
            logger.error(f"Dropping {url} as it has MultiIndex columns")
            if redis_insertion:
                r_drop_table.set(f'{url}@{idx}', "MultiIndex columns")
            continue
        if fetched_tbl['df'].columns.tolist() == list(range(len(fetched_tbl['df'].columns))):
            logger.error(f"Dropping {url} as it has empty header column names")
            if redis_insertion:
                r_drop_table.set(f'{url}@{idx}', "Default column names")
            continue

        if any([re.search('\w+\.\d', k) for k in fetched_tbl['df'].columns.tolist()]):
            logger.error(f"Dropping {url} as it has column names with numbers")
            if redis_insertion:
                r_drop_table.set(f'{url}@{idx}', "Column names with numbers")
            continue

        # if any([k for k in fetched_tbl['raw_html'].find_all('td') if  int(k.attrs.get('rowspan',1)) > 1]):
        #     logger.error(f"Dropping {url} as it has rowspan in table")
        #     if redis_insertion:
        #         r_drop_table.set(f'{url}@{idx}', "Rowspan in table")
        #     continue
        df = clean_df(fetched_tbl['df'], url, redis_insertion, idx)
        if df is not None:
            logger.ok(
                f'Passed: {url} ({[f"{k}:{df[k].dtype}" for k in df.columns]}) [{fetched_tbl["df"].shape}] ({idx})')
        else:
            logger.error(f"Failed {url} ({fetched_tbl['df'].columns}) [{fetched_tbl['df'].shape}]")
        return df, idx
    return None


def worker(url_list):
    """Worker procedure"""
    for url in url_list:
        if r_failed_and_passed.get(url + b'@0') != b'PASSED':
            logger.info(f"Already parsed {url}")
            continue
        df = parse_single_page(url.decode(), url.decode().split('/')[-1], redis_insertion=False)
        # if df is None:
        #     r_failed_and_passed.set(f'{url.decode()}@-1', 'FAILED')
        #     continue
        # if df[0] is None:
        #     r_failed_and_passed.set(f'{url.decode()}@{df[1]}', 'FAILED')
        #     continue
        # else:
        #     r_failed_and_passed.set(f'{url.decode()}@{df[1]}', 'PASSED')
        #     continue


if __name__ == "__main__":
    obj = WikiTableParser()
    obj.run('https://en.wikipedia.org/wiki/List_of_cities_in_Yamanashi_Prefecture_by_population')
    df = parse_single_page('https://en.wikipedia.org/wiki/List_of_cities_in_Yamanashi_Prefecture_by_population',
                           '2023–24 PGA Tour of Australasia', redis_insertion=False)
    exit(1)
    all_urls = r_all_tables.keys()
    worker(all_urls)
    # jobs = []  # list of jobs
    jobs_num = 10  # number of workers
    list_divided = array_split(all_urls, jobs_num)
    for url_chunk in list_divided:
        # Declare a new process and pass arguments to it
        p1 = multiprocessing.Process(target=worker, args=(url_chunk,))
        jobs.append(p1)
        p1.start()  # starting workers
    for job in jobs:
        job.join()
