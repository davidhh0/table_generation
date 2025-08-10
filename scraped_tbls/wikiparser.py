import requests
import yaml
from bs4 import BeautifulSoup
from io import StringIO
import re
from pandas import read_html, MultiIndex, DataFrame, isna, to_numeric, read_csv
from collections import Counter
from datetime import datetime
import numpy as np

unicode_dict = {'\xa0': ' '}


def get_type(obj, is_dtype=False):
    if is_dtype:
        if obj == 'object':
            return 'str'
        if obj == 'float64':
            return 'num'
        if obj == 'int64':
            return 'num'
        if obj == 'datetime64[ns]':
            return 'str'
    if isinstance(obj, str):
        return 'str'
    if isna(obj):
        return 'nan'
    return 'num'


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


class WikiTableParser:
    def __init__(self):
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # with open(os.path.join(dir_path, 'config.yaml'), 'r') as f:
        with open(os.path.join(dir_path, 'config.yaml'), 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
            self.link_char = self.cfg['special_chars']['link_char']
            self.style_char = self.cfg['special_chars']['style_char']
            self.nulls_char = self.cfg['null_chars']
            self.url = None
            self.df:DataFrame = DataFrame()
            self.logger = CustomFormatter()
            self.date_cols = {}
            self.error_msg = 'SUCCESS'

    def is_ascii(self, s):
        if s is None:
            return True
        if isinstance(s, int):
            return True
        if isinstance(s, float):
            return True
        if s in self.nulls_char:
            return True
        if s in [self.style_char, self.link_char]:
            return True
        return all(ord(c) < 128 or c in [self.style_char, self.nulls_char] for c in s)


    def is_null(self, obj):
        if isinstance(obj, (float, int)):
            return isna(obj)
        if isinstance(obj, str):
            return obj in self.nulls_char or obj == ''
        return False

    def clean_style_string(self, string):
        if isinstance(string, (float, int)) or string is None:
            return string
        style_string = re.search(
            f"{self.style_char}.+{self.style_char}", string
        ).group()
        if style_string == string:
            return string
        return re.sub(f"{self.style_char}.+{self.style_char}", '', string)

    def drop_irrelevant_columns(self):
        if self.df is None:
            return None
        columns_to_drop = self.cfg['drop_columns_blacklist']
        drop_column = [
            any(j == k for j in columns_to_drop) or k == ''
            for k in self.df.columns
            if not isinstance(k, (int, float))
        ]
        drop_column_idx = [
            self.df.columns[j] for j in range(len(drop_column)) if drop_column[j]
        ]
        if any(drop_column) and drop_column.index(True) in [
            0,
            len(self.df.columns) - 1,
        ]:
            self.df.drop(drop_column_idx, axis=1, inplace=True)
            self.logger.debug(
                f"Dropping column {drop_column_idx} as it is irrelevant ({self.url})"
            )

    def detect_date_format(self, date_str):
        for fmt in self.cfg['known_dates_format']:
            try:
                datetime.strptime(date_str, fmt)
                return fmt
            except ValueError:
                continue
        return None

    def try_cast(self, obj, col=None):
        import dateutil.parser as parser
        if isinstance(obj, float) and isna(obj):
            return None

        if str(obj) in self.nulls_char:
            return None

        if isinstance(obj, (int, float)):
            return obj

        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)

        obj = obj.replace('−','-').replace('–','-').replace('—','-').replace('"','').strip()
        try:
            return int(obj.__str__().replace(',', '').replace(' ', ''))
        except ValueError:
            pass

        try:
            return float(obj.__str__().replace(',', '').replace(' ', ''))
        except ValueError:
            pass
        if not any(sep in obj.__str__() for sep in [' ', '/', '-', '–', '—']):
            return obj.__str__()
        if ':' in obj.__str__():
            return obj.__str__()
        try:
            date_obj = parser.parse(obj, default=datetime(9999, 1, 1))
            date_format = self.detect_date_format(obj)
            if date_format is None:
                return obj.__str__()
                b = 5
                raise "Error in detecting date format"
            if col is not None:
                self.date_cols[col] = date_format
            if date_obj.year == 9999:
                return date_obj.strftime(date_format)
            return date_obj.strftime(date_format)  # parser.parse(obj)
        except (parser.ParserError, OverflowError):
            pass

        return obj.__str__()

    def clean_string(self, string, col=None):
        if isinstance(string, str):
            for unicode, value in unicode_dict.items():
                string = string.replace(unicode, value)
            string = string.replace(self.style_char, '')
            return self.try_cast(string.strip(f' {self.link_char}'), col)
        return string

    def different_style(self, _df):
        for col in _df.columns:
            if _df[col].dtype == 'object':
                if any(
                    [
                        re.search(f'{self.style_char}(.+){self.style_char}', k)
                        for k in self.df[col].tolist()
                        if k and not isinstance(k, (float, int))
                    ]
                ):
                    _df[col] = _df[col].apply(self.clean_style_string)
                    self.logger.debug(
                        f"Transforming rows in {self.url} as it has different style"
                    )
                _df[col] = self.df[col].apply(
                    lambda x: (
                        x.replace(f'{self.link_char}', '') if isinstance(x, str) else x
                    )
                )
                _df[col] = self.df[col].apply(
                    lambda x: (
                        x.replace(f'{self.style_char}', '') if isinstance(x, str) else x
                    )
                )


    def run(self, url, tbl_id=None, tbl_idx=None):
        def get_previous_paragraph(tbl):
            if tbl.find_previous(name='p'):
                p_tags = []
                for prev in tbl.find_all_previous():
                    if prev.name == 'div':
                        break
                    if prev.name == 'p':
                        p_tags.append(prev)
                previous_string = "\n".join([k.get_text(strip=True) for k in p_tags[::-1] ]) # tbl.find_previous(name='p').get_text(strip=True)
                return previous_string
            return None

        def get_table_from_html(tbls):
            return sorted(
                tbls, key=lambda tbl: tbl['df'].count(axis=1).sum(), reverse=True
            )

        page = requests.get(url)
        page_content = BeautifulSoup(page.text, 'html.parser')
        tbls = page_content.select('table[class*=wikitable]')
        self.date_cols.clear()
        for tbl in tbls:
            for a_class in tbl.find_all(['a', 'abbr']):
                list_of_strings = list(a_class.strings)
                if len(list_of_strings) == 0:
                    continue
                first_string = list_of_strings[0]
                # if len(list_of_strings) == 1:
                #     first_string.replace_with(
                #         f'{self.link_char}{first_string.text}{self.link_char}'
                #     )
                #     continue
                # last_string = list_of_strings[-1]
                # first_string.replace_with(f'{self.link_char}{first_string.text}')
                # last_string.replace_with(f'{last_string.text}{self.link_char}')
            for style in tbl.find_all('span', style=True):
                if 'none' in style.attrs.get('style'):
                    style.replace_with("")
                    continue
                style.replace_with(f'{self.style_char}{style.text}{self.style_char}')
        try:
            tbls = [
                {
                    'df': read_html(StringIO(k.__str__()))[0],
                    'paragraph': get_previous_paragraph(k),
                    'raw_html': k,
                }
                for k in tbls
            ]
        except Exception as e:
            self.logger.error(f"Error reading tables from {url}: {str(e)}")
            return None, None, 'general_error', None

        if tbl_idx is not None:
            if len(tbls) < tbl_idx or tbl_idx <= 0:
                self.logger.error(
                    f"Table index {tbl_idx} out of range for {url}. Total tables: {len(tbls)}"
                )
                return None, None, 'general_error', None
            fetched_tbls = [tbls[tbl_idx - 1]]
        else:
            fetched_tbls = get_table_from_html([k for k in tbls])
        min_rows, min_cols = (
            self.cfg['minimum_table_size']['rows'],
            self.cfg['minimum_table_size']['columns'],
        )
        for idx, fetched_tbl in enumerate(fetched_tbls):
            if (
                fetched_tbl['df'].shape[0] < min_rows
                or fetched_tbl['df'].shape[1] < min_cols
            ):
                self.logger.error(
                    f"Dropping {url} as it has less than {min_rows} rows or {min_cols} columns"
                )
                self.error_msg = f"less than {min_rows} rows or {min_cols} columns"
                if tbl_idx is not None:
                    return None, None, f"less than {min_rows} rows or {min_cols} columns", self.date_cols, fetched_tbl['paragraph']
                continue
            if isinstance(fetched_tbl['df'].columns, MultiIndex):
                self.logger.error(f"Dropping {url} as it has MultiIndex columns")
                self.error_msg = f"MultiIndex columns"
                if tbl_idx is not None:
                    return None, None, f"MultiIndex columns", self.date_cols, fetched_tbl['paragraph']
                continue
            if fetched_tbl['df'].columns.tolist() == list(
                range(len(fetched_tbl['df'].columns))
            ):
                self.logger.error(f"Dropping {url} as it has empty header column names")
                self.error_msg = f"empty header column names"
                if tbl_idx is not None:
                    return None, None, f"empty header column names", self.date_cols, fetched_tbl['paragraph']
                continue

            if any(
                [re.search('\w+\.\d', k) for k in fetched_tbl['df'].columns.tolist()]
            ):
                self.logger.error(f"Dropping {url} as it has column names with numbers")
                self.error_msg = f"column names with numbers"
                if tbl_idx is not None:
                    return None, None, f"column names with numbers", self.date_cols, fetched_tbl['paragraph']
                continue
            if self.cfg['only_ascii_chars']:
                if not all(self.is_ascii(k) for k in fetched_tbl['df'].columns.tolist()):
                    self.error_msg = f"non-ascii characters in column names"
                    self.logger.error(
                        f"Dropping {url} as it has non-ascii characters in column names"
                    )
                    if tbl_idx is not None:
                        return None, None, f"non-ascii characters in column names", self.date_cols, fetched_tbl['paragraph']
                    continue
                if not all(self.is_ascii(k) for k in fetched_tbl['df'].to_numpy().flatten() ):
                    self.error_msg = f"non-ascii characters in data"
                    self.logger.error(
                        f"Dropping {url} as it has non-ascii characters in data"
                    )
                    if tbl_idx is not None:
                        return None, None, f"non-ascii characters in data", self.date_cols, fetched_tbl['paragraph']
                    continue
            self.df = fetched_tbl['df']
            self.url = url
            self.clean_df()
            if self.df is not None:
                self.logger.ok(
                    f'Passed: {url} ({[f"{k}:{self.df[k].dtype}" for k in self.df.columns]}) [{fetched_tbl["df"].shape}] ({idx})'
                )
                if self.date_cols:
                    self.logger.info(
                        f"Detected date columns: {self.date_cols} for {url}"
                    )
            else:
                self.logger.error(
                    f"Failed {url} ({fetched_tbl['df'].columns}) [{fetched_tbl['df'].shape}]"
                )
            return self.df, idx, self.error_msg, self.date_cols, fetched_tbl['paragraph']
        return None, None, self.error_msg, self.date_cols, None

    def apply_rules_columns(self):
        if self.df is None:
            return None
        for col in self.df.columns:
            if self.style_char in col and self.cfg['remove_data_different_style_from_headers']:
                self.logger.debug(
                    f"Removing text with different style from {col} column [{self.url}]"
                )
                self.df.rename(
                    columns={col: self.clean_style_string(col).__str__().strip()},
                    inplace=True,
                )
            if re.search("[\[].*?[\]]", self.clean_string(col).__str__()):
                self.logger.debug(
                    f"Removing data inside brackets for {col} column [{self.url}]"
                )
                self.df.rename(
                    columns={
                        col: re.sub(
                            "[\[].*?[\]]", '', self.clean_string(col).__str__()
                        ).strip()
                    },
                    inplace=True,
                )
        return

    def apply_rules_rows(self):
        if self.df is None:
            return None
        # Repeating header columns
        header_columns = self.df.columns.tolist()
        rows_to_delete = [
            k[0] for k in enumerate(self.df.to_numpy()) if all(k[1] == header_columns)
        ]
        if rows_to_delete:
            self.df.drop(rows_to_delete, inplace=True)
            self.logger.debug(
                f"Dropping row {rows_to_delete} as it is repeating header columns ({self.url})"
            )
        # Row as divider or all NaN
        idx_to_delete = [
            k[0]
            for k in enumerate(self.df.to_numpy())
            if all(k[1][0] == j for j in k[1])
        ]
        if idx_to_delete:
            self.df.drop(idx_to_delete, inplace=True)
            self.logger.debug(
                f"Dropping row {idx_to_delete} as Row as divider or all NaN ({self.url})"
            )

    def apply_cell_rules(self):
        if self.df is None:
            return None
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].apply(lambda x: self.clean_string(x, col))

            if self.cfg['cell_manipulations']['remove_data_inside_parentheses']:

                if any(
                    [
                        re.search("\(.+\)", k)  # removes text in brackets
                        for k in self.df[col].tolist()
                        if k and not isinstance(k, (float, int))
                    ]
                ):
                    self.df[col] = self.df[col].apply(
                        lambda x: (
                            re.sub("\(.+?\)", '', x).strip()
                            if isinstance(x, str)
                            else x
                        )
                    )
                    self.logger.debug(
                        f"Removing data inside parentheses for every cell in {col} column [{self.url}]"
                    )
            if self.cfg['cell_manipulations']['remove_data_inside_square_brackets']:
                if any(
                    [
                        re.search("\[.+?\]", k)  # removes text in brackets
                        for k in self.df[col].tolist()
                        if k and not isinstance(k, (float, int))
                    ]
                ):
                    self.df[col] = self.df[col].apply(
                        lambda x: (
                            re.sub("\[.+\]", '', x).strip()
                            if isinstance(x, str)
                            else x
                        )
                    )
                    self.logger.debug(
                        f"Removing data inside square_brackets {self.url} ({col} column)"
                    )
            self.df[col] = self.df[col].apply(self.clean_string)
            numbers_counter = Counter(
                [
                    get_type(k)
                    for k in self.df[col].apply(self.clean_string).tolist()
                ]
            )
            if self.cfg['maximum_mixed_types']['type'] == 'percent':
                value = self.cfg['maximum_mixed_types']['value']
                if  max(min(numbers_counter['num'], numbers_counter['str']), 1) / max(numbers_counter['num'], numbers_counter['str']) >= value:
                    self.logger.error(
                        f"Dropping {self.url} as it has mixed data types with higher ratio than: {value} ({col} column)"
                    )
                    self.df.drop(col, axis=1, inplace=True)
                    if self.cfg['drop_table_on_column_removal']:
                        self.error_msg = f"mixed data types with higher ratio than: {value} ({col} column)"
                        self.df = None
                        return
            else:
                value = self.cfg['maximum_mixed_types']['value']
                if min(numbers_counter['num'], numbers_counter['str']) >= value:
                    self.logger.error(
                        f"Dropping {self.url} as it has mixed data types with higher absolute value than: {value} ({col} column)"
                    )
                    self.df.drop(col, axis=1, inplace=True)
                    if self.cfg['drop_table_on_column_removal']:
                        self.error_msg = f"mixed data types with higher absolute value than: {value} ({col} column)"
                        self.df = None
                        return

        return None

    def last_aggr_row(self):
        if self.df is None:
            return None
        df_without_last_row = read_csv(StringIO(self.df[:-1].to_csv().strip(',')))
        for col in [self.df.columns[0], self.df.columns[-1]]:
            if (
                    get_type(self.df[col].iloc[-1]) in ['str','nan'] and  get_type(df_without_last_row[col].dtype, is_dtype=True)=='num'
            ):
                try:
                    self.logger.debug(
                        f'Last row data type: {get_type(df_without_last_row[col].dtype, is_dtype=True)} - {get_type(self.df[col].iloc[-1])}'
                    )
                    self.df = df_without_last_row.copy(deep=True)
                except:
                    pass

        # df_without_first_row = read_csv(StringIO(self.df[1:].to_csv().strip(',')))
        # for col in [self.df.columns[0], self.df.columns[-1]]:
        #     if (
        #             get_type(self.df[col].iloc[0]) in ['str','nan'] and  get_type(df_without_first_row[col].dtype, is_dtype=True)== 'num'
        #     ):
        #         try:
        #             self.logger.debug(
        #                 f'First row data type: {get_type(df_without_first_row[col].dtype,is_dtype=True)} - {get_type(self.df[col].iloc[0])}'
        #             )
        #             self.df = df_without_first_row.copy(deep=True)
        #         except:
        #             pass



    def nulls_check(self):
        if self.df is None:
            return None
        maximum_row_null_values = self.cfg['maximum_row_null_values']['value']
        maximum_column_null_values = self.cfg['maximum_column_null_values']['value']

        for col in self.df.columns:
            if self.cfg['maximum_column_null_values']['type'] == 'percent':
                if  self.df[col].isna().sum() / self.df.shape[0]  > maximum_column_null_values:
                    self.logger.error(
                        f"Dropping {self.url} as it has more than {maximum_column_null_values} percent null values in column {col}"
                    )
                    self.df.drop(col, axis=1, inplace=True)
                    if self.cfg['drop_table_on_column_removal']:
                        self.error_msg = f"more than {maximum_column_null_values} percent null values in column {col}"
                        self.df = None
                        return None

            else:
                if self.df[col].isna().sum() > maximum_column_null_values:
                    self.logger.error(
                        f"Dropping {self.url} as it has more than {maximum_column_null_values} null values in column {col}"
                    )
                    self.df.drop(col, axis=1, inplace=True)
        for row in self.df.index:
            if self.cfg['maximum_row_null_values']['type'] == 'percent':
                if  self.df.loc[row].isna().sum() / self.df.shape[1]  > maximum_row_null_values:
                    self.logger.error(
                        f"Dropping {self.url} as it has more than {maximum_row_null_values} percent null values in row {row}"
                    )
                    self.df.drop(row, inplace=True)
            else:
                if self.df.loc[row].isna().sum() > maximum_row_null_values:
                    self.logger.error(
                        f"Dropping {self.url} as it has more than {maximum_row_null_values} null values in row {row}"
                    )
                    self.df.drop(row, inplace=True)

        pass
    def clean_df(self):
        # Check if the DataFrame has empty column headers
        if self.df.columns[0] == 'Unnamed: 0':
            np_array = np.array([re.sub("[\[].*?[\]]", '',k).strip() if type(k) == str else k for k in self.df['Unnamed: 0'].tolist()])
            try:
                np_array = to_numeric(np_array, errors='coerce')
            except ValueError:
                self.logger.error(f"Error converting column 'Unnamed: 0' to numeric for {self.url}")
                return None
            if np.all(np.logical_and(np_array >= 1000, np_array <= 3000)):
                self.logger.debug(f'Setting first empty column as Year column for {self.url}')
                self.df.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)
            pass
        if any(
            [
                isna(k) or 'Unnamed:' in k
                for k in self.df.columns.tolist()
                if not isinstance(k, (int, float))
            ]
        ):
            self.logger.error(f"Dropping {self.url} as it has empty column header")
            self.error_msg = f"empty column header"
            self.df = None
            return None
        self.drop_irrelevant_columns()
        self.apply_rules_rows()
        self.apply_cell_rules()
        self.apply_rules_columns()
        if self.cfg['last_row_agg_mechanism']:
            self.last_aggr_row()
        self.nulls_check()

        return self.df

# #
# obj = WikiTableParser()
# r = obj.run(
#     'https://en.wikipedia.org/wiki/1979_PGA_Tour'
# )
# b=5
