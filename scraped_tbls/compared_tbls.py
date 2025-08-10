import os
from collections import defaultdict

import pandas as pd
import numpy as np
import json
from unidecode import unidecode
from dateutil import parser as date_parser
from wikiparser import WikiTableParser

import warnings
warnings.filterwarnings('ignore')
wikiparser_obj = WikiTableParser()
def normalize_value(value, is_date=False):
    if value != value:
        return ''

    if is_date:
        try:
            return date_parser.parse(value)
        except:
            try:
                return pd.to_datetime(value)
            except:
                pass

    if type(value).__module__ == 'numpy':
        value = value.item()

    if isinstance(value, str):
        if value.startswith('-') and value.replace('-', '').replace(',', '').isdigit():
            return int(value.replace(',', ''))
        if value.replace(',', '').isdigit():
            return int(value.replace(',', ''))
        if value.startswith('=') and value.isdigit():
            return int(value.replace('=', ''))
        if (
            value.startswith('-')
            and value.replace('-', '').replace(',', '').replace('.', '').isdigit()
        ):
            return float(value.replace(',', ''))
        if value.replace(',', '').replace('.', '').isdigit():
            return float(value.replace(',', ''))

        value = value.strip().lower()

        if value in ('none', 'n/a', 'nan', '-'):
            return ''

        value = value.replace('&', 'and')

        if value == 'united states':
            return 'usa'
        if value == 'united kingdom':
            return 'uk'

        value = unidecode(value)
        value = ''.join(c for c in value if c.isalnum())
        return value

    return value


def normalize_key(value, is_date=False):
    if value != value:
        return ''

    if is_date:
        try:
            return str(date_parser.parse(value))
        except:
            try:
                return str(pd.to_datetime(value))
            except:
                pass

    if isinstance(value, str):
        value = value.strip().lower()

        if value in ('none', 'n/a', 'nan', '-', '--', 'unknown'):
            return ''

        value = value.replace('&', 'and')

        if value == 'united states':
            return 'usa'
        if value == 'united kingdom':
            return 'uk'

        value = unidecode(value)
        value = ''.join(c for c in value if c.isalnum())
        return value

    return str(value)


def normalize_primary_columns(
    df, norm_columns, date_columns, primary_columns, keys_type
):
    for col in norm_columns:
        df[col] = df[col].apply(normalize_key, col in date_columns)

    for col, key_type in zip(primary_columns, keys_type):
        if key_type == 'year':
            df[col] = df[col].astype(float).astype(int)

        df[col] = df[col].astype(str)

    return [tuple(r) for r in df[primary_columns].to_numpy()]


def find_row(df, columns, values):
    query = ' & '.join([f'(`{col}`=="{value}")' for col, value in zip(columns, values)])
    return df.query(query)


def evaluate_table(df_fetched, df_ref, primary_columns, keys_type, date_columns, epsilons, columns_dtypes):
    columns = df_ref.columns
    df_fetched.columns = columns
    df_fetched = df_fetched.drop_duplicates(subset=primary_columns)

    norm_columns = set(primary_columns)
    for pc in primary_columns:
        df_fetched = df_fetched[df_fetched[pc].notna()]

    fetched_entities = normalize_primary_columns(df_fetched, norm_columns, date_columns, primary_columns, keys_type)
    ref_entities = normalize_primary_columns(df_ref, norm_columns, date_columns, primary_columns, keys_type)

    total_matches = 0
    key_matches = 0
    matches_by_col = {col: 0 for col in columns}
    for fetched_entity in fetched_entities:
        if fetched_entity in ref_entities:
            row_fetched = find_row(df_fetched, primary_columns, fetched_entity)
            row_ref = find_row(df_ref, primary_columns, fetched_entity)
            key_matches += 1

            for column in columns:
                try:
                    value_fetched = row_fetched[column].values[0]
                    value_ref = row_ref[column].values[0]
                    norm_value_fetched, norm_value_ref = wikiparser_obj.try_cast(value_fetched), wikiparser_obj.try_cast(value_ref)

                    if norm_value_fetched == norm_value_ref:
                        total_matches += 1
                        matches_by_col[column] += 1
                        continue
                    if norm_value_fetched is None and norm_value_ref is None:
                        total_matches += 1
                        matches_by_col[column] += 1
                        continue
                    if norm_value_fetched is None or norm_value_ref is None:
                        continue
                    elif column in epsilons and norm_value_ref != '' and norm_value_fetched != '':
                        if norm_value_ref * 0.999 < norm_value_fetched < norm_value_ref * 1.001:
                            total_matches += 1
                            matches_by_col[column] += 1
                        continue
                except:
                    raise "Here"

    recall = total_matches / (df_ref.shape[0] * df_ref.shape[1])
    precision = total_matches / (df_fetched.shape[0] * df_fetched.shape[1])
    f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    keys_recall = key_matches / len(ref_entities)
    keys_precision = key_matches / len(fetched_entities)
    keys_f1_score = 2 * keys_recall * keys_precision / (keys_recall + keys_precision) if (
                                                                                                     keys_recall + keys_precision) > 0 else 0.0

    nk = len(primary_columns)

    non_keys_recall = (total_matches - key_matches * nk) / (df_ref.shape[0] * (df_ref.shape[1] - nk))
    non_keys_precision = (total_matches - key_matches * nk) / (df_fetched.shape[0] * (df_fetched.shape[1] - nk))
    non_keys_f1_score = 2 * non_keys_recall * non_keys_precision / (non_keys_recall + non_keys_precision) if (
                                                                                                                         non_keys_recall + non_keys_precision) > 0 else 0.0

    relative_non_key_accuracy = (total_matches - key_matches * nk) / (key_matches * (df_ref.shape[1] - nk))

    cols_recall =    {col: matches_by_col[col] / df_ref.shape[0] for col in columns}
    cols_precision = {col: matches_by_col[col] / df_fetched.shape[0] for col in columns}
    cols_f1_score =  {col: 2 * cols_recall[col] * cols_precision[col] / (cols_recall[col] + cols_precision[col]) if (cols_recall[col] + cols_precision[col]) > 0 else 0.0 for col in columns}
    score_by_col_type = defaultdict(dict)
    for r,p,f1 in zip(cols_recall.items(), cols_precision.items(), cols_f1_score.items()):
        col_type = columns_dtypes[r[0]]
        if col_type not in score_by_col_type:
            score_by_col_type[col_type] = {'recall': [], 'precision': [], 'f1_score': []}
        score_by_col_type[col_type]['recall'].append(r[1])
        score_by_col_type[col_type]['precision'].append(p[1])
        score_by_col_type[col_type]['f1_score'].append(f1[1])
    return keys_recall, keys_precision, keys_f1_score, non_keys_recall, non_keys_precision, non_keys_f1_score, recall, precision, f1_score, relative_non_key_accuracy, score_by_col_type

def run_compare(fetched_table, gt_table, key_column, key_column_type, epsilons, columns_dtypes):

    try:
        primary_columns = key_column
        keys_type = key_column_type
        date_columns = [] # md['dateColumns']
        epsilons = epsilons
        kr, kp, kf1, nkr, nkp, nkf1, r, p, f1, rnka, score_by_col_type  = evaluate_table(
            fetched_table,
            gt_table,
            primary_columns,
            keys_type,
            date_columns,
            epsilons,
            columns_dtypes
        )
    except Exception as e:
        kr, kp, kf1, nkr, nkp, nkf1, r, p, f1, rnka, score_by_col_type = 11 * [None]
    return kr, kp, kf1, nkr, nkp, nkf1, r, p, f1, rnka, score_by_col_type
    tables.append(md['name'])



