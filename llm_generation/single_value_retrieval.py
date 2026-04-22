import json
from utils.utils import get_llm_response
from prompts_generation import prompt_generation
import random
import os
random.seed(0)


def cell_retrieval():
    def get_random_sample(_df, _key_column, n=3):
        """Get a random sample of n rows from the DataFrame."""
        return list(_df.sample(n=n, random_state=1)[_key_column].items())
    def get_choices(_df, _answer_value, _desired_column):
        two_choices = set(df[_desired_column].tolist()).difference({_answer_value})
        random_two_choices = random.sample(list(two_choices), 2) + [_answer_value]
        random.shuffle(random_two_choices)
        return {
            'CHOICE_1': random_two_choices[0],
            'CHOICE_2': random_two_choices[1],
            'CHOICE_3': random_two_choices[2],
        }
    import matplotlib.pyplot as plt
    from wikiparser import WikiTableParser
    import seaborn as sns
    import git
    import pandas as pd
    import diskcache
    import numpy as np

    import yaml
    with open('../config.yaml', 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    key_model =  conf['title_and_key_model']
    model = conf['llm_model']
    rephrase = conf['rephrase']
    random.seed(10)
    tbl_statistics = {
        'popularity': [],
        'num_rows': [],
        'num_cols': [],
        'num_cells': [],
        'numerical_cols_ratio': [],
        'count': 0,
    }
    working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    parser_ins = WikiTableParser()
    with open(f'{working_dir}/llm_generation/prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)

    generated_tbl_cache = diskcache.Cache(
        f'{working_dir}/local_dbs/tables/generated_tables.db'
    )
    single_value_scores = {}
    MAX_ITER = 600
    count = 0
    scores_statistics = ""
    tbls_processed = 0
    for tbl in generated_tbl_cache.iterkeys():
        count += 1
        print(f'Retrieving {count}...')
        if count > MAX_ITER:
            break
        cfg = json.loads(generated_tbl_cache[tbl])
        if not os.path.isfile(f'../tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv'):
            generated_tbl_cache.delete(tbl)
            continue
        df = pd.read_csv(f'../tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv')
        if key_model not in cfg['llm_generated']:
            continue
        key = cfg['llm_generated'][key_model]['key']
        if len(str(key)) == 1 or key.isnumeric():
            continue
        if df[key].unique().__len__() < df.__len__():
            continue
        table_description = cfg['llm_generated'][key_model]['table_title']
        columns_without_key = [c for c in df.columns if c != key and c not in ['Total','Date']]
        comparable_columns = [
            c
            for c in columns_without_key
            if df[c].dtype in [np.int64, np.float64] and c not in ['Total']
        ]
        single_value_scores[tbl] = {}

        if not set(columns_without_key).intersection(set(cfg['columns'])):
            continue
        for row_index, key_value in get_random_sample(df, key, 3):
            single_value_scores[tbl][key_value] = {}

            for col in columns_without_key:
                if len(str(col)) == 1 or col.isnumeric():
                    continue
                real_value = df.iloc[row_index][col]
                if df[col].unique().__len__() < 3:
                    continue
                if set([type(k) for k in df[col].tolist()]).__len__() > 1:
                    continue
                if pd.isna(real_value):
                    continue
                if count == 416:
                    b=5
                three_variants = prompt_generation(
                    table_desc=table_description,
                    col=col,
                    key=key,
                    key_value=key_value,
                    real_value=real_value,
                    df=df,
                    parser_ins=parser_ins,
                    model=model,
                    instruction='single_value_retrieval_instruction',
                    url=cfg['url'],
                    rephrase=rephrase
                )
                if not three_variants['open_book']:
                    b=5
                if not rephrase:
                    single_value_scores[tbl] = three_variants
                else:
                    tmp_json = cfg.copy()
                    previous_tasks = tmp_json.get("tasks", {}).get('single_value', {})
                    three_variants['column'] = col
                    if 'tasks' not in tmp_json:
                        tmp_json['tasks'] = {}
                    tmp_json["tasks"]['single_value'] = {**previous_tasks, **three_variants}
                    generated_tbl_cache[tbl] = json.dumps(tmp_json)
                if isinstance(cfg['article_metadata'], str):
                    _tmp_json = json.loads(cfg['article_metadata'])
                    tbl_statistics['popularity'].append(_tmp_json['popularity'])
                else:
                    tbl_statistics['popularity'].append(cfg['article_metadata']['popularity'])
                tbl_statistics['num_rows'].append(df.shape[0])
                tbl_statistics['num_cols'].append(df.shape[1])
                tbl_statistics['num_cells'].append(df.shape[0] * df.shape[1])
                tbl_statistics['numerical_cols_ratio'].append(len(comparable_columns) / len(df.columns))
                tbl_statistics['count'] += 1

                for bd in three_variants:
                    score = three_variants[bd]
                    scores_statistics += f"{model},Single Retrieval,N/A,{bd},N/A,{df.shape[0]},{df.shape[1]},{tbl_statistics['popularity'][-1]},{tbl_statistics['numerical_cols_ratio'][-1]},{int(score)}\n"
                tbls_processed += 1

                break
            break

    print(f"Total tables processed: {len(generated_tbl_cache)}")


    score = {
        "closed_book": 0, "multiple_choices": 0, "open_book": 0
    }
    for record in single_value_scores.items():
        url = record[0]
        data = record[1]
        for variant, variant_details in data.items():
            if variant not in score: continue
            score[variant] += variant_details
    print(score)
    print(f"Total: {len(single_value_scores)}")
    print(f"Total: {tbls_processed}")
    print(f"Table statistics: ")
    print(
        f"Popularity - Min: {min(tbl_statistics['popularity'])}  Mean: {np.mean(tbl_statistics['popularity'])} - Max: {max(tbl_statistics['popularity'])} -"
        f" Std: {np.std(tbl_statistics['popularity'])}")
    print(
        f"Num rows - Min: {min(tbl_statistics['num_rows'])} Mean: {np.mean(tbl_statistics['num_rows'])} Max: {max(tbl_statistics['num_rows'])} - Std: {np.std(tbl_statistics['num_rows'])}")
    print(
        f"Num cols - Min: {min(tbl_statistics['num_cols'])}  Mean: {np.mean(tbl_statistics['num_cols'])} Max: {max(tbl_statistics['num_cols'])} - Std: {np.std(tbl_statistics['num_cols'])}")
    print(
        f"Num cells - Min: {min(tbl_statistics['num_cells'])} Mean: {np.mean(tbl_statistics['num_cells'])} Max: {max(tbl_statistics['num_cells'])} - Std: {np.std(tbl_statistics['num_cells'])}")
    print(
        f"Numerical cols ratio - Min: {min(tbl_statistics['numerical_cols_ratio'])} Mean: {np.mean(tbl_statistics['numerical_cols_ratio'])} Max: {max(tbl_statistics['numerical_cols_ratio'])} - Std: {np.std(tbl_statistics['numerical_cols_ratio'])}")
    print(f"Total tables with count task: {tbl_statistics['count']}")
    print(f"Total tables with count task: {tbl_statistics['count']}")

    with open("scores.csv","a") as f:
        f.write(f"{model},Single-retrieval,N/A,N/A,closed_book,={score["closed_book"]}/{tbls_processed}\n")
        f.write(f"{model},Single-retrieval,N/A,N/A,multiple_choices,={score["multiple_choices"]}/{tbls_processed}\n")
        f.write(f"{model},Single-retrieval,N/A,N/A,open_book,={score["open_book"]}/{tbls_processed}\n")
    return scores_statistics




if __name__ == '__main__':
    import time
    idx = 0
    cell_retrieval()

