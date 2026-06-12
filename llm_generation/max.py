import json

from llm_generation.prompts_generation import (
    prompt_generation,
    prompt_generation_count_categorical,
    prompt_generation_count_numerical,
    prompt_generation_max_categorical,
    prompt_generation_max_numerical
)
import random
import os
from collections import Counter

random.seed(0)


def max_retrieval():
    def get_random_sample(_df, _key_column, n=3):
        """Get a random sample of n rows from the DataFrame."""
        return list(_df.sample(n=n, random_state=1)[_key_column].items())

    def get_choices(_df, _answer_value, _desired_column):
        two_choices = set(df[_desired_column].tolist()).difference({_answer_value})
        random_two_choices = random.sample(list(two_choices), 2) + [_answer_value]
        random.shuffle(random_two_choices)
        return {
            "CHOICE_1": random_two_choices[0],
            "CHOICE_2": random_two_choices[1],
            "CHOICE_3": random_two_choices[2],
        }

    from wikiparser import WikiTableParser
    import git
    import pandas as pd
    import diskcache
    import numpy as np

    import yaml

    with open("../config.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    key_model = conf["title_and_key_model"]
    model = conf["llm_model"]
    rephrase = conf["rephrase"]
    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    parser_ins = WikiTableParser()
    generated_tbl_cache = diskcache.Cache(
        f"{working_dir}/local_dbs/tables/generated_tables.db"
    )
    single_value_scores = {}
    MAX_ITER = 550
    count = 0
    scores_statistics = ""
    categorical_tbls_processed = 0
    numerical_tbls_processed = 0
    tbl_statistics = {
        'popularity': [],
        'num_rows': [],
        'num_cols': [],
        'num_cells': [],
        'numerical_cols_ratio': [],
        'count': 0,
    }
    for tbl in generated_tbl_cache.iterkeys():
        count += 1
        print(f"Retrieving {count}...")
        if count > MAX_ITER:
            break
        cfg = json.loads(generated_tbl_cache[tbl])
        if not os.path.isfile(f'../tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv'):
            generated_tbl_cache.delete(tbl)
            continue
        df = pd.read_csv(f'../tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv')
        if key_model not in cfg["llm_generated"]:
            continue
        key = cfg["llm_generated"][key_model]["key"]
        if df[key].unique().__len__() < df.__len__():
            continue
        if len(str(key)) == 1 or key.isnumeric():
            continue
        table_description = cfg["llm_generated"][key_model]["table_title"]
        columns_without_key = [
            c for c in df.columns if c != key and c not in ["Total", "Date"]
        ]
        comparable_columns = [
            c
            for c in columns_without_key
            if df[c].dtype in [np.int64, np.float64] and c not in ["Total"]
        ]
        categorical_columns = [
            c
            for c in columns_without_key
            if df[c].dtype == object and c not in ["Total"]
        ]
        # Categorical:
        if not set(columns_without_key).intersection(set(cfg["columns"])):
            continue
        single_value_scores[tbl] = {
            "categorical": {},
            "numerical": {},
        }
        for col in comparable_columns:
            if df[col].unique().__len__() < 3:
                continue
            for cat_col in categorical_columns:
                if len(str(col)) == 1 or col.isnumeric():
                    continue
                try:
                    second_common, third_common = Counter(df[cat_col].tolist()).most_common(n=3)[
                        1:3
                    ]
                except ValueError:
                    continue
                if second_common[1] < 2:
                    continue
                if pd.isna(second_common[0]) or pd.isna(third_common[0]):
                    continue
                answer_1 = df[df[cat_col] == second_common[0]]
                answer_1 = answer_1[ answer_1[col] == answer_1[col].max()][key].tolist()
                answer_2 = df[df[cat_col] == third_common[0]]
                answer_2 = answer_2[ answer_2[col] == answer_2[col].max()][key].tolist()
                three_variants = prompt_generation_max_categorical(
                    table_desc=table_description,
                    comparable_col=col,
                    categorical_col=cat_col,
                    key_column=key,
                    cat_col_value_1=second_common[0],
                    cat_col_value_2=third_common[0],
                    df=df,
                    real_value_1=answer_1,
                    real_value_2=answer_2,
                    parser_ins=parser_ins,
                    model=model,
                    instruction="multiple_return_retrieval_instruction",
                    pk_column=key,
                    url=cfg["url"],
                    rephrase=rephrase
                )
                categorical_tbls_processed += 1
                if not rephrase:
                    single_value_scores[tbl]["categorical"][col] = three_variants
                else:
                    tmp_json = cfg.copy()
                    previous_tasks = tmp_json.get("tasks", {}).get('max', {})
                    if 'count' not in tmp_json.get("tasks", {}):
                        tmp_json["tasks"] = {}
                        tmp_json["tasks"]['max'] = {}
                    three_variants['column'] = col
                    tmp_json["tasks"]['max'] =  {**previous_tasks, **three_variants}
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
                    for _type, score in three_variants[bd].items():
                        # CSV for model, task type, task, variant, num_rows, num_cols, popularity, numerical_cols_ratio, score
                        scores_statistics += f"{model},Max,Categorical,{bd},{_type},{df.shape[0]},{df.shape[1]},{tbl_statistics['popularity'][-1]},{tbl_statistics['numerical_cols_ratio'][-1]},{score}\n"

                break
            break

        if len(comparable_columns) > 1:
            comparable_col_1 = comparable_columns[0]
            comparable_col_2 = comparable_columns[1]
            median_value = df[comparable_col_2].median()
            if median_value == 0 or pd.isna(median_value):
                continue
            if (df[comparable_col_2] > median_value).sum() < 2 or (df[comparable_col_2] < median_value).sum() < 2:
                continue
            if len(df[comparable_col_1].unique()) < 3 or len(df[comparable_col_2].unique()) < 3:
                continue
            three_variants = prompt_generation_max_numerical(
                table_desc=table_description,
                comparable_col_1=comparable_col_1,
                comparable_col_2=comparable_col_2,
                df=df,
                parser_ins=parser_ins,
                model=model,
                instruction="multiple_return_retrieval_instruction",
                pk_column=key,
                threshold=median_value,
                lower_threshold=int(median_value / 1.5),
                upper_threshold=int(median_value * 1.5),
                url=cfg["url"],
                rephrase=rephrase
            )
            numerical_tbls_processed += 1
            if not rephrase:
                single_value_scores[tbl]["numerical"][col] = three_variants
            else:
                tmp_json = cfg.copy()
                previous_tasks = tmp_json.get("tasks", {}).get('max', {})
                if 'count' not in tmp_json.get("tasks", {}):
                    tmp_json["tasks"] = {}
                    tmp_json["tasks"]['max'] = {}
                three_variants['column'] = col
                tmp_json["tasks"]['max'] = {**previous_tasks, **three_variants}
                generated_tbl_cache[tbl] = json.dumps(tmp_json)


    print(f"Total tables processed: {len(generated_tbl_cache)}")

    single_value_list = []
    categorical_score = {
        "categorical_equals": {"closed_book": 0, "multiple_choices": 0, "open_book": 0},
        "categorical_not_equals": {"closed_book": 0, "multiple_choices": 0, "open_book": 0},
        "categorical_in": {"closed_book": 0, "multiple_choices": 0, "open_book": 0},
    }
    numerical_score = {
        'numerical_greater_than': {"closed_book": 0, "multiple_choices": 0, "open_book": 0},
        'numerical_less_than': {"closed_book": 0, "multiple_choices": 0, "open_book": 0},
        'numerical_between': {"closed_book": 0, "multiple_choices": 0, "open_book": 0},
        'numerical_equals': {"closed_book": 0, "multiple_choices": 0, "open_book": 0},
    }
    categorical_count = 0
    numerical_count = 0
    for record in single_value_scores.items():
        url = record[0]
        data = record[1]
        if "categorical" in data and list(data["categorical"].keys()):
            categorical_col = list(data["categorical"].keys())[0]
            categorical_count += 1
            for variant, variant_details in data["categorical"][
                categorical_col
            ].items():
                for _type, score in variant_details.items():
                    categorical_score[variant][_type] += score
        if "numerical" in data and list(data["numerical"].keys()):
            numerical_col = list(data["numerical"].keys())[0]
            numerical_count += 1
            for variant, variant_details in data["numerical"][
                numerical_col
            ].items():
                for _type, score in variant_details.items():
                    numerical_score[variant][_type] += score
    # Categorical:
    #   Equals
    print(
        f"Categorical equals - Closed book: {categorical_score['categorical_equals']['closed_book']} out of {categorical_count}")
    print(
        f"Categorical equals - Multiple choices: {categorical_score['categorical_equals']['multiple_choices']} out of {categorical_count}")
    print(
        f"Categorical equals - Open book: {categorical_score['categorical_equals']['open_book']} out of {categorical_count}")
    #   Not equals
    print(
        f"Categorical not equals - Closed book: {categorical_score['categorical_not_equals']['closed_book']} out of {categorical_count}")
    print(
        f"Categorical not equals - Multiple choices: {categorical_score['categorical_not_equals']['multiple_choices']} out of {categorical_count}")
    print(
        f"Categorical not equals - Open book: {categorical_score['categorical_not_equals']['open_book']} out of {categorical_count}")
    #   In
    print(
        f"Categorical in - Closed book: {categorical_score['categorical_in']['closed_book']} out of {categorical_count}")
    print(
        f"Categorical in - Multiple choices: {categorical_score['categorical_in']['multiple_choices']} out of {categorical_count}")
    print(f"Categorical in - Open book: {categorical_score['categorical_in']['open_book']} out of {categorical_count}")
    # Numerical:
    #   Greater than
    print(
        f"Numerical Greater than - Closed book: {numerical_score['numerical_greater_than']['closed_book']} out of {numerical_count}")
    print(
        f"Numerical Greater than - Multiple choices: {numerical_score['numerical_greater_than']['multiple_choices']} out of {numerical_count}")
    print(
        f"Numerical Greater than - Open book: {numerical_score['numerical_greater_than']['open_book']} out of {numerical_count}")
    #   Less than
    print(
        f"Numerical Less than - Closed book: {numerical_score['numerical_less_than']['closed_book']} out of {numerical_count}")
    print(
        f"Numerical Less than - Multiple choices: {numerical_score['numerical_less_than']['multiple_choices']} out of {numerical_count}")
    print(
        f"Numerical Less than - Open book: {numerical_score['numerical_less_than']['open_book']} out of {numerical_count}")
    #   Between
    print(
        f"Numerical Between - Closed book: {numerical_score['numerical_between']['closed_book']} out of {numerical_count}")
    print(
        f"Numerical Between - Multiple choices: {numerical_score['numerical_between']['multiple_choices']} out of {numerical_count}")
    print(
        f"Numerical Between - Open book: {numerical_score['numerical_between']['open_book']} out of {numerical_count}")
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
    import utils.utils as _uu
    if _uu.COLLECT_MODE:
        return scores_statistics
    with open('scores.csv', 'a+') as f:
        # f.write("Model,Task,TaskType,Constraint,Variant,Score\n")
        for variant, details in categorical_score["categorical_equals"].items():
            f.write(f"{model},Max,Categorical,Equals,{variant},={details}/{categorical_tbls_processed}\n")
        for variant, details in categorical_score["categorical_not_equals"].items():
            f.write(f"{model},Max,Categorical,Not Equals,{variant},={details}/{categorical_tbls_processed}\n")
        for variant, details in categorical_score["categorical_in"].items():
            f.write(f"{model},Max,Categorical,In,{variant},={details}/{categorical_tbls_processed}\n")
        for variant, details in numerical_score["numerical_greater_than"].items():
            f.write(f"{model},Max,Numerical,Greater Than,{variant},={details}/{numerical_tbls_processed}\n")
        for variant, details in numerical_score["numerical_less_than"].items():
            f.write(f"{model},Max,Numerical,Less Than,{variant},={details}/{numerical_tbls_processed}\n")
        for variant, details in numerical_score["numerical_between"].items():
            f.write(f"{model},Max,Numerical,Between,{variant},={details}/{numerical_tbls_processed}\n")
        for variant, details in numerical_score["numerical_equals"].items():
            f.write(f"{model},Max,Numerical,Equals,{variant},={details}/{numerical_tbls_processed}\n")

    return scores_statistics



if __name__ == "__main__":
    import time

    idx = 0
    max_retrieval()
