import json
import random
import os
import sys
sys.path.append('/Users/david.harroch/PycharmProjects/table_generation')
from utils.utils import get_llm_response
from prompts_generation import f1_score_sets
from pprint import pprint

random.seed(0)


def count_retrieval():


    import git
    import pandas as pd
    import diskcache
    from wikiparser import WikiTableParser
    import yaml

    with open("../config.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    with open("prompts.yaml", "r") as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)
    key_model = conf["title_and_key_model"]
    model = conf["llm_model"]
    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir

    generated_tbl_cache = diskcache.Cache(
        f"{working_dir}/local_dbs/tables/generated_tables.db"
    )
    parser = WikiTableParser()
    MAX_ITER = 600
    count = 0
    MC = "Select one from the following options: ['{CHOICE_1}', '{CHOICE_2}', '{CHOICE_3}']"
    LIST_MC = "Select all that apply from the following options: {CHOICES_LIST}"
    scores = dict()
    count_tasks = dict()
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


        if cfg.get("tasks"):
            for _type in cfg.get("tasks"):
                if _type not in scores:
                    scores[_type] = {}
                    count_tasks[_type] = {}
                for _key in cfg["tasks"][_type]:
                    if _type == _key or _key in ['column','answer','options']:
                        continue
                    if _type in ['single_value','list']:
                        _key = 'N/A'
                        answer = cfg["tasks"][_type]['answer']
                        rephrased= cfg["tasks"][_type]['rephrased']
                        options = cfg["tasks"][_type]['options']

                    else:
                        answer = cfg["tasks"][_type][_key]['answer']
                        if 'rephrased' not in cfg["tasks"][_type][_key]:
                            continue
                        rephrased = cfg["tasks"][_type][_key]['rephrased']
                        options = cfg["tasks"][_type][_key]['options']

                    if _key not in scores[_type]:
                        scores[_type][_key] = {"closed_book": 0, "multiple_choices": 0, "open_book": 0}
                        count_tasks[_type][_key] = {"closed_book": 0, "multiple_choices": 0, "open_book": 0}

                    if _type in ['count','single_value']:
                        closed_book = rephrased + "\n" + prompts['single_value_retrieval_instruction']
                        multiple_choices = rephrased + "\n" + MC.format(**options) + "\n" + prompts['single_value_retrieval_instruction']
                    elif _type in ['list']:
                        closed_book = rephrased + "\n" + prompts['list_value_retrieval_instruction']
                        multiple_choices = rephrased + "\n" + LIST_MC.format(**options) + "\n" + prompts['list_value_retrieval_instruction']

                    else:
                        closed_book = rephrased + prompts['multiple_return_retrieval_instruction']
                        multiple_choices = rephrased + "\n" + MC.format(**options) + "\n" + prompts['multiple_return_retrieval_instruction']

                    closed_book_response = get_llm_response(prompt_string=closed_book, model=model)
                    if closed_book_response is None:
                        continue
                    multiple_choices_response = get_llm_response(prompt_string=multiple_choices, model=model)
                    if _type in ['list']:
                        closed_book_response = [str(k) for k in closed_book_response.split(',')]
                        multiple_choices_response = [str(k) for k in multiple_choices_response.split(',')]
                        scores[_type][_key]['closed_book'] += f1_score_sets(answer, closed_book_response)[-1]
                        scores[_type][_key]['multiple_choices'] += f1_score_sets(answer, multiple_choices_response)[-1]
                    elif isinstance(answer,list):
                        scores[_type][_key]['closed_book'] += 1 if str(closed_book_response) in answer else 0
                        scores[_type][_key]['multiple_choices'] += 1 if multiple_choices_response in answer else 0
                    else:
                        scores[_type][_key]['closed_book'] += 1 if (str(closed_book_response) == str(answer) or parser.try_cast(closed_book_response) == parser.try_cast(answer) ) else 0
                        scores[_type][_key]['multiple_choices'] += 1 if (str(multiple_choices_response) == str(answer) or parser.try_cast(multiple_choices_response) == parser.try_cast(answer)) else 0
                    count_tasks[_type][_key]['closed_book'] += 1
                    count_tasks[_type][_key]['multiple_choices'] += 1

        continue
    print(f"Total tables processed: {len(generated_tbl_cache)}")
    pprint(scores)

    with open("rephrase_scores.csv", "a+") as f:
        # Model, TaskType, Task, Constraint, Variant, Score
        for task in scores:
            for bd in scores[task]:
                for variant in scores[task][bd]:
                    total = count_tasks[task][bd][variant]
                    score = scores[task][bd][variant]
                    if total > 0:
                        f.write(f"{model},{task},{bd},{variant},={score}/{total}\n")
                    else:
                        scores[task][bd][variant] = "No data"
    # pprint(count_tasks)
    exit(0)


if __name__ == "__main__":
    import time

    idx = 0
    count_retrieval()

