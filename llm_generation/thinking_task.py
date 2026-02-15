import json
from utils.utils import get_llm_response
import os


def thinking_task_given_table():
    def get_random_sample(_df, _key_column, n=3):
        """Get a random sample of n rows from the DataFrame."""
        return list(_df.sample(n=n, random_state=1)[_key_column].items())
    import matplotlib.pyplot as plt
    from wikiparser import WikiTableParser
    import seaborn as sns
    import git
    import pandas as pd
    import diskcache
    import numpy as np
    import random
    import yaml
    with open('../config.yaml', 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    model = conf['llm_model'] # 'gemini-2.5-pro'
    key_model = model
    context = conf['context']
    random.seed(10)
    working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    parser_ins = WikiTableParser()
    with open(f'{working_dir}/llm_generation/prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)


    generated_tbl_cache = diskcache.Cache(
        f'{working_dir}/local_dbs/tables/generated_tables.db'
    )
    thinking_scores = {}
    MAX_ITER = 850
    count = 0
    for tbl in ['https://en.wikipedia.org/wiki/2018_Korean_Tour']: # generated_tbl_cache.iterkeys():
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
        table_description = cfg['llm_generated'][key_model]['table_title']
        columns_without_key = [c for c in df.columns if c != key and c not in prompts['ignore_comparable_cols']]
        comparable_columns = [
            c
            for c in columns_without_key
            if df[c].dtype in [np.int64, np.float64]
        ]
        thinking_scores[tbl] = {}
        if not set(columns_without_key).intersection(set(cfg['columns'])):
            continue
        if not comparable_columns:
            continue
        for comparable_col in comparable_columns:
            if len(set(df[comparable_col].to_list())) < (len(df) // 2):
                # Skip if there are duplicate values
                pass
                #continue
            min_val = df[comparable_col].min()
            second_from_bottom = df[df[comparable_col] > min_val][comparable_col].min()
            indices = df[df[comparable_col] == second_from_bottom].index
            if (
                len(comparable_col) == 1
                or comparable_col.isnumeric()
            ):
                # Skip if the value is not a valid key
                continue
            thinking_prompt_question = prompts['thinking'].format(
                Primary_key=key,
                Comparison_Column=comparable_col,
                PROVIDED_KEYS=''
            ).strip()
            thinking_prompt = prompts['table_wrapper'].format(
                TABLE_TITLE=table_description,
                QUESTION=thinking_prompt_question,
                TABLE=df.to_csv(index=False)
            ).strip()

            response = get_llm_response(thinking_prompt,MODEL=model,)
            parsed_response = parser_ins.try_cast(response)
            if response is None :
                continue

            llm_indices = list(df[df[key] == response][comparable_col].index) + list(
                df[df[key] == parsed_response][comparable_col].index
            )
            llm_indices = set(llm_indices)

            if parsed_response not in df[key].values:
                matched = -1
            else:
                matched = 0
                if len(llm_indices.intersection(indices)) > 0:
                    matched = 1
            if matched == 0:
                b=5


            thinking_scores[tbl][comparable_col] = {
                'response': response,
                'actual_result': list(
                    df[df[comparable_col] == second_from_bottom ][key]
                ),
                'correct': matched,
                'prompt': thinking_prompt,
            }
            break

    print(f"Total tables processed: {len(generated_tbl_cache)}")
    thinking_total_match = 0
    thinking_total_non_match = 0
    thinking_total_non_in = 0
    thinking_count = 0
    thinking_values_values = []
    for record in thinking_scores.items():
        url = record[0]
        data = record[1]
        for key, value in data.items():
            thinking_values_values.append(
                [
                    url,
                    key,
                    value['response'],
                    value['actual_result'],
                    value['correct'],
                    value['prompt'],
                ]
            )
            if value['correct'] == 1:
                thinking_total_match += 1
            elif value['correct'] == -1:
                thinking_total_non_in += 1
            else:
                thinking_total_non_match += 1
            thinking_count += 1

    print(
        f'Out of {thinking_count} records for thinking: match {thinking_total_match}  , non-match {thinking_total_non_match}, non-in {thinking_total_non_in} (match means exact match, non-match means wrong answer but the response is in the key column, non-in means none)'
    )
    df_thinking = pd.DataFrame(
        thinking_values_values,
        columns=[
            'URL',
            'Key',
            'Response',
            'actual_result',
            'correct',
            'Prompt',

        ],
    )
    plot_df = df_thinking[['correct']]

    mapping = {-1: "mistake", 0: "neutral", 1: "correct"}

    # Long-format DataFrame
    long_df = plot_df.melt(var_name="column", value_name="value")

    # Map numeric codes to labels
    long_df["value"] = long_df["value"].map(mapping)

    # Count occurrences per (column, value)
    counts = long_df.groupby(["column", "value"]).size().reset_index(name="count")

    # Plot
    plt.figure(figsize=(6, 3))
    sns.barplot(data=counts, x="column", y="count", hue="value")
    plt.title("Correct distribution per column [Thinking]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    df_thinking.to_csv('thinking_task.csv', index=False)









if __name__ == '__main__':
    thinking_task_given_table()

