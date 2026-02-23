import json
from utils.utils import get_llm_response
import os


def cell_retrieval():
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
    key_model =  conf['title_and_key_model']
    model = conf['llm_model']
    context = conf['context']
    random.seed(10)
    working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    parser_ins = WikiTableParser()
    with open(f'{working_dir}/llm_generation/prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)


    REPHRASED_SUFFIX = """{REPHRASE}
Return only the value, with no additional words, punctuation, or explanation."""
    generated_tbl_cache = diskcache.Cache(
        f'{working_dir}/local_dbs/tables/generated_tables.db'
    )
    single_value_scores = {}
    MAX_ITER = 300
    count = 0
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
                if pd.isna(real_value):
                    continue
                single_value_prompt = prompts['single_value'].format(
                    DESIRED_COLUMN_NAME=col, KEY_COLUMN=key, KEY_VALUE=key_value
                ).strip()
                prompt = (
                    prompts['wrapper']
                    .format(TABLE_TITLE=table_description, QUESTION=single_value_prompt)
                    .strip()
                )
                response = get_llm_response(prompt, MODEL=model,)

                if (
                    response is None
                    or response == ''
                ):
                    continue
                parsed_response = parser_ins.try_cast(response)
                context_response = None
                if context:
                    context_response = prompts['context_wrapper'].format(TABLE_TITLE=table_description, QUESTION=single_value_prompt,URL=cfg['url'],)
                    context_response = get_llm_response(context_response, MODEL=model,context=True)
                    context_response = parser_ins.try_cast(context_response)
                parsed_real_value = parser_ins.try_cast(real_value)
                single_value_scores[tbl][key_value][col] = {
                    'real_value': parsed_real_value,
                    'response': parsed_response,
                    'correct': int(parsed_response == parsed_real_value),
                    'prompt': single_value_prompt,
                    'context_response': context_response,
                    'context_correct': int(context_response == parsed_real_value) if context_response is not None else 'NA',
                }
                break
            break
        if not comparable_columns:
            continue

    print(f"Total tables processed: {len(generated_tbl_cache)}")

    # ====== Single values ======


    single_value_list = []
    single_value_count = 0
    single_value_match = 0
    single_value_context_match = 0
    for record in single_value_scores.items():
        url = record[0]
        data = record[1]
        for key, value in data.items():
            for col, result in value.items():
                single_value_list.append(
                    [
                        url,
                        key,
                        col,
                        result['real_value'],
                        result['response'],
                        result['correct'],
                        result['prompt'],
                        result['context_response'] if 'context_response' in result else 'NA',
                        result['context_correct'] if 'context_correct' in result else 'NA',
                    ]
                )
                single_value_match += result['correct']
                single_value_count += 1
                if 'context_correct' in result and result['context_correct'] != 'NA':
                    single_value_context_match += result['context_correct']

    print(f"Single value match: {single_value_match} out of {single_value_count}")
    print(f"Context matched: {single_value_context_match} out of {single_value_count}")

    df_single_value = pd.DataFrame(
        single_value_list,
        columns=[
            'URL',
            'Key',
            'Column',
            'Real Value',
            'Response',
            'Correct',
            'Prompt',
            'Context Response',
            'Context Correct',
        ],
    )
    plt.figure()
    plot_df = df_single_value[['Correct',]]
    if context:
        plot_df = df_single_value[['correct', 'Context Correct']]

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
    plt.title("Correct distribution per column [Single value]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    df_single_value.to_csv('single_value_results.csv', index=False)



if __name__ == '__main__':
    import time
    idx = 0
    cell_retrieval()

