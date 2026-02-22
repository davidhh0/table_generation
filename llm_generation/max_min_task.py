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
    max_scores = {}
    min_scores = {}
    MAX_ITER = 150
    count = 0
    rephrased_response = 'NA'
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
        columns_without_key = [c for c in df.columns if c != key and c not in ['Total']]
        comparable_columns = [
            c
            for c in columns_without_key
            if df[c].dtype in [np.int64, np.float64] and c not in ['Total']
        ]
        max_scores[tbl] = {}
        min_scores[tbl] = {}
        if not set(columns_without_key).intersection(set(cfg['columns'])):
            continue
        if not comparable_columns:
            continue
        # For max, we will take a comparable column and find the max value
        for comparable_col in comparable_columns:

            max_indices = df[df[comparable_col] == df[comparable_col].max()].index
            if (
                len(comparable_col) == 1
                or comparable_col.isnumeric()
            ):
                # Skip if the max value is not a valid key
                continue
            max_prompt_question = prompts['max_value'].format(
                Primary_key=key,
                Comparison_Column=comparable_col,
                PROVIDED_KEYS=''
            ).strip()
            max_prompt = prompts['wrapper'].format(
                TABLE_TITLE=table_description,
                QUESTION=max_prompt_question
            ).strip()

            max_prompt_with_keys_question = prompts['max_value'].format(
                Primary_key=key,
                Comparison_Column=comparable_col,
                PROVIDED_KEYS=f"Among the following possible values: {', '.join([str(k) for k in df[key].values])}",
            ).strip()

            max_prompt_with_keys = prompts['wrapper'].format(
                TABLE_TITLE=table_description,
                QUESTION=max_prompt_with_keys_question
            ).strip()
            response_with_keys = parser_ins.try_cast(
                get_llm_response(max_prompt_with_keys, MODEL=model,)
            )


            response = get_llm_response(max_prompt,MODEL=model,)
            parsed_response = parser_ins.try_cast(response)
            if response is None or response == '':
                continue

            llm_indices = list(df[df[key] == response][comparable_col].index) + list(
                df[df[key] == parsed_response][comparable_col].index
            )
            llm_indices = set(llm_indices)


            context_response,context_matched = None,None
            if context:
                context_response = prompts['context_wrapper'].format(TABLE_TITLE=table_description, QUESTION=max_prompt_question,URL=cfg['url'],)
                context_response = get_llm_response(context_response, MODEL=model,context=True)
                context_response = parser_ins.try_cast(context_response)




            if parsed_response not in df[key].values:
                matched = -1
            else:
                matched = 0
                if len(llm_indices.intersection(max_indices)) > 0:
                    matched = 1


            if response_with_keys not in df[key].values:
                max_with_keys = -1
            else:
                max_with_keys = 0
                if len(llm_indices.intersection(max_indices)) > 0:
                    max_with_keys = 1

            if context_response is not None:
                if context_response not in df[key].values:
                    context_matched = -1
                else:
                    context_matched = 0
                    if context_response in df[df[comparable_col] == df[comparable_col].max()][key].values:
                        context_matched = 1

            max_scores[tbl][comparable_col] = {
                'response': response,
                'actual_result': list(
                    df[df[comparable_col] == df[comparable_col].max()][key]
                ),
                'correct': matched,
                'prompt': max_prompt_question,
                'keys_provided': max_with_keys,
                'context_response': context_response,
                'context_correct': context_matched if context_response is not None else 'NA',
            }
            break
        # For min, we will take a comparable column and find the  min value
        for comparable_col in comparable_columns:
            min_indices = df[df[comparable_col] == df[comparable_col].min()].index
            if (
                len(comparable_col) == 1
                or comparable_col.isnumeric()
            ):
                # Skip if the min value is 'Total' as it is not a valid key
                continue

            min_prompt_question = prompts['min_value'].format(
                Primary_key=key,
                Comparison_Column=comparable_col,
                PROVIDED_KEYS='',
            ).strip()
            min_prompt = prompts['wrapper'].format(
                TABLE_TITLE=table_description,
                QUESTION=min_prompt_question
            ).strip()
            min_prompt_with_keys_question = prompts['min_value'].format(
                Primary_key=key,
                Comparison_Column=comparable_col,
                PROVIDED_KEYS=f"Among the following possible values: {', '.join([str(k) for k in df[key].values])}",
            ).strip()
            min_prompt_with_keys = prompts['wrapper'].format(
                TABLE_TITLE=table_description,
                QUESTION=min_prompt_with_keys_question
            ).strip()

            response_with_keys = parser_ins.try_cast(
                get_llm_response(min_prompt_with_keys, MODEL=model,)
            )



            response = get_llm_response(min_prompt,MODEL=model,)
            parsed_response = parser_ins.try_cast(response)


            llm_indices = list(df[df[key] == response][comparable_col].index) + list(
                df[df[key] == parsed_response][comparable_col].index
            )
            llm_indices = set(llm_indices)




            if context:
                context_response = prompts['context_wrapper'].format(TABLE_TITLE=table_description, QUESTION=min_prompt_question,URL=cfg['url'],)
                context_response = get_llm_response(context_response, MODEL=model,context=True)
                context_response = parser_ins.try_cast(context_response)
            context_matched = None
            if context_response is not None:
                if context_response not in df[key].values:
                    context_matched = -1
                else:
                    context_matched = 0
                    if context_response in df[df[comparable_col] == df[comparable_col].min()][key].values:
                        context_matched = 1

            if parsed_response not in df[key].values:
                matched = -1
            else:
                matched = 0
                if len(llm_indices.intersection(min_indices)) > 0:
                    matched = 1


            if response_with_keys not in df[key].values:
                min_with_keys = -1
            else:
                min_with_keys = 0
                if len(llm_indices.intersection(min_indices)) > 0:
                    min_with_keys = 1
            min_scores[tbl][comparable_col] = {
                'response': response,
                'actual_result': list(
                    df[df[comparable_col] == df[comparable_col].min()][key]
                ),
                'correct': matched,
                'prompt': min_prompt_question,
                'keys_provided': min_with_keys,
                'context_response': context_response,
                'context_correct': context_matched if context_response is not None else 'NA',
            }
            break

    print(f"Total tables processed: {len(generated_tbl_cache)}")
    max_total_match = 0
    max_total_non_match = 0
    max_total_non_in = 0
    max_count = 0
    max_values_values = []
    max_with_keys_match = 0
    max_context_matched = 0
    for record in max_scores.items():
        url = record[0]
        data = record[1]
        for key, value in data.items():
            max_values_values.append(
                [
                    url,
                    key,
                    value['response'],
                    value['actual_result'],
                    value['correct'],
                    value['keys_provided'],
                    value['prompt'],
                    value['context_response'],
                    value['context_correct'] if 'context_correct' in value else 'NA',
                ]
            )
            if value['correct'] == 1:
                max_total_match += 1
            elif value['correct'] == -1:
                max_total_non_in += 1
            else:
                max_total_non_match += 1
            max_count += 1
            if value['keys_provided'] == 1:
                max_with_keys_match += 1
            if 'context_correct' in value and value['context_correct'] == 1:
                max_context_matched += 1

    print(
        f'Out of {max_count} records for MAX: match {max_total_match} ({max_with_keys_match}) , non-match {max_total_non_match}, non-in {max_total_non_in} (match means exact match, non-match means wrong answer but the response is in the key column, non-in means none)'
    )
    print(f"Context: {max_context_matched} out of {max_count}")
    df_max = pd.DataFrame(
        max_values_values,
        columns=[
            'URL',
            'Key',
            'Response',
            'actual_result',
            'correct',
            'With keys',
            'Prompt',
            'Context Response',
            'Context Correct',
        ],
    )
    plot_df = df_max[['correct','With keys']]
    if context:
        plot_df = df_max[['correct', 'With keys', 'Context Correct']]

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
    plt.title("Correct distribution per column [Max]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    df_max.to_csv('max_values.csv', index=False)


    # ======= Min =======


    min_total_match = 0
    min_total_non_match = 0
    min_total_non_in = 0
    min_count = 0
    min_values_values = []
    min_with_keys_match = 0
    min_context_matched = 0
    for record in min_scores.items():
        url = record[0]
        data = record[1]
        for key, value in data.items():
            min_values_values.append(
                [
                    url,
                    key,
                    value['response'],
                    value['actual_result'],
                    value['correct'],
                    value['keys_provided'],
                    value['prompt'],
                    value['context_response'],
                    value['context_correct'] if 'context_correct' in value else 'NA',
                ]
            )
            if value['correct'] == 1:
                min_total_match += 1
            elif value['correct'] == -1:
                min_total_non_in += 1
            else:
                min_total_non_match += 1
            min_count += 1
            if value['keys_provided'] == 1:
                min_with_keys_match += 1
            if 'context_correct' in value and value['context_correct'] == 1:
                min_context_matched += 1
    print(
        f'Out of {min_count} records for MIN: match {min_total_match} ({min_with_keys_match}), non-match {min_total_non_match}, non-in {min_total_non_in} (match means exact match, non-match means wrong answer but the response is in the key column, non-in means none)'
    )
    print(f"Context matched: {min_context_matched} out of {min_count}")
    df_min = pd.DataFrame(
        min_values_values,
        columns=[
            'URL',
            'Key',
            'Response',
            'actual_result',
            'correct',
            'With keys',
            'Prompt',
            'Context Response',
            'Context Correct',
        ],
    )
    plot_df = df_min[['correct', 'With keys']]
    if context:
        plot_df = df_min[['correct', 'With keys', 'Context Correct']]

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
    plt.title("Correct distribution per column [Min]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    df_min.to_csv('min_values.csv', index=False)







if __name__ == '__main__':
    cell_retrieval()

