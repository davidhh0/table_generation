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
    key_model = 'gpt-4.1-2025-04-14' # conf['llm_model']
    model = 'gemini-2.5-pro'
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
    comparison_scores = {}
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
        comparison_scores[tbl] = {}
        if not set(columns_without_key).intersection(set(cfg['columns'])):
            continue

        if not comparable_columns:
            continue
        # For comparison, we will take triples of rows
        for entity_1, entity_2, entity_3 in zip(
            get_random_sample(df, key, 6)[::2],
            get_random_sample(df, key, 6)[1::2],
            get_random_sample(df, key, 6)[2::3],
        ):
            key_value_1 = entity_1[1]
            index_1 = entity_1[0]
            key_value_2 = entity_2[1]
            index_2 = entity_2[0]
            key_value_3 = entity_3[1]
            index_3 = entity_3[0]
            indices_list = [index_1, index_2, index_3]
            comparison_scores[tbl][
                str(key_value_1) + "," + str(key_value_2) + "," + str(key_value_3)
            ] = {}
            for comparable_col in comparable_columns:
                if len(str(comparable_col)) == 1 or comparable_col.isnumeric():
                    continue
                arg_max = df.iloc[indices_list][comparable_col].idxmax()
                actual_value_max = df.iloc[arg_max][key]
                arg_min = df.iloc[[index_1, index_2, index_3]][comparable_col].idxmin()
                actual_value_min = df.iloc[arg_min][key]
                ### MAX section:
                max_prompt_question = prompts['comparison_max'].format(
                    Primary_key=key,
                    Comparison_Column=comparable_col,
                    Value_1=key_value_1,
                    Value_2=key_value_2,
                    Value_3=key_value_3,
                ).strip()
                max_prompt = prompts['wrapper'].format(
                    TABLE_TITLE=table_description, QUESTION=max_prompt_question
                ).strip()
                max_response = get_llm_response(max_prompt, MODEL=model)
                max_rephrased_prompt = prompts['rephrase_wrapper'].format(
                    QUESTION=max_prompt_question,
                    TABLE_TITLE=table_description,
                    ANSWER=actual_value_max,
                    URL=cfg['url'],
                    MODEL=model,
                ).strip()
                max_rephrased_response = get_llm_response(max_rephrased_prompt,MODEL=model,)
                max_rephrased_llm_response = get_llm_response(
                    REPHRASED_SUFFIX.format(REPHRASE=max_rephrased_response),MODEL=model,
                )
                if max_response is None or max_response == '':
                    continue
                max_parsed_response = parser_ins.try_cast(max_response)
                max_rephrased_parsed_response = parser_ins.try_cast(
                    max_rephrased_llm_response
                )
                if max_parsed_response is None or max_rephrased_parsed_response is None:
                    continue

                # Min section:
                min_prompt_question = prompts['comparison_min'].format(
                    Primary_key=key,
                    Comparison_Column=comparable_col,
                    Value_1=key_value_1,
                    Value_2=key_value_2,
                    Value_3=key_value_3,
                ).strip()
                min_prompt = prompts['wrapper'].format(
                    TABLE_TITLE=table_description, QUESTION=min_prompt_question
                ).strip()

                min_response = get_llm_response(min_prompt, MODEL=model)
                min_rephrased_prompt = prompts['rephrase_wrapper'].format(
                    QUESTION=min_prompt_question,
                    TABLE_TITLE=table_description,
                    ANSWER=actual_value_min,
                    URL=cfg['url'],MODEL=model,
                ).strip()
                min_rephrased_response = get_llm_response(min_rephrased_prompt,MODEL=model,)
                min_rephrased_llm_response = get_llm_response(
                    REPHRASED_SUFFIX.format(REPHRASE=min_rephrased_response),MODEL=model,
                )
                if min_response is None or min_response == '':
                    continue
                min_parsed_response = parser_ins.try_cast(min_response)
                min_rephrased_parsed_response = parser_ins.try_cast(
                    min_rephrased_llm_response
                )
                if min_parsed_response is None or min_rephrased_parsed_response is None:
                    continue

                max_context_response = None
                min_context_response = None
                if context:
                    max_context_response = prompts['context_wrapper'].format(TABLE_TITLE=table_description, QUESTION=max_prompt_question,URL=cfg['url'],)
                    max_context_response = get_llm_response(max_context_response, MODEL=model,context=True)
                    max_context_response = parser_ins.try_cast(max_context_response)
                    min_context_response = prompts['context_wrapper'].format(TABLE_TITLE=table_description, QUESTION=min_prompt_question,URL=cfg['url'],)
                    min_context_response = get_llm_response(min_context_response, MODEL=model,context=True)
                    min_context_response = parser_ins.try_cast(min_context_response)

                comparison_scores[tbl][
                    str(key_value_1) + "," + str(key_value_2) + "," + str(key_value_3)
                ][comparable_col] = {
                    'max_response': max_response,
                    'max_actual_result': actual_value_max,
                    'max_correct': int(max_parsed_response == actual_value_max),
                    'max_rephrased_correct': int(
                        max_rephrased_parsed_response == actual_value_max
                    ),
                    'max_prompt': max_prompt_question,
                    'max_rephrased_question': max_rephrased_response,
                    'max_rephrased_response': max_rephrased_parsed_response,
                    # Min values:
                    'min_response': min_response,
                    'min_actual_result': actual_value_min,
                    'min_correct': int(min_parsed_response == actual_value_min),
                    'min_rephrased_correct': int(
                        min_rephrased_parsed_response == actual_value_min
                    ),
                    'min_prompt': min_prompt_question,
                    'min_rephrased_question': min_rephrased_response,
                    'min_rephrased_response': min_rephrased_llm_response,
                    # Context responses
                    'max_context_response': max_context_response,
                    'min_context_response': min_context_response,
                    'max_context_correct': int(max_context_response == actual_value_max) ,
                    'min_context_correct': int(min_context_response == actual_value_min),
                }
                break
            break

    print(f"Total tables processed: {len(generated_tbl_cache)}")

    # Now we will create a comparison list
    # The comparison list will contain the URL, key, column, response, actual result,
    max_comparison_list = []
    max_comparison_match = 0
    max_comparison_count = 0
    max_comparison_rephrased_match = 0
    max_comparison_context_match = 0
    for record in comparison_scores.items():
        url = record[0]
        data = record[1]
        for key, value in data.items():
            for col, result in value.items():
                max_comparison_list.append(
                    [
                        url,
                        key,
                        col,
                        result['max_response'],
                        result['max_actual_result'],
                        result['max_correct'],
                        result['max_rephrased_correct'],
                        result['max_prompt'],
                        result['max_rephrased_question'],
                        result['max_rephrased_response'],
                        result['max_context_response'] if 'max_context_response' in result else 'NA',
                        result['max_context_correct'] if 'max_context_correct' in result else 'NA',
                    ]
                )
                max_comparison_count += 1
                max_comparison_rephrased_match += result['max_rephrased_correct']
                max_comparison_match += result['max_correct']
                if 'max_context_correct' in result and result['max_context_correct'] != 'NA':
                    max_comparison_context_match += result['max_context_correct']

    print(
        f"Comparison match: {max_comparison_match} out of {max_comparison_count} for MAX comparison"
    )
    print(f"Rephrased matched: {max_comparison_rephrased_match} out of {max_comparison_count} for MAX comparison")
    print(f"Context matched: {max_comparison_context_match} out of {max_comparison_count} for MAX comparison")
    df_max_comparison = pd.DataFrame(
        max_comparison_list,
        columns=[
            'URL',
            'Key',
            'Column',
            'Response',
            'Actual Result',
            'Correct',
            'Rephrased Correct',
            'Prompt',
            'Rephrased Question',
            'Rephrased Response',
            'Context Response',
            'Context Correct',
        ],
    )
    plot_df = df_max_comparison[['Correct', 'Rephrased Correct']]
    if context:
        plot_df = df_max_comparison[['correct', 'Rephrased correct', 'Context Correct']]

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
    plt.title("Correct distribution per column [Max comparison]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    df_max_comparison.to_csv('max_comparison_results.csv', index=False)





    min_comparison_list = []
    min_comparison_match = 0
    min_comparison_count = 0
    min_comparison_rephrased_match = 0
    min_comparison_context_match = 0
    for record in comparison_scores.items():
        url = record[0]
        data = record[1]
        for key, value in data.items():
            for col, result in value.items():
                min_comparison_list.append(
                    [
                        url,
                        key,
                        col,
                        result['min_response'],
                        result['min_actual_result'],
                        result['min_correct'],
                        result['min_rephrased_correct'],
                        result['min_prompt'],
                        result['min_rephrased_question'],
                        result['min_rephrased_response'],
                        result['min_context_response'] if 'min_context_response' in result else 'NA',
                        result['min_context_correct'] if 'min_context_correct' in result else 'NA',
                    ]
                )
                min_comparison_count += 1
                min_comparison_rephrased_match += result['min_rephrased_correct']
                min_comparison_match += result['min_correct']
                if 'min_context_correct' in result and result['min_context_correct'] != 'NA':
                    min_comparison_context_match += result['min_context_correct']

    print(
        f"Comparison match: {min_comparison_match} out of {min_comparison_count} for MIN comparison"
    )
    print(f"Rephrased matched: {min_comparison_rephrased_match} out of {min_comparison_count} for MIN comparison")
    print(f"Context matched: {min_comparison_context_match} out of {min_comparison_count} for MIN comparison")
    df_min_comparison = pd.DataFrame(
        min_comparison_list,
        columns=[
            'URL',
            'Key',
            'Column',
            'Response',
            'Actual Result',
            'Correct',
            'Rephrased Correct',
            'Prompt',
            'Rephrased Question',
            'Rephrased Response',
            'Context Response',
            'Context Correct',
        ],
    )
    plot_df = df_min_comparison[['Correct', 'Rephrased Correct']]
    if context:
        plot_df = df_min_comparison[['correct', 'Rephrased correct', 'Context Correct']]

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
    plt.title("Correct distribution per column [Min comparison]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    df_min_comparison.to_csv('max_comparison_results.csv', index=False)



if __name__ == '__main__':
    cell_retrieval()
