import json
from utils.utils import get_llm_response
import os


def cell_retrieval():
    def get_random_sample(_df,_key_column, n=3):
        """Get a random sample of n rows from the DataFrame."""
        return list(_df.sample(n=n, random_state=1)[_key_column].items())
    from wikiparser import WikiTableParser
    import git
    import pandas as pd
    import diskcache
    import numpy as np
    import random
    random.seed(10)
    working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    parser_ins = WikiTableParser()
    PROMPT_SINGLE_VALUE = """Based on data regarding: 
{TABLE_TITLE}.
What is the value of '{DESIRED_COLUMN_NAME}' where '{KEY_COLUMN}' is '{KEY_VALUE}'?
Return only the value, with no additional words, punctuation, or explanation.
If you are unsure, return the most likely value from your knowledge; do not guess or make something up."""
    PROMPT_COMPARISON = """Based on data regarding: 
{TABLE_TITLE}.
Which '{Primary_key}' has higher '{Comparison_Column}', '{Value_1}' or '{Value_2}'?
Return only the value, with no additional words, punctuation, or explanation.
If you are unsure, return the most likely value from your knowledge; do not guess or make something up.
"""
    PROMPT_MAX = """Based on data regarding: 
{TABLE_TITLE}.
Which '{Primary_key}' has the maximum value in the '{Comparison_Column}' column?
Return only the value, with no additional words, punctuation, or explanation.
If you are unsure, return the most likely value from your knowledge; do not guess or make something up.
"""
    PROMPT_REPHRASE_SINGLE_VALUE = """Given the following template-based question: 
Based on data regarding: 
```
{TABLE_TITLE}.
What is the value of '{DESIRED_COLUMN_NAME}' where '{KEY_COLUMN}' is '{KEY_VALUE}'?
```
Regarding the table from wiki page {URL}, rephrase the question to be more natural and human-like, while keeping the same meaning.
For you guidance, the answer to the question is '{ANSWER}'.
Answer only with the rephrased question.
"""
    PROMPT_REPHRASE_COMPARISON = """Given the following template-based question:
Based on data regarding:
```
{TABLE_TITLE}.
Which '{Primary_key}' has higher '{Comparison_Column}', '{Value_1}' or '{Value_2}'?
```
Regarding the table from wiki page {URL}, rephrase the question to be more natural and human-like, while keeping the same meaning.
For you guidance, the answer to the question is '{ANSWER}'.
Answer only with the rephrased question.
"""
    PROMPT_REPHRASE_MAX = """Given the following template-based question:
Based on data regarding:
```
{TABLE_TITLE}.
Which '{Primary_key}' has the maximum value in the '{Comparison_Column}' column?
```
Regarding the table from wiki page {URL}, rephrase the question to be more natural and human-like, while keeping the same meaning.
For you guidance, the answer to the question is '{ANSWER}'.
Answer only with the rephrased question.
"""
    REPHRASED_SUFFIX = """{REPHRASE}
Return only the value, with no additional words, punctuation, or explanation."""
    generated_tbl_cache = diskcache.Cache(f'{working_dir}/local_dbs/tables/generated_tables.db')
    single_value_scores = {}
    comparison_scores = {}
    max_scores = {}
    MAX_ITER = 110
    count = 0
    for tbl in generated_tbl_cache.iterkeys():
        count += 1
        print(f'Retrieving {count}...')
        if count > MAX_ITER:
            break
        cfg = json.loads(generated_tbl_cache[tbl])
        if not os.path.isfile(f'../tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv'):
            continue
        df = pd.read_csv(f'../tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv')
        key = cfg['llm_generated']['key']
        if len(str(key)) == 1 or key.isnumeric():
            continue
        table_description = cfg['llm_generated']['table_title']
        columns_without_key = [c for c in df.columns if c != key]
        comparable_columns = [c for c in columns_without_key if df[c].dtype in [np.int64, np.float64]]
        single_value_scores[tbl] = {}
        comparison_scores[tbl] = {}
        max_scores[tbl] = {}
        if not set(columns_without_key).intersection(set(cfg['columns'])):
            continue
        for row_index, key_value in get_random_sample(df,key,3):
            single_value_scores[tbl][key_value] = {}

            for col in columns_without_key:
                if len(str(col)) == 1 or col.isnumeric():
                    continue
                real_value = df.iloc[row_index][col]
                if pd.isna(real_value):
                    continue
                prompt = PROMPT_SINGLE_VALUE.format(
                    TABLE_TITLE=table_description,
                    COLUMNS=cfg['columns'],
                    KEY_COLUMN=key,
                    KEY_VALUE=key_value,
                    DESIRED_COLUMN_NAME=col,
                    DESIRED_COLUMN_DTYPE=cfg['columns'][col]
                )
                response = get_llm_response(prompt)
                rephrased_prompt = PROMPT_REPHRASE_SINGLE_VALUE.format(
                    TABLE_TITLE=table_description,
                    URL=cfg['url'],
                    DESIRED_COLUMN_NAME=col,
                    KEY_COLUMN=key,
                    KEY_VALUE=key_value,
                    ANSWER=real_value,
                )
                rephrased_response = get_llm_response(rephrased_prompt)
                rephrased_llm_response = get_llm_response(REPHRASED_SUFFIX.format(REPHRASE=rephrased_response))
                if response is None or response == '' or rephrased_llm_response is None or rephrased_llm_response == '':
                    continue
                parsed_response = parser_ins.try_cast(response)
                rephrased_parsed_response = parser_ins.try_cast(rephrased_llm_response)

                parsed_real_value = parser_ins.try_cast(real_value)
                single_value_scores[tbl][key_value][col] = {
                    'real_value': parsed_real_value,
                    'response': parsed_response,
                    'correct': int(rephrased_parsed_response == parsed_real_value),
                    'rephrased_question': rephrased_response,
                    'rephrased_response': rephrased_parsed_response,
                    'prompt': prompt
                }
        if not comparable_columns:
            continue
        # For comparison, we will take pairs of rows
        for entity_1, entity_2 in zip(get_random_sample(df,key,6)[::2], get_random_sample(df,key,6)[1::2]):
            key_value_1 = entity_1[1]
            index_1 = entity_1[0]
            key_value_2 = entity_2[1]
            index_2 = entity_2[0]

            comparison_scores[tbl][str(key_value_1)+","+str(key_value_2)] = {}
            for comparable_col in comparable_columns:
                if df.iloc[index_1][comparable_col] > df.iloc[index_2][comparable_col]:
                    actual_result = key_value_1
                else:
                    actual_result = key_value_2
                prompt = PROMPT_COMPARISON.format(
                    TABLE_TITLE=table_description,
                    Primary_key=key,
                    Comparison_Column=comparable_col,
                    Value_1=key_value_1,
                    Value_2=key_value_2
                )
                response = get_llm_response(prompt)
                rephrased_prompt = PROMPT_REPHRASE_COMPARISON.format(
                    TABLE_TITLE=table_description,
                    URL=cfg['url'],
                    Primary_key=key,
                    Comparison_Column=comparable_col,
                    Value_1=key_value_1,
                    Value_2=key_value_2,
                    ANSWER=actual_result
                )
                rephrased_response = get_llm_response(rephrased_prompt)
                rephrased_llm_response = get_llm_response(REPHRASED_SUFFIX.format(REPHRASE=rephrased_response))
                if response is None or response == '':
                    continue
                parsed_response = parser_ins.try_cast(response)
                rephrased_parsed_response = parser_ins.try_cast(rephrased_llm_response)
                if parsed_response is None or rephrased_parsed_response is None:
                    continue
                comparison_scores[tbl][str(key_value_1)+","+str(key_value_2)][comparable_col] = {
                    'response': response,
                    'actual_result': actual_result,
                    'correct': int(parsed_response == actual_result),
                    'prompt': prompt,
                    'rephrased_question': rephrased_response,
                    'rephrased_response': rephrased_parsed_response
                }
        # For max, we will take a comparable column and find the max value
        for comparable_col in comparable_columns:
            max_indices = df[df[comparable_col] == df[comparable_col].max()].index
            if df.iloc[max_indices[0]][key] == 'Total' or len(comparable_col) == 1 or comparable_col.isnumeric():
                # Skip if the max value is 'Total' as it is not a valid key
                continue
            prompt = PROMPT_MAX.format(
                TABLE_TITLE=table_description,
                Primary_key=key,
                Comparison_Column=comparable_col
            )

            rephrased_prompt = PROMPT_REPHRASE_MAX.format(
                TABLE_TITLE=table_description,
                URL=cfg['url'],
                Primary_key=key,
                Comparison_Column=comparable_col,
                ANSWER=df[comparable_col].max()
            )
            rephrased_response = get_llm_response(rephrased_prompt)
            rephrased_llm_response = get_llm_response(REPHRASED_SUFFIX.format(REPHRASE=rephrased_response))


            response = get_llm_response(prompt)
            parsed_response = parser_ins.try_cast(response)
            rephrased_parsed_response = parser_ins.try_cast(rephrased_llm_response)
            parsed_response = rephrased_parsed_response
            llm_indices = list(df[df[key] == response][comparable_col].index) + list(df[df[key] == parsed_response][comparable_col].index)
            llm_indices = set(llm_indices)


            if parsed_response not in df[key].values:
                matched = -1
            else:
                matched = 0
                if len(llm_indices.intersection(max_indices)) > 0:
                    matched = 1
            max_scores[tbl][comparable_col] = {
                'response': response,
                'actual_result': list(df[df[comparable_col] == df[comparable_col].max()][key]),
                'correct': matched,
                'prompt': prompt
            }
    total_match = 0
    total_non_match = 0
    total_non_in = 0
    count = 0
    max_values_values = []
    for record in [k for k in max_scores.items() if 'http' in k[0]]:
        url = record[0]
        data = record[1]
        for key, value in data.items():
            max_values_values.append([url, key, value['response'], value['actual_result'], value['correct'], value['prompt']])
            if value['correct'] == 1:
                total_match += 1
            elif value['correct'] == -1:
                total_non_in += 1
            else:
                total_non_match += 1
            count += 1
    print(
        f'Out of {count} records: match {total_match}, non-match {total_non_match}, non-in {total_non_in} (match means exact match, non-match means wrong answer but the response is in the key column, non-in means none)')
    pd.DataFrame(max_values_values, columns=['URL', 'Key', 'Response', 'actual_result', 'correct', 'Prompt']).to_csv(
        'max_values.csv', index=False)
    single_value_list = []
    single_value_count = 0
    single_value_match = 0
    for record in [k for k in single_value_scores.items() if 'http' in k[0]]:
        url = record[0]
        data = record[1]
        for key, value in data.items():
            for col, result in value.items():
                single_value_list.append([url, key, col, result['real_value'], result['response'], result['correct'], result['prompt'], result['rephrased_question'], result['rephrased_response']])
                single_value_match += result['correct']
                single_value_count += 1
    print(f"Single value match: {single_value_match} out of {single_value_count}")
    pd.DataFrame(single_value_list, columns=['URL', 'Key', 'Column', 'Real Value', 'Response', 'Correct', 'Prompt', 'Rephrased_question', 'Rephased_response']).to_csv(
        'single_value_results.csv', index=False)

    comparison_list = []
    comparison_match = 0
    comparison_count = 0
    for record in [k for k in comparison_scores.items() if 'http' in k[0]]:
        url = record[0]
        data = record[1]
        for key, value in data.items():
            for col, result in value.items():
                comparison_list.append([url, key, col, result['response'], result['actual_result'], result['correct'], result['prompt']])
                comparison_count += 1
                comparison_match += result['correct']
    pd.DataFrame(comparison_list, columns=['URL', 'Key', 'Column', 'Response', 'Actual Result', 'Correct', 'Prompt']).to_csv(
        'comparison_results.csv', index=False)
    print(f"Comparison match: {comparison_match} out of {comparison_count}")


if __name__ == '__main__':
    cell_retrieval()