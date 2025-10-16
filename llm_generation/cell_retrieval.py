import json
from utils.utils import get_llm_response
import os


def cell_retrieval():
    def get_random_sample(_df, _key_column, n=3):
        """Get a random sample of n rows from the DataFrame."""
        return list(_df.sample(n=n, random_state=1)[_key_column].items())

    from wikiparser import WikiTableParser
    import git
    import pandas as pd
    import diskcache
    import numpy as np
    import random
    import yaml
    # model = 'gpt-4.1-2025-04-14'
    model = 'gemini-2.5-pro'
    context = True
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
    comparison_scores = {}
    max_scores = {}
    min_scores = {}
    MAX_ITER = 350
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
        key = cfg['llm_generated']['key']
        if len(str(key)) == 1 or key.isnumeric():
            continue
        table_description = cfg['llm_generated']['table_title']
        columns_without_key = [c for c in df.columns if c != key and c not in ['Total']]
        comparable_columns = [
            c
            for c in columns_without_key
            if df[c].dtype in [np.int64, np.float64] and c not in ['Total']
        ]
        single_value_scores[tbl] = {}
        comparison_scores[tbl] = {}
        max_scores[tbl] = {}
        min_scores[tbl] = {}
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
                rephrased_prompt = prompts['rephrase_wrapper'].format(
                    QUESTION=single_value_prompt,
                    TABLE_TITLE=table_description,
                    ANSWER=real_value,
                    URL=cfg['url'],

                ).strip()
                rephrased_response = get_llm_response(rephrased_prompt,MODEL=model,)
                rephrased_llm_response = get_llm_response(
                    REPHRASED_SUFFIX.format(REPHRASE=rephrased_response), MODEL=model,
                )
                if (
                    response is None
                    or response == ''
                    or rephrased_llm_response is None
                    or rephrased_llm_response == ''
                ):
                    continue
                parsed_response = parser_ins.try_cast(response)
                rephrased_parsed_response = parser_ins.try_cast(rephrased_llm_response)
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
                    'rephrased_correct': int(
                        rephrased_parsed_response == parsed_real_value
                    ),
                    'rephrased_question': rephrased_response,
                    'rephrased_response': rephrased_parsed_response,
                    'prompt': single_value_prompt,
                    'context_response': context_response,
                    'context_correct': int(context_response == parsed_real_value) if context_response is not None else 'NA',
                }
                break
            break
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
            rephrased_prompt = prompts['rephrase_wrapper'].format(
                QUESTION=max_prompt_question,
                TABLE_TITLE=table_description,
                ANSWER=df[comparable_col].max(),
                URL=cfg['url'],
            ).strip()
            rephrased_response = get_llm_response(rephrased_prompt,MODEL=model,)
            rephrased_llm_response = get_llm_response(
                REPHRASED_SUFFIX.format(REPHRASE=rephrased_response),MODEL=model,
            )

            response = get_llm_response(max_prompt,MODEL=model,)
            parsed_response = parser_ins.try_cast(response)
            if response is None or response == '' or rephrased_llm_response is None or rephrased_llm_response == '':
                continue
            rephrased_parsed_response = parser_ins.try_cast(rephrased_llm_response)

            llm_indices = list(df[df[key] == response][comparable_col].index) + list(
                df[df[key] == parsed_response][comparable_col].index
            )
            llm_indices = set(llm_indices)

            rephrased_llm_indices = list(
                df[df[key] == rephrased_response][comparable_col].index
            ) + list(df[df[key] == rephrased_parsed_response][comparable_col].index)
            rephrased_llm_indices = set(rephrased_llm_indices)

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

            if rephrased_parsed_response not in df[key].values:
                rephrased_matched = -1
            else:
                rephrased_matched = 0
                if len(rephrased_llm_indices.intersection(max_indices)) > 0:
                    rephrased_matched = 1

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
                'rephrased_response': rephrased_llm_response,
                'rephrased_question': rephrased_response,
                'rephrased_correct': rephrased_matched,
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
            rephrased_prompt = prompts['rephrase_wrapper'].format(
                QUESTION=min_prompt_question,
                TABLE_TITLE=table_description,
                ANSWER=df[comparable_col].min(),
                URL=cfg['url'],
            ).strip()
            rephrased_response = get_llm_response(rephrased_prompt,MODEL=model,)
            rephrased_llm_response = get_llm_response(
                REPHRASED_SUFFIX.format(REPHRASE=rephrased_response),MODEL=model,
            )
            if rephrased_llm_response is None or rephrased_llm_response == '':
                continue
            response = get_llm_response(min_prompt,MODEL=model,)
            parsed_response = parser_ins.try_cast(response)

            rephrased_parsed_response = parser_ins.try_cast(rephrased_llm_response)

            llm_indices = list(df[df[key] == response][comparable_col].index) + list(
                df[df[key] == parsed_response][comparable_col].index
            )
            llm_indices = set(llm_indices)

            rephrased_llm_indices = list(
                df[df[key] == rephrased_response][comparable_col].index
            ) + list(df[df[key] == rephrased_parsed_response][comparable_col].index)
            rephrased_llm_indices = set(rephrased_llm_indices)


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

            if rephrased_parsed_response not in df[key].values:
                rephrased_matched = -1
            else:
                rephrased_matched = 0
                if len(rephrased_llm_indices.intersection(min_indices)) > 0:
                    rephrased_matched = 1
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
                'rephrased_response': rephrased_llm_response,
                'rephrased_question': rephrased_response,
                'rephrased_correct': rephrased_matched,
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
    max_rephrased_matched = 0
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
                    value['rephrased_correct'],
                    value['keys_provided'],
                    value['prompt'],
                    value['rephrased_question'],
                    value['rephrased_response'],
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
            if value['rephrased_correct'] == 1:
                max_rephrased_matched += 1
            if 'context_correct' in value and value['context_correct'] == 1:
                max_context_matched += 1

    print(
        f'Out of {max_count} records for MAX: match {max_total_match} ({max_with_keys_match}) , non-match {max_total_non_match}, non-in {max_total_non_in} (match means exact match, non-match means wrong answer but the response is in the key column, non-in means none)'
    )
    print(f"Rephrased matched: {max_rephrased_matched} out of {max_count}")
    print(f"Context: {max_context_matched} out of {max_count}")
    pd.DataFrame(
        max_values_values,
        columns=[
            'URL',
            'Key',
            'Response',
            'actual_result',
            'correct',
            'Rephrased correct',
            'With keys',
            'Prompt',
            'Rephrased Question',
            'Rephrased Response',
            'Context Response',
            'Context Correct',
        ],
    ).to_csv('max_values.csv', index=False)

    min_total_match = 0
    min_total_non_match = 0
    min_total_non_in = 0
    min_count = 0
    min_values_values = []
    min_with_keys_match = 0
    min_rephrased_matched = 0
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
                    value['rephrased_correct'],
                    value['keys_provided'],
                    value['prompt'],
                    value['rephrased_question'],
                    value['rephrased_response'],
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
            if value['rephrased_correct'] == 1:
                min_rephrased_matched += 1
            if 'context_correct' in value and value['context_correct'] == 1:
                min_context_matched += 1
    print(
        f'Out of {min_count} records for MIN: match {min_total_match} ({min_with_keys_match}), non-match {min_total_non_match}, non-in {min_total_non_in} (match means exact match, non-match means wrong answer but the response is in the key column, non-in means none)'
    )
    print(f"Rephrased matched: {min_rephrased_matched} out of {min_count}")
    print(f"Context matched: {min_context_matched} out of {min_count}")
    pd.DataFrame(
        min_values_values,
        columns=[
            'URL',
            'Key',
            'Response',
            'actual_result',
            'correct',
            'Rephrased correct',
            'With keys',
            'Prompt',
            'Rephrased Question',
            'Rephrased Response',
            'Context Response',
            'Context Correct',
        ],
    ).to_csv('min_values.csv', index=False)

    single_value_list = []
    single_value_count = 0
    single_value_match = 0
    single_value_rephrased_match = 0
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
                        result['rephrased_question'],
                        result['rephrased_response'],
                        result['context_response'] if 'context_response' in result else 'NA',
                        result['context_correct'] if 'context_correct' in result else 'NA',
                    ]
                )
                single_value_match += result['correct']
                single_value_count += 1
                single_value_rephrased_match += result['rephrased_correct']
                if 'context_correct' in result and result['context_correct'] != 'NA':
                    single_value_context_match += result['context_correct']

    print(f"Single value match: {single_value_match} out of {single_value_count}")
    print(f"Single value rephrased match: {single_value_rephrased_match} out of {single_value_count}")
    print(f"Context matched: {single_value_context_match} out of {single_value_count}")
    pd.DataFrame(
        single_value_list,
        columns=[
            'URL',
            'Key',
            'Column',
            'Real Value',
            'Response',
            'Correct',
            'Prompt',
            'Rephrased_question',
            'Rephased_response',
            'Context Response',
            'Context Correct',
        ],
    ).to_csv('single_value_results.csv', index=False)
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
    pd.DataFrame(
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
    ).to_csv('max_comparison_results.csv', index=False)
    print(
        f"Comparison match: {max_comparison_match} out of {max_comparison_count} for MAX comparison"
    )
    print(f"Rephrased matched: {max_comparison_rephrased_match} out of {max_comparison_count} for MAX comparison")
    print(f"Context matched: {max_comparison_context_match} out of {max_comparison_count} for MAX comparison")
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
    pd.DataFrame(
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
    ).to_csv('min_comparison_results.csv', index=False)
    print(
        f"Comparison match: {min_comparison_match} out of {min_comparison_count} for MIN comparison"
    )
    print(f"Rephrased matched: {min_comparison_rephrased_match} out of {min_comparison_count} for MIN comparison")
    print(f"Context matched: {min_comparison_context_match} out of {min_comparison_count} for MIN comparison")

if __name__ == '__main__':
    import time
    idx = 0
    while True:
        try:
            cell_retrieval()
        except Exception as e:
            print(f'Error occurred: {e}, restarting...')
        time.sleep(30)
        idx += 1
        print(f'Iteration {idx} completed, restarting...')

