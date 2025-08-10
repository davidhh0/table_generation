import os.path
from wikiparser import WikiTableParser
import pandas as pd
from io import StringIO, BufferedReader
import json
import redis
import re

from compared_tbls import run_compare
from utils.utils import get_revid, page_details, get_sample, get_llm_response
import diskcache

# cache = diskcache.Cache('llm_cache.db')
# openai.api_key = os.environ["openai_api_key"]
# MODEL = 'gpt-4.1-2025-04-14'
r_generated_tbls = redis.Redis(
        host='localhost', port=6379, db=4, decode_responses=True
    )
r_generated_tbls_with_example = redis.Redis(
        host='localhost', port=6379, db=9, decode_responses=True
    )



TITLE_PROMPT = """
Given the following article name: '{ARTICLE_NAME}', and the following columns along with some metadata about each column:
{DATA_SAMPLE}
I want you to describe the table title in one single sentence.
Don't repeat any columns in your response.
Response only the sentence concisely.
Examples of table descriptions:
    "Riders who took part in 2009 Tour de France"
    "Delegates of Miss New York USA 2012"
    "Vital statistics of Serbia's demographics"
    """

KEY_PROMPT = """
Based on the table title: {ARTICLE_NAME}, table description: {TABLE_TITLE}
and some sample data: {DATA_SAMPLE}
I need you to use Chain of Thought, first figure out the subject of the table based on its title,
for example if the title is "List of countries by population" the subject is "countries".

what is the most probable key-column out of the following columns (I mentioned for each column if it's unique or not): 
{TABLE_COLUMNS} ?

Response with the following format: "Key: 'YOUR_ANSWER'"
without any further explanation.
    """

TEMPLATE = """
List {TABLE_TITLE} - as many as possible to fit into response.
The response will be formatted as JSON shown below.
Each element of the response will contain {NUM_FIELDS} fields: {COLUMNS}.

Do not output any additional text that is not in the following format:
RESPONSE FORMAT:
{RESPONSE_FORMAT} \\n
"""

TEMPLATE_ROW_EXAMPLE = """
List {TABLE_TITLE} - as many as possible to fit into response.
The response will be formatted as JSON shown below.
Each element of the response will contain {NUM_FIELDS} fields: {COLUMNS}.
You are given the first row of the table as an example:
{EXAMPLE_ROW}
Do not output any additional text that is not in the following format:
RESPONSE FORMAT:
{RESPONSE_FORMAT} \\n
"""

def run(chunk):
    counter = 0
    parser_ins = WikiTableParser()
    tables = []
    keys_recall_scores = []
    keys_precision_scores = []
    keys_f1_scores = []
    non_keys_recall_scores = []
    non_keys_precision_scores = []
    non_keys_f1_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    rel_nk_acc_scores = []
    for key,value in {k: v for d in chunk.tolist() for k, v in d.items()}.items():
        cfg = json.loads(value)
        try:
            info = get_revid(cfg['page_id'])
            url = "https://en.wikipedia.org/wiki/" + info['e_title']
            tbl_metadata = page_details(url, cfg['page_id'])
        except:
            continue
        if not os.path.isfile(f'tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv'):
            continue
        df = pd.read_csv(f'tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv')
        article_name = cfg["article_name"]
        title_prompt = (
            TITLE_PROMPT.format(
                ARTICLE_NAME=cfg['article_name'],
                DATA_SAMPLE=get_sample(df, parser_ins.try_cast),
                TABLE_COLUMNS=cfg['columns'],
            )
            .strip()
            .strip('"')
        )
        title_response = get_llm_response(title_prompt)

        uniqueness = ", ".join(
            [f"{k} ({'unique' if df[k].is_unique else 'not-unique'})" for k in df]
        )

        response_format = "|".join([f'"<{k}>"' for k in df.columns.tolist()])

        key_prompt = KEY_PROMPT.format(
            ARTICLE_NAME=article_name,
            TABLE_TITLE=title_response,
            DATA_SAMPLE=df.sample(n=3, random_state=1).to_dict('list').__str__(),
            TABLE_COLUMNS=uniqueness,
        )
        key_response = get_llm_response(key_prompt,)
        try:
            tbl_key = re.search(r"'(.+)'", key_response).group(1)
        except TypeError as e:
            continue
        cfg['table_key'] = key

        prompt = TEMPLATE.format(
            NUM_FIELDS=cfg['shape'][1],
            COLUMNS=cfg['columns'],
            TABLE_TITLE=title_response,
            RESPONSE_FORMAT=response_format,
            EXAMPLE_ROW=df.iloc[0].to_dict(),
        )
        response = get_llm_response(prompt)
        if response is None:
            print(f"LLM response is None for {cfg['page_id']}. Skipping.")
            continue
        try:
            llm_tbl = pd.read_csv(
                StringIO(response), sep='|', names=df.columns.tolist()
            )
        except pd.errors.ParserError as e:
            print(f"Error parsing table for {cfg['page_id']}: {e}")
            continue
        llm_tbl.to_csv(
            f'llm_tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv', index=False
        )
        kr, kp, kf1, nkr, nkp, nkf1, r, p, f1, rnka, dtype_scores = run_compare(
            fetched_table=llm_tbl,
            gt_table=df.copy(),
            key_column=[tbl_key],
            key_column_type=['text'],
            epsilons=[
                k for k, v in cfg['columns'].items() if v in ['float64', 'int64']
            ],
            columns_dtypes={**cfg['columns'], **{k:'date' for k in cfg['dates'].keys()}}
        )
        print(kr, kp, kf1, nkr, nkp, nkf1, r, p, f1, rnka)

        if None in [kr, kp, kf1, nkr, nkp, nkf1, r, p, f1, rnka]:
            print(f'Skipping table {cfg["page_id"]} due to None values in scores.')
            continue
        score_json = {
            'key_recall': kr,
            'key_precision': kp,
            'key_f1_score': kf1,
            'non_key_recall': nkr,
            'non_key_precision': nkp,
            'non_key_f1_score': nkf1,
            'recall': r,
            'precision': p,
            'f1_score': f1,
            'relative_non_key_accuracy': rnka,
            'dtype_scores': dtype_scores,
        }
        generated_tbl_data = {
            **cfg,
            **{
                'llm_generated': {
                    'key': tbl_key,
                    'table_title': title_response,
                    'shape': llm_tbl.shape,
                }
            },
            **{'article_metadata': json.loads(tbl_metadata)},
            **{'scores': score_json},
        }
        r_generated_tbls.set(key, json.dumps(generated_tbl_data))
        counter += 1
        print(f'Processed {counter} tables out of {chunk.__len__()} chunks.')
        tables.append(cfg['page_id'])
        keys_recall_scores.append(kr)
        keys_precision_scores.append(kp)
        keys_f1_scores.append(kf1)
        non_keys_recall_scores.append(nkr)
        non_keys_precision_scores.append(nkp)
        non_keys_f1_scores.append(nkf1)
        recall_scores.append(r)
        precision_scores.append(p)
        f1_scores.append(f1)
        rel_nk_acc_scores.append(rnka)
    res_df = pd.DataFrame(
        [
            tables,
            keys_recall_scores,
            keys_precision_scores,
            keys_f1_scores,
            non_keys_recall_scores,
            non_keys_precision_scores,
            non_keys_f1_scores,
            rel_nk_acc_scores,
            recall_scores,
            precision_scores,
            f1_scores,
        ]
    ).T
    res_df.columns = [
        'Table',
        'Keys_Recall',
        'Keys_Precision',
        'Keys_F1_Score',
        'Non_Keys_Recall',
        'Non_Keys_Precision',
        'Non_Keys_F1_Score',
        'Rel_Non_Keys_Accuracy',
        'Recall',
        'Precision',
        'F1_Score',
    ]

    res_df['Keys_Recall'] = res_df['Keys_Recall'].astype(float).round(4)
    res_df['Keys_Precision'] = res_df['Keys_Precision'].astype(float).round(4)
    res_df['Keys_F1_Score'] = res_df['Keys_F1_Score'].astype(float).round(4)
    res_df['Non_Keys_Recall'] = res_df['Non_Keys_Recall'].astype(float).round(4)
    res_df['Non_Keys_Precision'] = res_df['Non_Keys_Precision'].astype(float).round(4)
    res_df['Non_Keys_F1_Score'] = res_df['Non_Keys_F1_Score'].astype(float).round(4)
    res_df['Rel_Non_Keys_Accuracy'] = (
        res_df['Rel_Non_Keys_Accuracy'].astype(float).round(4)
    )
    res_df['Recall'] = res_df['Recall'].astype(float).round(4)
    res_df['Precision'] = res_df['Precision'].astype(float).round(4)
    res_df['F1_Score'] = res_df['F1_Score'].astype(float).round(4)

    means = pd.DataFrame(['All'] + res_df.mean(axis=0, numeric_only=True).tolist()).T
    means.columns = res_df.columns
    res_df = pd.concat([res_df, means], axis=0)
    print(res_df.iloc[-1])


    pass


if __name__ == "__main__":
    from numpy import array_split
    import multiprocessing
    r_tbl_details = redis.Redis(
        host='localhost', port=6379, db=4, decode_responses=True
    )
    tbls_to_generate = [{k:r_tbl_details[k]} for k in r_tbl_details.scan_iter()]
    jobs_num = 1  # number of workers
    jobs = []

    list_divided = array_split(tbls_to_generate, jobs_num)
    for _chunk in list_divided:
        # Declare a new process and pass arguments to it
        p1 = multiprocessing.Process(target=run, args=(_chunk,))
        jobs.append(p1)
        p1.start()  # starting workers
    for job in jobs:
        job.join()