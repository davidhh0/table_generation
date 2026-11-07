"""Numerical-constraint List Retrieval task (=, >=, <=, between), templated questions.

Run from llm_generation/ (like the other tasks):
    EVAL_MODEL=gpt-5.2-2025-12-11 python list_retrieval_numeric.py [--limit N] [--dry]

--dry enumerates the questions (and their gold answers) without any LLM call.
Rows are emitted in the same CSV layout as the other tasks:
    model,List,Numerical,<break-down>,<setting>,rows,cols,popularity,numerical_cols_ratio,score
"""
import json
import os
import sys

try:
    from llm_generation import list_numeric as ln
except ImportError:
    import list_numeric as ln


MAX_PER_OPERATOR = 200  # each table yields one question per operator, so tables <= 200 caps every operator


def iter_numeric_list_tables(limit=None, max_iter=450):
    """Yield (tbl_key, cfg, df, key, table_title, col, variants) for every table that
    admits all four numeric List break-downs. Same table filters as list_retrieval."""
    import diskcache
    import git
    import pandas as pd
    import yaml

    with open("../config.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    key_model = conf["title_and_key_model"]
    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    cache = diskcache.Cache(f"{working_dir}/local_dbs/tables/generated_tables.db")
    n = 0
    for i, tbl in enumerate(cache.iterkeys(), 1):
        if i > max_iter:
            break
        cfg = json.loads(cache[tbl])
        path = f'../tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv'
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        if key_model not in cfg["llm_generated"]:
            continue
        key = cfg["llm_generated"][key_model]["key"]
        if key not in df.columns or len(str(key)) == 1 or str(key).isnumeric():
            continue
        if df[key].nunique() < len(df):
            continue
        cols_wo_key = [c for c in df.columns if c != key and c not in ["Total", "Date"]]
        if not set(cols_wo_key).intersection(set(cfg["columns"])):
            continue
        col, variants = ln.pick_column(df, key)
        if col is None:
            continue
        n += 1
        yield tbl, cfg, df, key, cfg["llm_generated"][key_model]["table_title"], col, variants
        if n >= min(limit or MAX_PER_OPERATOR, MAX_PER_OPERATOR):
            return


def _popularity(cfg):
    meta = cfg["article_metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return meta["popularity"]


def list_numeric_retrieval(limit=None, dry=False):
    import numpy as np
    from wikiparser import WikiTableParser
    import yaml
    from llm_generation.prompts_generation import prompt_generation_list_numerical

    with open("../config.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    model = os.environ.get("EVAL_MODEL") or conf["llm_model"]
    parser_ins = WikiTableParser()
    rows = ""
    for tbl, cfg, df, key, title, col, variants in iter_numeric_list_tables(limit):
        if dry:
            for bd, v in variants.items():
                print(f"{cfg['url']} | {title} | {col} | {bd} {v['format_args']} -> K={len(v['answer'])} {v['answer']}")
            continue
        scores = prompt_generation_list_numerical(
            table_desc=title, col=col, df=df, parser_ins=parser_ins, model=model,
            instruction="list_value_retrieval_instruction", pk_column=key,
            url=cfg["url"], variants=variants,
        )
        num_cols = [c for c in df.columns if df[c].dtype in [np.int64, np.float64] and c != "Total"]
        ratio = len(num_cols) / len(df.columns)
        for bd, per_setting in scores.items():
            for setting, score in per_setting.items():
                rows += (f"{model},List,Numerical,{bd},{setting},{df.shape[0]},{df.shape[1]},"
                         f"{_popularity(cfg)},{ratio},{score}\n")
    return rows


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, os.path.dirname(script_dir))
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    out = list_numeric_retrieval(limit=lim, dry="--dry" in sys.argv)
    if out:
        print(out)
        if "--out" in sys.argv:
            with open(sys.argv[sys.argv.index("--out") + 1], "a") as f:
                f.write(out)
