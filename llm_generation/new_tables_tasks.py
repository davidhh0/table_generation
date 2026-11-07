"""Question generator for tables NOT used by the main benchmark (no tasks in the cache).

Two task families, built deterministically (no LLM, no cache writes, no API calls):
  * single_k1 -- Single Retrieval with a numeric filter that isolates exactly one row
                 ('=' and 'between'); the answer is that row's value of a non-key column.
  * max_ties / min_ties -- Max/Min that keeps ties: gold = every key attaining the extremum,
                 kept only when |gold| == 2 and gold <= 25% of the qualifying rows (paper,
                 Sec. 5.1), plus an equal number of untied questions so the answer size is
                 not predictable.
Writes one JSON line per question with the four prompts (CB, MC, OB, OB-MC).

    python new_tables_tasks.py [--out questions.jsonl] [--seed 0]
"""
import json
import os
import random
import re
import sys
from collections import Counter

import diskcache
import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEY_MODEL = "gpt-4.1-2025-04-14"  # config.yaml: title_and_key_model
MAX_TIES, MAX_TIE_SHARE = 2, 0.25
PER_OP = 3  # Single rows per operator and table (the existing Single task samples 3 rows)

SINGLE_INSTR = "Answer with the exact value only. No sentences or punctuation."
TIES_INSTR = "Return all matching values as a comma-separated list without any additional text."
TOTAL = re.compile(r"^\s*(grand\s+)?(total|all|overall|sum)\b", re.I)
RANKLIKE = re.compile(r"\b(rank|pos|no|id|place|code|zip|phone|isbn|number)\b|#", re.I)

with open(os.path.join(HERE, "prompts.yaml")) as _f:
    P = yaml.safe_load(_f)


def num(v):
    v = float(v)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(round(v, 6))


def is_num(s):
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)


def filter_cols(df, key):
    """numeric, non-rank-like, non-sequence columns usable as a filter"""
    out = []
    for c in df.columns:
        if c == key or c in ("Total", "Date") or len(str(c)) == 1 or str(c).isnumeric():
            continue
        if not is_num(df[c]) or df[c].nunique() < 3 or RANKLIKE.search(str(c)):
            continue
        out.append(c)
    return out


def target_cols(df, key, fc):
    return [c for c in df.columns if c not in (key, fc, "Total", "Date") and len(str(c)) > 1
            and not str(c).isnumeric() and df[c].nunique() >= 3
            and len({type(v) for v in df[c].tolist()}) == 1]


def between_window(df, fc, v):
    """[lo, hi] around the unique value v containing no other row (None for min/max rows)."""
    vals = sorted(df[fc].dropna().unique())
    j = vals.index(v)
    if j == 0 or j == len(vals) - 1:
        return None
    prev, nxt = vals[j - 1], vals[j + 1]
    if is_num(df[fc]) and pd.api.types.is_integer_dtype(df[fc]):
        lo, hi = v - (v - prev - 1) // 2, v + (nxt - v - 1) // 2
    else:
        lo, hi = round(v - (v - prev) / 2, 2), round(v + (nxt - v) / 2, 2)
    if not lo < hi or int(df[fc].between(lo, hi).sum()) != 1:
        return None
    return lo, hi


def build_single(df, key, rng):
    fcs = filter_cols(df, key)
    rng.shuffle(fcs)
    for fc in fcs:
        tcs = target_cols(df, key, fc)
        if not tcs:
            continue
        tc = rng.choice(tcs)
        vc = df[fc].value_counts()
        uniq = [v for v in vc[vc == 1].index if not pd.isna(df.loc[df[fc] == v, tc].iloc[0])
                and not TOTAL.match(str(df.loc[df[fc] == v, key].iloc[0]))]
        if not uniq:
            continue
        rng.shuffle(uniq)
        qs = []
        for v in uniq[:PER_OP]:
            gold = df.loc[df[fc] == v, tc].tolist()
            qs.append(dict(op="numerical_equals", fc=fc, tc=tc, gold=gold,
                           clause=P["numerical_equals"].format(COMPARISON_COLUMN=fc, VALUE=num(v)).strip(), nq=1))
        got = 0
        for v in uniq:
            w = between_window(df, fc, v)
            if w and got < PER_OP:
                gold = df.loc[df[fc].between(*w), tc].tolist()
                qs.append(dict(op="numerical_between", fc=fc, tc=tc, gold=gold, nq=1, clause=P["numerical_between"].format(
                    COMPARISON_COLUMN=fc, LOWER_THRESHOLD=num(w[0]), UPPER_THRESHOLD=num(w[1])).strip()))
                got += 1
        return qs
    return []


def build_maxmin(df, key):
    """every constructible Max/Min question (same space as max.py / min.py) with its tie set."""
    cols = [c for c in df.columns if c != key and c not in ("Total", "Date")]
    comp = [c for c in cols if df[c].dtype in (np.int64, np.float64) and not RANKLIKE.search(str(c))]
    cat = [c for c in cols if df[c].dtype == object]
    out = []

    def add(fn, c1, op, clause, d):
        if not len(d):
            return
        ext = d[c1].max() if fn == "max" else d[c1].min()
        gold = d.loc[d[c1] == ext, key].tolist()
        if any(TOTAL.match(str(g)) for g in gold):
            return
        out.append(dict(fn=fn, op=op, c1=c1, clause=clause, gold=gold, nq=len(d)))

    for fn in ("max", "min"):
        for col in comp:
            if df[col].nunique() < 3 or len(str(col)) == 1 or str(col).isnumeric():
                continue
            for cc in cat:
                mc = Counter(df[cc].tolist()).most_common(3)
                if len(mc) < 3 or mc[1][1] < 2 or pd.isna(mc[1][0]) or pd.isna(mc[2][0]):
                    continue
                v1, v2 = mc[1][0], mc[2][0]
                fmt = dict(DESIRED_COLUMN=cc, DESIRED_VALUE=v1, CHOICE_1=v1, CHOICE_2=v2)
                add(fn, col, "categorical_equals", P["categorical_equals"].format(**fmt).strip(), df[df[cc] == v1])
                add(fn, col, "categorical_not_equals", P["categorical_not_equals"].format(**fmt).strip(), df[df[cc] != v1])
                add(fn, col, "categorical_in", P["categorical_in"].format(**fmt).strip(), df[df[cc].isin([v1, v2])])
        if len(comp) > 1:
            c1, c2 = comp[0], comp[1]
            med = df[c2].median()
            if (med == 0 or pd.isna(med) or (df[c2] > med).sum() < 2 or (df[c2] < med).sum() < 2
                    or df[c1].nunique() < 3 or df[c2].nunique() < 3):
                continue
            ev = Counter(df[c2].dropna()).most_common()[0][0]
            lo, hi = int(med / 1.5), int(med * 1.5)
            fmt = dict(COMPARISON_COLUMN=c2, THRESHOLD=med, LOWER_THRESHOLD=lo, UPPER_THRESHOLD=hi, VALUE=ev)
            for op, d in (("numerical_greater_than", df[df[c2] >= med]), ("numerical_less_than", df[df[c2] <= med]),
                          ("numerical_between", df[df[c2].between(lo, hi)]), ("numerical_equals", df[df[c2] == ev])):
                add(fn, c1, op, P[op].format(**fmt).strip(), d)
    return out


def tied(q):
    return len(q["gold"]) == MAX_TIES and len(q["gold"]) / q["nq"] <= MAX_TIE_SHARE


def render(title, key, df, question, instr, gold, rng, multi):
    """the four prompts. multi=True -> select-all MC with 2K options, else single-answer MC with 3."""
    csv = df.to_csv(index=False)
    col = key if multi else None
    pool = [v for v in df[col].tolist() if v not in gold] if multi else None
    if multi:
        choices = list(gold) + rng.sample(pool, len(gold))
        rng.shuffle(choices)
        mc = dict(CHOICES_LIST=str(choices))
        w_mc, w_ob_mc = "list_retrieval_multiple_choices_wrapper", "open_book_list_retrieval_multiple_choices_wrapper"
    else:
        raise AssertionError  # single MC is built by the caller (needs the target column)
    base = dict(TABLE_TITLE=title, QUESTION=question, INSTRUCTION=instr, PRIMARY_KEY_COLUMN=key)
    return {"closed_book": P["closed_book_wrapper"].format(**base).strip(),
            "multiple_choices": P[w_mc].format(**base, **mc).strip(),
            "open_book": P["open_book_wrapper"].format(**base, FULL_CSV_TABLE=csv).strip(),
            "open_book_multiple_choices": P[w_ob_mc].format(**base, FULL_CSV_TABLE=csv, **mc).strip()}


def render_single(title, key, df, question, tc, gold, rng):
    csv = df.to_csv(index=False)
    pool = sorted({v for v in df[tc].dropna().tolist()} - set(gold), key=str)
    ch = rng.sample(pool, 2) + list(gold)
    rng.shuffle(ch)
    mc = dict(CHOICE_1=ch[0], CHOICE_2=ch[1], CHOICE_3=ch[2])
    base = dict(TABLE_TITLE=title, QUESTION=question, INSTRUCTION=SINGLE_INSTR, PRIMARY_KEY_COLUMN=key)
    return {"closed_book": P["closed_book_wrapper"].format(**base).strip(),
            "multiple_choices": P["multiple_choice_wrapper"].format(**base, **mc).strip(),
            "open_book": P["open_book_wrapper"].format(**base, FULL_CSV_TABLE=csv).strip(),
            "open_book_multiple_choices": P["open_book_multiple_choice_wrapper"].format(**base, FULL_CSV_TABLE=csv, **mc).strip()}


def unused_tables():
    cache = diskcache.Cache(os.path.join(ROOT, "local_dbs/tables/generated_tables.db"))
    for k in cache.iterkeys():
        cfg = json.loads(cache[k])
        if cfg.get("tasks"):
            continue
        g = cfg["llm_generated"].get(KEY_MODEL)
        path = os.path.join(ROOT, f'tbls/{cfg["page_id"]}_{cfg["table_idx"]}.csv')
        if not g or not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        key = g["key"]
        if (key not in df.columns or df[key].nunique() < len(df) or len(str(key)) == 1 or str(key).isnumeric()
                or not set(c for c in df.columns if c != key and c not in ("Total", "Date")) & set(cfg["columns"])):
            continue
        yield k, cfg, g["table_title"], key, df


def generate(seed=0):
    rng = random.Random(seed)
    rows, untied = [], []
    for k, cfg, title, key, df in unused_tables():
        meta = dict(table_id=k, url=cfg["url"], title=title, key=key)
        for q in build_single(df, key, rng):
            q_text = "Identify the value of the column `%s` %s" % (q["tc"], q["clause"])
            rows.append(dict(meta, task="single_k1", op=q["op"], gold=[str(g) for g in q["gold"]],
                             prompts=render_single(title, key, df, q_text, q["tc"], q["gold"], rng)))
        for q in build_maxmin(df, key):
            if tied(q):
                rows.append(dict(meta, task=q["fn"] + "_ties", op=q["op"], gold=[str(g) for g in q["gold"]],
                                 _q=q, _df=df))
            elif len(q["gold"]) == 1:
                untied.append(dict(meta, task=q["fn"] + "_untied", op=q["op"], gold=[str(g) for g in q["gold"]],
                                   _q=q, _df=df))
    n_tied = sum(r["task"].endswith("_ties") for r in rows)
    by_tbl = Counter()
    rng.shuffle(untied)
    keep = []
    for u in untied:  # equal number of untied questions, at most 2 per table
        if len(keep) >= n_tied:
            break
        if by_tbl[u["table_id"]] < 2:
            keep.append(u)
            by_tbl[u["table_id"]] += 1
    out = []
    for r in rows + keep:
        if "_q" in r:
            q, df = r.pop("_q"), r.pop("_df")
            word = "highest" if r["task"].startswith("max") else "lowest"
            question = "Name all the `%s` values with the %s `%s` column %s" % (r["key"], word, q["c1"], q["clause"])
            gold_vals = [v for v in df[r["key"]].tolist() if str(v) in set(r["gold"])]
            r["prompts"] = render(r["title"], r["key"], df, question, TIES_INSTR, gold_vals, rng, multi=True)
        out.append(r)
    return out


if __name__ == "__main__":
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    rows = generate(seed)
    if "--out" in sys.argv:
        with open(sys.argv[sys.argv.index("--out") + 1], "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    c = Counter(r["task"] for r in rows)
    print(dict(c), "questions:", len(rows), "tables:", len({r["table_id"] for r in rows}))
