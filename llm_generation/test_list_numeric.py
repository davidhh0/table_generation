"""Offline tests for numerical List Retrieval. Run from llm_generation/:
    ../.venv/bin/python -m pytest test_list_numeric.py -q
No LLM calls: the real tables are loaded, gold answers recomputed independently
(pure-python loops, not the pandas masks used by the builder)."""
import os
import random
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import list_numeric as ln  # noqa: E402
import list_retrieval_numeric as lrn  # noqa: E402


# ---------- synthetic edge cases ----------
def _df(vals, nan_at=()):
    v = list(vals)
    for i in nan_at:
        v[i] = float("nan")
    return pd.DataFrame({"k": [f"r{i}" for i in range(len(v))], "x": v})


def test_fmt_num():
    assert ln.fmt_num(5.0) == "5"
    assert ln.fmt_num(3) == "3"
    assert ln.fmt_num(2.5) == "2.5"
    assert ln.fmt_num(-0.0) == "0"


def test_equals_picks_group_in_range_and_skips_oversize_groups():
    d = _df([1] * 12 + [2] * 3 + [3, 4, 5, 6])
    r = ln.pick_equals(d, "k", "x")
    assert r and r[0]["VALUE"] == 2          # 12-group too big, 1-groups too small
    assert all(c["VALUE"] != 1 for c in r)


def test_gte_lte_inclusive_and_ties():
    d = _df([10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    res = ln.build_list_numeric(d, "k", "x")
    g = res["numerical_greater_than"]
    t = float(g["format_args"]["THRESHOLD"])
    assert sorted(g["answer"]) == sorted(d.k[d.x >= t])       # inclusive
    assert 2 <= len(g["answer"]) <= 7


def test_nan_never_in_answers():
    d = _df(list(range(1, 13)), nan_at=(0, 5))
    for v in ln.build_list_numeric(d, "k", "x").values():
        assert "r0" not in v["answer"] and "r5" not in v["answer"]


def test_k_bounds_small_table():
    d = _df([1, 2, 3, 4, 5, 6, 7])           # N=7: 2K <= N -> K <= 3
    for v in ln.build_list_numeric(d, "k", "x").values():
        assert 2 <= len(v["answer"]) <= 3


def test_constant_column_rejected():
    assert ln.build_list_numeric(_df([5] * 10), "k", "x") == {}
    assert ln.pick_column(_df([5] * 10), "k") == (None, None)


def test_string_column_rejected():
    d = pd.DataFrame({"k": list("abcdefgh"), "x": list("aabbccdd")})
    assert ln.eligible_numeric_columns(d, "k") == []


def test_between_is_two_sided():
    d = _df(list(range(1, 21)))
    b = ln.build_list_numeric(d, "k", "x")["numerical_between"]
    lo, hi = float(b["format_args"]["LOWER_THRESHOLD"]), float(b["format_args"]["UPPER_THRESHOLD"])
    assert lo < hi and lo > 1 and hi < 20
    assert sorted(b["answer"]) == sorted(d.k[(d.x >= lo) & (d.x <= hi)])


def test_distractors():
    d = _df(list(range(1, 21)))
    ans = ["r1", "r2", "r3"]
    ch = ln.make_distractors(d, "k", ans, random.Random(1))
    assert len(ch) == 6 and len(set(ch)) == 6 and set(ans) <= set(ch)
    assert ch == ln.make_distractors(d, "k", ans, random.Random(1))   # deterministic


# ---------- real tables ----------
@pytest.fixture(scope="module")
def real():
    rows = list(lrn.iter_numeric_list_tables())
    if not rows:
        pytest.skip("needs the local table data (local_dbs/ and tbls/), which is not in the repo")
    return rows


def test_enough_real_questions(real):
    n = sum(len(v) for *_, v in real)
    assert n >= 300, n
    assert all(len(v) == 4 for *_, v in real)


def test_real_gold_independent_recompute(real):
    for tbl, cfg, df, key, title, col, variants in real:
        keys, vals = df[key].tolist(), df[col].tolist()
        for bd, v in variants.items():
            fa = v["format_args"]
            f = lambda x: x is not None and x == x  # not NaN
            if bd == "numerical_equals":
                t = float(fa["VALUE"]); exp = [k for k, x in zip(keys, vals) if f(x) and x == t]
            elif bd == "numerical_greater_than":
                t = float(fa["THRESHOLD"]); exp = [k for k, x in zip(keys, vals) if f(x) and x >= t]
            elif bd == "numerical_less_than":
                t = float(fa["THRESHOLD"]); exp = [k for k, x in zip(keys, vals) if f(x) and x <= t]
            else:
                lo, hi = float(fa["LOWER_THRESHOLD"]), float(fa["UPPER_THRESHOLD"])
                assert lo < hi
                exp = [k for k, x in zip(keys, vals) if f(x) and lo <= x <= hi]
            assert v["answer"] == exp, (cfg["url"], col, bd)
            assert 2 <= len(exp) <= 7 and 2 * len(exp) <= len(df)
            assert len(set(map(str, exp))) == len(exp)               # keys unique
            assert not any(pd.isna(k) for k in exp)


def test_real_distinct_sets_and_columns(real):
    for *_, df, key, title, col, variants in [(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in real]:
        sigs = [frozenset(map(str, v["answer"])) for v in variants.values()]
        assert len(set(sigs)) == 4
        assert col != key and df[col].dtype.kind in "if"


def test_real_distractors(real):
    for tbl, cfg, df, key, title, col, variants in real:
        for bd, v in variants.items():
            ch = ln.make_distractors(df, key, v["answer"], random.Random(f"{cfg['url']}_{col}_{bd}_list_numerical"))
            assert len(ch) == 2 * len(v["answer"]) and len(set(ch)) == len(ch)
            assert set(v["answer"]) <= set(ch)
            assert set(ch) <= set(df[key])


def test_prompt_texts(real):
    import yaml
    prompts = yaml.safe_load(open("prompts.yaml"))
    tbl, cfg, df, key, title, col, variants = real[0]
    for bd, v in variants.items():
        q = prompts[bd].format(COLUMN=col, COMPARISON_COLUMN=col, **v["format_args"]).strip()
        assert col in q and "{" not in q
        assert "nan" not in q.lower().split()
