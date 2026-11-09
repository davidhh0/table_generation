"""Deterministic construction of numerical-constraint List Retrieval questions.

Pure functions only (no LLM, no cache, no randomness except an explicit ``rng``),
so the gold answers can be unit-tested against the real tables.

A question lists the primary-key values of the rows satisfying a numerical
constraint on one column: ``=``, ``>=`` (inclusive), ``<=`` (inclusive) or
``between`` (inclusive). The answer size K is kept in [K_MIN, K_MAX] (the paper
evaluates List questions with K <= 7) and never exceeds half the table (2K <= N, so MC options can pair each answer with a distinct distractor).
"""

import numpy as np
import pandas as pd

K_MIN = 2
K_MAX = 7
K_TARGET = 4  # prefer mid-range answer sizes so equals/>=/<=/between are comparable

BREAK_DOWNS = ("numerical_equals", "numerical_greater_than",
               "numerical_less_than", "numerical_between")


def fmt_num(v):
    """Render a number the way the question text and a reader would: 5.0 -> '5'."""
    if isinstance(v, (np.integer,)):
        return str(int(v))
    v = float(v)
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _numeric_series(df, col):
    s = df[col]
    if not (pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)):
        return None
    s = s[s.notna()]
    s = s[np.isfinite(s.astype(float))]
    return s


def _keys(df, key, mask):
    return df.loc[mask, key].tolist()


def _ok(k, n):
    # 2k <= n: every correct key can be paired with a distinct non-answer distractor
    return K_MIN <= k <= K_MAX and 2 * k <= n


def _ranked(cands):
    """cands: list of (k, tiebreak, payload), best first: |k - K_TARGET| then tiebreak."""
    return sorted(cands, key=lambda c: (abs(c[0] - K_TARGET), c[1]))


def pick_equals(df, key, col):
    s = _numeric_series(df, col)
    if s is None:
        return []
    n = len(df)
    cands = []
    for v, k in s.value_counts().items():
        if _ok(int(k), n):
            cands.append((int(k), float(v), v))
    return [{"VALUE": v, "mask": df[col] == v} for _, _, v in _ranked(cands)]


def pick_gte(df, key, col):
    s = _numeric_series(df, col)
    if s is None:
        return []
    n = len(df)
    cands = []
    for t in sorted(s.unique()):
        k = int((s >= t).sum())
        if _ok(k, n):
            cands.append((k, -float(t), t))
    return [{"THRESHOLD": t, "mask": df[col] >= t} for _, _, t in _ranked(cands)]


def pick_lte(df, key, col):
    s = _numeric_series(df, col)
    if s is None:
        return []
    n = len(df)
    cands = []
    for t in sorted(s.unique()):
        k = int((s <= t).sum())
        if _ok(k, n):
            cands.append((k, float(t), t))
    return [{"THRESHOLD": t, "mask": df[col] <= t} for _, _, t in _ranked(cands)]


def pick_between(df, key, col):
    """Window [lo, hi] over distinct values, lo < hi, centred on the median so it is
    a genuine two-sided range (not a disguised >= / <=)."""
    s = _numeric_series(df, col)
    if s is None:
        return []
    n = len(df)
    vals = sorted(s.unique())
    if len(vals) < 3:
        return []
    med = float(s.median())
    cands = []
    for i in range(1, len(vals) - 1):          # lo is not the global min
        for j in range(i + 1, len(vals) - 1):  # hi is not the global max
            lo, hi = vals[i], vals[j]
            k = int(((s >= lo) & (s <= hi)).sum())
            if _ok(k, n):
                centre = abs((float(lo) + float(hi)) / 2 - med)
                cands.append((k, centre, (lo, hi)))
    return [{"LOWER_THRESHOLD": lo, "UPPER_THRESHOLD": hi,
             "mask": df[col].between(lo, hi, inclusive="both")}
            for _, _, (lo, hi) in _ranked(cands)]


_PICKERS = {
    "numerical_equals": pick_equals,
    "numerical_greater_than": pick_gte,
    "numerical_less_than": pick_lte,
    "numerical_between": pick_between,
}


def build_list_numeric(df, key, col):
    """Return {bd: {'format_args': {...}, 'answer': [key values]}} for every break-down
    constructible on ``col`` (possibly empty)."""
    out = {}
    used = set()  # gold sets already taken: each operator gets a distinct answer set
    for bd in BREAK_DOWNS:
        for r in _PICKERS[bd](df, key, col):
            mask = r.pop("mask")
            answer = _keys(df, key, mask)
            sig = frozenset(map(str, answer))
            if sig in used:
                continue
            used.add(sig)
            out[bd] = {"format_args": {k: fmt_num(v) for k, v in r.items()},
                       "answer": answer}
            break
    return out


def eligible_numeric_columns(df, key, skip=("Total", "Date")):
    cols = []
    for c in df.columns:
        if c == key or c in skip:
            continue
        if len(str(c)) == 1 or str(c).isnumeric():
            continue
        s = _numeric_series(df, c)
        if s is None or s.nunique() < 3:
            continue
        cols.append(c)
    return cols


def pick_column(df, key):
    """First eligible numeric column for which ALL four break-downs are constructible."""
    for c in eligible_numeric_columns(df, key):
        r = build_list_numeric(df, key, c)
        if len(r) == len(BREAK_DOWNS):
            return c, r
    return None, None


def make_distractors(df, key, answer, rng):
    """One non-answer key per correct key (distinct where the table allows), so the
    option list has 2K entries as in the categorical List MC design."""
    pool = [k for k in df[key].tolist() if k not in set(answer)]
    assert len(pool) >= len(answer), "need 2K <= N"
    d = rng.sample(pool, len(answer))
    choices = list(answer) + d
    rng.shuffle(choices)
    return choices
