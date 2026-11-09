"""Collect -> warm -> score for the new-table Single-K=1 / Max-Min-ties questions.

    python run_new_tables.py --jsonl questions.jsonl --model gpt-5.2-2025-12-11 \
        [--tables 25] [--seed 0] [--out rows.csv] [--dry]

Models listed in VARIANTS (Opus 4.8: anti-refusal, Llama: no-SQL) automatically get their
reworded closed-book / select-all multiple-choice wrappers, as in the main pipeline;
--plain disables that.

--tables N runs a pilot on N tables (all of their questions, so open-book bundling stays
realistic; tied-Max/Min tables are over-sampled). --dry only prints how many prompts would
be sent. Resumable: finished prompts are cache hits. Does not import prompts_generation
(its atexit hook rewrites lr_chance_summary.txt).
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import yaml

import utils.utils as U
from batch import warm_cache

SETTINGS = ("closed_book", "multiple_choices", "open_book", "open_book_multiple_choices")

# Keep in sync with prompts_generation._ANTI_REFUSAL_MODELS / _NO_SQL_MODELS (not imported:
# that module's atexit hook rewrites lr_chance_summary.txt).
VARIANTS = (("claude-opus-4-8", "_anti_refusal"), ("llama", "_no_sql"))
_MC_MARK = "Select all that apply from the following options: "


def variant_suffix(model):
    return next((s for m, s in VARIANTS if m in model.lower()), "")


def variant_prompts(rec, suffix, P):
    """The tableless closed-book and select-all multiple-choice prompts, rebuilt with the
    model's reworded wrapper exactly as the main pipeline does (prompts.yaml *_anti_refusal /
    *_no_sql). Single K=1 multiple-choice has no variant wrapper (same as the main pipeline).
    The stored plain prompt is parsed back into its parts and re-rendered with the plain
    wrapper first; any mismatch aborts instead of silently altering a prompt."""
    base = dict(TABLE_TITLE=rec["title"], PRIMARY_KEY_COLUMN=rec["key"])
    out = dict(rec["prompts"])
    cb = rec["prompts"]["closed_book"].split("\n")
    parts = dict(QUESTION="\n".join(cb[1:-1]), INSTRUCTION=cb[-1])
    assert P["closed_book_wrapper"].format(**base, **parts).strip() == rec["prompts"]["closed_book"], rec["table_id"]
    out["closed_book"] = P["closed_book_wrapper" + suffix].format(**base, **parts).strip()
    mc = rec["prompts"]["multiple_choices"]
    if _MC_MARK in mc:
        lines = mc.split("\n")
        choices = lines[-2].split(_MC_MARK, 1)[1]
        mparts = dict(QUESTION="\n".join(lines[1:-2]), INSTRUCTION=lines[-1], CHOICES_LIST=choices)
        assert P["list_retrieval_multiple_choices_wrapper"].format(**base, **mparts).strip() == mc, rec["table_id"]
        out["multiple_choices"] = P["list_retrieval_multiple_choices_wrapper" + suffix].format(**base, **mparts).strip()
    return out


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def norm(s):
    s = str(s).strip().strip("'\"[]").strip().lower().rstrip(".")
    t = s.replace(",", "")
    try:
        f = float(t)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return re.sub(r"\s+", " ", s)


def parse(ans, gold):
    """prediction -> set of normalized items; gold items that contain commas are matched whole."""
    if ans is None:
        return None
    text, items = ans.strip(), []
    for g in sorted(gold, key=len, reverse=True):
        if "," in g and g.lower() in text.lower():
            text = re.sub(re.escape(g), "\x00", text, flags=re.I)
            items.append(g)
    parts = [p for p in re.split(r",|\n|;", text) if p.strip()]
    items += [p for p in parts if "\x00" not in p]
    return {norm(i) for i in items if i.strip()}


def f1(pred, gold):
    gold = {norm(g) for g in gold}
    if pred is None:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def load(path, n_tables, seed):
    rows = [json.loads(l) for l in open(path)]
    if n_tables:
        rng = random.Random(seed)
        by = defaultdict(list)
        for r in rows:
            by[r["table_id"]].append(r)
        tied = [t for t, v in by.items() if any(r["task"].endswith("_ties") for r in v)]
        rest = [t for t in by if t not in tied]
        rng.shuffle(tied)
        rng.shuffle(rest)
        pick = tied[: n_tables // 3] + rest[: n_tables - n_tables // 3]
        rows = [r for t in pick for r in by[t]]
    return rows


def main():
    model = arg("--model", "gpt-5.2-2025-12-11")
    rows = load(arg("--jsonl"), int(arg("--tables", 0)), int(arg("--seed", 0)))
    with open("../config.yaml") as f:
        conf = yaml.safe_load(f)
    suffix = "" if "--plain" in sys.argv else variant_suffix(model)
    if suffix:
        with open("prompts.yaml") as f:
            P = yaml.safe_load(f)
        for r in rows:
            r["prompts"] = variant_prompts(r, suffix, P)
        print(f"using {suffix} wrappers for closed-book / select-all multiple-choice", flush=True)
    print(f"{len(rows)} questions on {len({r['table_id'] for r in rows})} tables "
          f"-> {len(rows) * 4} prompts | {dict(Counter(r['task'] for r in rows))}", flush=True)
    U.COLLECT_MODE, U.COLLECTED = True, {}
    for r in rows:
        for s in SETTINGS:
            U.get_llm_response(model, r["prompts"][s])
    U.COLLECT_MODE = False
    missing = sorted(U.COLLECTED.get(model, ()))
    print(f"=== {len(missing)} uncached prompt(s) to send ===", flush=True)
    if "--dry" in sys.argv:
        return
    if missing:
        warm_cache(model, missing, poll_interval=conf.get("batch_poll_interval", 30),
                   chunk_size=int(os.environ.get("BATCH_CHUNK") or conf.get("batch_chunk_size", 50)))
    U.SCORE_ONLY = True
    out = []
    for r in rows:
        for s in SETTINGS:
            ans = U.get_llm_response(model, r["prompts"][s])
            pred = parse(ans, r["gold"])
            gold = {norm(g) for g in r["gold"]}
            out.append(dict(model=model, task=r["task"], op=r["op"], setting=s, table_id=r["table_id"],
                            k=len(r["gold"]), answered=ans is not None, exact=int(pred == gold) if pred is not None else 0,
                            f1=round(f1(pred, r["gold"]), 4), gold="|".join(r["gold"]), answer=(ans or "").replace("\n", " ")))
    if arg("--out"):
        with open(arg("--out"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    agg = defaultdict(list)
    for o in out:
        agg[(o["task"], o["setting"])].append(o)
    print("task, setting: n, answered, exact, F1")
    for k in sorted(agg):
        v = agg[k]
        print(f"{k[0]:14s} {k[1]:27s} n={len(v):4d} answered={sum(x['answered'] for x in v):4d} "
              f"exact={sum(x['exact'] for x in v) / len(v):.2f} f1={sum(x['f1'] for x in v) / len(v):.2f}")


if __name__ == "__main__":
    main()
