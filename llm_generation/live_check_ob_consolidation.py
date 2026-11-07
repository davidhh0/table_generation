"""Live sanity check for _hf_consolidated on ONE table before trusting it for a
full run.

Picks the table (URL) with the most cached open_book sub-questions, bundles all
of them into a single _hf_consolidated call (bypassing warm_cache/chunking --
this talks to the real HF endpoint directly), and prints each answer next to
prompts_cache's known-correct value ('a') so a mismatch is obvious by eye.

Writes to the SAME response cache _hf_consolidated always writes to
(local_dbs/cache/llm_cache/<model>.db) -- so a good run here already primes
the cache for that table; nothing here is a dry run.

Run from llm_generation/:  python live_check_ob_consolidation.py
Needs hf_api set.
"""
import sys

import git
import yaml
import diskcache

wd = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.insert(0, wd)

from llm_generation.batch import _group_open_book, _consolidated_prompt, _hf_consolidated  # noqa: E402

conf = yaml.safe_load(open(f'{wd}/config.yaml'))
MODEL = conf['llm_model']
BUNDLE_SIZE = conf.get('open_book_bundle_size', 15)

VARIANTS = ('_anti_refusal', '_no_sql')


def pick_table():
    """Return (url, [(prompt, gold_answer), ...]) for the table with the most
    cached open_book sub-questions, restricted to this model's variant tier."""
    pc = diskcache.Cache(f'{wd}/local_dbs/cache/prompts_cache.db')
    best_url, best_items = None, []
    for url in pc.iterkeys():
        d = pc[url]
        if not isinstance(d, dict):
            continue
        items = []
        for sub, v in d.items():
            if any(sub.endswith(var) for var in VARIANTS):
                continue
            if not sub.endswith('open_book') or not isinstance(v, dict) or 'q' not in v:
                continue
            items.append((str(v['q']).strip(), v.get('a')))
        if len(items) > len(best_items):
            best_url, best_items = url, items
    return best_url, best_items


def main():
    print(f"model={MODEL}  bundle_size={BUNDLE_SIZE}\n")
    url, items = pick_table()
    if not items:
        sys.exit("No open_book prompts found in prompts_cache -- run pass 1 (COLLECT_MODE) first")
    print(f"table: {url}\n{len(items)} open_book question(s) on this table\n")

    prompts = [p for p, _ in items]
    gold = dict(items)  # prompt -> gold answer, for the printout below

    units = _group_open_book(prompts, bundle_size=max(BUNDLE_SIZE, len(prompts)))
    if not units:
        sys.exit("_group_open_book returned nothing -- prompts didn't parse as open_book "
                  "(check _OB_PREFIX/_OB_MARKER against the actual prompt text)")
    if len(units) > 1:
        print(f"NOTE: {len(units)} bundles produced (bundle_size cap) -- testing the first only\n")
    unit = units[0]

    print("=== consolidated prompt sent to the model (single call) ===")
    print(_consolidated_prompt(unit))
    print("=== end prompt ===\n")

    conf_dict = conf
    results = _hf_consolidated(MODEL, [unit], poll_interval=None, conf=conf_dict)

    print(f"\n{len(results)}/{len(unit['originals'])} question(s) resolved by the single call\n")
    for i, orig in enumerate(unit['originals'], 1):
        ans = results.get(orig, '<UNRESOLVED -- falls back to per-prompt>')
        print(f"[{i}] model={ans!r}   gold={gold.get(orig)!r}")

    missing = len(unit['originals']) - len(results)
    if missing:
        print(f"\nWARNING: {missing} question(s) unresolved -- would fall back to the "
              "per-prompt HF path in a real warm_cache run (not a hard failure, just no "
              "consolidation saving for those).")


if __name__ == '__main__':
    main()
