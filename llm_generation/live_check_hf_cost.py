"""Smoke-test an HF model's output profile before committing to a full run.

HuggingFace has no Batch API -- `_hf_parallel` (batch.py) is a 12-worker thread
pool at full synchronous price -- so a cold run of this benchmark is 12,186
prompts billed one by one. The cost of that run is dominated by *output* tokens,
and how many of those a model emits is not knowable from its name: cached answers
average 10 chars for the non-thinking Qwen3.5-397B but 4,335 (p50 3,879) for
Qwen3-32B, which streams visible chain-of-thought. That is a ~400x spread.

So: sample a few real prompts per condition, call the model live, and measure.
Extrapolate to the full run so the bill is a number you saw before you paid it.

Run from llm_generation/:  python live_check_hf_cost.py [N_PER_CONDITION] [CONDITION]
CONDITION restricts the run to one of closed_book/multiple_choices/open_book
(aliases: cb/mc/ob). Omit it to run all three, as before.
Makes N (or 3*N with no CONDITION) live HF calls (needs hf_api). Default N=8.
Reads prompts straight from prompts_cache and calls general_hf directly, so
nothing is written to the response cache -- this never pollutes scored data.
"""
import re
import sys
import time
import statistics

import git
import yaml
import diskcache

wd = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.insert(0, wd)

from utils.utils import general_hf  # noqa: E402

conf = yaml.safe_load(open(f'{wd}/config.yaml'))
MODEL = conf['llm_model']
CAP = conf.get('max_completion_tokens', 1024)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

ALL_CONDITIONS = ('closed_book', 'multiple_choices', 'open_book')
_ALIASES = {'cb': 'closed_book', 'mc': 'multiple_choices', 'ob': 'open_book'}


def _resolve_condition(arg):
    key = _ALIASES.get(arg.lower(), arg.lower())
    if key not in ALL_CONDITIONS:
        sys.exit(f"Unknown condition {arg!r}; choose from "
                  f"{ALL_CONDITIONS} (or aliases {tuple(_ALIASES)})")
    return key


CONDITIONS = (_resolve_condition(sys.argv[2]),) if len(sys.argv) > 2 else ALL_CONDITIONS

# Prompt counts for a fully cold cache, measured with a COLLECT_MODE-only pass
# (run_batch.py pass 1, zero API calls). Even per condition by construction: every
# question is asked in all three. Override if the task set changes.
COLD_COUNTS = {'closed_book': 4062, 'multiple_choices': 4062, 'open_book': 4062}

# Variant sub-keys belong to other models' wrapper tiers (prompts_generation.py
# _variant_suffix); sampling them would measure a prompt this model never sees.
VARIANTS = ('_anti_refusal', '_no_sql')

# Visible chain-of-thought giveaways. Cheap heuristics -- the length stats below
# are the real signal; these just help explain *why* an answer was long.
COT = re.compile(r'<think>|</think>|\bLet me\b|\bFirst,|\bStep 1\b|\bWe need to\b', re.I)

CHARS_PER_TOKEN = 4  # rough, provider-agnostic; fine for an order-of-magnitude call


def sample_prompts():
    """N prompts per condition, chosen deterministically so reruns are comparable."""
    pc = diskcache.Cache(f'{wd}/local_dbs/cache/prompts_cache.db')
    pools = {c: [] for c in CONDITIONS}
    for url in sorted(pc.iterkeys()):
        d = pc[url]
        if not isinstance(d, dict):
            continue
        for sub in sorted(d):
            if any(sub.endswith(v) for v in VARIANTS):
                continue
            v = d[sub]
            if not isinstance(v, dict) or 'q' not in v:
                continue
            for c in CONDITIONS:
                if sub.endswith(c):
                    pools[c].append(str(v['q']).strip())
                    break
    out = {}
    for c in CONDITIONS:
        uniq = list(dict.fromkeys(pools[c]))
        # even stride across the whole pool rather than the first N, so we don't
        # sample one table's questions over and over
        step = max(len(uniq) // max(N, 1), 1)
        out[c] = uniq[::step][:N]
    return out


def main():
    samples = sample_prompts()
    print(f"model={MODEL}  max_completion_tokens={CAP}  n={N} per condition\n")

    stats = {}
    for c in CONDITIONS:
        prompts = samples[c]
        if not prompts:
            print(f"[{c}] no prompts found in prompts_cache -- skipped")
            continue
        in_chars, out_chars, secs, cot, truncated, errors = [], [], [], 0, 0, 0
        for i, p in enumerate(prompts, 1):
            t0 = time.time()
            try:
                r = general_hf(MODEL, p)
            except Exception as e:  # noqa: BLE001 -- report, don't abort the survey
                errors += 1
                print(f"  [{c} {i}/{len(prompts)}] ERROR {type(e).__name__}: {e}")
                continue
            dt = time.time() - t0
            r = '' if r is None else str(r)
            in_chars.append(len(p))
            out_chars.append(len(r))
            secs.append(dt)
            if COT.search(r):
                cot += 1
            # within 5% of the cap means the new max_tokens is clipping answers,
            # which would silently corrupt scores -- worth knowing immediately
            if len(r) >= CAP * CHARS_PER_TOKEN * 0.95:
                truncated += 1
            print(f"  [{c} {i}/{len(prompts)}] in={len(p):>6} out={len(r):>6} "
                  f"{dt:>5.1f}s  {r[:70]!r}")
        if not out_chars:
            continue
        stats[c] = {
            'n': len(out_chars),
            'in_mean': statistics.mean(in_chars),
            'out_mean': statistics.mean(out_chars),
            'out_p50': statistics.median(out_chars),
            'out_max': max(out_chars),
            'sec_mean': statistics.mean(secs),
            'cot': cot,
            'truncated': truncated,
            'errors': errors,
        }
        print()

    if not stats:
        print("No successful calls -- is hf_api set and the model served?")
        return

    print('=== measured per call ===')
    for c, s in stats.items():
        print(f"  {c:<18} n={s['n']:<3} in~{s['in_mean']:>7.0f}ch  "
              f"out mean={s['out_mean']:>7.0f}ch p50={s['out_p50']:>6.0f} "
              f"max={s['out_max']:>6}  {s['sec_mean']:>5.1f}s  "
              f"cot={s['cot']}/{s['n']} truncated={s['truncated']}")

    print('\n=== extrapolated to a full cold run ===')
    tot_in = tot_out = tot_calls = 0
    for c, s in stats.items():
        n = COLD_COUNTS.get(c, 0)
        ti = n * s['in_mean'] / CHARS_PER_TOKEN
        to = n * s['out_mean'] / CHARS_PER_TOKEN
        tot_in += ti
        tot_out += to
        tot_calls += n
        print(f"  {c:<18} {n:>6} calls  in~{ti / 1e6:>6.2f}M tok  out~{to / 1e6:>6.2f}M tok")
    wall = sum(COLD_COUNTS.get(c, 0) * s['sec_mean'] for c, s in stats.items()) / 12 / 3600
    print(f"  {'TOTAL':<18} {tot_calls:>6} calls  in~{tot_in / 1e6:>6.2f}M tok  "
          f"out~{tot_out / 1e6:>6.2f}M tok")
    print(f"\n  ~{wall:.1f}h wall-clock at 12 concurrent workers (_hf_parallel), "
          "billed at full synchronous rate -- HF has no batch discount.")

    if any(s['truncated'] for s in stats.values()):
        print(f"\n  WARNING: some answers hit the {CAP}-token cap. Raise "
              "max_completion_tokens or those responses are being cut off "
              "mid-answer and will score as wrong.")
    if any(s['cot'] for s in stats.values()):
        print("\n  NOTE: chain-of-thought detected in the visible output. Output "
              "tokens dominate this bill; consider whether the reasoning can be "
              "disabled for this model before running.")


if __name__ == '__main__':
    main()
