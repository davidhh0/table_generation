"""Collect -> warm -> score for the numerical List Retrieval task (same flow as run_batch.py).

    EVAL_MODEL=Qwen/Qwen3.8-2.4T-A95B python run_batch_numeric.py [--limit N] [--out rows.csv]
Safe to re-run / resume: finished prompts are cache hits, only missing ones are fetched.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import yaml

import utils.utils as U
from batch import warm_cache
from list_retrieval_numeric import list_numeric_retrieval


def main():
    # per-run override of config.yaml's hf_max_workers (shared config left untouched)
    if os.environ.get("HF_WORKERS"):
        import batch
        _orig = batch._load_conf
        batch._load_conf = lambda: {**_orig(), "hf_max_workers": int(os.environ["HF_WORKERS"]),
                                       "hf_chunk_size": int(os.environ.get("HF_CHUNK", 96))}
    with open("../config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    U.COLLECT_MODE = True
    U.COLLECTED = {}
    list_numeric_retrieval(limit=lim)
    U.COLLECT_MODE = False
    print(f"=== collected {sum(len(v) for v in U.COLLECTED.values())} prompt(s) ===", flush=True)
    for model, prompts in U.COLLECTED.items():
        warm_cache(model, sorted(prompts), poll_interval=conf.get("batch_poll_interval", 30),
                   chunk_size=int(os.environ.get("BATCH_CHUNK") or conf.get("batch_chunk_size", 50)))
    out = list_numeric_retrieval(limit=lim)
    print(out)
    if "--out" in sys.argv:
        with open(sys.argv[sys.argv.index("--out") + 1], "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
