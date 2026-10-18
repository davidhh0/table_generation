"""Two-pass batch orchestrator.

Pass 1 (collect): run every task with utils.utils.COLLECT_MODE = True so that
get_llm_response records cache-missing prompts instead of calling the provider.
This also populates prompts_cache / choices_cache, making the prompt set
deterministic across passes.

Warm: submit the collected prompts per model through that provider's Batch API
(or HF concurrency) and write the responses into the per-model diskcache.

Pass 2 (score): run the same tasks with COLLECT_MODE = False. Every prompt is
now a cache hit, so the existing scoring + CSV-writing code runs unchanged.

Run from the llm_generation/ directory (tasks use ../config.yaml, ../tbls/...):
    python run_batch.py
"""
import yaml

import utils.utils as U
from batch import warm_cache

from count import count_retrieval
from min import min_retrieval
from max import max_retrieval
from single_value_retrieval import cell_retrieval
from list_retrieval import list_retrieval

# Same task set as thinking_task.py's __main__ full-run block.
TASKS = [count_retrieval, min_retrieval, max_retrieval, cell_retrieval, list_retrieval]


def main():
    with open("../config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    poll_interval = conf.get("batch_poll_interval", 30)
    chunk_size = conf.get("batch_chunk_size", 50)

    # --- Pass 1: collect prompts -------------------------------------------
    print("=== Pass 1: collecting prompts ===")
    U.COLLECT_MODE = True
    U.COLLECTED = {}
    for task in TASKS:
        print(f"--- collecting: {task.__name__} ---")
        task()
    U.COLLECT_MODE = False

    total = sum(len(v) for v in U.COLLECTED.values())
    print(f"=== Collected {total} unique prompt(s) across {len(U.COLLECTED)} model(s) ===")

    # --- Warm: batch per model ---------------------------------------------
    # sorted() groups open-book prompts by their shared title->CSV prefix, so
    # same-table prompts sit adjacent in the batch -> better OpenAI prompt-cache
    # locality (the repeated table prefix is billed at the cached rate). Keep it.
    for model, prompts in U.COLLECTED.items():
        warm_cache(model, sorted(prompts), poll_interval=poll_interval, chunk_size=chunk_size)

    # --- Pass 2: score (all cache hits) ------------------------------------
    print("=== Pass 2: scoring ===")
    scores_statistics = ""
    for task in TASKS:
        print(f"--- scoring: {task.__name__} ---")
        result = task()
        if isinstance(result, str):
            scores_statistics += result

    with open("scores_statistics.csv", "w") as file:
        file.write(scores_statistics)
    print("=== Done. scores.csv / scores_statistics.csv written. ===")


if __name__ == "__main__":
    main()
