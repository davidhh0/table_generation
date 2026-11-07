"""Re-score every model from the per-model LLM cache -- no provider calls.

The merged results CSV used to be assembled from separate per-model runs made at
different times, so rows could reflect an older scorer or a cache that was not yet
fully populated. This runs the *current* scorer over the *current* cache for every
model in one pass, so all rows are consistent.

Run from llm_generation/:
    python rescore.py [out.csv] [model ...]
"""
import os
import sys

import utils.utils as U
from count import count_retrieval
from min import min_retrieval
from max import max_retrieval
from single_value_retrieval import cell_retrieval
from list_retrieval import list_retrieval

TASKS = [count_retrieval, min_retrieval, max_retrieval, cell_retrieval, list_retrieval]
MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.8-2.4T-A95B",
    "deepseek-ai/DeepSeek-V3", "claude-opus-4-6", "claude-opus-4-8", "gemini-2.5-pro",
    "gemini-3.1-pro-preview", "gpt-5.2-2025-12-11", "gpt-5.5-2026-04-23",
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "scores_statistics_rescored.csv"
    models = sys.argv[2:] or MODELS
    U.SCORE_ONLY = True
    rows = ""
    for model in models:
        os.environ["EVAL_MODEL"] = model
        for task in TASKS:
            r = task()
            if isinstance(r, str):
                rows += r
        print(f"=== {model}: cache misses so far {U.MISSES.get(model, 0)} ===")
    with open(out, "w") as f:
        f.write("model,type,constraint_family,break_down,test_type,number_of_rows,number_of_columns,popularity,numerical_cols_ratio,score\n" + rows)
    print("misses per model:", U.MISSES)


if __name__ == "__main__":
    main()
