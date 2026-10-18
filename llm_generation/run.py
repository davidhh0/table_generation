import os
import sys
import yaml

if __name__ == "__main__":
    # Ensure current working directory is llm_generation/ for relative asset paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    with open("../config.yaml", "r") as f:
        conf = yaml.safe_load(f)

    if conf.get("use_batch", False):
        # Two-pass batch flow: collect prompts, batch per model, score from cache.
        from run_batch import main
        main()
    else:
        # Direct single-call flow across all benchmark tasks.
        from count import count_retrieval
        from min import min_retrieval
        from max import max_retrieval
        from single_value_retrieval import cell_retrieval
        from list_retrieval import list_retrieval

        for task in [count_retrieval, min_retrieval, max_retrieval, cell_retrieval, list_retrieval]:
            print(f"--- Running task: {task.__name__} ---")
            task()
