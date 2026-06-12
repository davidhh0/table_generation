import yaml

if __name__ == "__main__":
    with open("../config.yaml", "r") as f:
        conf = yaml.safe_load(f)

    if conf.get("use_batch", False):
        # Two-pass batch flow: collect prompts, batch per model, score from cache.
        from run_batch import main
        main()
    else:
        # Legacy single-call flow.
        from single_value_retrieval import cell_retrieval
        cell_retrieval()
        # from list_retrieval import list_retrieval
        # list_retrieval()
        # from count import count_retrieval
        # count_retrieval()
        # from max import max_retrieval
        # max_retrieval()
        # from min import min_retrieval
        # min_retrieval()
