import argparse
from datetime import datetime
import yaml

def main():
    parser = argparse.ArgumentParser(
        description="WikiFacTables benchmark construction and evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["fetch", "generate", "both"],
        default="fetch",
        help="Pipeline phase: 'fetch' (download and curate Wikipedia tables), 'generate' (LLM metadata and table generation), or 'both'",
    )
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    ts = datetime.now()

    if args.mode in ("fetch", "both"):
        from utils.utils import get_articles_to_parse
        print("=== Phase 1: Fetching & curating Wikipedia tables ===")
        get_articles_to_parse(conf, ts)

    if args.mode in ("generate", "both"):
        from llm_tbl_generate import llm_table_generation
        print("=== Phase 2: Generating table metadata and LLM tables ===")
        llm_table_generation(conf, ts)

if __name__ == "__main__":
    main()
