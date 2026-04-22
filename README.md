# WikiFacTables

A dynamic benchmark for evaluating LLM tabular reasoning over live Wikipedia tables.

WikiFacTables automatically fetches, cleans, and validates Wikipedia wikitables, then generates structured QA tasks (retrieval, aggregation, counting) to evaluate large language models across three evaluation modes: **Closed-Book** (parametric knowledge), **Multiple-Choice**, and **Open-Book** (contextual reasoning). Unlike static benchmarks built on frozen data dumps, WikiFacTables operates on live Wikipedia content, ensuring the benchmark evolves with the underlying data and exposing staleness in LLM training corpora.

## Architecture

```
Wikipedia  ──►  WikiTableParser  ──►  Table Storage (tbls/)
                (wikiparser.py)           │
                                          ▼
                                  LLM Table Generation  ──►  Generated Tables (llm_tbls/)
                                  (llm_tbl_generate.py)          │
                                          │                      ▼
                                          │              Structural Comparison
                                          │              (compared_tbls.py)
                                          ▼
                                  Task Evaluation  ──►  Score Outputs
                                  (llm_generation/)     (scores.csv, scores_statistics.csv)
                                          │
                                          ▼
                                  Metrics & Analysis
                                  (metrics.py)
```

## Project Structure

```
table_generation/
├── main.py                     # Entry point: article discovery + LLM table generation loop
├── wikiparser.py               # WikiTableParser: fetch, parse, clean, validate wikitables
├── llm_tbl_generate.py         # LLM-driven table generation + structural scoring
├── compared_tbls.py            # Table comparison metrics (precision, recall, F1)
├── metrics.py                  # Post-hoc analysis and visualizations
├── config.yaml                 # Pipeline configuration (models, rules, thresholds)
├── requirements.txt            # Python dependencies
├── .gitignore
│
├── utils/
│   ├── utils.py                # Wikipedia API, LLM client wrappers, article pipeline
│   ├── .env                    # API keys (not committed)
│   └── .env.example            # Template for required API keys
│
├── llm_generation/             # Task evaluation module
│   ├── run.py                  # Batch orchestrator for all task evaluators
│   ├── prompts_generation.py   # Prompt construction for all task types
│   ├── count.py                # Count aggregation evaluation
│   ├── min.py                  # Min aggregation evaluation
│   ├── max.py                  # Max aggregation evaluation
│   ├── single_value_retrieval.py  # Single-cell retrieval evaluation
│   ├── list_retrieval.py       # List retrieval evaluation
│   ├── rephrase_task.py        # Rephrase robustness evaluation (standalone script)
│   ├── prompts.yaml            # Prompt templates and instructions
│   └── tasks.yaml              # Task type taxonomy
│
└── quality_check/              # Parser validation module
    ├── benchmark_quality_check.py  # Validates parser against curated benchmark
    ├── benchmark_tbls.json     # Benchmark table manifest
    └── tables/                 # 97 curated benchmark CSVs
```

**Generated at runtime** (gitignored):

| Directory | Contents |
|-----------|----------|
| `tbls/` | Fetched ground-truth Wikipedia table CSVs |
| `llm_tbls/` | LLM-generated table CSVs |
| `local_dbs/` | DiskCache databases (metadata, generated tables, prompt cache) |
| `llm_generation/*.csv` | Score output files |

## Setup

### Prerequisites

- Python 3.11+
- API keys for at least one LLM provider (see below)

### Installation

```bash
git clone <repository-url>
cd table_generation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API Keys

Copy the environment template and fill in your keys:

```bash
cp utils/.env.example utils/.env
```

Required keys (provide at least those matching the models in `config.yaml`):

| Variable | Provider |
|----------|----------|
| `gemini_api_key` | Google Gemini |
| `openai_api_key` | OpenAI (GPT) |
| `claude_api` | Anthropic (Claude) |
| `hf_api` | Hugging Face (Llama, etc.) |

## Configuration

All pipeline behavior is controlled through `config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `llm_model` | Model used for table generation and task evaluation |
| `rephrase_model` | Model used for rephrasing structured prompts |
| `title_and_key_model` | Model used for title generation and primary key detection |
| `rephrase` | Enable rephrase mode (`true`/`false`) |
| `context` | Enable context mode for LLM calls |
| `run_tables_to_fetch` | Number of tables to fetch per run |
| `minimum_table_size` | Minimum rows/columns for a valid table |
| `minimum_popularity` | Minimum Wikipedia article popularity threshold |
| `maximum_cell_length` | Maximum allowed cell string length |
| `drop_columns_blacklist` | Column names to strip (Notes, References, etc.) |
| `null_chars` | Characters treated as NULL values |
| `maximum_row_null_values` | NULL threshold per row (absolute or percentage) |
| `maximum_column_null_values` | NULL threshold per column (absolute or percentage) |
| `header_manipulations` | Remove parentheses/brackets from headers |
| `cell_manipulations` | Remove parentheses/brackets from cell values |
| `maximum_mixed_types` | Mixed data-type threshold per column |
| `known_dates_format` | Date format patterns for temporal validation |
| `last_row_agg_mechanism` | Drop aggregation/summary rows |

## Pipeline Stages

### 1. Table Fetching (`main.py` + `utils/utils.py`)

Randomly samples Wikipedia articles containing wikitables via the MediaWiki API. Each table is parsed by `WikiTableParser` and checked for temporal consistency (current vs. historical revision) to ensure the table has been stable over time.

```bash
python main.py
```

### 2. Table Cleaning & Validation (`wikiparser.py`)

`WikiTableParser.run()` applies a configurable chain of validation and cleaning rules:

- **Pre-filtering**: minimum size, MultiIndex rejection, empty headers, max cell length, ASCII enforcement
- **Column cleaning**: drop blacklisted columns (Notes, References), remove parenthesized/bracketed content from headers
- **Row cleaning**: remove repeated-header rows, divider rows, aggregation rows
- **Cell cleaning**: strip special characters, normalize unicode, cast values (int/float/date), remove parenthesized/bracketed content
- **NULL handling**: drop rows/columns exceeding NULL thresholds
- **Type validation**: drop columns with excessive mixed types

### 3. LLM Table Generation (`llm_tbl_generate.py`)

For each validated table, the pipeline prompts an LLM to:
1. Generate a natural-language table title from article name and column metadata
2. Identify the primary key column using chain-of-thought reasoning
3. Generate table rows matching the original schema

Generated tables are saved to `llm_tbls/` and scored against ground truth.

### 4. Structural Comparison (`compared_tbls.py`)

Evaluates LLM-generated tables against ground truth using:

- **Key metrics**: precision, recall, F1 on primary key column matches
- **Non-key metrics**: precision, recall, F1 on non-key cell values (conditioned on key match)
- **Overall metrics**: combined precision, recall, F1 across all cells
- **Per-type scores**: breakdown by column data type
- **Epsilon matching**: approximate numeric comparison for float/int columns

### 5. Task Evaluation (`llm_generation/`)

Generates and evaluates five task types over the benchmark tables:

| Task | Description | Breakdowns |
|------|-------------|------------|
| **Single Retrieval** | Retrieve a specific cell value | None |
| **List Retrieval** | List all values matching a filter | Categorical (Equals, In) |
| **Count** | Count rows matching a condition | Categorical (Equals, Not Equals, In) + Numerical (GT, LT, Between, Equals) |
| **Max** | Find the entity with the highest value | Categorical + Numerical |
| **Min** | Find the entity with the lowest value | Categorical + Numerical |

Each task is evaluated in three modes:
- **Closed-Book**: LLM must answer from parametric knowledge (no table provided)
- **Multiple-Choice**: LLM selects from provided options
- **Open-Book**: Full table CSV is included in the prompt

Run all evaluations:

```bash
cd llm_generation
python run.py
```

This produces `scores_statistics.csv` with per-instance results.

### 6. Rephrase Evaluation (`llm_generation/rephrase_task.py`)

Tests model robustness to natural-language rephrasings of structured prompts. A separate model (`rephrase_model` in config) converts structured task descriptions into natural questions, which are then evaluated.

> **Note**: This script uses hardcoded paths and must be run from the `llm_generation/` directory.

```bash
cd llm_generation
python rephrase_task.py
```

### 7. Metrics & Analysis (`metrics.py`)

Reads `generated_tables.db` and produces visualizations and a raw metrics CSV, breaking down F1 scores by:
- Number of cells
- Numeric column ratio
- Article popularity
- Number of rows / columns
- Column data type

```bash
python metrics.py
```

## Quality Check

The `quality_check/` module validates `WikiTableParser` against a curated set of 97 benchmark tables:

```bash
cd quality_check
python benchmark_quality_check.py
```

It re-fetches each benchmark table from Wikipedia (using historical revision IDs) and compares the parsed output against the stored CSV ground truth.

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `scores.csv` | `llm_generation/` | Aggregated scores per model/task/constraint/variant |
| `scores_statistics.csv` | `llm_generation/` | Per-instance scores with table property metadata |
| `rephrase_scores.csv` | `llm_generation/` | Rephrase evaluation results |
| `raw_metrics.csv` | Root | Per-table structural F1 scores with metadata |
| `generated_tables.db` | `local_dbs/tables/` | DiskCache DB with all table metadata and scores |
| `tbl_metadata.db` | `local_dbs/` | DiskCache DB with fetched article/table metadata |

## License

This project is part of a Master's thesis research. Please contact the author for usage terms.
