# Clover

**A clean, updatable benchmark of factual Wikipedia tables for tabular inference.**

**Clover** (earlier names: WikiFactual, WikiFacTables) is a framework that builds a tabular-reasoning benchmark directly from **live
Wikipedia** rather than from a frozen data dump. It fetches structured Wikipedia
tables, curates them into clean type-consistent relational data, enriches them with
primary keys and descriptive titles, and automatically generates reasoning tasks
that are evaluated against large language models.

Because the benchmark is regenerated from the live encyclopedia and gated by a
one-year temporal-stability check, it avoids the *staleness* problem of static
benchmarks (WTQ, WikiSQL, FeTaQA, …), where edits to Wikipedia silently turn
correct model answers into "wrong" ones.

This repository contains the reference implementation accompanying the paper
*"Clover: Benchmarking Closed-book Relational Reasoning over Durable (Always-True) Wikipedia Facts."*

---

## What it does

The system runs in two phases.

### 1. Benchmark construction (four-stage curation pipeline)

Implemented mainly in `wikiparser.py`, `utils/utils.py`, `main.py`, and
`llm_tbl_generate.py`.

| Stage | Module | What happens |
|-------|--------|--------------|
| **A. Fetching** | `utils.get_articles_to_parse`, `wikiparser.WikiTableParser` | Query the MediaWiki API for pages containing `wikitable` HTML tables and parse them into DataFrames. |
| **B. Pre-filtering** | `wikiparser.py` | Coarse structural filters: minimum size, popularity (at least 1 average monthly view over two years, inclusive), empty/MultiIndex headers, basic type detection. |
| **C. Table editing** | `wikiparser.py` (config-driven) | Drop noise columns (more than 40% NULL; rows are never dropped for sparsity, only all-NULL rows), strip bracket/parenthesis notes without colliding column names but keep unit annotations such as `(US$)` or `(km2)`, normalize NULLs, enforce type consistency, remove aggregation rows. |
| **D. Post-filtering** | `utils.is_consistent` | Quality assurance + a **temporal-validation** check that compares each table against its Wikipedia revision from one year prior and drops volatile tables. |
| **Metadata** | `llm_tbl_generate.py` | An LLM identifies the **primary key** column and writes a short descriptive **table title**. |

All curation thresholds (minimum rows/columns, NULL ratios, mixed-type limits,
date formats, blacklisted columns, …) are driven by **`config.yaml`**.

### 2. Task generation & LLM evaluation

Implemented in `llm_generation/`. For each curated table the framework
instantiates SQL-equivalent task templates and evaluates them under four
settings.

**Task types** (`tasks.yaml`):

- **Single (Value) Retrieval** — fetch one cell given a key.
- **List Retrieval** — return *all* values satisfying a constraint (scored by F1).
- **Count** — count rows matching a constraint.
- **Max / Min Aggregation** — entity with the extreme value in a column.

**Break-down constraints** applied to each task (the SQL `WHERE` clause):
categorical `=`, `≠`, `∈`; numerical `=`, `≥`, `≤`, `[a, b]`.

**Evaluation tests** (`prompts.yaml` wrappers):

- **Closed-Book** — question + table title only; probes parametric recall.
- **Multiple-Choice** — question + distractor options; probes recognition.
- **Open-Book** — question + full table serialized as CSV; probes reasoning.
- **Open-Book Multiple-Choice** — full table *and* the multiple-choice options.

In open-book mode all questions on the same table are consolidated into one
structured call (`consolidate_open_book`, bundle size `open_book_bundle_size`), so
the CSV is sent once per table instead of once per question.

Each task/test pair is scored against deterministic ground truth (exact match,
or F1 for List Retrieval). An optional **natural-language rephrasing** step
(`rephrase: true` in `config.yaml`, prompt `prompts.yaml: rephrasing`) rewrites
each question in conversational phrasing.

---

## Repository layout

```
.
├── config.yaml                # All pipeline + evaluation knobs
├── main.py                    # Entry point for benchmark construction & metadata
├── wikiparser.py              # Wikipedia HTML table fetching + cleaning
├── llm_tbl_generate.py        # LLM primary-key + title metadata extraction
├── compared_tbls.py           # Table comparison / scoring library
├── requirements.txt
├── utils/
│   └── utils.py               # Multi-provider LLM dispatch, caching, MediaWiki API
├── llm_generation/            # Task generation + EVALUATION
│   ├── run.py                 # Entry point (dispatches to batch or direct flow)
│   ├── run_batch.py           # Two-pass batch orchestrator (collect → warm → score)
│   ├── batch.py               # Provider Batch-API submission & polling
│   ├── prompts_generation.py  # Builds CB / MC / OB prompts for every task
│   ├── prompts.yaml           # Prompt wrappers & instructions
│   ├── tasks.yaml             # Task types × break-down constraints
│   ├── single_value_retrieval.py
│   ├── list_retrieval.py      # List Retrieval, categorical constraints
│   ├── list_numeric.py        # List Retrieval, numerical constraints: pure question builder
│   ├── list_retrieval_numeric.py  # ... and its runner (templated questions)
│   ├── run_batch_numeric.py   # Collect → warm → score for numerical List Retrieval
│   ├── count.py
│   ├── max.py
│   ├── min.py
│   ├── new_tables_tasks.py    # Single K=1 and Max/Min-with-ties questions on unused tables
│   ├── run_new_tables.py      # Collect → warm → score for those questions
│   ├── rescore.py             # Re-score every model from its cache (no provider calls)
│   └── test_list_numeric.py   # Offline pytest suite for the numerical List builder
├── results/new_tables/        # Question set + per-model results for the new-table tasks
├── quality_check/             # Curation-quality validation harness
│   ├── benchmark_quality_check.py
│   ├── benchmark_tbls.json    # Registry of curated sample tables
│   └── tables/                # ~100 curated ground-truth CSVs (sample data)
└── test_wikiparser_rules.py   # Offline tests of the table-cleaning rules
```

Runtime artifacts (the diskcache databases under `local_dbs/`, fetched tables,
and result CSVs such as `scores.csv` / `raw_metrics.csv`) are written locally and
are **git-ignored** — they are regenerated by running the pipeline.

---

## Supported model providers

`utils/utils.py` dispatches by model name:

| Provider | Trigger / Example | Env var |
|----------|-------------------|---------|
| OpenAI (GPT-4.1 / GPT-5.x) | `gpt`, `o1`, `o3` | `openai_api_key` |
| Google Gemini | `gemini` (e.g. `gemini-2.5-pro`) | `gemini_api_key` |
| Anthropic Claude | `claude` (e.g. `claude-opus-4-8`) | `claude_api` |
| Hugging Face / vLLM | e.g. `llama`, `Qwen/Qwen3.8-2.4T-A95B` | `hf_api` |

OpenAI, Anthropic, and Gemini are additionally driven through their **Batch
APIs** (`llm_generation/batch.py`) for cheaper, higher-throughput evaluation.
Open-weight Hugging Face models are executed with bounded concurrency and optional
thinking suppression.

---

## Setup

Requires **Python 3.14** (see `requirements.txt`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Provide API keys for the providers you intend to use
cp .env.example .env        # then edit .env
set -a; source .env; set +a
```

---

## Usage

### 1. Construct the benchmark

```bash
# Fetch and curate Wikipedia tables directly from live MediaWiki API:
python main.py --mode fetch

# Extract table primary-keys & titles using an LLM:
python main.py --mode generate

# Or run both phases in sequence:
python main.py --mode both
```

This fetches and curates Wikipedia tables into `tbls/` and stores their metadata
in a local diskcache under `local_dbs/`.

### 2. Verify curation quality

Run the quality check harness against the included 97 sample ground-truth tables
without querying the live Wikipedia API:

```bash
python quality_check/benchmark_quality_check.py --offline
```

To run online against historical revisions of those tables:

```bash
python quality_check/benchmark_quality_check.py
```

### 3. Run task generation & evaluation

Run evaluation from the repository root or from `llm_generation/`:

```bash
python llm_generation/run.py
```

- With `use_batch: true` in `config.yaml`, `run.py` uses the two-pass batch
  orchestrator (`run_batch.py`): **collect** all prompts → **warm** the per-model
  cache via each provider's Batch API → **score** from cache.
- With `use_batch: false`, it runs the direct single-call flow across all tasks.

Results are appended to `scores.csv` / `scores_statistics.csv`.

To evaluate a model other than `llm_model`, set `EVAL_MODEL` (read by every task
module and the numerical-List runner), e.g. `EVAL_MODEL=gpt-5.2-2025-12-11 python llm_generation/run.py`.

Finished prompts are cache hits, so every runner below is **resumable**; a crashed
or interrupted run only sends what is still missing. To re-score all models from
their caches without any provider call, run `python rescore.py [out.csv] [model ...]`
from `llm_generation/`.

#### Numerical List Retrieval

List questions with numerical constraints (`=`, `≥`, `≤`, inclusive `between`),
answer size K in 2–7:

```bash
cd llm_generation
python list_retrieval_numeric.py --dry              # enumerate questions + gold, no LLM
EVAL_MODEL=Qwen/Qwen3.8-2.4T-A95B python run_batch_numeric.py --out rows.csv
python -m pytest test_list_numeric.py -q            # offline gold-answer tests
```

#### Tasks on tables outside the main benchmark

`new_tables_tasks.py` builds, without any LLM call, **Single K=1** questions
(numerical `=` / `between` that isolate exactly one row) and **Max/Min with ties**
(gold = all tied keys, kept only when exactly 2 and ≤ 25 % of the qualifying rows,
mixed with an equal number of untied questions) on distilled tables that have no
tasks in the cache. Each question carries the four prompts (CB, MC, OB, OB-MC).

```bash
cd llm_generation
python new_tables_tasks.py --out questions.jsonl --seed 0       # full generator output
python run_new_tables.py --jsonl ../results/new_tables/questions_500.jsonl \
    --model gpt-5.2-2025-12-11 --out ../results/new_tables/gpt52.csv [--dry]
```

`results/new_tables/questions_500.jsonl` is the exact 500-question set (2,000
prompts, 116 tables) evaluated for the models whose per-prompt results are stored
next to it. `--dry` only reports how many prompts would be sent.

### Configuration

`config.yaml` controls everything: the evaluation model (`llm_model`), batch
settings, output-token / reasoning caps per provider, the open-book
consolidation strategy, and every curation threshold. It is heavily commented —
read it before changing behavior.

---

## Notes

- API keys are read from environment variables only; **no secrets are stored in
  the repo**.
- Provider clients (Gemini, Claude, OpenAI, Hugging Face) initialize lazily only
  when requested, allowing offline operations and single-provider runs without dummy keys.
- `quality_check/tables/` ships a curated sample of ground-truth tables so the
  quality-check harness can run offline without re-fetching from Wikipedia.

---

## License

The code is released under the [MIT License](LICENSE). Tables derived from
Wikipedia (including the samples in `quality_check/tables/`) are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation

If you use this code or the benchmark, please cite the paper (see also
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{harroch2026clover,
  title  = {Clover: Benchmarking Closed-book Relational Reasoning over Durable (Always-True) Wikipedia Facts},
  author = {Harroch, David and Glickman, Oren and Somech, Amit},
  year   = {2026},
  note   = {Under review}
}
```
