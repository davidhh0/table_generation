# WikiFacTables (WikiFactual)

**A clean, updatable benchmark of factual Wikipedia tables for tabular inference.**

WikiFacTables (referenced in the accompanying research paper as **WikiFactual**)
is a framework that builds a tabular-reasoning benchmark directly from **live
Wikipedia** rather than from a frozen data dump. It fetches structured Wikipedia
tables, curates them into clean type-consistent relational data, enriches them with
primary keys and descriptive titles, and automatically generates reasoning tasks
that are evaluated against large language models.

Because the benchmark is regenerated from the live encyclopedia and gated by a
one-year temporal-stability check, it avoids the *staleness* problem of static
benchmarks (WTQ, WikiSQL, FeTaQA, …), where edits to Wikipedia silently turn
correct model answers into "wrong" ones.

This repository contains the reference implementation accompanying the paper
*"WikiFactual: Benchmarking Closed-book Relational Reasoning over Always-True Wikipedia Facts."*

---

## What it does

The system runs in two phases.

### 1. Benchmark construction (four-stage curation pipeline)

Implemented mainly in `wikiparser.py`, `utils/utils.py`, `main.py`, and
`llm_tbl_generate.py`.

| Stage | Module | What happens |
|-------|--------|--------------|
| **A. Fetching** | `utils.get_articles_to_parse`, `wikiparser.WikiTableParser` | Query the MediaWiki API for pages containing `wikitable` HTML tables and parse them into DataFrames. |
| **B. Pre-filtering** | `wikiparser.py` | Coarse structural filters: minimum size, popularity, empty/MultiIndex headers, basic type detection. |
| **C. Table editing** | `wikiparser.py` (config-driven) | Drop noise columns, strip bracket/parenthesis clutter without colliding column names, normalize NULLs, enforce type consistency, remove aggregation rows. |
| **D. Post-filtering** | `utils.is_consistent` | Quality assurance + a **temporal-validation** check that compares each table against its Wikipedia revision from one year prior and drops volatile tables. |
| **Metadata** | `llm_tbl_generate.py` | An LLM identifies the **primary key** column and writes a short descriptive **table title**. |

All curation thresholds (minimum rows/columns, NULL ratios, mixed-type limits,
date formats, blacklisted columns, …) are driven by **`config.yaml`**.

### 2. Task generation & LLM evaluation

Implemented in `llm_generation/`. For each curated table the framework
instantiates SQL-equivalent task templates and evaluates them under three
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

Each task/test pair is scored against deterministic ground truth (exact match,
or F1 for List Retrieval). An optional **natural-language rephrasing** step
(`rephrase_task.py`, `prompts.yaml: rephrasing`) tests robustness to
conversational phrasing.

---

## Repository layout

```
.
├── config.yaml                # All pipeline + evaluation knobs
├── main.py                    # Entry point for benchmark construction & metadata
├── wikiparser.py              # Wikipedia HTML table fetching + cleaning
├── llm_tbl_generate.py        # LLM primary-key + title metadata extraction
├── compared_tbls.py           # Table comparison / scoring library
├── metrics.py                 # Analysis & matplotlib plots over results
├── requirements.txt
├── utils/
│   └── utils.py               # Multi-provider LLM dispatch, caching, MediaWiki API
├── llm_generation/            # Task generation + EVALUATION
│   ├── run.py                 # Entry point (dispatches to batch or legacy flow)
│   ├── run_batch.py           # Two-pass batch orchestrator (collect → warm → score)
│   ├── batch.py               # Provider Batch-API submission & polling
│   ├── prompts_generation.py  # Builds CB / MC / OB prompts for every task
│   ├── prompts.yaml           # Prompt wrappers & instructions
│   ├── tasks.yaml             # Task types × break-down constraints
│   ├── single_value_retrieval.py
│   ├── list_retrieval.py
│   ├── count.py
│   ├── max.py
│   ├── min.py
│   ├── rephrase_task.py       # Natural-language rephrasing evaluation
│   └── test_anti_refusal.py   # Sanity check for the Claude anti-refusal wrapper
├── quality_check/             # Curation-quality validation harness
│   ├── benchmark_quality_check.py
│   ├── benchmark_tbls.json    # Registry of curated sample tables
│   └── tables/                # ~100 curated ground-truth CSVs (sample data)
└── legacy/                    # Archived / superseded code (see legacy/README.md)
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

Results are appended to `scores.csv` / `scores_statistics.csv`. Use
`metrics.py` to aggregate and plot them:

```bash
python metrics.py
```

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
- `legacy/` holds superseded experiments and is not part of the active pipeline.
