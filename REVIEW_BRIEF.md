# Review brief: WikiFactual / `table_generation`

Hand this file to a reviewer (human or LLM) who has no prior context. It explains what the
repository is, what the paper behind it claims, what was changed recently, and what a review
should check. Paths are absolute for the author's machine; adapt as needed.

- Code repo: `/Users/david.harroch/PycharmProjects/table_generation`
- Paper (LaTeX) checkout: `/Users/david.harroch/IdeaProjects/688758a9141016b6ba0d7832`
  (`task_generation.tex`, `results.tex`, `results_explained.md`, `analysis/`, `figures/`)

## 1. The article

**"WikiFactual: Benchmarking Closed-book Relational Reasoning over Always-True Wikipedia Facts."**
The benchmark is also called WikiFacTables in the code and README.

Claim: existing table-reasoning benchmarks (WTQ, WikiSQL, FeTaQA, ...) are built from a frozen
Wikipedia dump, so edits to Wikipedia silently turn correct model answers into "wrong" ones
(staleness). WikiFactual instead:

1. Fetches structured tables from live Wikipedia.
2. Curates them into clean, type-consistent relational tables, gated by a one-year
   temporal-stability check (a table is dropped if it differs from its revision one year earlier).
3. Has an LLM identify each table's primary key and write a short descriptive title.
4. Auto-generates SQL-equivalent reasoning tasks with deterministic gold answers.
5. Evaluates LLMs under four settings that separate recall, recognition and reasoning.

The headline question is how much relational reasoning models can do from parametric
knowledge alone (closed-book), compared with having the table in the prompt (open-book).

## 2. What the repository does

Two phases (see `README.md` for the module map):

**Benchmark construction**: `main.py`, `wikiparser.py`, `utils/utils.py`, `llm_tbl_generate.py`.
Fetch, pre-filter, edit, post-filter (`utils.is_consistent`), then LLM metadata. Thresholds
live in `config.yaml`.

**Task generation and evaluation**: `llm_generation/`.

- Task types (`tasks.yaml`): Single (value) retrieval, List retrieval (F1), Count, Max/Min.
- Constraints: categorical `=`, `!=`, `in`; numerical `=`, `>=`, `<=`, `[a, b]`.
- Settings: Closed-Book (CB, question + table title only), Multiple-Choice (MC),
  Open-Book (OB, full CSV table), Open-Book Multiple-Choice (OB-MC).
- Providers: OpenAI, Anthropic, Gemini via Batch APIs; Hugging Face-hosted open models
  (e.g. Llama 3.3 70B, DeepSeek V3) per prompt with bounded concurrency.
- Flow: collect all prompts, warm a per-model diskcache (`local_dbs/cache/llm_cache/<model>.db`,
  keyed by the full prompt string), then score from cache. Everything is resumable.
- Open-book bundling: all questions of one table go in one structured call for
  OpenAI/Anthropic/Gemini (`consolidate_open_book`); HF models run per prompt.

## 3. The recent mission (what was done and why)

Goal of the recent work: extend the benchmark with new task variants, run more models, and make
the repo clean and reproducible enough to commit and push.

New task variants (built without any LLM call, on distilled tables that had no tasks):

- **Numerical List retrieval** (`list_numeric.py`, `list_retrieval_numeric.py`,
  `run_batch_numeric.py`; offline tests in `test_list_numeric.py`).
- **Single K=1** (numerical `=` / `between` isolating exactly one row) and **Max/Min with
  ties** (gold = all tied keys; kept only if exactly 2 tied and at most 25% of qualifying
  rows; mixed with equal numbers of untied questions): `new_tables_tasks.py`.
- Runner: `run_new_tables.py`; evaluated set: `results/new_tables/questions_500.jsonl`
  (500 questions, 2,000 prompts, 116 tables).

Models run on the 500-question set: GPT-5.2, GPT-5.5, Claude Opus 4.6, Claude Opus 4.8,
DeepSeek V3, and Llama 3.3 70B (Llama and the Opus 4.8 rerun may still be in progress).

Notable fixes made along the way:

- Anthropic bundled-answer parser tolerates an `answers` field returned as a sloppy JSON string.
- Opus 4.8 refused most closed-book / multiple-choice prompts (about 97% of those rows were
  scored wrong because of refusals, not capability). `run_new_tables.py` now selects the
  `*_anti_refusal` wrappers for Opus 4.8 and the `*_no_sql` wrappers for Llama, as the main
  pipeline does (`prompts_generation._ANTI_REFUSAL_MODELS` / `_NO_SQL_MODELS`). `--plain`
  disables this. Refusals still count as wrong in the main numbers and are reported
  separately as a refusal rate.
- Hugging Face `max_tokens` limit: Llama's endpoint caps at 12,288 while `config.yaml` asks for
  16,000; `utils.general_hf` now parses the error and clamps/retries.
- Cleanup: removed unused functions/imports and no-op placeholders; renamed live-API scripts
  `test_*.py` to `live_check_*.py` so pytest collection works; updated `.gitignore`
  (`results/`, `local_dbs.zip`, `scores_statistics_all*.csv`) and the README.

State: all changes are staged in git, **not committed or pushed**. Results CSVs are local only.

## 4. What a review should check

Please be skeptical and verify against the code rather than trusting this brief.

1. **Gold-answer correctness** in `new_tables_tasks.py` and `list_numeric.py`: tie detection,
   the exactly-2 and 25% rules, numeric parsing and normalization (commas, floats, units),
   inclusive `between`, K ranges, and that no question has an ambiguous or duplicated key.
2. **Scoring** in `run_new_tables.py` (`norm`, `parse`, `f1`): comma-containing gold items,
   case/whitespace/trailing-period handling, and whether exact-match is too strict or too lenient
   for any model's output style. DeepSeek's open-book Max/Min scores (0.10 to 0.33) are far below
   its OB-MC scores and look suspicious; this has not been investigated.
3. **Prompt wrappers**: `variant_prompts` re-renders stored prompts with the variant wrapper and
   asserts the plain re-render equals the stored prompt. Check that this cannot silently change
   a prompt, and that Single K=1 MC intentionally has no variant wrapper.
4. **Batch/cache logic** (`llm_generation/batch.py`, `utils/utils.py`): resumability, the
   journal, `COLLECT_MODE` / `SCORE_ONLY`, retry and clamp logic, and any place a failed request
   could be cached as a valid answer.
5. **Table-count consistency**: the paper says 492 tables; the data shows 530 tables with
   tasks recorded in `generated_tables.db`, 495 URLs with evaluated prompts in
   `prompts_cache.db`, plus 304 new tables from the generator (834 distinct tables participated
   in task generation; 611 were evaluated). Explain or reconcile the 530 / 495 / 492 gap.
6. **Reproducibility**: `requirements.txt`, `.env.example`, config defaults, README commands
   actually run, and nothing secret or machine-specific in tracked files.
7. **Paper-code agreement**: numbers and definitions in `results.tex`, `task_generation.tex`
   and `analysis/generated/*.txt` should match what the code computes.
8. **Dead or duplicated code** still left behind (`ruff check --select F`, `compileall`).

## 5. How to run the checks

```bash
cd /Users/david.harroch/PycharmProjects/table_generation
.venv/bin/python -m compileall -q . -x '\.venv|local_dbs|legacy'
cd llm_generation && ../.venv/bin/python -m pytest -q test_list_numeric.py     # 14 tests
../.venv/bin/python new_tables_tasks.py --out /tmp/q.jsonl --seed 0             # no LLM calls
../.venv/bin/python run_new_tables.py --jsonl ../results/new_tables/questions_500.jsonl \
    --model gpt-5.2-2025-12-11 --dry                                            # cache-hit check
```

Notes: use `.venv/bin/python` (Python 3.14). API keys load from `.env` only after
`import utils.utils`. Do not run `live_check_*.py` or any non-`--dry` runner without intent: they
make paid API calls. `prompts_generation.py` has an atexit hook that rewrites
`lr_chance_summary.txt`, so avoid importing it for read-only checks.

## 6. Results so far (500-question set, exact match; CB / MC / OB / OB-MC)

Opus 4.8 CB/MC values are invalid (refusals under plain wrappers); a rerun with anti-refusal
wrappers is in progress.

| Task | GPT-5.5 | Opus 4.6 | DeepSeek V3 |
|---|---|---|---|
| Single K=1 | .15 / .54 / 1.00 / 1.00 | .04 / .47 / 1.00 / 1.00 | .04 / .39 / .79 / .94 |
| Max ties | .14 / .31 / .97 / 1.00 | .00 / .34 / .79 / .86 | .00 / .14 / .10 / .79 |
| Min ties | .22 / .41 / .93 / .93 | .00 / .22 / .69 / .69 | .00 / .20 / .33 / .61 |
| Max untied | .18 / .64 / .90 / .97 | .00 / .44 / .77 / .92 | .00 / .21 / .10 / .74 |
| Min untied | .26 / .76 / .95 / .97 | .00 / .47 / .89 / .97 | .00 / .32 / .32 / .79 |

## 7. Known open items

- Llama 3.3 70B run and Opus 4.8 anti-refusal rerun were still running when this was written.
- Llama's low numeric open-book scores may be a symptom of the earlier token-cap issue; unverified.
- DeepSeek's low open-book Max/Min scores (see section 4, item 2).
- The 530 / 495 / 492 table-count gap (section 4, item 5).
- Commit and push are pending the author's go-ahead; do not push on the author's behalf.
