# legacy/

Archived, superseded code kept for reference only. **Nothing here is imported by
the active pipeline** (`main.py`, `llm_generation/run.py`), and these files may
reference prompt keys or modules that no longer exist. Do not expect them to run
as-is.

| File | Superseded by | Notes |
|------|---------------|-------|
| `cell_retrieval.py` | `llm_generation/single_value_retrieval.py` | Older single-file implementation that also bundled max/min/comparison. The live `cell_retrieval()` is the one in `single_value_retrieval.py`. |
| `comparison_task.py` | `llm_generation/max.py`, `llm_generation/min.py` | Early combined max/min/comparison task; references prompt keys not present in the current `prompts.yaml`. |
| `thinking_task.py` | `llm_generation/run_batch.py` | Standalone "thinking" experiment; not part of the task set run by the batch orchestrator. |
| `count_example.py` | — | Hard-coded scratch list of example count prompts. |
| `lama.bash` | Hugging Face Inference API path in `utils/utils.py` | Local `torchrun` launch script for self-hosted Llama 4. |
