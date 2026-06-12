"""Provider batch dispatcher for two-pass cache-warming.

``warm_cache(model, prompts)`` submits a list of prompts to the appropriate
provider's Batch API (or, for HuggingFace, a bounded thread pool), waits for
completion, and writes each response into the *same* per-model diskcache that
``utils.utils.get_llm_response`` reads from -- keyed by the exact stripped
prompt string. After warming, a normal (non-collect) pass over the tasks is all
cache hits.

Routing mirrors ``get_llm_response``:
    'gpt' in model      -> OpenAI batch
    'gemini' in model   -> Gemini batch
    'claude' in model   -> Anthropic Message Batches
    otherwise           -> HuggingFace (concurrency, no true batch endpoint)
"""
import io
import json
import os
import time

import diskcache
import git

working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

# Default poll interval (seconds); overridable by callers / config.
POLL_INTERVAL = 30
MAX_TOKENS = 1024
# Submit prompts in chunks of this size, caching each chunk before the next.
# Smaller chunks = more frequent persistence (better crash recovery) but more
# round trips; larger chunks = fewer submissions. Provider caps are far higher
# (OpenAI 50k/200MB, Anthropic 100k/256MB per batch).
CHUNK_SIZE = 50


def _cache_for(model):
    return diskcache.Cache(f'{working_dir}/local_dbs/cache/llm_cache/{model}.db')


def warm_cache(model, prompts, poll_interval=POLL_INTERVAL, chunk_size=CHUNK_SIZE):
    """Batch-resolve ``prompts`` for ``model`` and write them into the cache.

    Prompts are submitted in chunks of ``chunk_size``; each chunk's responses are
    written to the cache before the next chunk is submitted. So a crash mid-run
    only loses the single in-flight chunk -- the next run's collect pass sees the
    already-cached chunks as hits and re-batches only the remainder.

    Resilient by design: any prompt that errors or is missing from a chunk's
    output is simply left out of the cache, so the subsequent scoring pass falls
    back to a live ``get_llm_response`` call rather than scoring a bogus answer.
    """
    prompts = [p for p in dict.fromkeys(p.strip() for p in prompts) if p]
    if not prompts:
        print(f"[batch] nothing to warm for {model}")
        return {}

    if 'gpt' in model:
        provider = _openai_batch
    elif 'gemini' in model:
        provider = _gemini_batch
    elif 'claude' in model.lower():
        provider = _anthropic_batch
    else:
        provider = _hf_parallel

    total = len(prompts)
    n_chunks = (total + chunk_size - 1) // chunk_size
    print(f"[batch] warming {total} prompt(s) for model {model} "
          f"in {n_chunks} chunk(s) of up to {chunk_size}")

    cache = _cache_for(model)
    all_results = {}
    written = 0
    for ci in range(n_chunks):
        chunk = prompts[ci * chunk_size:(ci + 1) * chunk_size]
        print(f"[batch] {model}: chunk {ci + 1}/{n_chunks} ({len(chunk)} prompt(s))")
        # _hf_parallel has no poll_interval argument.
        if provider is _hf_parallel:
            results = provider(model, chunk)
        else:
            results = provider(model, chunk, poll_interval)
        for prompt, response in results.items():
            if response is None:
                continue
            cache[prompt] = response.strip()
            written += 1
        all_results.update(results)
        print(f"[batch] {model}: chunk {ci + 1} cached "
              f"(total written {written}/{total})")
    print(f"[batch] done: wrote {written}/{total} response(s) to cache for {model}")
    return all_results


# --------------------------------------------------------------------------- #
# Anthropic Claude -- Message Batches API
# --------------------------------------------------------------------------- #
def _anthropic_batch(model, prompts, poll_interval):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    from utils.utils import claude_client

    requests = [
        Request(
            custom_id=f"req-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        for i, prompt in enumerate(prompts)
    ]
    batch = claude_client.messages.batches.create(requests=requests)
    print(f"[batch][claude] submitted {batch.id}")

    while True:
        batch = claude_client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"[batch][claude] {batch.processing_status} "
              f"processing={batch.request_counts.processing}")
        time.sleep(poll_interval)

    results = {}
    for result in claude_client.messages.batches.results(batch.id):
        idx = int(result.custom_id.split("-")[1])
        prompt = prompts[idx]
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), None)
            results[prompt] = text
        else:
            print(f"[batch][claude] {result.custom_id}: {result.result.type}")
    return results


# --------------------------------------------------------------------------- #
# OpenAI GPT -- /v1/chat/completions batch
# --------------------------------------------------------------------------- #
def _openai_batch(model, prompts, poll_interval):
    from openai import OpenAI

    oai = OpenAI(api_key=os.environ["openai_api_key"])

    lines = []
    for i, prompt in enumerate(prompts):
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
        }))
    buf = io.BytesIO("\n".join(lines).encode("utf-8"))
    buf.name = "batch_input.jsonl"

    uploaded = oai.files.create(file=buf, purpose="batch")
    batch = oai.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[batch][openai] submitted {batch.id}")

    while True:
        batch = oai.batches.retrieve(batch.id)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        print(f"[batch][openai] {batch.status}")
        time.sleep(poll_interval)

    results = {}
    if batch.status != "completed" or not batch.output_file_id:
        print(f"[batch][openai] batch ended as {batch.status}")
        return results

    content = oai.files.content(batch.output_file_id).read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    for line in content.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        idx = int(rec["custom_id"].split("-")[1])
        prompt = prompts[idx]
        try:
            results[prompt] = rec["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            print(f"[batch][openai] {rec['custom_id']}: no content")
    return results


# --------------------------------------------------------------------------- #
# Google Gemini -- batch jobs
# --------------------------------------------------------------------------- #
def _gemini_batch(model, prompts, poll_interval):
    from google.genai.types import GenerateContentConfig
    from utils.utils import client as gemini_client

    src = [
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "config": GenerateContentConfig(temperature=0.0),
        }
        for prompt in prompts
    ]
    job = gemini_client.batches.create(model=model, src=src)
    print(f"[batch][gemini] submitted {job.name}")

    terminal = {
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    }
    while True:
        job = gemini_client.batches.get(name=job.name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state in terminal:
            break
        print(f"[batch][gemini] {state}")
        time.sleep(poll_interval)

    results = {}
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    if state != "JOB_STATE_SUCCEEDED":
        print(f"[batch][gemini] job ended as {state}")
        return results

    # Inlined responses come back in submission order.
    dest = job.dest
    inlined = getattr(dest, "inlined_responses", None) if dest else None
    if not inlined:
        print("[batch][gemini] no inlined responses (file-based dest not handled)")
        return results
    for prompt, item in zip(prompts, inlined):
        resp = getattr(item, "response", None)
        if resp is not None and getattr(resp, "text", None):
            results[prompt] = resp.text
    return results


# --------------------------------------------------------------------------- #
# HuggingFace / DeepSeek -- NO batch endpoint; use bounded concurrency.
# This is parallelism, not a true Batch API: no async discount.
# --------------------------------------------------------------------------- #
def _hf_parallel(model, prompts, max_workers=12):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from utils.utils import general_hf

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(general_hf, model, p): p for p in prompts}
        done = 0
        for fut in as_completed(futures):
            prompt = futures[fut]
            try:
                results[prompt] = fut.result()
            except Exception as e:  # noqa: BLE001 -- skip; falls back to live call
                print(f"[batch][hf] error: {e}")
            done += 1
            if done % 25 == 0:
                print(f"[batch][hf] {done}/{len(prompts)}")
    return results
