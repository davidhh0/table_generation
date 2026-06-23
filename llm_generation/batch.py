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


# --------------------------------------------------------------------------- #
# Batch journal -- restart tolerance (chunk-size independent)
# A submitted batch can take minutes/hours. If the program dies mid-poll, the
# batch_id would otherwise be lost and the work re-submitted on restart (the old
# batch keeps running, unread -- wasted cost + time).
#
# The journal is keyed by the OpenAI ``batch_id`` itself (one entry per submitted
# batch), with value {model, kind, items}:
#   * kind   -- "per_prompt" | "consolidated", selects how to parse the output.
#   * items  -- the submit-time ordering used to map custom_id "<prefix>-<i>"
#               back to the prompt(s); survives restart so reattach never depends
#               on regenerating the same order.
# On restart, ``_drain_in_flight`` resumes EVERY in-flight batch for the model
# (poll + fetch + parse), writes the answers to the cache, and deletes the entry --
# BEFORE any re-chunking. Because reattach is by batch_id, not by a hash of the
# chunk's prompt set, it is independent of ``batch_chunk_size``: you can restart
# with a different chunk size (or bundle size) and still recover every running
# batch instead of orphaning it. The entry is removed once fetched (or once the
# batch ends non-completed, so the next run resubmits that work fresh).
# --------------------------------------------------------------------------- #
def _journal():
    return diskcache.Cache(f'{working_dir}/local_dbs/cache/batch_jobs.db')


def _fetch_completed(oai, batch_id, poll_interval, label):
    """Poll ``batch_id`` to a terminal state; return its parsed JSONL records.

    Returns ``[]`` if the batch ended non-completed (failed/expired/cancelled or
    no output file), so the caller leaves that work unwarmed for a fresh resubmit.
    """
    while True:
        batch = oai.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        print(f"[batch]{label} {batch.status}")
        time.sleep(poll_interval)

    if batch.status != "completed" or not batch.output_file_id:
        print(f"[batch]{label} ended as {batch.status}")
        return []

    content = oai.files.content(batch.output_file_id).read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    return [json.loads(ln) for ln in content.splitlines() if ln.strip()]


def _openai_run_batch(oai, model, kind, items, lines, poll_interval, label):
    """Submit one OpenAI batch, journal it by batch_id, poll, fetch.

    Returns ``(records, items)`` where ``records`` is the parsed JSONL output
    (empty if the batch ended non-completed) and ``items`` is the submit-time
    ordering to map each ``custom_id`` index back. Caller does the result parsing.

    The journal entry is written immediately after ``create`` (so a crash right
    after submission is still resumable via ``_drain_in_flight``) and removed
    once the batch reaches a terminal state.
    """
    buf = io.BytesIO("\n".join(lines).encode("utf-8"))
    buf.name = "batch_input.jsonl"
    uploaded = oai.files.create(file=buf, purpose="batch")
    batch = oai.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch.id
    with _journal() as jr:
        jr[batch_id] = {"model": model, "provider": "openai", "kind": kind, "items": items}
    print(f"[batch]{label} submitted {batch_id}")

    records = _fetch_completed(oai, batch_id, poll_interval, label)
    with _journal() as jr:
        jr.pop(batch_id, None)
    return records, items


def _drain_in_flight(model, poll_interval):
    """Resume any in-flight batches journaled for ``model`` and harvest them.

    Called at the start of ``warm_cache`` before any new submission. Reattaches by
    batch_id, so it recovers batches submitted by a previous run REGARDLESS of the
    chunk/bundle size that run used. Dispatches by the entry's ``provider`` (missing
    provider defaults to "openai" for entries written before this field existed), so
    a single drain covers OpenAI and Anthropic alike. Returns {original_prompt:
    answer_string} for every answer recovered; the caller writes these to the cache
    and drops the corresponding prompts so they are never resubmitted.
    """
    with _journal() as jr:
        pending = [(bid, jr.get(bid)) for bid in list(jr)]
    pending = [(bid, e) for bid, e in pending if e and e.get("model") == model]
    if not pending:
        return {}

    print(f"[batch][resume] {len(pending)} in-flight batch(es) journaled for "
          f"{model}; reattaching (chunk-size independent)")
    oai = None
    resolved = {}
    for batch_id, entry in pending:
        provider = entry.get("provider", "openai")
        kind = entry.get("kind", "per_prompt")
        items = entry.get("items", [])
        print(f"[batch][resume] reattaching to {batch_id} (provider={provider}, kind={kind})")
        if provider == "anthropic":
            results = _anthropic_fetch_completed(batch_id, poll_interval, "[resume]")
            if results:
                if kind == "consolidated":
                    resolved.update(_parse_anthropic_consolidated(results, items))
                else:
                    resolved.update(_parse_anthropic_per_prompt(results, items))
        elif provider == "gemini":
            texts = _gemini_fetch_completed(batch_id, poll_interval, "[resume]")
            if texts:
                if kind == "consolidated":
                    resolved.update(_parse_gemini_consolidated(texts, items))
                else:
                    resolved.update(_parse_gemini_per_prompt(texts, items))
        else:
            if oai is None:
                from openai import OpenAI
                oai = OpenAI(api_key=os.environ["openai_api_key"])
            records = _fetch_completed(oai, batch_id, poll_interval, "[resume]")
            if records:
                if kind == "consolidated":
                    resolved.update(_parse_consolidated_records(records, items))
                else:
                    resolved.update(_parse_per_prompt_records(records, items))
        with _journal() as jr:
            jr.pop(batch_id, None)
    return resolved


# --------------------------------------------------------------------------- #
# Open-Book consolidation
# All open-book prompts share the template (prompts.yaml open_book_wrapper):
#     Given a factual table titled '<TITLE>':
#     <CSV>
#     Question: <Q>.
#     <INSTRUCTION>
# so every question on one table shares the prefix up to "\nQuestion: ". We bundle
# a table's questions into one structured-JSON call, then split the answers back
# to each question's own cache key, leaving scoring untouched.
# --------------------------------------------------------------------------- #
_OB_PREFIX = "Given a factual table titled"
_OB_MARKER = "\nQuestion: "

# The structured-output contract shared by both providers, so a bundle answers
# identically whether resolved via OpenAI response_format or Anthropic tool-use.
_OB_ANSWERS_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "answer": {"type": "string"},
                },
                "required": ["id", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

# OpenAI response_format wrapper around the shared schema.
_OB_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "table_answers",
        "strict": True,
        "schema": _OB_ANSWERS_SCHEMA,
    },
}

# Anthropic forced tool-use mirror of the same schema.
_OB_TOOL_NAME = "table_answers"
_OB_TOOL = {
    "name": _OB_TOOL_NAME,
    "description": "Return one answer per question id using only the given table.",
    "input_schema": _OB_ANSWERS_SCHEMA,
}


def _load_conf():
    import yaml
    with open(f"{working_dir}/config.yaml", "r") as f:
        return yaml.safe_load(f) or {}


def _split_open_book(prompt):
    """Return (table_block, question_tail) or None if not an open-book prompt.

    table_block = "Given a factual table titled '<TITLE>':\\n<CSV>" (the shared
    prefix to group on); question_tail = "Question: <Q>.\\n<INSTRUCTION>".
    Splits on the first "\\nQuestion: " after the CSV. (Wikipedia factual cells
    don't contain that marker, so the first occurrence is the wrapper's.)
    """
    i = prompt.find(_OB_MARKER)
    if not prompt.startswith(_OB_PREFIX) or i == -1:
        return None
    return prompt[:i], prompt[i + 1:]  # drop the leading newline from the tail


def _group_open_book(prompts, bundle_size):
    """Group open-book prompts by table block, chunked to <= bundle_size.

    Returns a list of units: {"table_block", "originals": [prompt...],
    "questions": [question_tail...]}. Prompts that don't parse are skipped here
    and stay in the caller's per-prompt list (resolved live in pass 2).
    """
    groups = {}  # table_block -> list of (original_prompt, question_tail)
    for p in prompts:
        split = _split_open_book(p)
        if split is None:
            continue
        table_block, tail = split
        groups.setdefault(table_block, []).append((p, tail))

    units = []
    for table_block, items in groups.items():
        for start in range(0, len(items), bundle_size):
            chunk = items[start:start + bundle_size]
            units.append({
                "table_block": table_block,
                "originals": [o for o, _ in chunk],
                "questions": [q for _, q in chunk],
            })
    return units


def _consolidated_prompt(unit):
    """Build one prompt: the table once + a numbered list of its questions."""
    lines = [
        unit["table_block"],
        "",
        f"Answer the following {len(unit['questions'])} independent questions "
        "using ONLY the table above. Treat each question separately and follow "
        "its own answer-format instruction. Return a JSON object with an "
        '"answers" array containing one {"id", "answer"} entry per question id '
        "below; \"answer\" is the exact answer value as a string.",
        "",
    ]
    for idx, q in enumerate(unit["questions"], start=1):
        lines.append(f"[{idx}] {q}")
    return "\n".join(lines)


def _openai_consolidated(model, units, poll_interval, conf):
    """Resolve open-book ``units`` via one OpenAI batch of structured-JSON calls.

    Returns {original_prompt: answer_string}. Any id missing from a response is
    simply absent from the result, so the caller leaves it unwarmed (live fallback).
    """
    if not units:
        return {}
    from openai import OpenAI
    oai = OpenAI(api_key=os.environ["openai_api_key"])

    base_cap = conf.get("max_completion_tokens", 4096)
    effort = conf.get("openai_reasoning_effort", "low")
    is_reasoning = model.lower().startswith("gpt-5") or model.lower().startswith(("o1", "o3", "o4"))

    lines = []
    for i, unit in enumerate(units):
        # Scale the ceiling with bundle size (reasoning + N visible answers).
        cap = min(base_cap + 256 * len(unit["questions"]), 32000)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": _consolidated_prompt(unit)}],
            "max_completion_tokens": cap,
            "response_format": _OB_SCHEMA,
        }
        if is_reasoning:
            body["reasoning_effort"] = effort
        lines.append(json.dumps({
            "custom_id": f"bundle-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))

    # items[i] = the originals list for bundle i, used to map "bundle-<i>" answers
    # back to per-question cache keys (survives restart via the journal).
    items = [u["originals"] for u in units]
    records, items = _openai_run_batch(
        oai, model, "consolidated", items, lines, poll_interval, "[consolidate]")
    return _parse_consolidated_records(records, items)


def _parse_consolidated_records(records, items):
    """Map consolidated-batch JSONL records -> {original_prompt: answer_string}.

    ``items[i]`` is the originals list for bundle ``i`` (custom_id "bundle-<i>").
    Any id missing/out-of-range/unparseable is omitted, so the caller leaves that
    prompt unwarmed (live fallback in the scoring pass).
    """
    results = {}
    for rec in records:
        uidx = int(rec["custom_id"].split("-")[1])
        originals = items[uidx]
        try:
            choice = rec["response"]["body"]["choices"][0]
            text = choice["message"]["content"]
            finish = choice.get("finish_reason")
            if not text or finish == "length":
                print(f"[batch][consolidate] {rec['custom_id']}: empty/truncated "
                      f"(finish_reason={finish}) -- {len(originals)} q(s) "
                      "fall back to live calls")
                continue
            parsed = json.loads(text)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"[batch][consolidate] {rec['custom_id']}: unparseable ({e})")
            continue
        for item in parsed.get("answers", []):
            qid = item.get("id")
            ans = item.get("answer")
            if not isinstance(qid, int) or not (1 <= qid <= len(originals)):
                continue
            results[originals[qid - 1]] = ans
    return results


def warm_cache(model, prompts, poll_interval=POLL_INTERVAL, chunk_size=CHUNK_SIZE):
    """Batch-resolve ``prompts`` for ``model`` and write them into the cache.

    Prompts are submitted in chunks of ``chunk_size``; each chunk's responses are
    written to the cache before the next chunk is submitted. So a crash mid-run
    only loses the single in-flight chunk -- the next run's collect pass sees the
    already-cached chunks as hits and re-batches only the remainder.

    Restart-safe across chunk-size changes (OpenAI + Anthropic): each submitted
    batch is journaled by its batch_id, and ``_drain_in_flight`` resumes every
    in-flight batch up front. So you may restart with a different ``batch_chunk_size``
    / ``open_book_bundle_size`` and still recover running batches rather than
    orphaning them.

    Resilient by design: any prompt that errors or is missing from a chunk's
    output is simply left out of the cache, so the subsequent scoring pass falls
    back to a live ``get_llm_response`` call rather than scoring a bogus answer.
    """
    prompts = [p for p in dict.fromkeys(p.strip() for p in prompts) if p]
    if not prompts:
        print(f"[batch] nothing to warm for {model}")
        return {}

    cache = _cache_for(model)
    all_results = {}
    written = 0

    # --- Resume in-flight batches from a previous run (OpenAI/Anthropic/Gemini) --
    # Reattach by batch_id to any batch a prior run left running, harvest its
    # answers into the cache, and drop those prompts from the to-submit set. This
    # is independent of batch_chunk_size / open_book_bundle_size, so restarting
    # with a different chunk size recovers running batches instead of orphaning
    # them (and paying twice). Completed-and-cached chunks were already filtered
    # out by the collect pass; this covers the batches that were still in flight.
    if 'gpt' in model or 'claude' in model.lower() or 'gemini' in model:
        drained = _drain_in_flight(model, poll_interval)
        resumed = 0
        for prompt, ans in drained.items():
            if ans is None or not str(ans).strip():
                continue
            cache[prompt] = str(ans).strip()
            written += 1
            resumed += 1
        if drained:
            all_results.update(drained)
            done = {p for p, a in drained.items() if a is not None and str(a).strip()}
            prompts = [p for p in prompts if p not in done]
            print(f"[batch][resume] harvested {resumed} answer(s) from in-flight "
                  f"batch(es); {len(prompts)} prompt(s) remain to submit")
        if not prompts:
            print(f"[batch] done: wrote {written} response(s) to cache for {model} "
                  "(all recovered from in-flight batches)")
            return all_results

    # --- Open-Book consolidation (OpenAI + Anthropic + Gemini) -------------
    # All open-book questions on the same table share one CSV. Bundle them into a
    # single structured-JSON call per table (CSV sent once, N requests -> 1), then
    # split the answers back to each question's own cache key. Closed-Book / MC
    # carry no table and are NOT bundled (would contaminate parametric-recall).
    # Methodologically sound because open-book already gives the model the full table.
    conf = _load_conf()
    consolidator = None
    if conf.get("consolidate_open_book", True):
        if 'gpt' in model:
            consolidator = _openai_consolidated
        elif 'claude' in model.lower():
            consolidator = _anthropic_consolidated
        elif 'gemini' in model:
            consolidator = _gemini_consolidated
    if consolidator is not None:
        ob = [p for p in prompts if p.startswith(_OB_PREFIX)]
        if ob:
            bundle_size = conf.get("open_book_bundle_size", 15)
            units = _group_open_book(ob, bundle_size)
            # Submit the table-bundles in chunks of `chunk_size` table(s) per batch
            # (not all at once); each chunk is cached before the next, so a crash
            # only loses the in-flight chunk.
            n_uchunks = (len(units) + chunk_size - 1) // chunk_size
            print(f"[batch][consolidate] {len(ob)} open-book prompt(s) -> "
                  f"{len(units)} table-bundle(s) for {model}, submitting in "
                  f"{n_uchunks} chunk(s) of up to {chunk_size} bundle(s)")
            resolved = set()
            for ui in range(n_uchunks):
                uchunk = units[ui * chunk_size:(ui + 1) * chunk_size]
                print(f"[batch][consolidate] {model}: bundle-chunk {ui + 1}/{n_uchunks} "
                      f"({len(uchunk)} table-bundle(s))")
                res = consolidator(model, uchunk, poll_interval, conf)
                for orig, ans in res.items():
                    if ans is None or not str(ans).strip():
                        continue
                    cache[orig] = str(ans).strip()
                    written += 1
                    resolved.add(orig)
                all_results.update(res)
            # Unresolved open-book prompts fall through to the per-prompt path below.
            prompts = [p for p in prompts if p not in resolved]
            print(f"[batch][consolidate] resolved {len(resolved)}/{len(ob)} "
                  f"open-book via bundles; {len(prompts)} prompt(s) left for per-prompt warm")

    if not prompts:
        print(f"[batch] done: wrote {written} response(s) to cache for {model} (all via consolidation)")
        return all_results

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
    print(f"[batch] warming {total} per-prompt request(s) for model {model} "
          f"in {n_chunks} chunk(s) of up to {chunk_size}")

    for ci in range(n_chunks):
        chunk = prompts[ci * chunk_size:(ci + 1) * chunk_size]
        print(f"[batch] {model}: chunk {ci + 1}/{n_chunks} ({len(chunk)} prompt(s))")
        # _hf_parallel has no poll_interval argument.
        if provider is _hf_parallel:
            results = provider(model, chunk)
        else:
            results = provider(model, chunk, poll_interval)
        for prompt, response in results.items():
            # Skip None AND empty strings: an empty answer cached as "" is later
            # read back by get_llm_response as a (useless) hit. Leaving it unwarmed
            # lets pass 2 fall back to a live call.
            if response is None or not response.strip():
                continue
            cache[prompt] = response.strip()
            written += 1
        all_results.update(results)
        print(f"[batch] {model}: chunk {ci + 1} cached "
              f"(total written {written})")
    print(f"[batch] done: wrote {written} response(s) to cache for {model}")
    return all_results


# --------------------------------------------------------------------------- #
# Anthropic Claude -- Message Batches API
# --------------------------------------------------------------------------- #
def _claude_request_params(model, conf, answer_cap):
    """Per-request param dict for Claude: max_tokens (+ optional extended thinking).

    ``answer_cap`` is the visible-output budget (mirrors OpenAI max_completion_tokens).
    Extended thinking is off by default (config ``claude_extended_thinking``); when on,
    the thinking budget is added on top of the answer budget because Anthropic requires
    ``max_tokens`` to exceed the thinking ``budget_tokens`` and they share the window.
    """
    if conf.get("claude_extended_thinking", False):
        budget = int(conf.get("claude_thinking_budget", 4096))
        return {
            "model": model,
            "max_tokens": budget + answer_cap,
            "thinking": {"type": "enabled", "budget_tokens": budget},
        }
    return {"model": model, "max_tokens": answer_cap}


def _anthropic_fetch_completed(batch_id, poll_interval, label):
    """Poll an Anthropic message batch to ``ended``; return its results list.

    Mirrors ``_fetch_completed`` (OpenAI). The batch ends regardless of per-request
    success; success/error is decided per result in the parse helpers.
    """
    from utils.utils import claude_client
    while True:
        batch = claude_client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = getattr(batch, "request_counts", None)
        print(f"[batch]{label} {batch.processing_status}"
              + (f" processing={counts.processing}" if counts else ""))
        time.sleep(poll_interval)
    return list(claude_client.messages.batches.results(batch_id))


def _anthropic_run_batch(model, kind, items, requests, poll_interval, label):
    """Submit one Anthropic batch, journal it by batch_id, poll, fetch.

    Analog of ``_openai_run_batch``: the journal entry (provider="anthropic") is
    written right after create -- so a crash mid-poll is recoverable by
    ``_drain_in_flight`` -- and removed once the batch reaches ``ended``.
    """
    from utils.utils import claude_client
    batch = claude_client.messages.batches.create(requests=requests)
    batch_id = batch.id
    with _journal() as jr:
        jr[batch_id] = {"model": model, "provider": "anthropic", "kind": kind, "items": items}
    print(f"[batch]{label} submitted {batch_id}")

    results = _anthropic_fetch_completed(batch_id, poll_interval, label)
    with _journal() as jr:
        jr.pop(batch_id, None)
    return results, items


def _anthropic_batch(model, prompts, poll_interval):
    from utils.utils import _config
    conf = _config()
    cap = conf.get("max_completion_tokens", MAX_TOKENS)
    requests = []
    for i, prompt in enumerate(prompts):
        params = _claude_request_params(model, conf, cap)
        params["messages"] = [{"role": "user", "content": prompt}]
        requests.append({"custom_id": f"req-{i}", "params": params})
    results, items = _anthropic_run_batch(
        model, "per_prompt", list(prompts), requests, poll_interval, "[claude]")
    return _parse_anthropic_per_prompt(results, items)


def _parse_anthropic_per_prompt(results, items):
    """Map per-prompt Anthropic batch results -> {prompt: answer_string}.

    ``items[i]`` is the prompt for custom_id "req-<i>". Errored, truncated
    (stop_reason="max_tokens"), or empty results are logged and omitted, so the
    caller leaves them unwarmed (live fallback) rather than caching a bad answer.
    """
    out = {}
    for result in results:
        idx = int(result.custom_id.split("-")[1])
        prompt = items[idx]
        if result.result.type != "succeeded":
            print(f"[batch][claude] {result.custom_id}: {result.result.type}")
            continue
        msg = result.result.message
        if getattr(msg, "stop_reason", None) == "max_tokens":
            print(f"[batch][claude] {result.custom_id}: truncated "
                  "(stop_reason=max_tokens) -- raise max_completion_tokens")
            continue
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if not text or not text.strip():
            print(f"[batch][claude] {result.custom_id}: empty content")
            continue
        out[prompt] = text
    return out


def _anthropic_consolidated(model, units, poll_interval, conf):
    """Resolve open-book ``units`` via one Anthropic batch of forced tool-use calls.

    Analog of ``_openai_consolidated`` but using tool-use (Anthropic has no
    response_format): each bundle forces the ``table_answers`` tool whose input_schema
    is the shared ``_OB_ANSWERS_SCHEMA``. Returns {original_prompt: answer_string};
    any missing id is absent so the caller leaves it unwarmed (live fallback).

    Extended thinking is intentionally NOT applied here: Anthropic forbids forced
    tool_choice while thinking is enabled, and bundled single-value answers don't need it.
    """
    if not units:
        return {}
    base_cap = conf.get("max_completion_tokens", 4096)
    requests = []
    for i, unit in enumerate(units):
        # Scale the visible-answer budget with bundle size (N answers in one call).
        cap = min(base_cap + 256 * len(unit["questions"]), 32000)
        requests.append({
            "custom_id": f"bundle-{i}",
            "params": {
                "model": model,
                "max_tokens": cap,
                "messages": [{"role": "user", "content": _consolidated_prompt(unit)}],
                "tools": [_OB_TOOL],
                "tool_choice": {"type": "tool", "name": _OB_TOOL_NAME},
            },
        })
    items = [u["originals"] for u in units]
    results, items = _anthropic_run_batch(
        model, "consolidated", items, requests, poll_interval, "[consolidate]")
    return _parse_anthropic_consolidated(results, items)


def _parse_anthropic_consolidated(results, items):
    """Map consolidated Anthropic tool-use results -> {original_prompt: answer_string}.

    ``items[i]`` is the originals list for bundle ``i`` (custom_id "bundle-<i>").
    Reads the forced ``table_answers`` tool_use block's ``answers`` array; any id
    missing/out-of-range is omitted (live fallback). Mirrors _parse_consolidated_records.
    """
    out = {}
    for result in results:
        uidx = int(result.custom_id.split("-")[1])
        originals = items[uidx]
        if result.result.type != "succeeded":
            print(f"[batch][consolidate] {result.custom_id}: {result.result.type} -- "
                  f"{len(originals)} q(s) fall back to live calls")
            continue
        msg = result.result.message
        block = next((b for b in msg.content
                      if b.type == "tool_use" and getattr(b, "name", None) == _OB_TOOL_NAME), None)
        if block is None:
            print(f"[batch][consolidate] {result.custom_id}: no tool_use block")
            continue
        for item in (block.input or {}).get("answers", []):
            qid = item.get("id")
            ans = item.get("answer")
            if not isinstance(qid, int) or not (1 <= qid <= len(originals)):
                continue
            out[originals[qid - 1]] = ans
    return out


# --------------------------------------------------------------------------- #
# OpenAI GPT -- /v1/chat/completions batch
# --------------------------------------------------------------------------- #
def _openai_batch(model, prompts, poll_interval):
    from openai import OpenAI
    from utils.utils import _config, _is_openai_reasoning_model

    oai = OpenAI(api_key=os.environ["openai_api_key"])

    # Mirror the live-call caps (utils.chatgpt): cap total output tokens and, for
    # reasoning models, set a low reasoning_effort. Without this the batch path
    # runs reasoning unbounded -- the same hidden cost as single calls.
    conf = _config()
    max_out_tokens = conf.get("max_completion_tokens", 512)

    lines = []
    for i, prompt in enumerate(prompts):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_out_tokens,
        }
        if _is_openai_reasoning_model(model):
            body["reasoning_effort"] = conf.get("openai_reasoning_effort", "minimal")
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))
    records, items = _openai_run_batch(
        oai, model, "per_prompt", list(prompts), lines, poll_interval, "[openai]")
    return _parse_per_prompt_records(records, items)


def _parse_per_prompt_records(records, items):
    """Map per-prompt-batch JSONL records -> {prompt: answer_string}.

    ``items[i]`` is the prompt for custom_id "req-<i>". Empty/truncated/errored
    lines are logged and omitted, so the caller leaves them unwarmed (live fallback).
    """
    results = {}
    for rec in records:
        idx = int(rec["custom_id"].split("-")[1])
        prompt = items[idx]
        try:
            choice = rec["response"]["body"]["choices"][0]
            text = choice["message"]["content"]
            finish = choice.get("finish_reason")
            # Explain empties: a reasoning model that hits the token ceiling during
            # reasoning returns finish_reason="length" with empty content. Make the
            # reason visible so a "wrote 0/N" is never a silent mystery.
            if not text or finish == "length":
                print(f"[batch][openai] {rec['custom_id']}: empty/truncated "
                      f"(finish_reason={finish}) -- raise max_completion_tokens")
                continue
            results[prompt] = text
        except (KeyError, IndexError, TypeError):
            err = rec.get("error") or (rec.get("response") or {}).get("body", {}).get("error")
            print(f"[batch][openai] {rec['custom_id']}: no content ({err})")
    return results


# --------------------------------------------------------------------------- #
# Google Gemini -- batch jobs
# --------------------------------------------------------------------------- #
# Gemini structured-output schema. Mirrors the shared _OB_ANSWERS_SCHEMA but drops
# `additionalProperties` (Gemini's response_schema uses the OpenAPI subset, which
# doesn't accept it). Paired with response_mime_type="application/json".
_GEMINI_OB_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "answer": {"type": "string"},
                },
                "required": ["id", "answer"],
            },
        }
    },
    "required": ["answers"],
}

_GEMINI_TERMINAL = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}


def _gemini_gen_config(conf, schema=None):
    """Build a GenerateContentConfig: temperature 0, max_output_tokens from the
    shared cap, optional thinking budget, and (for consolidation) a JSON schema.

    Gemini's max_output_tokens caps VISIBLE output only; thinking has its own
    budget (thinking_config), so a generous cap never truncates the answer. Thinking
    is left at the SDK default unless ``gemini_thinking_budget`` is set in config."""
    from google.genai.types import GenerateContentConfig
    kwargs = {"temperature": 0.0}
    cap = conf.get("max_completion_tokens")
    if cap:
        kwargs["max_output_tokens"] = cap
    budget = conf.get("gemini_thinking_budget")
    if budget is not None:
        from google.genai.types import ThinkingConfig
        kwargs["thinking_config"] = ThinkingConfig(thinking_budget=int(budget))
    if schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = schema
    return GenerateContentConfig(**kwargs)


def _gemini_fetch_completed(job_name, poll_interval, label):
    """Poll a Gemini batch job to a terminal state; return its inlined response
    texts IN SUBMISSION ORDER (list aligned to the submitted src; None per slot
    with no text). Returns [] if the job ended non-succeeded or used a file dest.

    Gemini has no custom_id, so callers map results back to prompts by index via
    the journaled ``items`` list -- order is what makes reattach work."""
    from utils.utils import client as gemini_client
    while True:
        job = gemini_client.batches.get(name=job_name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state in _GEMINI_TERMINAL:
            break
        print(f"[batch]{label} {state}")
        time.sleep(poll_interval)

    if state != "JOB_STATE_SUCCEEDED":
        print(f"[batch]{label} ended as {state}")
        return []
    dest = job.dest
    inlined = getattr(dest, "inlined_responses", None) if dest else None
    if not inlined:
        print(f"[batch]{label} no inlined responses (file-based dest not handled)")
        return []
    texts = []
    for item in inlined:
        resp = getattr(item, "response", None)
        texts.append(getattr(resp, "text", None) if resp is not None else None)
    return texts


def _gemini_run_batch(model, kind, items, src, poll_interval, label):
    """Submit one Gemini batch, journal it by job.name, poll, fetch.

    Analog of ``_openai_run_batch``: the journal entry (provider="gemini") is
    written right after create -- so a crash mid-poll is recoverable by
    ``_drain_in_flight`` -- and removed once the job reaches a terminal state.
    Returns ``(texts, items)`` where texts is submission-ordered."""
    from utils.utils import client as gemini_client
    job = gemini_client.batches.create(model=model, src=src)
    job_name = job.name
    with _journal() as jr:
        jr[job_name] = {"model": model, "provider": "gemini", "kind": kind, "items": items}
    print(f"[batch]{label} submitted {job_name}")

    texts = _gemini_fetch_completed(job_name, poll_interval, label)
    with _journal() as jr:
        jr.pop(job_name, None)
    return texts, items


def _gemini_batch(model, prompts, poll_interval):
    conf = _load_conf()
    cfg = _gemini_gen_config(conf)
    src = [
        {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "config": cfg}
        for prompt in prompts
    ]
    texts, items = _gemini_run_batch(
        model, "per_prompt", list(prompts), src, poll_interval, "[gemini]")
    return _parse_gemini_per_prompt(texts, items)


def _parse_gemini_per_prompt(texts, items):
    """Map submission-ordered Gemini texts -> {prompt: answer_string}.

    ``items[i]`` is the prompt for response slot ``i``. Empty/missing slots are
    logged and omitted, so the caller leaves them unwarmed (live fallback)."""
    out = {}
    for i, prompt in enumerate(items):
        text = texts[i] if i < len(texts) else None
        if text and text.strip():
            out[prompt] = text
        else:
            print(f"[batch][gemini] response {i}: empty/no content")
    return out


def _gemini_consolidated(model, units, poll_interval, conf):
    """Resolve open-book ``units`` via one Gemini batch of JSON-schema calls.

    Analog of ``_openai_consolidated`` using Gemini native structured output
    (response_mime_type=application/json + response_schema). Returns
    {original_prompt: answer_string}; any missing id is absent (live fallback)."""
    if not units:
        return {}
    base_cap = conf.get("max_completion_tokens", 4096)
    src = []
    for unit in units:
        # Scale the visible-answer budget with bundle size (N answers in one call).
        cap = min(base_cap + 256 * len(unit["questions"]), 32000)
        cfg = _gemini_gen_config({**conf, "max_completion_tokens": cap},
                                 schema=_GEMINI_OB_SCHEMA)
        src.append({
            "contents": [{"role": "user", "parts": [{"text": _consolidated_prompt(unit)}]}],
            "config": cfg,
        })
    items = [u["originals"] for u in units]
    texts, items = _gemini_run_batch(
        model, "consolidated", items, src, poll_interval, "[consolidate]")
    return _parse_gemini_consolidated(texts, items)


def _parse_gemini_consolidated(texts, items):
    """Map submission-ordered Gemini JSON texts -> {original_prompt: answer_string}.

    ``items[i]`` is the originals list for bundle ``i``. Any id missing/out-of-range
    /unparseable is omitted (live fallback). Mirrors _parse_consolidated_records."""
    out = {}
    for i, originals in enumerate(items):
        text = texts[i] if i < len(texts) else None
        if not text or not text.strip():
            print(f"[batch][consolidate] bundle-{i}: empty -- "
                  f"{len(originals)} q(s) fall back to live calls")
            continue
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError) as e:
            print(f"[batch][consolidate] bundle-{i}: unparseable ({e})")
            continue
        for item in parsed.get("answers", []):
            qid = item.get("id")
            ans = item.get("answer")
            if not isinstance(qid, int) or not (1 <= qid <= len(originals)):
                continue
            out[originals[qid - 1]] = ans
    return out


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
