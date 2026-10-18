"""Quick 5-prompt live check that the dedicated anti-refusal closed-book wrapper
stops Claude Opus 4.8 from refusing tableless recall prompts.

For each sampled closed-book question we:
  1. read the OLD cached Opus response (built with the plain closed_book_wrapper),
  2. rebuild the prompt with `closed_book_wrapper_anti_refusal` (prompts.yaml),
  3. call Opus LIVE with the new prompt (no cache),
and print refusal/answer/correctness side by side.

Run from llm_generation/:  python test_anti_refusal.py
Makes 5 live API calls (needs claude_api in env).
"""
import re
import sys

import git
import yaml
import diskcache

wd = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.insert(0, wd)

prompts = yaml.safe_load(open(f'{wd}/llm_generation/prompts.yaml'))
pc = diskcache.Cache(f'{wd}/local_dbs/cache/prompts_cache.db')
resp_cache = diskcache.Cache(f'{wd}/local_dbs/cache/llm_cache/claude-opus-4-8.db')

from utils.utils import claude  # noqa: E402

MODEL = 'claude-opus-4-8'
N = 5
_FIRST_LINE = re.compile(
    r"^Assume a factual table titled '(.*)' with a primary key column named '(.*)':$")
_REFUSAL = re.compile(
    r"don't have access|do not have access|wasn't provided|was not provided|"
    r"no (actual|such) (table|data)|cannot (count|provide|determine|identify|verify)|"
    r"I don't have|share the (table|data)", re.I)


def _rebuild_anti_refusal(old_q):
    """Reconstruct the anti-refusal closed-book prompt from a cached plain one."""
    lines = old_q.split('\n')
    m = _FIRST_LINE.match(lines[0])
    if not m:
        return None
    title, pk = m.group(1), m.group(2)
    question = lines[1] if len(lines) > 1 else ''
    instruction = '\n'.join(lines[2:])
    return prompts['closed_book_wrapper_anti_refusal'].format(
        TABLE_TITLE=title, PRIMARY_KEY_COLUMN=pk,
        QUESTION=question, INSTRUCTION=instruction).strip()


def main():
    examples = []
    for url in pc.iterkeys():
        d = pc[url]
        sub = d.get('single_value_closed_book')
        if not sub:
            continue
        new_prompt = _rebuild_anti_refusal(sub['q'])
        if not new_prompt:
            continue
        examples.append((url, sub['a'], sub['q'], new_prompt))
        if len(examples) >= N:
            break

    print(f"Testing {len(examples)} closed-book prompts with the anti-refusal wrapper\n")
    old_refusals = new_refusals = correct = 0
    for i, (url, ans, old_q, new_p) in enumerate(examples, 1):
        old_r = str(resp_cache.get(old_q, '<not cached>'))
        new_r = str(claude(MODEL, new_p))
        old_ref = bool(_REFUSAL.search(old_r))
        new_ref = bool(_REFUSAL.search(new_r))
        is_correct = new_r.strip().lower() == str(ans).strip().lower()
        old_refusals += old_ref
        new_refusals += new_ref
        correct += is_correct
        print(f"--- #{i}  expected={ans!r}")
        print(f"    OLD (plain)   refusal={old_ref}  resp={old_r[:80]!r}")
        print(f"    NEW (anti-ref) refusal={new_ref}  correct={is_correct}  resp={new_r[:80]!r}")
        print()

    n = len(examples)
    print(f"SUMMARY: old refusals {old_refusals}/{n} -> new refusals {new_refusals}/{n}; "
          f"new correct {correct}/{n}")


if __name__ == '__main__':
    main()
