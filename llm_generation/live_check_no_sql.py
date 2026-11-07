"""Quick N-prompt live check that the dedicated no-SQL List-Retrieval wrapper stops
Llama from answering closed-book recall prompts with a SQL query.

Llama-3.3-70B answered 361/388 (93%) of its cached closed-book LR prompts with a
`SELECT ...` statement instead of the recalled values, scoring an automatic F1 of 0
(329/336 exact zeros in scores_statistics.csv). For each sampled question we:
  1. read the OLD cached Llama response (built with the plain closed_book_wrapper),
  2. rebuild the prompt with `closed_book_wrapper_no_sql` (prompts.yaml),
  3. call Llama LIVE with the new prompt (no cache),
and print SQL/answer/F1 side by side.

Run from llm_generation/:  python live_check_no_sql.py [N]
Makes N live HF calls (needs hf_api in env). Default N=5.
"""
import re
import sys

import git
import yaml
import diskcache

wd = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.insert(0, wd)

prompts = yaml.safe_load(open(f'{wd}/llm_generation/prompts.yaml'))
conf = yaml.safe_load(open(f'{wd}/config.yaml'))
pc = diskcache.Cache(f'{wd}/local_dbs/cache/prompts_cache.db')

from utils.utils import general_hf  # noqa: E402
from wikiparser import WikiTableParser  # noqa: E402
from prompts_generation import f1_score_sets  # noqa: E402

MODEL = conf['llm_model']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
_FIRST_LINE = re.compile(
    r"^Assume a factual table titled '(.*)' with a primary key column named '(.*)':$")
_SQL = re.compile(r'^(select|with)\b', re.I)

parser_ins = WikiTableParser()


def _is_sql(resp):
    return bool(_SQL.match(str(resp).strip().strip('"').strip()))


def _parse_list(resp):
    """Same parse chain as prompt_generation_list_categorical."""
    x = parser_ins.try_cast(resp)
    try:
        return [parser_ins.try_cast(k) for k in x.split(',')]
    except AttributeError:
        return []


def _rebuild_no_sql(old_q):
    """Reconstruct the no-SQL closed-book prompt from a cached plain one."""
    lines = old_q.split('\n')
    m = _FIRST_LINE.match(lines[0])
    if not m:
        return None
    title, pk = m.group(1), m.group(2)
    question = lines[1] if len(lines) > 1 else ''
    instruction = '\n'.join(lines[2:])
    return prompts['closed_book_wrapper_no_sql'].format(
        TABLE_TITLE=title, PRIMARY_KEY_COLUMN=pk,
        QUESTION=question, INSTRUCTION=instruction).strip()


def main():
    resp_cache = diskcache.Cache(f'{wd}/local_dbs/cache/llm_cache/{MODEL}.db')
    examples = []
    for url in pc.iterkeys():
        d = pc[url]
        # plain (non-variant) sub-keys hold the prompts Llama already answered
        for bd in ('categorical_equals', 'categorical_in'):
            sub = d.get(f'list_categorical_{bd}_closed_book')
            if not sub:
                continue
            new_prompt = _rebuild_no_sql(sub['q'])
            if new_prompt is None or sub['q'] not in resp_cache:
                continue
            examples.append((url, sub['a'], sub['q'], new_prompt))
            break
        if len(examples) >= N:
            break

    print(f"Testing {len(examples)} closed-book LR prompts with the no-SQL wrapper "
          f"({MODEL})\n")
    old_sql = new_sql = 0
    old_f1 = new_f1 = 0.0
    for i, (url, ans, old_q, new_p) in enumerate(examples, 1):
        old_r = str(resp_cache[old_q])
        new_r = str(general_hf(MODEL, new_p))
        o_sql, n_sql = _is_sql(old_r), _is_sql(new_r)
        o_f1 = f1_score_sets(ans, _parse_list(old_r))[-1]
        n_f1 = f1_score_sets(ans, _parse_list(new_r))[-1]
        old_sql += o_sql
        new_sql += n_sql
        old_f1 += o_f1
        new_f1 += n_f1
        print(f"--- #{i}  expected={str(ans)[:70]!r}")
        print(f"    OLD (plain)  sql={o_sql}  f1={o_f1:.2f}  resp={old_r[:90]!r}")
        print(f"    NEW (no-SQL) sql={n_sql}  f1={n_f1:.2f}  resp={new_r[:90]!r}")
        print()

    n = len(examples) or 1
    print(f"SUMMARY: SQL answers {old_sql}/{len(examples)} -> {new_sql}/{len(examples)}; "
          f"mean F1 {old_f1 / n:.3f} -> {new_f1 / n:.3f}")


if __name__ == '__main__':
    main()
