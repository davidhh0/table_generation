import os
import random
import git
from pandas import isna
import yaml
from utils.utils import get_llm_response
import diskcache
from collections import Counter
working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

random.seed(0)
choices_cache = diskcache.Cache(f'{working_dir}/local_dbs/cache/choices_cache.db')
prompts_cache = diskcache.Cache(f'{working_dir}/local_dbs/cache/prompts_cache.db')

# Models that refuse tableless closed-book / list-MC probes ("I don't have access
# to the table") instead of guessing from parametric memory. For these, swap in the
# dedicated *_anti_refusal wrappers (prompts.yaml). Keyed per-model so every other
# model keeps its exact original prompt string -> existing cached answers are reused
# untouched; only these models recompute the affected prompts.
_ANTI_REFUSAL_MODELS = ("claude-opus-4-8",)


def _is_anti_refusal_model(model):
    return any(m in str(model).lower() for m in _ANTI_REFUSAL_MODELS)


def _wrapper_key(base_key, model):
    """Return the anti-refusal variant of ``base_key`` for refusal-prone models,
    else the base key unchanged (so cached prompts for other models stay valid)."""
    return f"{base_key}_anti_refusal" if _is_anti_refusal_model(model) else base_key


def _pc_key(subkey, model):
    """Per-URL prompts_cache sub-key. For anti-refusal models the closed-book /
    list-MC prompts differ, so they live under a SEPARATE ``*_anti_refusal`` sub-key
    -- the shared base sub-key (and every other model's cached prompt+answer) is
    left untouched, and only these prompts are rebuilt+recomputed for Opus."""
    return f"{subkey}_anti_refusal" if _is_anti_refusal_model(model) else subkey

# K/N logging for List Retrieval — used to compute the LR random-baseline F1
LR_KN_LOG_PATH = f'{working_dir}/llm_generation/lr_kn_log.csv'
if not os.path.exists(LR_KN_LOG_PATH):
    with open(LR_KN_LOG_PATH, 'w') as _f:
        _f.write('url,bd,K,N,K_over_N\n')


def _log_lr_kn(url, bd, K, N):
    """Append one row per LR question's (K, N) for downstream chance analysis."""
    if N == 0:
        return
    with open(LR_KN_LOG_PATH, 'a') as _f:
        _f.write(f'{url},{bd},{K},{N},{K / N:.6f}\n')


# Run-of-runs random-baseline simulation for List Retrieval.
#
# We collect every LR question encountered during the pipeline as a
# (url, bd, answer_value, options) tuple in LR_QUESTIONS. At interpreter
# shutdown we simulate the LR task LR_CHANCE_NUM_RUNS times:
#
#   for each of LR_CHANCE_NUM_RUNS runs:
#       for each LR question (Q questions in total):
#           n ~ Uniform{0, 1, ..., |options|}
#           sampled = random sample of n options (no replacement)
#           f1_q = F1(answer_value, sampled)
#       run_avg = mean of the Q f1_q values
#   final estimate = mean of the LR_CHANCE_NUM_RUNS run_avg values
#
# No LLM is invoked. The result and the question count are written to
# lr_chance_summary.txt and printed at shutdown.
LR_CHANCE_NUM_RUNS = 1000
LR_CHANCE_SUMMARY_PATH = f'{working_dir}/llm_generation/lr_chance_summary.txt'
LR_QUESTIONS = []  # list of (url, bd, tuple(answer_value), tuple(options))


def _record_lr_question(url, bd, answer_value, options):
    """Collect one LR question for the end-of-run chance simulation."""
    if not options:
        return
    LR_QUESTIONS.append((url, bd, tuple(answer_value), tuple(options)))


def simulate_lr_chance_f1(n_runs=LR_CHANCE_NUM_RUNS, verbose=True):
    """Run the run-of-runs random-baseline simulation and return the mean.

    Returns a dict with keys: num_questions, num_runs, mean, std, run_means.
    """
    q = len(LR_QUESTIONS)
    if q == 0:
        if verbose:
            print('\n[LR chance simulation] no LR questions recorded; nothing to do')
        return None

    run_means = []
    for _ in range(n_runs):
        run_f1_sum = 0.0
        for (_url, _bd, answer_value, options) in LR_QUESTIONS:
            n = random.randint(0, len(options))            # inclusive
            sampled = random.sample(options, n)
            run_f1_sum += f1_score_sets(answer_value, sampled)[-1]
        run_means.append(run_f1_sum / q)

    grand_mean = sum(run_means) / n_runs
    variance = sum((m - grand_mean) ** 2 for m in run_means) / n_runs
    std = variance ** 0.5

    with open(LR_CHANCE_SUMMARY_PATH, 'w') as f:
        f.write(f'NUMBER_OF_QUESTIONS = {q}\n')
        f.write(f'NUMBER_OF_RUNS      = {n_runs}\n')
        f.write(f'MEAN_OF_RUN_MEANS   = {grand_mean:.6f}\n')
        f.write(f'STD_OF_RUN_MEANS    = {std:.6f}\n')
        f.write(f'MIN_RUN_MEAN        = {min(run_means):.6f}\n')
        f.write(f'MAX_RUN_MEAN        = {max(run_means):.6f}\n')

    if verbose:
        print('\n=== LR random-baseline simulation ===')
        print(f'  NUMBER_OF_QUESTIONS : {q}')
        print(f'  NUMBER_OF_RUNS      : {n_runs}')
        print(f'  mean of run-means   : {grand_mean:.4f}')
        print(f'  std  of run-means   : {std:.4f}')
        print(f'  range of run-means  : [{min(run_means):.4f}, {max(run_means):.4f}]')
        print(f'  summary written to  : {LR_CHANCE_SUMMARY_PATH}')

    return {
        'num_questions': q,
        'num_runs':      n_runs,
        'mean':          grand_mean,
        'std':           std,
        'run_means':     run_means,
    }


# Automatically run the simulation when the process exits, so the user does
# not need to modify list_retrieval.py.
import atexit
atexit.register(simulate_lr_chance_f1)

def f1_score_sets(set_a, set_b):
    """
    set_a: ground truth set
    set_b: predicted set
    Returns (precision, recall, f1)
    """
    a = set(set_a)
    b = set(set_b)

    tp = len(a & b)  # intersection
    fp = len(b - a)  # predicted but not in truth
    fn = len(a - b)  # in truth but not predicted

    precision = tp / (tp + fp) if (tp + fp) else 1.0  # if b empty: define as 1 if both empty else 0 handled below
    recall = tp / (tp + fn) if (tp + fn) else 1.0  # if a empty

    # More explicit empty-set handling:
    if len(a) == 0 and len(b) == 0:
        return 1.0, 1.0, 1.0
    if len(b) == 0 and len(a) != 0:
        return 0.0, 0.0, 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1

def prompt_generation(
        table_desc, col, key, key_value, real_value, df, parser_ins, model, instruction,url, rephrase=False
):
    def get_choices(_df, _answer_value, _desired_column):
        if f'{url}_{_desired_column}_{_answer_value}_single' in choices_cache:
            cached_choices = choices_cache[f'{url}_{_desired_column}_{_answer_value}_single']
            return {
                "CHOICE_1": str(cached_choices[0]),
                "CHOICE_2": str(cached_choices[1]),
                "CHOICE_3": str(cached_choices[2]),
            }
        unique_choices = set(_df[_desired_column].dropna().tolist()).difference({_answer_value})
        if any([isna(k) for k in unique_choices]):
            b=5
        sorted_choices = sorted(list(unique_choices))
        random_two_choices = random.sample(sorted_choices, 2) + [_answer_value]

        random.shuffle(random_two_choices)
        choices_cache[f'{url}_{_desired_column}_{_answer_value}_single'] = random_two_choices
        return {
            "CHOICE_1": str(random_two_choices[0]),
            "CHOICE_2": str(random_two_choices[1]),
            "CHOICE_3": str(random_two_choices[2]),
        }

    from utils.utils import get_llm_response

    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    with open(f"{working_dir}/llm_generation/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    with open(f"{working_dir}/config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    based_prompt = (
        prompts["single_value"]
        .format(DESIRED_COLUMN=col, KEY_COLUMN=key, KEY_VALUE=key_value)
        .strip()
    )
    prompt_cache = prompts_cache.get(url, {})
    # Closed book

    closed_book_prompt = (
        prompts[_wrapper_key("closed_book_wrapper", model)]
        .format(
            TABLE_TITLE=table_desc,
            QUESTION=based_prompt,
            INSTRUCTION=prompts[instruction],
            PRIMARY_KEY_COLUMN=key,
        )
        .strip()
    )

    if _pc_key('single_value_closed_book', model) in prompt_cache:
        closed_book_prompt = prompt_cache[_pc_key('single_value_closed_book', model)]['q']
        real_value = prompt_cache[_pc_key('single_value_closed_book', model)]['a']
    else:
        print("Caching prompt")
        prompt_cache[_pc_key('single_value_closed_book', model)] = {'q': closed_book_prompt, 'a': real_value}

    if not rephrase:
        closed_book_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=closed_book_prompt,
                model=model,
            )
        )
    else:
        closed_book_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=closed_book_prompt),
                model=conf["rephrase_model"],
            )
        )
        prompts_cache[url] = prompt_cache
        return {
            'rephrased': str(closed_book_response),
            'answer': str(real_value),
            'options': get_choices(_df=df, _answer_value=real_value, _desired_column=col,)
        }


    # Multiple choices
    multiple_choices_single_value_prompt = (
        prompts["multiple_choice_wrapper"]
        .format(
            TABLE_TITLE=table_desc,
            QUESTION=based_prompt,
            INSTRUCTION=prompts[instruction],
            PRIMARY_KEY_COLUMN=key,
            **get_choices(_df=df, _answer_value=real_value, _desired_column=col,),
        )
        .strip()
    )

    if 'single_value_multiple_choices' in prompt_cache:
        multiple_choices_single_value_prompt = prompt_cache['single_value_multiple_choices']['q']
        real_value = prompt_cache['single_value_multiple_choices']['a']
    else:
        print("Caching prompt")
        prompt_cache['single_value_multiple_choices'] = {'q': multiple_choices_single_value_prompt, 'a': real_value}

    multiple_choices_response = parser_ins.try_cast(
        get_llm_response(
            prompt_string=multiple_choices_single_value_prompt,
            model=model,
            answer=real_value,
        )
    )

    # Open book
    open_book_prompt = prompts["open_book_wrapper"].format(
        TABLE_TITLE=table_desc,
        QUESTION=based_prompt,
        INSTRUCTION=prompts[instruction],
        FULL_CSV_TABLE=df.to_csv(index=False),
    )

    if 'single_value_open_book' in prompt_cache:
        open_book_prompt = prompt_cache['single_value_open_book']['q']
        real_value = prompt_cache['single_value_open_book']['a']
    else:
        print("Caching prompt")
        prompt_cache['single_value_open_book'] = {'q': open_book_prompt, 'a': real_value}

    open_book_response = parser_ins.try_cast(
        get_llm_response(
            prompt_string=open_book_prompt,
            model=model,
        )
    )
    if not bool(str(open_book_response) == str(real_value) or open_book_response == real_value):
        b=5
    prompts_cache[url] = prompt_cache
    return {
        "closed_book": bool(str(closed_book_response) == str(real_value) or closed_book_response == real_value),
        "multiple_choices": bool( str(multiple_choices_response) == str(real_value) or multiple_choices_response == real_value),
        "open_book": bool(str(open_book_response) == str(real_value) or open_book_response == real_value),
    }


def prompt_generation_list_categorical(
        table_desc,
        col,
        col_value_1,
        col_value_2,
        real_value_1,
        real_value_2,
        df,
        parser_ins,
        model,
        instruction,
        pk_column,
        url,
        rephrase=False,
):
    def get_choices(_answer_value, _df, _pk_column):
        cache_key = f'{url}_{_answer_value}_list_categorical'
        if cache_key in choices_cache:
            cached_choices = choices_cache[cache_key]
            _log_lr_kn(url=url, bd=bd,
                       K=len(_answer_value), N=len(cached_choices))
            _record_lr_question(url=url, bd=bd,
                                answer_value=_answer_value,
                                options=cached_choices)
            return {
                "CHOICES_LIST": cached_choices,
            }
        choices = random.choices([k for k in _df[_pk_column].tolist() if k not in _answer_value],
                                 k=len(_answer_value)) + _answer_value
        random.shuffle(choices)
        choices_cache[cache_key] = choices
        _log_lr_kn(url=url, bd=bd, K=len(_answer_value), N=len(choices))
        _record_lr_question(url=url, bd=bd,
                            answer_value=_answer_value,
                            options=choices)
        return {
            "CHOICES_LIST": choices,
        }



    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    with open(f"{working_dir}/llm_generation/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    with open(f"{working_dir}/config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    based_prompt = prompts["list_value"].format(KEY_COLUMN=pk_column).strip()
    response_dict = {}
    prompt_cache = prompts_cache.get(url, {})
    for bd in ["categorical_equals",
               # "categorical_not_equals",
               "categorical_in"]:
        response_dict[bd] = {}
        _based_prompt = (
                based_prompt
                + " "
                + prompts[bd]
                .format(
            COLUMN=col,
            DESIRED_COLUMN=col,
            DESIRED_VALUE=col_value_1,
            CHOICE_1=col_value_1,
            CHOICE_2=col_value_2,
        )
                .strip()
        )

        if bd == "categorical_equals":
            answer_value = real_value_1
        elif bd == "categorical_not_equals":
            answer_value = df[df[col] != col_value_1][pk_column].tolist()
        elif bd == "categorical_in":
            answer_value = df[df[col].isin([col_value_1, col_value_2])][pk_column].tolist()
        # Closed book
        closed_book_prompt = (
            prompts[_wrapper_key("closed_book_wrapper", model)]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if _pc_key(f'list_categorical_{bd}_closed_book', model) in prompt_cache:
            closed_book_prompt = prompt_cache[_pc_key(f'list_categorical_{bd}_closed_book', model)]['q']
            answer_value = prompt_cache[_pc_key(f'list_categorical_{bd}_closed_book', model)]['a']
        else:
            print("Caching prompt")
            prompt_cache[_pc_key(f'list_categorical_{bd}_closed_book', model)] = {'q': closed_book_prompt, 'a': answer_value}

        if not rephrase:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=closed_book_prompt,
                    model=model,
                )
            )
            try:
                closed_book_response = [parser_ins.try_cast(k) for k in closed_book_response.split(',')]
            except:
                closed_book_response = []
        else:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=closed_book_prompt),
                    model=conf["rephrase_model"],
                )
            )
            prompts_cache[url] = prompt_cache
            return {
                'rephrased': str(closed_book_response),
                'answer': answer_value,
                'options': get_choices(_answer_value=answer_value, _df=df, _pk_column=pk_column)
            }

        # Multiple choices
        multiple_choices_single_value_prompt = (
            prompts[_wrapper_key("list_retrieval_multiple_choices_wrapper", model)]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                **get_choices(_answer_value=answer_value, _df=df, _pk_column=pk_column),
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if _pc_key(f'list_categorical_{bd}_multiple_choices', model) in prompt_cache:
            multiple_choices_single_value_prompt = prompt_cache[_pc_key(f'list_categorical_{bd}_multiple_choices', model)]['q']
            answer_value = prompt_cache[_pc_key(f'list_categorical_{bd}_multiple_choices', model)]['a']
        else:
            print("Caching prompt")
            prompt_cache[_pc_key(f'list_categorical_{bd}_multiple_choices', model)] = {'q': multiple_choices_single_value_prompt, 'a': answer_value}

        multiple_choices_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=multiple_choices_single_value_prompt,
                model=model,
                answer=answer_value,
            )
        )
        try:
            multiple_choices_response = [parser_ins.try_cast(k) for k in multiple_choices_response.split(',')]
        except AttributeError:
            multiple_choices_response = []

        # Open book
        open_book_prompt = prompts["open_book_wrapper"].format(
            TABLE_TITLE=table_desc,
            QUESTION=_based_prompt,
            INSTRUCTION=prompts[instruction],
            FULL_CSV_TABLE=df.to_csv(index=False),
        )

        if f'list_categorical_{bd}_open_book' in prompt_cache:
            open_book_prompt = prompt_cache[f'list_categorical_{bd}_open_book']['q']
            answer_value = prompt_cache[f'list_categorical_{bd}_open_book']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'list_categorical_{bd}_open_book'] = {'q': open_book_prompt, 'a': answer_value}

        open_book_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=open_book_prompt,
                model=model,
            )
        )
        try:
            open_book_response = [parser_ins.try_cast(k) for k in open_book_response.split(',')]
        except AttributeError:
            open_book_response = []
        response_dict[bd] = {
            "closed_book": f1_score_sets(answer_value, closed_book_response)[-1],
            "multiple_choices": f1_score_sets(answer_value, multiple_choices_response)[-1],
            "open_book": f1_score_sets(answer_value, open_book_response)[-1],
        }

    prompts_cache[url] = prompt_cache
    return response_dict


def prompt_generation_count_categorical(
        table_desc,
        col,
        col_value_1,
        col_value_2,
        real_value_1,
        real_value_2,
        df,
        parser_ins,
        model,
        instruction,
        pk_column,
        url,
        rephrase=False,
):
    def get_choices(_answer_value):
        if f'{url}_{_answer_value}_count_categorical' in choices_cache:
            cached_choices = choices_cache[f'{url}_{_answer_value}_count_categorical']
            return {
                "CHOICE_1": str(cached_choices[0]),
                "CHOICE_2": str(cached_choices[1]),
                "CHOICE_3": str(cached_choices[2]),
            }
        choices = [int(_answer_value / 1.5), _answer_value, int(_answer_value * 1.5)]
        random.shuffle(choices)
        choices_cache[f'{url}_{_answer_value}_count_categorical'] = choices
        return {
            "CHOICE_1": str(choices[0]),
            "CHOICE_2": str(choices[1]),
            "CHOICE_3": str(choices[2]),
        }

    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    with open(f"{working_dir}/llm_generation/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    with open(f"{working_dir}/config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    based_prompt = prompts["count_rows"].strip()
    response_dict = {}
    prompt_cache = prompts_cache.get(url,{})
    for bd in ["categorical_equals", "categorical_not_equals", "categorical_in"]:
        response_dict[bd] = {}
        _based_prompt = (
                based_prompt
                + " "
                + prompts[bd]
                .format(
            COLUMN=col,
            DESIRED_COLUMN=col,
            DESIRED_VALUE=col_value_1,
            CHOICE_1=col_value_1,
            CHOICE_2=col_value_2,
        )
                .strip()
        )

        if bd == "categorical_equals":
            answer_value = real_value_1
        elif bd == "categorical_not_equals":
            answer_value = len(df[df[col] != col_value_1])
        elif bd == "categorical_in":
            answer_value = real_value_1 + real_value_2
        # Closed book
        closed_book_prompt = (
            prompts[_wrapper_key("closed_book_wrapper", model)]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if _pc_key(f'count_categorical_{bd}_closed_book', model) in prompt_cache:
            closed_book_prompt = prompt_cache[_pc_key(f'count_categorical_{bd}_closed_book', model)]['q']
            answer_value = prompt_cache[_pc_key(f'count_categorical_{bd}_closed_book', model)]['a']
        else:
            print("Caching prompt")
            prompt_cache[_pc_key(f'count_categorical_{bd}_closed_book', model)] = {'q': closed_book_prompt, 'a': answer_value}


        if not rephrase:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=closed_book_prompt,
                    model=model,
                )
            )
        if rephrase:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=closed_book_prompt),
                    model=conf["rephrase_model"],
                )
            )
            response_dict[bd] = {
                'rephrased': closed_book_response,
                'answer': answer_value,
                'options': get_choices(_answer_value=answer_value)
            }
            continue

        # Multiple choices
        multiple_choices_single_value_prompt = (
            prompts["multiple_choice_wrapper"]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                **get_choices(_answer_value=answer_value),
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if f'count_categorical_{bd}_multiple_choices' in prompt_cache:
            multiple_choices_single_value_prompt = prompt_cache[f'count_categorical_{bd}_multiple_choices']['q']
            answer_value = prompt_cache[f'count_categorical_{bd}_multiple_choices']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'count_categorical_{bd}_multiple_choices'] = {'q': multiple_choices_single_value_prompt, 'a': answer_value}

        if not rephrase:
            multiple_choices_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=multiple_choices_single_value_prompt,
                    model=model,
                    answer=answer_value,
                )
            )
        if rephrase:
            multiple_choices_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=multiple_choices_single_value_prompt),
                    model=conf["rephrase_model"],
                )
            )
        # Open book
        open_book_prompt = prompts["open_book_wrapper"].format(
            TABLE_TITLE=table_desc,
            QUESTION=_based_prompt,
            INSTRUCTION=prompts[instruction],
            FULL_CSV_TABLE=df.to_csv(index=False),
        )

        if f'count_categorical_{bd}_open_book' in prompt_cache:
            open_book_prompt = prompt_cache[f'count_categorical_{bd}_open_book']['q']
            answer_value = prompt_cache[f'count_categorical_{bd}_open_book']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'count_categorical_{bd}_open_book'] = {'q': open_book_prompt, 'a': answer_value}

        if not rephrase:
            open_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=open_book_prompt,
                    model=model,
                )
            )


        if not rephrase:
            response_dict[bd] = {
                "closed_book": str(closed_book_response) == str(answer_value) or closed_book_response == answer_value,
                "multiple_choices": str(multiple_choices_response) == str(answer_value) or multiple_choices_response == answer_value,
                "open_book": str(open_book_response) == str(answer_value) or open_book_response == answer_value,
            }
        else:
            response_dict[bd] = {
                "closed_book": closed_book_response,
                "multiple_choices": multiple_choices_response,
                "open_book": open_book_prompt,
                "answer": answer_value,
                'raw':
                    {
                        'closed_book': closed_book_prompt,
                        'multiple_choices': multiple_choices_single_value_prompt,
                        'open_book': open_book_prompt,
                    }
            }
    prompts_cache[url] = prompt_cache
    return response_dict


def prompt_generation_count_numerical(
        table_desc,
        col,
        threshold,
        lower_threshold,
        upper_threshold,
        df,
        parser_ins,
        model,
        instruction,
        pk_column,
        url,
        rephrase=False
):
    def get_choices(_answer_value):
        if f'{url}_{_answer_value}_count_numerical' in choices_cache:
            cached_choices = choices_cache[f'{url}_{_answer_value}_count_numerical']
            return {
                "CHOICE_1": cached_choices[0],
                "CHOICE_2": cached_choices[1],
                "CHOICE_3": cached_choices[2],
            }
        choices = [int(_answer_value / 1.5), _answer_value, int(_answer_value * 1.5)]
        random.shuffle(choices)
        choices_cache[f'{url}_{_answer_value}_count_numerical'] = choices
        return {
            "CHOICE_1": choices[0],
            "CHOICE_2": choices[1],
            "CHOICE_3": choices[2],
        }

    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    with open(f"{working_dir}/llm_generation/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    with open(f"{working_dir}/config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    based_prompt = prompts["count_rows"].strip()
    response_dict = {}
    prompt_cache = prompts_cache.get(url, {})
    equals_value, equals_answer = Counter(df[col].dropna()).most_common()[0]
    for bd in ["numerical_greater_than", "numerical_less_than", "numerical_between", "numerical_equals"]:
        response_dict[bd] = {}
        _based_prompt = (
                based_prompt
                + " "
                + prompts[bd]
                .format(
            COLUMN=col,
            COMPARISON_COLUMN=col,
            THRESHOLD=threshold,
            LOWER_THRESHOLD=lower_threshold,
            UPPER_THRESHOLD=upper_threshold,
            VALUE=equals_value,
        )
                .strip()
        )

        if bd == "numerical_greater_than":
            answer_value = len(df[df[col] >= threshold])
        elif bd == "numerical_less_than":
            answer_value = len(df[df[col] <= threshold])
        elif bd == "numerical_between":
            answer_value = int(
                df[col]
                .between(lower_threshold, upper_threshold, inclusive="both")
                .sum()
            )
        elif bd == "numerical_equals":
            answer_value = equals_answer
        # Closed book
        closed_book_prompt = (
            prompts[_wrapper_key("closed_book_wrapper", model)]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if _pc_key(f'count_numerical_{bd}_closed_book', model) in prompt_cache:
            closed_book_prompt = prompt_cache[_pc_key(f'count_numerical_{bd}_closed_book', model)]['q']
            answer_value = prompt_cache[_pc_key(f'count_numerical_{bd}_closed_book', model)]['a']
        else:
            print("Caching prompt")
            prompt_cache[_pc_key(f'count_numerical_{bd}_closed_book', model)] = {'q': closed_book_prompt, 'a': answer_value}

        if not rephrase:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=closed_book_prompt,
                    model=model,
                )
            )
        else:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=closed_book_prompt),
                    model=conf["rephrase_model"],
                )
            )
            response_dict[bd] = {
                'rephrased' :closed_book_response,
                'answer': answer_value,
                'options': get_choices(_answer_value=answer_value)
            }
            continue

        # Multiple choices
        multiple_choices_single_value_prompt = (
            prompts["multiple_choice_wrapper"]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                **get_choices(_answer_value=answer_value),
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if f'count_numerical_{bd}_multiple_choices' in prompt_cache:
            multiple_choices_single_value_prompt = prompt_cache[f'count_numerical_{bd}_multiple_choices']['q']
            answer_value = prompt_cache[f'count_numerical_{bd}_multiple_choices']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'count_numerical_{bd}_multiple_choices'] = {'q': multiple_choices_single_value_prompt, 'a': answer_value}

        if not rephrase:
            multiple_choices_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=multiple_choices_single_value_prompt,
                    model=model,
                    answer=answer_value,
                )
            )
        else:
            multiple_choices_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=multiple_choices_single_value_prompt),
                    model=conf["rephrase_model"],
                )
            )

        # Open book
        open_book_prompt = prompts["open_book_wrapper"].format(
            TABLE_TITLE=table_desc,
            QUESTION=_based_prompt,
            INSTRUCTION=prompts[instruction],
            FULL_CSV_TABLE=df.to_csv(index=True),
        )

        if f'count_numerical_{bd}_open_book' in prompt_cache:
            open_book_prompt = prompt_cache[f'count_numerical_{bd}_open_book']['q']
            answer_value = prompt_cache[f'count_numerical_{bd}_open_book']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'count_numerical_{bd}_open_book'] = {'q': open_book_prompt, 'a': answer_value}

        if not rephrase:
            open_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=open_book_prompt,
                    model=model,
                )
            )

        if not rephrase:
            response_dict[bd] = {
                "closed_book": str(closed_book_response) == str(answer_value) or closed_book_response == answer_value,
                "multiple_choices": str(multiple_choices_response) == str(answer_value) or multiple_choices_response == answer_value,
                "open_book": str(open_book_response) == str(answer_value) or open_book_response == answer_value,
            }
        else:
            response_dict[bd] = {
                "closed_book": closed_book_response,
                "multiple_choices": multiple_choices_response,
                "open_book": open_book_prompt,
                "answer": answer_value
            }
    prompts_cache[url] = prompt_cache
    return response_dict


def prompt_generation_max_categorical(
        table_desc,
        comparable_col,
        categorical_col,
        cat_col_value_1,
        cat_col_value_2,
        key_column,
        real_value_1,
        real_value_2,
        df,
        parser_ins,
        model,
        instruction,
        pk_column,
        url,
        rephrase=False
):
    def get_choices(_df, _answer_value, _desired_column):
        if f'{url}_{_desired_column}_{_answer_value}_max_categorical' in choices_cache:
            cached_choices = choices_cache[f'{url}_{_desired_column}_{_answer_value}_max_categorical']
            return {
                "CHOICE_1": cached_choices[0],
                "CHOICE_2": cached_choices[1],
                "CHOICE_3": cached_choices[2],
            }
        unique_choices = set(_df[_desired_column].dropna().tolist()).difference(
            set(_answer_value)
        )
        sorted_choices = sorted(list(unique_choices))

        random_two_choices = random.sample(sorted_choices, 2) + [_answer_value[0]]
        random.shuffle(random_two_choices)
        choices_cache[f'{url}_{_desired_column}_{_answer_value}_max_categorical'] = random_two_choices

        return {
            "CHOICE_1": random_two_choices[0],
            "CHOICE_2": random_two_choices[1],
            "CHOICE_3": random_two_choices[2],
        }

    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    with open(f"{working_dir}/llm_generation/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    with open(f"{working_dir}/config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    based_prompt = (
        prompts["max_value"]
        .format(
            PRIMARY_COLUMN=key_column,
            COMPARISON_COLUMN=comparable_col,
        )
        .strip()
    )
    response_dict = {}
    prompt_cache = prompts_cache.get(url, {})
    for bd in ["categorical_equals", "categorical_not_equals", "categorical_in"]:
        response_dict[bd] = {}
        _based_prompt = (
                based_prompt
                + " "
                + prompts[bd]
                .format(
            DESIRED_COLUMN=categorical_col,
            DESIRED_VALUE=cat_col_value_1,
            CHOICE_1=cat_col_value_1,
            CHOICE_2=cat_col_value_2,
        )
                .strip()
        )

        if bd == "categorical_equals":
            answer_value = real_value_1
        elif bd == "categorical_not_equals":
            max_val = df[df[categorical_col] != cat_col_value_1][comparable_col].max()
            answer_value = df[df[comparable_col] == max_val][key_column].tolist()
        elif bd == "categorical_in":
            max_val = df[df[categorical_col].isin([cat_col_value_1, cat_col_value_2])][
                comparable_col
            ].max()
            answer_value = df[df[comparable_col] == max_val][key_column].tolist()
        if answer_value == []:
            continue
        # Closed book
        closed_book_prompt = (
            prompts[_wrapper_key("closed_book_wrapper", model)]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if _pc_key(f'max_categorical_{bd}_closed_book', model) in prompt_cache:
            closed_book_prompt = prompt_cache[_pc_key(f'max_categorical_{bd}_closed_book', model)]['q']
            answer_value = prompt_cache[_pc_key(f'max_categorical_{bd}_closed_book', model)]['a']
        else:
            print("Caching prompt")
            prompt_cache[_pc_key(f'max_categorical_{bd}_closed_book', model)] = {'q': closed_book_prompt, 'a': answer_value}

        if not rephrase:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=closed_book_prompt,
                    model=model,
                )
            )
        else:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=closed_book_prompt),
                    model=conf["rephrase_model"],
                )
            )
            response_dict[bd] = {
                'rephrased' :closed_book_response,
                'answer': answer_value,
                'options': get_choices(_answer_value=answer_value, _df=df, _desired_column=key_column)
            }
            continue

        # Multiple choices
        multiple_choices_single_value_prompt = (
            prompts["multiple_choice_wrapper"]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                **get_choices(
                    _answer_value=answer_value, _desired_column=key_column, _df=df
                ),
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if f'max_categorical_{bd}_multiple_choices' in prompt_cache:
            multiple_choices_single_value_prompt = prompt_cache[f'max_categorical_{bd}_multiple_choices']['q']
            answer_value = prompt_cache[f'max_categorical_{bd}_multiple_choices']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'max_categorical_{bd}_multiple_choices'] = {'q': multiple_choices_single_value_prompt, 'a': answer_value}

        if not rephrase:
            multiple_choices_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=multiple_choices_single_value_prompt,
                    model=model,
                    answer=answer_value,
                )
            )
        else:
            multiple_choices_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=multiple_choices_single_value_prompt),
                    model=conf["rephrase_model"],
                )
            )

        # Open book
        open_book_prompt = prompts["open_book_wrapper"].format(
            TABLE_TITLE=table_desc,
            QUESTION=_based_prompt,
            INSTRUCTION=prompts[instruction],
            FULL_CSV_TABLE=df.to_csv(index=False),
        )

        if f'max_categorical_{bd}_open_book' in prompt_cache:
            open_book_prompt = prompt_cache[f'max_categorical_{bd}_open_book']['q']
            answer_value = prompt_cache[f'max_categorical_{bd}_open_book']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'max_categorical_{bd}_open_book'] = {'q': open_book_prompt, 'a': answer_value}

        open_book_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=open_book_prompt,
                model=model,
            )
        )
        if not rephrase:
            response_dict[bd] = {
                "closed_book": closed_book_response in answer_value,
                "multiple_choices": multiple_choices_response in answer_value,
                "open_book": open_book_response in answer_value,
            }
        else:
            response_dict[bd] = {
                "closed_book": closed_book_response,
                "multiple_choices": multiple_choices_response,
                "open_book": open_book_prompt,
                "answer": answer_value
            }
    prompts_cache[url] = prompt_cache
    return response_dict


def prompt_generation_max_numerical(
        table_desc,
        threshold,
        lower_threshold,
        upper_threshold,
        df,
        parser_ins,
        model,
        instruction,
        pk_column,
        comparable_col_1,
        comparable_col_2,
        url,
        rephrase=False
):
    def get_choices(_df, _answer_value, _desired_column):
        if f'{url}_{_desired_column}_{_answer_value}_max_numerical' in choices_cache:
            cached_choices = choices_cache[f'{url}_{_desired_column}_{_answer_value}_max_numerical']
            return {
                "CHOICE_1": cached_choices[0],
                "CHOICE_2": cached_choices[1],
                "CHOICE_3": cached_choices[2],
            }
        unique_choices = set(_df[_desired_column].dropna().tolist()).difference(
            set(_answer_value)
        )
        sorted_choices = sorted(list(unique_choices))
        random_two_choices = random.sample(sorted_choices, 2) + [_answer_value[0]]
        random.shuffle(random_two_choices)

        choices_cache[f'{url}_{_desired_column}_{_answer_value}_max_numerical'] = random_two_choices
        return {
            "CHOICE_1": random_two_choices[0],
            "CHOICE_2": random_two_choices[1],
            "CHOICE_3": random_two_choices[2],
        }

    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    with open(f"{working_dir}/llm_generation/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    with open(f"{working_dir}/config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    based_prompt = prompts["max_value"].format(
        COMPARISON_COLUMN=comparable_col_1,
        PRIMARY_COLUMN=pk_column,
    ).strip()
    response_dict = {}
    prompt_cache = prompts_cache.get(url, {})
    equals_value, equals_answer = Counter(df[comparable_col_2].dropna()).most_common()[0]
    for bd in ["numerical_greater_than", "numerical_less_than", "numerical_between", "numerical_equals"]:
        response_dict[bd] = {}
        _based_prompt = (
                based_prompt
                + " "
                + prompts[bd]
                .format(
            COMPARISON_COLUMN=comparable_col_2,
            THRESHOLD=threshold,
            LOWER_THRESHOLD=lower_threshold,
            UPPER_THRESHOLD=upper_threshold,
            VALUE=equals_value,

        )
                .strip()
        )

        if bd == "numerical_greater_than":
            _df = df[df[comparable_col_2] >= threshold]
            max_val = _df[comparable_col_1].max()
            answer_value = _df[_df[comparable_col_1] == max_val][pk_column].tolist()
        elif bd == "numerical_less_than":
            _df = df[df[comparable_col_2] <= threshold]
            max_val = _df[comparable_col_1].max()
            answer_value = _df[_df[comparable_col_1] == max_val][pk_column].tolist()
        elif bd == "numerical_between":
            _df = df[df[comparable_col_2].between(lower_threshold, upper_threshold, inclusive="both")]
            max_val = _df[comparable_col_1].max()
            answer_value = _df[_df[comparable_col_1] == max_val][pk_column].tolist()
        elif bd == "numerical_equals":
            _df = df[df[comparable_col_2] == equals_value]
            max_val = _df[comparable_col_1].max()
            answer_value = _df[_df[comparable_col_1] == max_val][pk_column].tolist()
        if answer_value == []:
            continue
        # Closed book
        closed_book_prompt = (
            prompts[_wrapper_key("closed_book_wrapper", model)]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if _pc_key(f'max_numerical_{bd}_closed_book', model) in prompt_cache:
            closed_book_prompt = prompt_cache[_pc_key(f'max_numerical_{bd}_closed_book', model)]['q']
            answer_value = prompt_cache[_pc_key(f'max_numerical_{bd}_closed_book', model)]['a']
        else:
            print("Caching prompt")
            prompt_cache[_pc_key(f'max_numerical_{bd}_closed_book', model)] = {'q': closed_book_prompt, 'a': answer_value}

        if not rephrase:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=closed_book_prompt,
                    model=model,
                )
            )
        else:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=closed_book_prompt),
                    model=conf["rephrase_model"],
                )
            )
            response_dict[bd] = {
                'rephrased': closed_book_response,
                'answer': answer_value,
                'options': get_choices(_df=df, _answer_value=answer_value, _desired_column=pk_column)
            }
            continue

        # Multiple choices
        multiple_choices_single_value_prompt = (
            prompts["multiple_choice_wrapper"]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
                **get_choices(_answer_value=answer_value, _df=df, _desired_column=pk_column),
            )
            .strip()
        )

        if f'max_numerical_{bd}_multiple_choices' in prompt_cache:
            multiple_choices_single_value_prompt = prompt_cache[f'max_numerical_{bd}_multiple_choices']['q']
            answer_value = prompt_cache[f'max_numerical_{bd}_multiple_choices']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'max_numerical_{bd}_multiple_choices'] = {'q': multiple_choices_single_value_prompt, 'a': answer_value}

        multiple_choices_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=multiple_choices_single_value_prompt,
                model=model,
                answer=answer_value,
            )
        )

        # Open book
        open_book_prompt = prompts["open_book_wrapper"].format(
            TABLE_TITLE=table_desc,
            QUESTION=_based_prompt,
            INSTRUCTION=prompts[instruction],
            FULL_CSV_TABLE=df.to_csv(index=True),
        )

        if f'max_numerical_{bd}_open_book' in prompt_cache:
            open_book_prompt = prompt_cache[f'max_numerical_{bd}_open_book']['q']
            answer_value = prompt_cache[f'max_numerical_{bd}_open_book']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'max_numerical_{bd}_open_book'] = {'q': open_book_prompt, 'a': answer_value}

        open_book_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=open_book_prompt,
                model=model,
            )
        )
        response_dict[bd] = {
            "closed_book": closed_book_response in answer_value,
            "multiple_choices": multiple_choices_response in answer_value,
            "open_book": open_book_response in answer_value,
        }

    prompts_cache[url] = prompt_cache
    return response_dict





def prompt_generation_min_categorical(
        table_desc,
        comparable_col,
        categorical_col,
        cat_col_value_1,
        cat_col_value_2,
        key_column,
        real_value_1,
        real_value_2,
        df,
        parser_ins,
        model,
        instruction,
        pk_column,
        url,
rephrase=False
):
    def get_choices(_df, _answer_value, _desired_column, url):
        if f'{url}_{_desired_column}_{_answer_value}_min_categorical' in choices_cache:
            cached_choices = choices_cache[f'{url}_{_desired_column}_{_answer_value}_min_categorical']
            return {
                "CHOICE_1": cached_choices[0],
                "CHOICE_2": cached_choices[1],
                "CHOICE_3": cached_choices[2],
            }
        unique_choices = set(_df[_desired_column].dropna().tolist()).difference(
            set(_answer_value)
        )
        sorted_choices = sorted(list(unique_choices))

        random_two_choices = random.sample(sorted_choices, 2) + [_answer_value[0]]
        random.shuffle(random_two_choices)

        choices_cache[f'{url}_{_desired_column}_{_answer_value}_min_categorical'] = random_two_choices

        return {
            "CHOICE_1": random_two_choices[0],
            "CHOICE_2": random_two_choices[1],
            "CHOICE_3": random_two_choices[2],
        }

    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    with open(f"{working_dir}/llm_generation/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    with open(f"{working_dir}/config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    based_prompt = (
        prompts["min_value"]
        .format(
            PRIMARY_COLUMN=key_column,
            COMPARISON_COLUMN=comparable_col,
        )
        .strip()
    )
    response_dict = {}
    prompt_cache = prompts_cache.get(url, {})
    for bd in ["categorical_equals", "categorical_not_equals", "categorical_in"]:
        response_dict[bd] = {}
        _based_prompt = (
                based_prompt
                + " "
                + prompts[bd]
                .format(
            DESIRED_COLUMN=categorical_col,
            DESIRED_VALUE=cat_col_value_1,
            CHOICE_1=cat_col_value_1,
            CHOICE_2=cat_col_value_2,
        )
                .strip()
        )

        if bd == "categorical_equals":
            answer_value = real_value_1
        elif bd == "categorical_not_equals":
            min_val = df[df[categorical_col] != cat_col_value_1][comparable_col].min()
            answer_value = df[df[comparable_col] == min_val][key_column].tolist()
        elif bd == "categorical_in":
            min_val = df[df[categorical_col].isin([cat_col_value_1, cat_col_value_2])][
                comparable_col
            ].min()
            answer_value = df[df[comparable_col] == min_val][key_column].tolist()
        if answer_value == []:
            b = 5
        # Closed book
        closed_book_prompt = (
            prompts[_wrapper_key("closed_book_wrapper", model)]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if _pc_key(f'min_categorical_{bd}_closed_book', model) in prompt_cache:
            closed_book_prompt = prompt_cache[_pc_key(f'min_categorical_{bd}_closed_book', model)]['q']
            answer_value = prompt_cache[_pc_key(f'min_categorical_{bd}_closed_book', model)]['a']
        else:
            print("Caching prompt")
            prompt_cache[_pc_key(f'min_categorical_{bd}_closed_book', model)] = {'q': closed_book_prompt, 'a': answer_value}

        if not rephrase:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=closed_book_prompt,
                    model=model,
                )
            )
        else:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=closed_book_prompt),
                    model=conf["rephrase_model"],
                )
            )
            response_dict[bd] = {
                'rephrased': closed_book_response,
                'answer': answer_value,
                'options': get_choices(_df=df, _answer_value=answer_value, _desired_column=key_column, url=url)
            }
            continue


        # Multiple choices
        multiple_choices_single_value_prompt = (
            prompts["multiple_choice_wrapper"]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                **get_choices(
                    _answer_value=answer_value, _desired_column=key_column, _df=df, url=url
                ),
                PRIMARY_KEY_COLUMN=pk_column,

            )
            .strip()
        )

        if f'min_categorical_{bd}_multiple_choices' in prompt_cache:
            multiple_choices_single_value_prompt = prompt_cache[f'min_categorical_{bd}_multiple_choices']['q']
            answer_value = prompt_cache[f'min_categorical_{bd}_multiple_choices']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'min_categorical_{bd}_multiple_choices'] = {'q': multiple_choices_single_value_prompt, 'a': answer_value}

        multiple_choices_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=multiple_choices_single_value_prompt,
                model=model,
                answer=answer_value,
            )
        )

        # Open book
        open_book_prompt = prompts["open_book_wrapper"].format(
            TABLE_TITLE=table_desc,
            QUESTION=_based_prompt,
            INSTRUCTION=prompts[instruction],
            FULL_CSV_TABLE=df.to_csv(index=False),
        )

        if f'min_categorical_{bd}_open_book' in prompt_cache:
            open_book_prompt = prompt_cache[f'min_categorical_{bd}_open_book']['q']
            answer_value = prompt_cache[f'min_categorical_{bd}_open_book']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'min_categorical_{bd}_open_book'] = {'q': open_book_prompt, 'a': answer_value}

        open_book_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=open_book_prompt,
                model=model,
            )
        )
        response_dict[bd] = {
            "closed_book": closed_book_response in answer_value,
            "multiple_choices": multiple_choices_response in answer_value,
            "open_book": open_book_response in answer_value,
        }
        if not open_book_response in answer_value:
            b=5

    prompts_cache[url] = prompt_cache
    return response_dict


def prompt_generation_min_numerical(
        table_desc,
        threshold,
        lower_threshold,
        upper_threshold,
        df,
        parser_ins,
        model,
        instruction,
        pk_column,
        comparable_col_1,
        comparable_col_2,
        url,
        rephrase=False
):
    def get_choices(_df, _answer_value, _desired_column):
        if f'{url}_{_desired_column}_{_answer_value}_min_numerical' in choices_cache:
            cached_choices = choices_cache[f'{url}_{_desired_column}_{_answer_value}_min_numerical']
            return {
                "CHOICE_1": cached_choices[0],
                "CHOICE_2": cached_choices[1],
                "CHOICE_3": cached_choices[2],
            }
        unique_choices = set(_df[_desired_column].dropna().tolist()).difference(
            set(_answer_value)
        )
        sorted_choices = sorted(list(unique_choices))
        random_two_choices = random.sample(sorted_choices, 2) + [_answer_value[0]]

        random.shuffle(random_two_choices)
        choices_cache[f'{url}_{_desired_column}_{_answer_value}_min_numerical'] = random_two_choices

        return {
            "CHOICE_1": random_two_choices[0],
            "CHOICE_2": random_two_choices[1],
            "CHOICE_3": random_two_choices[2],
        }

    working_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
    with open(f"{working_dir}/llm_generation/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    with open(f"{working_dir}/config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    based_prompt = prompts["min_value"].format(
        COMPARISON_COLUMN=comparable_col_1,
        PRIMARY_COLUMN=pk_column,
    ).strip()
    response_dict = {}
    prompt_cache = prompts_cache.get(url, {})
    equals_value, equals_answer = Counter(df[comparable_col_2].dropna()).most_common()[0]
    for bd in ["numerical_greater_than", "numerical_less_than", "numerical_between", "numerical_equals"]:
        response_dict[bd] = {}
        _based_prompt = (
                based_prompt
                + " "
                + prompts[bd]
                .format(
            COMPARISON_COLUMN=comparable_col_2,
            THRESHOLD=threshold,
            LOWER_THRESHOLD=lower_threshold,
            UPPER_THRESHOLD=upper_threshold,
            VALUE=equals_value,
        )
                .strip()
        )

        if bd == "numerical_greater_than":
            _df = df[df[comparable_col_2] >= threshold]
            min_val = _df[comparable_col_1].min()
            answer_value = _df[_df[comparable_col_1] == min_val][pk_column].tolist()
        elif bd == "numerical_less_than":
            _df = df[df[comparable_col_2] <= threshold]
            min_val = _df[comparable_col_1].min()
            answer_value = _df[_df[comparable_col_1] == min_val][pk_column].tolist()
        elif bd == "numerical_between":
            _df = df[df[comparable_col_2].between(lower_threshold, upper_threshold, inclusive="both")]
            min_val = _df[comparable_col_1].min()
            answer_value = _df[_df[comparable_col_1] == min_val][pk_column].tolist()
        elif bd == "numerical_equals":
            _df = df[df[comparable_col_2] == equals_value]
            min_val = _df[comparable_col_1].min()
            answer_value = _df[_df[comparable_col_1] == min_val][pk_column].tolist()
        if answer_value == []:
            continue
        # Closed book
        closed_book_prompt = (
            prompts[_wrapper_key("closed_book_wrapper", model)]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
            )
            .strip()
        )

        if _pc_key(f'min_numerical_{bd}_closed_book', model) in prompt_cache:
            closed_book_prompt = prompt_cache[_pc_key(f'min_numerical_{bd}_closed_book', model)]['q']
            answer_value = prompt_cache[_pc_key(f'min_numerical_{bd}_closed_book', model)]['a']
        else:
            print("Caching prompt")
            prompt_cache[_pc_key(f'min_numerical_{bd}_closed_book', model)] = {'q': closed_book_prompt, 'a': answer_value}

        if not rephrase:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=closed_book_prompt,
                    model=model,
                )
            )
        else:
            closed_book_response = parser_ins.try_cast(
                get_llm_response(
                    prompt_string=prompts['rephrasing'].format(STRUCTURED_TASK=closed_book_prompt),
                    model=conf["rephrase_model"],
                )
            )
            response_dict[bd] = {
                'rephrased': closed_book_response,
                'answer': answer_value,
                'options': get_choices(_df=df, _answer_value=answer_value, _desired_column=pk_column)
            }
            continue

        # Multiple choices
        multiple_choices_single_value_prompt = (
            prompts["multiple_choice_wrapper"]
            .format(
                TABLE_TITLE=table_desc,
                QUESTION=_based_prompt,
                INSTRUCTION=prompts[instruction],
                PRIMARY_KEY_COLUMN=pk_column,
                **get_choices(_answer_value=answer_value, _df=df, _desired_column=pk_column),
            )
            .strip()
        )

        if f'min_numerical_{bd}_multiple_choices' in prompt_cache:
            multiple_choices_single_value_prompt = prompt_cache[f'min_numerical_{bd}_multiple_choices']['q']
            answer_value = prompt_cache[f'min_numerical_{bd}_multiple_choices']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'min_numerical_{bd}_multiple_choices'] = {'q': multiple_choices_single_value_prompt, 'a': answer_value}

        multiple_choices_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=multiple_choices_single_value_prompt,
                model=model,
                answer=answer_value,
            )
        )

        # Open book
        open_book_prompt = prompts["open_book_wrapper"].format(
            TABLE_TITLE=table_desc,
            QUESTION=_based_prompt,
            INSTRUCTION=prompts[instruction],
            FULL_CSV_TABLE=df.to_csv(index=True),
        )

        if f'min_numerical_{bd}_open_book' in prompt_cache:
            open_book_prompt = prompt_cache[f'min_numerical_{bd}_open_book']['q']
            answer_value = prompt_cache[f'min_numerical_{bd}_open_book']['a']
        else:
            print("Caching prompt")
            prompt_cache[f'min_numerical_{bd}_open_book'] = {'q': open_book_prompt, 'a': answer_value}

        open_book_response = parser_ins.try_cast(
            get_llm_response(
                prompt_string=open_book_prompt,
                model=model,
            )
        )
        response_dict[bd] = {
            "closed_book": closed_book_response in answer_value,
            "multiple_choices": multiple_choices_response in answer_value,
            "open_book": open_book_response in answer_value,
        }

    prompts_cache[url] = prompt_cache
    return response_dict