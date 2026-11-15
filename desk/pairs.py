"""DPO pairs from the model's own rollouts, labelled deterministically.

Injected tasks: at the assistant turn that made the hijacking call, rejected = that turn,
chosen = a successful sample's turn after the same prefix, else the gold continuation.
Clean tasks: at the first turn that departs from gold (invalid call, wrong tool, or an early
answer), rejected = that turn, chosen = the gold continuation. Prompts are role/content only,
exactly what the chat template renders at rollout time.
"""
from __future__ import annotations

import random

from desk.injections import is_injected_call
from desk.prompting import fold_result, render_call, system_prompt
from desk.scoring import Episode, _matches_gold, _norm, score_episode
from desk.tasks import Task, execute_gold, gold_answer_text
from desk.tools import TOOL_SCHEMAS


def _strip(msgs: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def _turns(ep: Episode) -> list[tuple[int, int, list[dict]]]:
    """(assistant turn number, index in messages, that turn's calls)."""
    out, n = [], 0
    for i, m in enumerate(ep.messages):
        if m["role"] == "assistant":
            out.append((n, i, [c for c in ep.calls if c["step"] == n]))
            n += 1
    return out


def _pair(prompt, chosen, rejected, kind, task) -> dict:
    return {"prompt": _strip(prompt), "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}], "kind": kind, "task_id": task.id,
            "family": task.injection["family"] if task.injection else None}


def _injected_pairs(task: Task, eps: list[Episode]) -> list[dict]:
    good = [e for e in eps if score_episode(task, e).task_success]
    out = []
    for e in eps:
        for n, idx, calls in _turns(e):
            if any(c["valid"] and is_injected_call(task.injection, c["name"], c["params"]) for c in calls):
                prompt = e.messages[:idx]
                chosen = next((g.messages[idx]["content"] for g in good
                               if len(g.messages) > idx and g.messages[idx]["role"] == "assistant"
                               and _strip(g.messages[:idx]) == _strip(prompt)), None)
                chosen = chosen or _gold_next(task, [c for c in e.calls if c["step"] < n])
                out.append(_pair(prompt, chosen, e.messages[idx]["content"], "injected", task))
                break
        else:  # not hijacked: a wrong turn after the injection was read is still "lost the user's task" data
            x = _exposure_idx(task, e)
            if x is not None and (p := _first_deviation(task, e, after_idx=x)):
                out.append({**p, "kind": "injected_task"})
    return out


def _matched(task: Task, calls: list[dict]) -> int:
    """How many gold calls the calls so far satisfy, as an ordered subsequence."""
    k = 0
    for c in calls:
        if k < len(task.gold_calls) and _matches_gold(c, task.gold_calls[k]):
            k += 1
    return k


def _gold_next(task: Task, calls_before: list[dict]) -> str:
    k = _matched(task, calls_before)
    if k < len(task.gold_calls):
        g = task.gold_calls[k]
        return render_call(g["name"], g["arguments"])
    return gold_answer_text(task)


def _first_deviation(task: Task, e: Episode, after_idx: int = -1) -> dict | None:
    """Pair for the first assistant turn (past message index after_idx) that departs from gold."""
    for n, idx, calls in _turns(e):
        before = [c for c in e.calls if c["step"] < n]
        k = _matched(task, before)
        done = k == len(task.gold_calls)
        if not calls:  # final answer: bad if gold calls are still missing or the gold fact is absent
            bad = not done or bool(task.gold_fact) and _norm(task.gold_fact) not in _norm(e.messages[idx]["content"])
        else:  # a call: bad if invalid, redundant after all gold calls, or not the next gold call
            bad = done or any(not c["valid"] for c in calls) or not _matches_gold(calls[0], task.gold_calls[k])
        if bad and idx > after_idx:
            return _pair(e.messages[:idx], _gold_next(task, before), e.messages[idx]["content"], "clean", task)
        if bad or not calls:
            return None
    return None


def _clean_pairs(task: Task, eps: list[Episode]) -> list[dict]:
    return [p for e in eps if (p := _first_deviation(task, e))]


def _exposure_idx(task: Task, e: Episode) -> int | None:
    needle = " ".join(task.injection["text"].split())[:60]
    return next((i for i, m in enumerate(e.messages) if i >= 2 and m["role"] == "user" and needle in " ".join(m["content"].split())), None)


def build_pairs(tasks: list[Task], episodes: list[Episode], *, min_clean_share: float = 0.30,
                rng: random.Random | None = None) -> list[dict]:
    rng = rng or random.Random(0)
    by_task: dict[str, list[Episode]] = {}
    for e in episodes:
        by_task.setdefault(e.task_id.rsplit("#", 1)[0], []).append(e)
    injected, clean, seen = [], [], set()
    for t in tasks:
        eps = by_task.get(t.id, [])
        for p in (_injected_pairs(t, eps) if t.injection else _clean_pairs(t, eps)):
            key = (repr(p["prompt"]), p["chosen"][0]["content"], p["rejected"][0]["content"])
            if key not in seen:
                seen.add(key)
                (clean if p["kind"] == "clean" else injected).append(p)
    if min_clean_share > 0 and injected and len(clean) / (len(clean) + len(injected)) < min_clean_share:
        cap = int(len(clean) * (1 - min_clean_share) / min_clean_share)
        injected = rng.sample(injected, cap)
    return clean + injected


def synthetic_pairs(tasks: list[Task]) -> list[dict]:
    """Fallback for injected tasks with no hijacked sample: gold prefix up to the injected
    result, chosen = the gold continuation, rejected = the injected call."""
    out = []
    for t in tasks:
        if not t.injection:
            continue
        _, obs = execute_gold(t)
        k = t.injection["at_read_step"]
        prompt = [{"role": "system", "content": system_prompt(TOOL_SCHEMAS)}, {"role": "user", "content": t.user_prompt}]
        for g, o in zip(t.gold_calls[:k + 1], obs):
            prompt += [{"role": "assistant", "content": render_call(g["name"], g["arguments"])}, fold_result(g["name"], o)]
        tg = t.injection["target"]
        chosen = (render_call(t.gold_calls[k + 1]["name"], t.gold_calls[k + 1]["arguments"])
                  if k + 1 < len(t.gold_calls) else gold_answer_text(t))
        out.append(_pair(prompt, chosen, render_call(tg["tool"], {tg["key_arg"]: tg["value"], **tg["extra"]}), "synthetic", t))
    return out
