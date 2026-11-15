"""AuditKIT bridge: recorded-episode agent eval, before/after compare, paired bootstrap CIs."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import auditkit as ak
from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, CustomPredicate, episode_from_openai_messages

from desk.scoring import Episode, score_episode
from desk.tasks import Task
from desk.tools import TOOL_SCHEMAS

AGENT_SCORERS = ["tool_call_f1", "tool_call_validity", "trajectory_match", "redundant_tool_calls"]


def _task_of(tasks: list[Task]):
    by_id = {t.id: t for t in tasks}
    return lambda ep: by_id[ep.task_id.rsplit("#", 1)[0]]


def to_auditkit_episode(task: Task, ep: Episode):
    s = score_episode(task, ep)
    ae = episode_from_openai_messages(ep.messages, case_id=ep.task_id, final_answer=ep.final_answer)
    ae.final_state = {"asr": s.asr, "task_success": s.task_success}
    return ae


def _done_not_hijacked(episode, case) -> bool:
    return episode.final_state.get("task_success") == 1


def to_case(task: Task, case_id: str) -> AgentCase:
    return AgentCase(id=case_id, task=task.user_prompt, allowed_tools=[s["name"] for s in TOOL_SCHEMAS],
                     reference_turns=[[{"name": c["name"], "arguments": c["arguments"]}] for c in task.gold_calls],
                     outcome=CustomPredicate(_done_not_hijacked, version="1", name="task_done_not_hijacked"))


def run_agent_eval(tasks: list[Task], episodes: list[Episode]) -> dict:
    task_of = _task_of(tasks)
    spec = AgentEvalSpec(cases=[to_case(task_of(e), e.task_id) for e in episodes], mode="recorded",
                         episodes=[to_auditkit_episode(task_of(e), e) for e in episodes], scorers=AGENT_SCORERS, trials=1)
    res = AgentEvalRunner().run(spec).to_json()
    res = json.loads(res) if isinstance(res, str) else res
    sums, counts = defaultdict(float), defaultdict(int)
    for row in res["rows"]:
        for k, v in row["diagnostics"].items():
            if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
                sums[k] += v["value"]
                counts[k] += 1
    verdicts = [row["outcome"]["verdict"] for row in res["rows"]]
    return {"rows": res["rows"], "means": {k: sums[k] / counts[k] for k in sums},
            "outcome_success_rate": verdicts.count("success") / len(verdicts) if verdicts else None}


@ak.scorer(direction=ak.Direction.MINIMIZE, name="asr")
def asr_scorer(sample, output) -> float:
    return float(sample.metadata["asr"])


@ak.scorer(direction=ak.Direction.MAXIMIZE, name="task_success")
def task_success_scorer(sample, output) -> float:
    return float(sample.metadata["task_success"])


@ak.scorer(direction=ak.Direction.MAXIMIZE, name="validity")
def validity_scorer(sample, output) -> float:
    return float(sample.metadata["validity"])


def per_task_scores(tasks: list[Task], episodes: list[Episode]) -> dict[str, dict]:
    """Mean score per task over its samples (so base and tuned pair up by task id)."""
    task_of, acc = _task_of(tasks), defaultdict(list)
    for e in episodes:
        acc[task_of(e).id].append(score_episode(task_of(e), e))
    return {tid: {"asr": sum(s.asr for s in ss) / len(ss), "task_success": sum(s.task_success for s in ss) / len(ss),
                  "validity": sum(s.validity_rate for s in ss) / len(ss)} for tid, ss in acc.items()}


def _run(scores: dict[str, dict], ids: list[str], scorers):
    samples = [ak.Sample(input=i, target="1", id=i, actual_output=str(scores[i]["task_success"]), metadata=scores[i]) for i in ids]
    return ak.evaluate(samples, model="precomputed", scorers=scorers)


def _v(x):  # RunComparison exposes some results as methods
    return x() if callable(x) else x


def run_compare(tasks: list[Task], base_eps: list[Episode], tuned_eps: list[Episode], out_dir=None) -> dict:
    b, t = per_task_scores(tasks, base_eps), per_task_scores(tasks, tuned_eps)
    inj = {x.id for x in tasks if x.injection}
    groups = {"injected": ([i for i in b if i in inj and i in t], [asr_scorer, task_success_scorer]),
              "clean": ([i for i in b if i not in inj and i in t], [task_success_scorer, validity_scorer])}
    out = {}
    for name, (ids, scorers) in groups.items():
        if not ids:
            continue
        rb, rt = _run(b, ids, scorers), _run(t, ids, scorers)
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            rb.save(str(Path(out_dir) / f"base_{name}.json"), fmt="json")
            rt.save(str(Path(out_dir) / f"tuned_{name}.json"), fmt="json")
        c = ak.compare(rb, rt)
        out[name] = {
            "n_tasks": len(ids), "overall_grade": _v(c.grade).value, "grades": {k: v.value for k, v in _v(c.grades).items()},
            "metric_deltas": {d.metric: {"baseline": d.baseline, "candidate": d.candidate, "delta": d.delta, "grade": d.grade.value}
                              for d in _v(c.metric_deltas)},
            "auditkit_significance": _v(c.significance),  # AuditKIT's own test (first metric of the group)
            "bootstrap": {m: paired_bootstrap([b[i][m] for i in ids], [t[i][m] for i in ids])
                          for m in ("asr", "task_success", "validity") if any(s.name == m for s in scorers)},
            "summary": str(_v(c.summary)),
        }
    return out


def paired_bootstrap(base: list[float], tuned: list[float], n: int = 10000, seed: int = 0) -> dict:
    """95% CI of mean(tuned - base), resampling tasks with replacement."""
    diffs = [t - b for b, t in zip(base, tuned)]
    rng, k = random.Random(seed), len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return {"delta": sum(diffs) / k, "ci_low": means[int(0.025 * n)], "ci_high": means[int(0.975 * n) - 1], "n": k, "resamples": n}
