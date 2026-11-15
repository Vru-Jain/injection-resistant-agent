"""Batched lockstep agent loop: every active episode advances one assistant turn per round."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Callable

from desk.prompting import fold_result, parse_calls, system_prompt
from desk.scoring import Episode
from desk.tasks import Task
from desk.tools import TOOL_SCHEMAS, DeskState, call_tool

GenerateFn = Callable[[list[list[dict]]], list[str]]

PARSE_ERROR = "error: could not parse tool call"


class _Run:
    def __init__(self, task: Task, sample: int):
        self.task = task
        self.state = task.state.copy()
        self.ep = Episode(task_id=f"{task.id}#{sample}", calls=[], final_answer=None, state_after=self.state, stop_reason="max_steps",
                          messages=[{"role": "system", "content": system_prompt(TOOL_SCHEMAS)}, {"role": "user", "content": task.user_prompt}])
        self.rounds = 0
        self.done = False

    def advance(self, text: str, max_steps: int) -> None:
        step, self.rounds = self.rounds, self.rounds + 1
        calls = parse_calls(text)
        msg = {"role": "assistant", "content": text}
        if calls is None and text.lstrip()[:1] in ("[", "{"):
            self.ep.messages.append(msg)
            self.ep.calls.append({"step": step, "name": None, "params": None, "valid": False, "observation": PARSE_ERROR})
            self.ep.messages.append(fold_result("unknown", PARSE_ERROR))
        elif calls is None or (len(calls) == 1 and calls[0][0] == "finish"):
            self.ep.messages.append(msg)
            args = calls[0][1] if calls else None
            self.ep.final_answer = str(args.get("answer", "")) if isinstance(args, dict) else text
            self.ep.stop_reason, self.done = "finished", True
            return
        else:
            msg["tool_calls"] = [{"id": f"call_{step}_{i}", "type": "function",
                                  "function": {"name": n, "arguments": json.dumps(p, ensure_ascii=False)}}
                                 for i, (n, p) in enumerate(calls)]
            self.ep.messages.append(msg)
            for n, p in calls:
                obs = call_tool(self.state, n, p)
                self.ep.calls.append({"step": step, "name": n, "params": p, "valid": not obs.startswith("error:"), "observation": obs})
                self.ep.messages.append(fold_result(n, obs))
        if self.rounds >= max_steps:
            self.done = True


def run_episodes(tasks: list[Task], generate: GenerateFn, *, max_steps: int = 8, n_samples: int = 1,
                 batch_size: int = 32) -> list[Episode]:
    runs = [_Run(t, s) for t in tasks for s in range(n_samples)]
    while active := [r for r in runs if not r.done]:
        for i in range(0, len(active), batch_size):
            chunk = active[i:i + batch_size]
            for r, text in zip(chunk, generate([r.ep.messages for r in chunk])):
                r.advance(text or "", max_steps)
    return [r.ep for r in runs]


def episode_to_dict(ep: Episode) -> dict:
    return asdict(ep)


def episode_from_dict(d: dict) -> Episode:
    return Episode(**{**d, "state_after": DeskState(**d["state_after"])})
