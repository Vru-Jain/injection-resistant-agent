"""Score a rollout file: python scripts/summarise_rollouts.py data/rollouts/X.jsonl data/tasks/Y.jsonl [--out results/X.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.datasets_io import read_jsonl  # noqa: E402
from desk.loop import episode_from_dict  # noqa: E402
from desk.summary import summarise  # noqa: E402
from desk.tasks import task_from_dict  # noqa: E402


def main(rollouts: Path, tasks: Path, out: Path | None) -> dict:
    s = summarise([task_from_dict(d) for d in read_jsonl(tasks)], [episode_from_dict(d) for d in read_jsonl(rollouts)])
    s["source"] = {"rollouts": rollouts.as_posix(), "tasks": tasks.as_posix()}
    out = out or Path("results") / f"{rollouts.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    o = s["overall"]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"{rollouts.stem}: n={o['n']} asr={fmt(o['asr'])} attempted={fmt(o['attempted'])} clean_success={fmt(o['clean_task_success'])} "
          f"injected_success={fmt(o['injected_task_success'])} validity={fmt(o['validity'])} over_refusal={fmt(o['over_refusal'])} -> {out}")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts", type=Path)
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    main(a.rollouts, a.tasks, a.out)
