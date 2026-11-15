"""Build DPO pairs from base rollouts.

python scripts/build_pairs.py --rollouts data/rollouts/train_base_s4.jsonl data/rollouts/train_base_forced_s8.jsonl --tasks data/tasks/train.jsonl --out data/pairs/train_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.datasets_io import read_jsonl, write_jsonl  # noqa: E402
from desk.loop import episode_from_dict  # noqa: E402
from desk.pairs import build_pairs, synthetic_pairs  # noqa: E402
from desk.tasks import task_from_dict  # noqa: E402

SYNTHETIC_FLOOR = 800  # below this many on-policy pairs, add gold-vs-injected synthetic pairs


def main(rollouts: list[Path], tasks_path: Path, out: Path, min_clean_share: float) -> list[dict]:
    tasks = [task_from_dict(d) for d in read_jsonl(tasks_path)]
    eps = [episode_from_dict(d) for r in rollouts for d in read_jsonl(r)]
    pairs = build_pairs(tasks, eps, min_clean_share=min_clean_share, rng=random.Random(0))
    # keep the hijack pairs visible: at most two "lost the task under attack" pairs per hijack pair
    hijack = [p for p in pairs if p["kind"] == "injected"]
    task_pairs = [p for p in pairs if p["kind"] == "injected_task"]
    if len(task_pairs) > 2 * len(hijack):
        keep = {id(p) for p in random.Random(1).sample(task_pairs, 2 * len(hijack))}
        pairs = [p for p in pairs if p["kind"] != "injected_task" or id(p) in keep]
    if len(pairs) < SYNTHETIC_FLOOR:
        covered = {p["task_id"] for p in pairs if p["kind"] == "injected"}
        extra = synthetic_pairs([t for t in tasks if t.injection and t.id not in covered])
        pairs += extra[:SYNTHETIC_FLOOR - len(pairs)]
    write_jsonl(out, pairs)
    by_kind = Counter(p["kind"] for p in pairs)
    by_family = Counter(str(p["family"]) for p in pairs if p["family"] is not None)
    stats = {"n": len(pairs), "by_kind": dict(by_kind), "by_family": dict(sorted(by_family.items())),
             "clean_share": by_kind["clean"] / max(1, len(pairs)), "n_tasks_with_pairs": len({p["task_id"] for p in pairs})}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, nargs="+", required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/pairs/train_pairs.jsonl"))
    ap.add_argument("--min-clean-share", type=float, default=0.30)
    a = ap.parse_args()
    main(a.rollouts, a.tasks, a.out, a.min_clean_share)
