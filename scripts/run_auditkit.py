"""AuditKIT over recorded episodes: agent-eval scorers per model and split, and base-vs-tuned compare.

python scripts/run_auditkit.py --tuned dpo_v1 dpo_clean
Writes results/auditkit_<tag>_<split>_<mode>.json and results/auditkit_compare_<tuned>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.audit import run_agent_eval, run_compare  # noqa: E402
from desk.datasets_io import read_jsonl  # noqa: E402
from desk.loop import episode_from_dict  # noqa: E402
from desk.tasks import task_from_dict  # noqa: E402

SPLITS = ["test_in_dist", "test_new_template", "test_new_family", "benign"]
MODES = ["greedy", "s4", "forced_greedy", "forced_s4"]


def eps(tag, split, mode):
    p = Path(f"data/rollouts/{split}_{tag}_{mode}.jsonl")
    return [episode_from_dict(d) for d in read_jsonl(p)] if p.exists() else None


def main(base: str, tuned: list[str]) -> None:
    for t in tuned:
        report = {}
        for split in SPLITS:
            tasks = [task_from_dict(d) for d in read_jsonl(f"data/tasks/{split}.jsonl")]
            for mode in MODES:
                b, u = eps(base, split, mode), eps(t, split, mode)
                if not (b and u):
                    continue
                for tag, e in ((base, b), (t, u)):
                    out = Path(f"results/auditkit_{tag}_{split}_{mode}.json")
                    if not out.exists():
                        r = run_agent_eval(tasks, e)
                        out.write_text(json.dumps({"means": r["means"], "outcome_success_rate": r["outcome_success_rate"],
                                                   "n": len(r["rows"])}, indent=2), encoding="utf-8")
                c = run_compare(tasks, b, u, out_dir=Path(f"results/auditkit_runs/{t}/{split}_{mode}"))
                report[f"{split}/{mode}"] = {g: {k: v for k, v in d.items() if k != "summary"} for g, d in c.items()}
                for g, d in c.items():
                    bs = "; ".join(f"{m} {x['delta']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]" for m, x in d["bootstrap"].items())
                    print(f"{t} {split}/{mode} {g}: grade {d['overall_grade']} | {bs}")
        Path(f"results/auditkit_compare_{t}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base")
    ap.add_argument("--tuned", nargs="+", required=True)
    a = ap.parse_args()
    main(a.base, a.tuned)
