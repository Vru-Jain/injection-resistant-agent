"""Cost of 10,000 desk tasks/day: self-hosted Tiny Aya (measured on H100) vs a closed API (assumed prices).

python scripts/cost_table.py --tag dpo_v1  ->  results/cost_table.md
Token counts and H100 throughput come from data/rollouts/<split>_<tag>_greedy.stats.json (batched HF generate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.cost import cost_table  # noqa: E402

SPLITS = ["test_in_dist", "test_new_template", "test_new_family", "benign"]
H100_PER_HOUR = 3.95  # Modal on-demand H100, as quoted in the design doc (Nov 2026)
API_IN, API_OUT = 2.50, 10.0  # assumption: USD per 1M tokens for a frontier closed model; check current list prices


def main(tag: str, tasks_per_day: int = 10_000) -> str:
    st = [json.loads(Path(f"data/rollouts/{s}_{tag}_greedy.stats.json").read_text()) for s in SPLITS]
    n = sum(x["n_tasks"] * x["n_samples"] for x in st)
    p, c = sum(x["prompt_tokens"] for x in st) / n, sum(x["completion_tokens"] for x in st) / n
    gen_minutes = sum(x["seconds"] for x in st) / 60
    h100_tpm = n / gen_minutes
    rows = {
        f"Self-host, H100 measured ({h100_tpm:.0f} tasks/min, ${H100_PER_HOUR}/h)":
            cost_table(prompt_tokens_per_task=p, completion_tokens_per_task=c, tasks_per_day=tasks_per_day, api_in_per_m=API_IN,
                       api_out_per_m=API_OUT, gpu_per_hour=H100_PER_HOUR, tasks_per_minute=h100_tpm),
    }
    api = next(iter(rows.values()))
    md = [f"# Cost of {tasks_per_day:,} desk tasks per day\n",
          f"Measured on `{tag}` greedy rollouts over the 4 eval splits ({n} tasks): {p:.0f} prompt tokens and {c:.0f} completion "
          "tokens per task (prompt tokens summed over turns, since every turn resends the conversation).\n",
          "| option | USD per day | USD per 1k tasks | GPU hours per day |", "|---|---|---|---|"]
    for name, r in rows.items():
        md.append(f"| {name} | {r['selfhost_per_day']:.2f} | {r['selfhost_per_1k']:.3f} | {r['selfhost_gpu_hours_per_day']:.2f} |")
    md.append(f"| Closed API (assumed ${API_IN}/M input, ${API_OUT}/M output) | {api['api_per_day']:.2f} | {api['api_per_1k']:.3f} | - |")
    md += ["", "The API prices are an assumption, not a measurement (check current list prices before quoting). The H100 row is "
           "our own measured batched HF `generate` throughput, which vLLM would raise. Token counts are as measured; a model "
           "that stops after the task instead of running to the 8-turn cap would use fewer.",
           "Self-hosting also keeps emails and files on your own hardware, which matters for an assistant that reads them."]
    out = "\n".join(md) + "\n"
    Path("results/cost_table.md").write_text(out, encoding="utf-8")
    print(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    main(ap.parse_args().tag)
