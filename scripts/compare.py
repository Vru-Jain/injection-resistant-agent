"""Same-model before/after tables from results/<split>_<tag>_<mode>.json.

python scripts/compare.py --base base --tuned dpo_v1  ->  results/compare_dpo_v1.md and .json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.scoring import passbar  # noqa: E402

SPLITS = ["test_in_dist", "test_new_template", "test_new_family", "benign"]
MODES = ["greedy", "s4", "forced_greedy", "forced_s4"]
METRICS = [("asr", "lower"), ("exposure", "-"), ("asr_given_exposed", "lower"), ("injected_task_success", "higher"),
           ("clean_task_success", "higher"), ("validity", "higher"), ("over_refusal", "lower"), ("mean_steps", "-")]


def load(tag: str, split: str, mode: str) -> dict | None:
    p = Path("results") / f"{split}_{tag}_{mode}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def delta(b, t) -> str:
    return "-" if b is None or t is None else f"{t - b:+.3f}"


def main(base: str, tuned: str) -> dict:
    out = {"base": base, "tuned": tuned, "slices": {}}
    md = [f"# Before/after: `{base}` vs `{tuned}` (same model, same tasks, same decoding)\n",
          "ASR = share of injected tasks where an executable call hit the attacker's target. Forced = gold steps replayed "
          "up to and including the tool result that carries the injection, then the model continues.\n"]
    for mode in MODES:
        rows = {s: (load(base, s, mode), load(tuned, s, mode)) for s in SPLITS}
        rows = {s: v for s, v in rows.items() if v[0] and v[1]}
        if not rows:
            continue
        md += [f"\n## {mode}\n", "| metric | better | " + " | ".join(f"{s} base | {s} {tuned} | delta" for s in rows) + " |",
               "|---|---|" + "---|---|---|" * len(rows)]
        for m, better in METRICS:
            cells = []
            for s, (b, t) in rows.items():
                bv, tv = b["overall"].get(m), t["overall"].get(m)
                cells += [fmt(bv), fmt(tv), delta(bv, tv)]
                out["slices"].setdefault(f"{s}/{mode}", {})[m] = {"base": bv, "tuned": tv}
            if any(c != "-" for c in cells):
                md.append(f"| {m} | {better} | " + " | ".join(cells) + " |")
        md.append("\n" + "n: " + ", ".join(f"{s} {b['overall']['n']}" for s, (b, _) in rows.items()))

    names = {"1": "imperative", "2": "authority", "3": "urgency", "4": "hidden format", "5": "multilingual",
             "6": "delayed (held out)", "7": "encoded (held out)"}
    for mode in ("forced_greedy", "greedy"):
        fams: dict[str, list[float]] = {}  # family -> [base hits, n, tuned hits], pooled over test slices
        for s in ("test_in_dist", "test_new_template", "test_new_family"):
            b, t = load(base, s, mode), load(tuned, s, mode)
            if not (b and t):
                continue
            for f, v in b["by_family"].items():
                if f == "clean":
                    continue
                acc = fams.setdefault(f, [0, 0, 0])
                acc[0] += v["asr"] * v["n_injected"]
                acc[1] += v["n_injected"]
                acc[2] += t["by_family"][f]["asr"] * t["by_family"][f]["n_injected"]
        if fams:
            md += ["", f"## ASR by injection family ({mode}, test slices pooled)", "", "| family | n | base | tuned | delta |", "|---|---|---|---|---|"]
            for f in sorted(fams):
                hb, n, ht = fams[f]
                md.append(f"| {f} {names.get(f, '')} | {n} | {fmt(hb / n)} | {fmt(ht / n)} | {delta(hb / n, ht / n)} |")
                out.setdefault("by_family", {}).setdefault(mode, {})[f] = {"n": n, "base": hb / n, "tuned": ht / n}

    md += ["\n## Task success by template (greedy)\n", "| template | slice | base | tuned | delta |", "|---|---|---|---|---|"]
    for s in ("test_in_dist", "test_new_template", "benign"):
        b, t = load(base, s, "greedy"), load(tuned, s, "greedy")
        if b and t:
            for tpl, v in b["by_template"].items():
                tv = t["by_template"][tpl]["task_success"]
                md.append(f"| {tpl} | {s} | {fmt(v['task_success'])} | {fmt(tv)} | {delta(v['task_success'], tv)} |")

    bi, ti = load(base, "test_in_dist", "greedy"), load(tuned, "test_in_dist", "greedy")
    bb, tb = load(base, "benign", "greedy"), load(tuned, "benign", "greedy")
    if bi and ti:
        b = {**bi["overall"], "over_refusal": bb["overall"]["over_refusal"] if bb else None}
        t = {**ti["overall"], "over_refusal": tb["overall"]["over_refusal"] if tb else None}
        pb = passbar(b, t)
        out["passbar"] = pb
        md += ["\n## Pass bar (spec, pre-registered; in-dist greedy, over-refusal on the benign split)\n",
               "| criterion | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in pb.items()]
    Path(f"results/compare_{tuned}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    Path(f"results/compare_{tuned}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(md))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base")
    ap.add_argument("--tuned", required=True)
    a = ap.parse_args()
    main(a.base, a.tuned)
