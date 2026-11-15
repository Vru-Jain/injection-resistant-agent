"""Export the injection suite (seed sentences, targets, families, rendered instructions) as JSON for the HF dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.injections import _INSTR, FAMILIES, HELD_OUT_FAMILIES, SEED_LANG, SEEDS, TARGETS, TRAIN_FAMILIES  # noqa: E402


def main(out: Path = Path("data/injection_suite")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"family": f, "family_name": FAMILIES[f], "split": "train" if f in TRAIN_FAMILIES else "held_out",
             "language": SEED_LANG.get(s, "en"), "seed": s, "tool_format": "{target_call}" in s}
            for f in sorted(SEEDS) for s in SEEDS[f]]
    (out / "seeds.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "targets.json").write_text(json.dumps({"targets": TARGETS, "instructions_by_language": _INSTR,
                                                  "train_families": sorted(TRAIN_FAMILIES), "held_out_families": sorted(HELD_OUT_FAMILIES)},
                                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} seeds -> {out}")


if __name__ == "__main__":
    main()
