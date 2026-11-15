"""Generate the fixed task splits and the leak check.

Splits are decided before any rollout: held-out templates forward_to_contact and
event_from_file, held-out injection families 6 (delayed) and 7 (encoded).
Seeds: train 1, tests 2, benign 3.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.datasets_io import write_jsonl  # noqa: E402
from desk.injections import TARGETS, inject  # noqa: E402
from desk.leakcheck import leak_check  # noqa: E402
from desk.tasks import BENIGN_TEMPLATES, HELD_OUT_TEMPLATES, TEMPLATES, generate_tasks, task_to_dict  # noqa: E402

SEEN = [t for t in TEMPLATES if t not in HELD_OUT_TEMPLATES]
HELD = sorted(HELD_OUT_TEMPLATES)
FAMILY_CYCLE = [1, 2, 3, 4] * 3 + [5] * 4  # family 5 (multilingual) is 25% of injected tasks


def inject_share(tasks, families, rng, num, den, per_template=True):
    """Inject num of every den tasks (counted per template, or by global position), rotating family and target."""
    seen, k, out = Counter(), 0, []
    for i, t in enumerate(tasks):
        occ = seen[t.template] if per_template else i
        seen[t.template] += 1
        if occ % den >= den - num:
            out.append(inject(t, families[k % len(families)], TARGETS[k % len(TARGETS)], rng))
            k += 1
        else:
            out.append(t)
    return out


def main(out: Path) -> dict:
    train = inject_share(generate_tasks(600, SEEN, seed=1, split="train"), FAMILY_CYCLE, random.Random(101), 1, 2)
    used = {t.user_prompt for t in train}

    def fresh(n, templates, split, seed=2):
        ts = generate_tasks(n, templates, seed=seed, split=split, exclude_prompts=used)
        used.update(t.user_prompt for t in ts)
        return ts

    in_dist = inject_share(fresh(150, SEEN, "test_in_dist"), FAMILY_CYCLE, random.Random(102), 2, 3)
    new_template = inject_share(fresh(75, HELD, "test_new_template"), FAMILY_CYCLE, random.Random(103), 2, 3, per_template=False)
    rng = random.Random(104)
    new_family = ([inject(t, 6, TARGETS[i % 3], rng) for i, t in enumerate(fresh(25, ["multi_hop"], "test_new_family_delayed"))]
                  + [inject(t, 7, TARGETS[i % 3], rng) for i, t in enumerate(fresh(50, SEEN, "test_new_family_encoded"))])
    benign = fresh(100, sorted(BENIGN_TEMPLATES), "benign", seed=3)

    def n_inj(ts):
        return sum(bool(t.injection) for t in ts)

    assert (len(train), n_inj(train)) == (600, 300)
    assert (len(in_dist), n_inj(in_dist)) == (150, 100)
    assert (len(new_template), n_inj(new_template)) == (75, 50)
    assert (len(new_family), n_inj(new_family)) == (75, 75) and (len(benign), n_inj(benign)) == (100, 0)
    assert all(t.injection["family"] in (1, 2, 3, 4, 5) for t in train + in_dist + new_template if t.injection)
    assert not {t.template for t in train} & HELD_OUT_TEMPLATES

    splits = {"train": train, "test_in_dist": in_dist, "test_new_template": new_template, "test_new_family": new_family, "benign": benign}
    for name, ts in splits.items():
        write_jsonl(out / f"{name}.jsonl", [task_to_dict(t) for t in ts])

    leaks = leak_check(train, {k: v for k, v in splits.items() if k != "train"})
    assert all(v["exact_prompt_overlap"] == 0 for v in leaks.values()) and leaks["test_new_family"]["injection_text_overlap"] == 0
    (out.parent / "leakcheck.json").write_text(json.dumps(leaks, indent=2), encoding="utf-8")

    print(f"{'split':<20}{'n':>5}{'clean':>7}{'injected':>9}  families              exact  norm  inj_text")
    for name, ts in splits.items():
        fam = Counter(t.injection["family"] for t in ts if t.injection)
        lk = leaks.get(name, {})
        print(f"{name:<20}{len(ts):>5}{sum(not t.injection for t in ts):>7}{sum(bool(t.injection) for t in ts):>9}  "
              f"{dict(sorted(fam.items()))!s:<22}{lk.get('exact_prompt_overlap', '-'):>5}{lk.get('normalised_prompt_overlap', '-'):>6}"
              f"{lk.get('injection_text_overlap', '-'):>9}")
    return {"splits": splits, "leaks": leaks}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tasks", type=Path)
    main(ap.parse_args().out)
