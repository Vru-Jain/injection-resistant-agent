"""Modal app: rollouts (HF generate) and DPO training for Tiny Aya on H100.

  modal run modal_app.py --cmd smoke
  modal run modal_app.py --cmd rollout --tasks data/tasks/train.jsonl --out train_base_s4 --n-samples 4 --temperature 0.7
  modal run modal_app.py --cmd baseline                  # train rollouts + every eval rollout of the base model, in parallel
  modal run modal_app.py --cmd evals --tag dpo_v1 --model-id Vru-Jain/tiny-aya-global-injection-dpo
Outputs land in data/rollouts/<out>.jsonl and data/rollouts/<out>.stats.json locally (and /vol/rollouts on the volume).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

BASE_MODEL = "CohereLabs/tiny-aya-global"
AGENTTUNE = "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git@2aecde44d5b3f8e9752af42dc9900b7646647583"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.11.0", "transformers>=5.15,<6", "trl==1.7.1", "peft", "datasets", "accelerate",
                    "huggingface_hub", "auditkit", "psutil", AGENTTUNE)
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("desk")
)
vol = modal.Volume.from_name("ira-data", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token")]
app = modal.App("injection-resistant-agent", image=image)


@app.function(gpu="H100", timeout=3600, volumes={"/vol": vol}, secrets=secrets)
def rollout(task_rows: list[dict], out_name: str, model_id: str = BASE_MODEL, adapter: str | None = None,
            n_samples: int = 1, temperature: float = 0.0, batch_size: int = 64, prefill: bool = False) -> dict:
    from desk.hf_generate import load_model, make_generate
    from desk.loop import episode_to_dict, run_episodes
    from desk.tasks import task_from_dict

    t0 = time.time()
    model, tok = load_model(model_id, adapter)
    gen = make_generate(model, tok, temperature=temperature, seed=0)
    eps = run_episodes([task_from_dict(r) for r in task_rows], gen, n_samples=n_samples, batch_size=batch_size, prefill=prefill)
    lines = "".join(json.dumps(episode_to_dict(e), ensure_ascii=False) + "\n" for e in eps)
    stats = {**gen.stats, "out": out_name, "model_id": model_id, "adapter": adapter, "n_tasks": len(task_rows),
             "n_samples": n_samples, "temperature": temperature, "prefill": prefill, "wall_seconds": time.time() - t0}
    Path("/vol/rollouts").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/rollouts/{out_name}.jsonl").write_text(lines, encoding="utf-8")
    Path(f"/vol/rollouts/{out_name}.stats.json").write_text(json.dumps(stats, indent=2))
    vol.commit()
    return {"jsonl": lines, "stats": stats}


def _save(res: dict) -> None:
    out = Path("data/rollouts")
    out.mkdir(parents=True, exist_ok=True)
    name = res["stats"]["out"]
    (out / f"{name}.jsonl").write_text(res["jsonl"], encoding="utf-8")
    (out / f"{name}.stats.json").write_text(json.dumps(res["stats"], indent=2))
    s = res["stats"]
    print(f"{name}: {s['n_tasks']} tasks x {s['n_samples']}, {s['wall_seconds'] / 60:.1f} min wall, "
          f"{s['completion_tokens']} completion tokens")


def _rows(path: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


EVAL_SPLITS = ["test_in_dist", "test_new_template", "test_new_family", "benign"]
INJECTED_SPLITS = ["test_in_dist", "test_new_template", "test_new_family"]


def eval_specs(tag: str, limit: int = 0, only: str = "") -> list[tuple]:
    """(rows, out_name, n_samples, temperature, prefill) for every eval rollout of one model."""
    specs = []
    for split in [s for s in EVAL_SPLITS if not only or s in only.split(",")]:
        rows = _rows(f"data/tasks/{split}.jsonl", limit)
        specs += [(rows, f"{split}_{tag}_greedy", 1, 0.0, False), (rows, f"{split}_{tag}_s4", 4, 0.7, False)]
        if split in INJECTED_SPLITS:  # forced exposure: gold prefix through the injected read, then the model
            inj = [r for r in rows if r["injection"]]
            specs += [(inj, f"{split}_{tag}_forced_greedy", 1, 0.0, True), (inj, f"{split}_{tag}_forced_s4", 4, 0.7, True)]
    return specs


def train_specs(limit: int = 0) -> list[tuple]:
    rows = _rows("data/tasks/train.jsonl", limit)
    return [(rows, "train_base_s4", 4, 0.7, False),
            ([r for r in rows if r["injection"]], "train_base_forced_s8", 8, 1.0, True)]


@app.local_entrypoint()
def main(cmd: str = "smoke", tasks: str = "", out: str = "", n_samples: int = 1, temperature: float = 0.0,
         model_id: str = BASE_MODEL, adapter: str = "", limit: int = 0, tag: str = "base", prefill: bool = False, splits: str = ""):
    if cmd == "smoke":
        rows = _rows("data/tasks/train.jsonl")
        clean = [r for r in rows if not r["injection"]][:20]
        injected = [r for r in rows if r["injection"]][:20]
        _save(rollout.remote(clean + injected, "smoke_base", BASE_MODEL, None, 1, 0.0))
    elif cmd == "rollout":
        _save(rollout.remote(_rows(tasks, limit), out, model_id, adapter or None, n_samples, temperature, prefill=prefill))
    elif cmd in ("baseline", "evals"):  # one container per rollout, all in parallel
        specs = (train_specs(limit) if cmd == "baseline" else []) + eval_specs(tag, limit, splits)
        calls = [rollout.spawn(rows, name, model_id, adapter or None, n, temp, prefill=pf) for rows, name, n, temp, pf in specs]
        for c in calls:
            _save(c.get())
    else:
        raise SystemExit(f"unknown cmd {cmd}")
