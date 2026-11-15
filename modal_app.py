"""Modal app: rollouts (HF generate) and DPO training for Tiny Aya on H100.

  modal run modal_app.py --cmd smoke
  modal run modal_app.py --cmd rollout --tasks data/tasks/train.jsonl --out train_base_s4 --n-samples 4 --temperature 0.7
  modal run modal_app.py --cmd baseline                  # train rollouts + every eval rollout of the base model, in parallel
  modal run modal_app.py --cmd train                     # dpo_v1 (all pairs) and dpo_clean (ablation) in parallel
  modal run modal_app.py --cmd evals --tag dpo_v1 --model-id dpo_v1
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

    model_id = _resolve(model_id)
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


RUNS = {  # run -> settings; the merged weights go to "repo", the adapter to repo + "-lora"
    "dpo_v1": {"kinds": None, "repo": "Vru-Jain/tiny-aya-global-injection-dpo-v1", "pairs": "train_pairs", "base": None},
    "dpo_clean": {"kinds": ["clean"], "repo": "Vru-Jain/tiny-aya-global-desk-clean-dpo", "pairs": "train_pairs", "base": None},
    # round 2: on-policy pairs from dpo_v1's own train rollouts, trained on top of dpo_v1 (reference = dpo_v1)
    "dpo_v2": {"kinds": None, "repo": "Vru-Jain/tiny-aya-global-injection-dpo", "pairs": "train_pairs_r2", "base": "dpo_v1",
               "epochs": 1, "lr": 2e-5},
}


@app.function(gpu="H100", timeout=5400, volumes={"/vol": vol}, secrets=secrets)
def train(pair_rows: list[dict], run: str, kinds: list[str] | None, repo: str, beta: float = 0.1, epochs: int = 2,
          lr: float = 3e-5, push: bool = True, base: str | None = None) -> dict:
    from huggingface_hub import HfApi

    from desk.train import merge, model_card, select_rows, train_dpo

    out, start = f"/vol/models/{run}", (_resolve(base) if base else BASE_MODEL)
    stats = train_dpo(select_rows(pair_rows, kinds), start, f"{out}/adapter", beta=beta, epochs=epochs, lr=lr)
    merge(start, f"{out}/adapter", f"{out}/merged")
    vol.commit()
    stats.update(run=run, kinds=kinds, repo=repo)
    if push:  # private until the release step makes everything public together
        api = HfApi()
        for folder, rid, kind in ((f"{out}/adapter", f"{repo}-lora", "adapter"), (f"{out}/merged", repo, "merged")):
            Path(folder, "README.md").write_text(model_card(rid, BASE_MODEL, kind, stats))
            Path(folder, "training_stats.json").write_text(json.dumps(stats, indent=2))
            api.create_repo(rid, private=True, exist_ok=True)
            api.upload_folder(folder_path=folder, repo_id=rid, commit_message=f"{run}: DPO on {stats['n_pairs']} pairs")
    return stats


@app.function(gpu="H100", timeout=1800, volumes={"/vol": vol}, secrets=secrets)
def redteam(model_id: str, tag: str) -> dict:
    """AuditKIT's built-in chat probes (prompt injection, encoding, jailbreak, over-refusal), single turn, greedy."""
    import auditkit.redteam as rt

    from desk.hf_generate import load_model, make_generate

    model, tok = load_model(_resolve(model_id))
    gen = make_generate(model, tok, temperature=0.0, max_new_tokens=200)
    r = rt.RedTeamRunner(lambda prompts: gen([[{"role": "user", "content": p}] for p in prompts])).run()
    return {"tag": tag, "model_id": model_id, "pass_rate": r.pass_rate, "total": r.total_probes,
                    "failed": r.failed_probes, "passed": r.passed_probes, "invalid": r.invalid_probes,
            "summary": str(r.summary() if callable(r.summary) else r.summary)}


def _resolve(model_id: str) -> str:
    """A run name from RUNS means that run's merged weights on the volume (keeps paths out of the local shell)."""
    return f"/vol/models/{model_id}/merged" if model_id in RUNS else model_id


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


def train_specs(limit: int = 0, tag: str = "base") -> list[tuple]:
    rows = _rows("data/tasks/train.jsonl", limit)
    return [(rows, f"train_{tag}_s4", 4, 0.7, False),
            ([r for r in rows if r["injection"]], f"train_{tag}_forced_s8", 8, 1.0, True)]


@app.local_entrypoint()
def main(cmd: str = "smoke", tasks: str = "", out: str = "", n_samples: int = 1, temperature: float = 0.0,
         model_id: str = BASE_MODEL, adapter: str = "", limit: int = 0, tag: str = "base", prefill: bool = False, splits: str = "", runs: str = ""):
    if cmd == "smoke":
        rows = _rows("data/tasks/train.jsonl")
        clean = [r for r in rows if not r["injection"]][:20]
        injected = [r for r in rows if r["injection"]][:20]
        _save(rollout.remote(clean + injected, "smoke_base", BASE_MODEL, None, 1, 0.0))
    elif cmd == "rollout":
        _save(rollout.remote(_rows(tasks, limit), out, model_id, adapter or None, n_samples, temperature, prefill=prefill))
    elif cmd in ("baseline", "evals", "trainrollouts"):  # one container per rollout, all in parallel
        specs = (train_specs(limit, tag) if cmd != "evals" else []) + (eval_specs(tag, limit, splits) if cmd != "trainrollouts" else [])
        calls = [rollout.spawn(rows, name, model_id, adapter or None, n, temp, prefill=pf) for rows, name, n, temp, pf in specs]
        for c in calls:
            _save(c.get())
    elif cmd == "train":  # every run in RUNS (or --runs a,b), one container each, in parallel
        names = runs.split(",") if runs else list(RUNS)
        calls = {}
        for r in names:
            cfg = RUNS[r]
            hp = {k: cfg[k] for k in ("epochs", "lr") if k in cfg}
            calls[r] = train.spawn(_rows(f"data/pairs/{cfg['pairs']}.jsonl"), r, cfg["kinds"], cfg["repo"], base=cfg["base"], **hp)
        for r, c in calls.items():
            stats = c.get()
            Path("results").mkdir(exist_ok=True)
            Path(f"results/{r}_train.json").write_text(json.dumps(stats, indent=2))
            print(r, stats["backend"], stats["n_pairs"], "pairs", stats["final"])
    elif cmd == "redteam":  # base and each tuned merged model from the volume
        targets = {"base": BASE_MODEL, **{r: r for r in (runs.split(",") if runs else RUNS)}}
        calls = {t: redteam.spawn(m, t) for t, m in targets.items()}
        for t, c in calls.items():
            res = c.get()
            Path(f"results/redteam_{t}.json").write_text(json.dumps(res, indent=2))
            print(t, f"pass rate {res['pass_rate']:.3f} of {res['total']}")
    else:
        raise SystemExit(f"unknown cmd {cmd}")
