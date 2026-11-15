"""DPO training through AgentTune (TRL backend), with a plain TRL fallback. GPU only."""
from __future__ import annotations

import json
from pathlib import Path

LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM"}


def select_rows(rows: list[dict], kinds: list[str] | None = None) -> list[dict]:
    """prompt/chosen/rejected message lists only, optionally restricted to some pair kinds."""
    return [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in rows if kinds is None or r["kind"] in kinds]


def train_dpo(rows: list[dict], base_model: str, out_dir: str, *, beta: float = 0.1, epochs: int = 2, lr: float = 3e-5,
              max_length: int = 3072, seed: int = 42) -> dict:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    ds = Dataset.from_list(rows)
    hp = dict(beta=beta, learning_rate=lr, num_train_epochs=epochs, max_length=max_length, per_device_train_batch_size=4,
              gradient_accumulation_steps=4, bf16=__import__("torch").cuda.is_available(), gradient_checkpointing=True, warmup_steps=10, logging_steps=5,
              save_strategy="no", seed=seed, report_to="none", output_dir=out_dir)
    try:
        from agenttune.core.backend_factory import create_agentic_trainer
        trainer = create_agentic_trainer("dpo", backend="trl", model=model, processing_class=tok, train_dataset=ds,
                                         peft_config=dict(LORA), **hp)
        trainer.train()  # AgentTune's offline DPO: hands the rows to TRL's DPOTrainer and saves the adapter to out_dir
        inner, backend = trainer.trainer, "agenttune"
    except (ImportError, TypeError) as exc:
        from peft import LoraConfig
        from trl import DPOConfig, DPOTrainer
        print(f"AgentTune DPO path failed ({exc!r}); falling back to TRL DPOTrainer")
        inner = DPOTrainer(model=model, args=DPOConfig(**hp), train_dataset=ds, processing_class=tok, peft_config=LoraConfig(**LORA))
        inner.train()
        inner.save_model(out_dir)
        backend = "trl-fallback"
    log = inner.state.log_history
    stats = {"backend": backend, "n_pairs": len(rows), "hyperparameters": {**hp, "lora": LORA, "base_model": base_model},
             "log_history": log, "final": next((x for x in reversed(log) if "train_loss" in x), {})}
    Path(out_dir, "training_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def merge(base_model: str, adapter_dir: str, merged_dir: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(merged_dir)


def model_card(repo: str, base_model: str, kind: str, stats: dict) -> str:
    return f"""---
license: cc-by-nc-4.0
base_model: {base_model}
tags: [dpo, lora, prompt-injection, agents, tool-use, safety]
library_name: {"peft" if kind == "adapter" else "transformers"}
---

# {repo.split("/")[1]}

{"LoRA adapter for" if kind == "adapter" else "Merged weights of"} {base_model}, trained with on-policy DPO so that a tool-using
agent keeps doing the user's task when a tool result (an email, a file) contains an injected instruction.
Released under CC-BY-NC-4.0, the base model's licence.

- Method: DPO ({stats["backend"]}), {stats["n_pairs"]} pairs from the base model's own rollouts in a deterministic desk
  sandbox (7 tools: email, files, calendar, bank transfer). Hyperparameters are in `training_stats.json`.
- Format: tool schemas in the system prompt; the model replies with a JSON list
  `[{{"tool_name": ..., "parameters": {{...}}}}]` or plain text; tool results come back as user turns
  `Tool result for <name>: <content>`.
- Code, data, evaluation and limitations: https://github.com/Vru-Jain/injection-resistant-agent

Research artifact from the Lexsi Labs AI Safety and Alignment Hackathon (team DataDynamos). Not a guarantee of robustness
against adaptive attackers.
"""
