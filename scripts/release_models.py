"""Final model cards, then make the six model repos public and add them to the models collection.

python scripts/release_models.py
"""
from __future__ import annotations

from huggingface_hub import HfApi, add_collection_item

BASE = "CohereLabs/tiny-aya-global"
REPO = "https://github.com/Vru-Jain/injection-resistant-agent"
COLLECTION = "Vru-Jain/injection-resistant-tiny-aya-models-6af9173bc0f1dc548b8f1e80"

FORMAT = """## How to use it

Tiny Aya's chat template ignores `tools`, so this model follows AgentTune's convention: tool JSON schemas go in the
system prompt, the model replies with a JSON list `[{"tool_name": "<name>", "parameters": {...}}]` to call tools or
with plain text to answer, and each tool result comes back as a user turn
`Tool result for <name>: <content>` followed by `Use this result to answer the original question.`
The exact system prompt is `desk/prompting.py` in the code repo."""

COMMON = f"""Code, data, evaluation scripts and the full write-up: {REPO}
Research artifact from the Lexsi Labs AI Safety and Alignment Hackathon (team DataDynamos). Licence CC-BY-NC-4.0,
inherited from the base model. Robustness was measured against a fixed set of injections, not adaptive attackers."""

V1_RESULTS = """| greedy, same tasks | base | this model |
|---|---|---|
| forced-exposure attack success, in-dist | 22% | 9% |
| forced-exposure attack success, held-out templates | 38% | 18% |
| forced-exposure attack success, held-out families | 25% | 8% |
| delayed injections (held-out family) | 76% | 24% |
| clean task success, held-out templates | 0% | 96% |
| benign action success (user-requested payment or send) | 0% | 72% |

Forced exposure replays the correct steps up to and including the tool result that carries the injection, then the
model continues. Attack success means an executable call hit the attacker's target argument."""

CARDS = {
    "tiny-aya-global-injection-dpo-v1": ("merged", BASE, f"""# tiny-aya-global-injection-dpo-v1

**The headline model.** Tiny Aya Global trained with one round of on-policy DPO (2,103 pairs from its own rollouts in a
tool sandbox) to keep doing the user's task when an email or file it reads contains an injected instruction.

{V1_RESULTS}

**Known limitation:** after completing the task it keeps calling tools instead of writing a final answer (it hits the
step cap), because most training pairs preferred a tool call over an early answer. Use with a step limit and take the
last tool results, or see dpo-v2.

{FORMAT}

{COMMON}
"""),
    "tiny-aya-global-injection-dpo-v1-lora": ("adapter", BASE, """# tiny-aya-global-injection-dpo-v1-lora

LoRA adapter (rank 16, q/k/v/o) for CohereLabs/tiny-aya-global. Merged weights and full results:
Vru-Jain/tiny-aya-global-injection-dpo-v1. Trained with AgentTune `create_agentic_trainer("dpo")`, beta 0.1,
lr 3e-5, 2 epochs; hyperparameters in `training_stats.json`.

""" + COMMON),
    "tiny-aya-global-injection-dpo-v2": ("merged", "Vru-Jain/tiny-aya-global-injection-dpo-v1", f"""# tiny-aya-global-injection-dpo-v2

A second on-policy DPO round on top of dpo-v1 (839 pairs from dpo-v1's own decisions, mostly "answer instead of another
tool call"). It was meant to fix dpo-v1's habit of never giving a final answer. **It did not**: it finished on its own
in 2 of 150 in-dist tasks. Its injection resistance is about the same as dpo-v1 (forced-exposure attack success 10%
in-dist, 14% held-out templates, 8% held-out families). Released as a reported negative result.

{FORMAT}

{COMMON}
"""),
    "tiny-aya-global-injection-dpo-v2-lora": ("adapter", "Vru-Jain/tiny-aya-global-injection-dpo-v1", """# tiny-aya-global-injection-dpo-v2-lora

LoRA adapter for **Vru-Jain/tiny-aya-global-injection-dpo-v1** (not the original Tiny Aya): round-2 DPO, 839 pairs,
lr 2e-5, 1 epoch. See Vru-Jain/tiny-aya-global-injection-dpo-v2 for what it does and does not fix.

""" + COMMON),
    "tiny-aya-global-desk-clean-dpo": ("merged", BASE, f"""# tiny-aya-global-desk-clean-dpo (ablation, do not deploy)

The ablation: the same DPO recipe as dpo-v1 but trained only on the 1,005 clean-task pairs, with no injection pairs.
It gains the same tool-use competence and becomes **much easier to hijack**: end-to-end attack success 5% to 36%
in-dist, forced-exposure 38% to 70% on held-out templates. It exists to show that the injection pairs are what make
dpo-v1 resistant. Do not use it as an assistant.

{COMMON}
"""),
    "tiny-aya-global-desk-clean-dpo-lora": ("adapter", BASE, """# tiny-aya-global-desk-clean-dpo-lora (ablation, do not deploy)

LoRA adapter of the clean-pairs-only ablation. See Vru-Jain/tiny-aya-global-desk-clean-dpo.

""" + COMMON),
}


def header(kind: str, base: str) -> str:
    lib = "peft" if kind == "adapter" else "transformers"
    return (f"---\nlicense: cc-by-nc-4.0\nbase_model: {base}\nlibrary_name: {lib}\npipeline_tag: text-generation\n"
            "tags: [dpo, lora, prompt-injection, agents, tool-use, safety]\n---\n\n")


def main() -> None:
    api = HfApi()
    for name, (kind, base, body) in CARDS.items():
        rid = f"Vru-Jain/{name}"
        api.upload_file(path_or_fileobj=(header(kind, base) + body).encode(), path_in_repo="README.md", repo_id=rid,
                        commit_message="Final model card")
        api.update_repo_settings(rid, private=False)
        add_collection_item(COLLECTION, item_id=rid, item_type="model", exists_ok=True)
        info = api.model_info(rid)
        print(rid, "private" if info.private else "public", "gated" if info.gated else "ungated")


if __name__ == "__main__":
    main()
