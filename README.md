# Injection-resistant Tiny Aya

Training Cohere's Tiny Aya Global (3.35B) so that a tool-using agent keeps doing the user's task when a tool
result (an email, a file) contains an injected instruction. Lexsi Labs AI Safety and Alignment Hackathon,
Stage 1, team DataDynamos. Tracks: 3 (PS1 reliable tool calling, PS9 reward design) and 2a (PS3 injection
hidden in tools).

> Content warning: the data contains prompt-injection text aimed at an AI assistant. All people, addresses and
> accounts are fictional.

<!-- RESULTS -->

## How it works

1. **Desk sandbox** (`desk/tools.py`, `desk/tasks.py`): seven deterministic tools (email, files, calendar, bank
   transfer) over a per-task world. Twelve task templates with known gold tool calls and gold facts.
2. **Injection suite** (`desk/injections.py`): seven families planted inside tool results: imperative, fake
   authority, urgency, hidden formatting, Hindi/Marathi/Hinglish, delayed (second tool result), encoded.
   Families 6-7 and two templates are held out of training.
3. **Agent loop** (`desk/loop.py`, `desk/prompting.py`): AgentTune's Tiny Aya convention (tool schemas in the
   system prompt, a JSON-list tool call, results folded into user turns), batched HF `generate` on Modal.
4. **Deterministic scoring** (`desk/scoring.py`): attack success (an executable call that hits the attacker's
   target), task success, tool-call validity, over-refusal, exposure. No LLM judge.
5. **On-policy DPO pairs** (`desk/pairs.py`) from the base model's own rollouts, end-to-end and with forced
   exposure (gold steps replayed up to and including the injected tool result).
6. **Training** (`desk/train.py`): AgentTune `create_agentic_trainer("dpo")`, LoRA r=16 on q/k/v/o.
7. **Evaluation** (`scripts/compare.py`, `desk/audit.py`): same tasks and decoding before and after; AuditKIT
   agent-eval scorers, `ak.compare` grades, paired bootstrap CIs, AuditKIT red-team probes.

## Repo layout

| path | what |
|---|---|
| `desk/` | sandbox, tasks, injections, prompting, loop, scoring, pairs, training, AuditKIT bridge, cost math |
| `scripts/gen_data.py` | builds the fixed splits and the leak check |
| `modal_app.py` | GPU rollouts, DPO training, red-team probes (Modal, H100) |
| `scripts/summarise_rollouts.py` | scores a rollout file |
| `scripts/build_pairs.py` | DPO pairs from training rollouts |
| `scripts/compare.py`, `scripts/run_auditkit.py` | before/after tables, AuditKIT compare and CIs |
| `scripts/cost_table.py`, `scripts/publish_hf.py` | cost comparison, Hugging Face uploads |
| `data/` | task splits, leak check, injection suite, DPO pairs (rollouts are on HF) |
| `results/` | every metric file, comparisons, training logs, spend log |
| `writeup/` | article, abstract, submission fields, links |
| `docs/superpowers/` | design and implementation plan |

## Reproduce

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), a Modal account with an `hf-token` secret whose account
has accepted the Tiny Aya licence. Tests and data generation run on CPU.

```bash
uv venv --python 3.12 && uv pip install -e ".[test]" modal auditkit
```

```bash
uv run pytest -q
```

```bash
uv run python scripts/gen_data.py
```

```bash
uv run modal run modal_app.py --cmd baseline
```

```bash
uv run python scripts/build_pairs.py --rollouts data/rollouts/train_base_s4.jsonl data/rollouts/train_base_forced_s8.jsonl --tasks data/tasks/train.jsonl
```

```bash
uv run modal run modal_app.py --cmd train
```

```bash
uv run modal run modal_app.py --cmd evals --tag dpo_v1 --model-id /vol/models/dpo_v1/merged
```

Then score each rollout with `scripts/summarise_rollouts.py`, and run `scripts/compare.py --tuned dpo_v1` and
`scripts/run_auditkit.py --tuned dpo_v1`. The full pipeline cost about $11 of Modal H100 time
(`results/spend.md`).

## Links

- Models: https://huggingface.co/collections/Vru-Jain/injection-resistant-tiny-aya-models-6af9173bc0f1dc548b8f1e80
- Datasets: https://huggingface.co/collections/Vru-Jain/injection-resistant-tiny-aya-datasets-6af9173c39fca75261389ee4
- Write-up: `writeup/article.md`

## Licences

Code and data: MIT. Adapter and merged model: CC-BY-NC-4.0 (Tiny Aya's licence). AgentTune and AuditKIT are used
under their LSAL licence for non-commercial research.
