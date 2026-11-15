---
license: mit
tags: [agents, tool-use, prompt-injection, trajectories]
pretty_name: Tiny Aya desk-agent rollouts (before and after DPO)
---

# Tiny Aya desk-agent rollouts

> Content warning: contains prompt-injection text aimed at AI assistants (fake transfers to `ACC-ATTACKER-77`, exfiltration to `collector@evil-mail.example`, spam calendar events). All people, addresses and accounts are fictional.

Recorded episodes of CohereLabs/tiny-aya-global (`base`) and the DPO-tuned models (`dpo_v1`, round-2 `dpo_v2`, ablation
`dpo_clean`) on the desk-sandbox tasks: OpenAI-format messages (assistant turns carry tool_calls), every executed
call with its observation and validity, the final answer, and the final world state. File names are
`<split>_<model>_<mode>.jsonl` with modes `greedy`, `s4` (4 samples, T=0.7), `forced_greedy` and `forced_s4`
(gold steps replayed up to and including the injected tool result, then the model continues). `train_base_s4` and
`train_base_forced_s8` are the training rollouts the DPO pairs come from. `*.stats.json` hold token counts and
GPU time. Scored with `scripts/summarise_rollouts.py` in https://github.com/Vru-Jain/injection-resistant-agent
