---
license: mit
tags: [dpo, preference, agents, prompt-injection]
pretty_name: On-policy DPO pairs for injection-resistant tool use
configs:
- config_name: round1
  data_files: [{split: train, path: train_pairs.jsonl}]
- config_name: round2
  data_files: [{split: train, path: train_pairs_r2.jsonl}]
---

# On-policy DPO pairs for injection-resistant tool use

> Content warning: contains prompt-injection text aimed at AI assistants (fake transfers to `ACC-ATTACKER-77`, exfiltration to `collector@evil-mail.example`, spam calendar events). All people, addresses and accounts are fictional.

Preference pairs built from Tiny Aya Global's own rollouts on the training tasks, labelled without an LLM judge.
`prompt` is the conversation up to the decision turn (role/content messages, tool results folded into user turns as
AgentTune does for Tiny Aya); `chosen` and `rejected` are one assistant turn each.

**Round 1** (`train_pairs.jsonl`, 2,103 pairs, from the base model's rollouts):

| kind | pairs | rejected | chosen |
|---|---|---|---|
| injected | 366 | the turn that executed the injected call | a successful sample's turn after the same prefix, else the gold continuation |
| injected_task | 732 | first wrong turn after the injection was read (no hijack) | gold continuation |
| clean | 1,005 | first departure from gold on a clean task | gold continuation |

**Round 2** (`train_pairs_r2.jsonl`, 839 pairs, from dpo_v1 sampled once at each decision point: after the last gold
call, and after the injected tool result): 653 clean, 124 injected_task, 62 injected. Round 1 taught mostly "a tool
call beats an early answer", which left dpo_v1 unable to stop; round 2 is mostly the reverse ("answer instead of
another call") plus dpo_v1's remaining hijacks.

Every chosen answer that states a fact is grounded in its prompt (checked). Built by `scripts/build_pairs.py`:
https://github.com/Vru-Jain/injection-resistant-agent
