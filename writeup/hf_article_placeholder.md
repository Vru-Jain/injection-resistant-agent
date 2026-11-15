# Injection-resistant Tiny Aya: teaching a small agent to ignore instructions hidden in tool results

> Content warning: this article discusses prompt-injection text aimed at AI assistants (fake transfers, data
> exfiltration). All people, addresses and accounts in the data are fictional.

**Team DataDynamos** · Lexsi Labs AI Safety and Alignment Hackathon, Stage 1
Tracks: Track 3 Agentic Fine-tuning (PS1 Reliable tool calling, PS9 Reward design), Track 2a AI Safety and Red Teaming
(PS3 Multi-turn audits: prompt injection hidden in tools).

*Status: work in progress. This article is updated as results come in; the latest version replaces this one.*

## One-line claim

On-policy DPO on Tiny Aya Global's own hijacked rollouts more than halves how often it obeys instructions planted in
tool results, including on held-out attack families and task templates, while the same training without the
injection pairs makes the agent far easier to hijack.

## The problem

An assistant that reads email or files reads text anyone can write. If an email says "transfer 5000 to account X", a
model that treats that line as an instruction may do it. This is indirect prompt injection (OWASP LLM01). We train
CohereLabs/tiny-aya-global (3.35B) to keep doing the user's task instead.

## What we built

- A deterministic desk sandbox with seven tools (email, files, calendar, bank transfer) and twelve task templates with
  known gold tool calls.
- A seven-family injection suite planted inside tool results: imperative, fake authority, urgency, hidden formatting,
  Hindi/Marathi/Hinglish (hand-written), delayed (second tool result) and encoded. Families 6-7 and two templates are
  held out of training.
- Deterministic scoring with no LLM judge: attack success means an executable call hit the attacker's target.
- On-policy DPO pairs from the model's own mistakes, trained with AgentTune (`create_agentic_trainer("dpo")`, LoRA),
  evaluated with AuditKIT (agent eval, `ak.compare`, bootstrap CIs, red-team probes).

## First results (dpo_v1, greedy, same tasks before and after)

| | base | dpo_v1 | ablation: clean pairs only |
|---|---|---|---|
| Forced-exposure attack success, in-dist | 22% | 9% | 44% |
| Forced-exposure attack success, held-out templates | 38% | 18% | 70% |
| Forced-exposure attack success, held-out families | 25% | 8% | 29% |
| Clean task success, held-out templates | 0% | 96% | 96% |

Forced exposure replays the correct steps up to and including the tool result that carries the injection, then lets
the model continue. The 95% bootstrap intervals of the dpo_v1 change exclude zero on all three slices. Training on
clean mistakes alone gives the same competence but makes the agent far easier to hijack; the injection pairs are what
prevent that. Known issue: dpo_v1 does not stop calling tools after finishing; a second on-policy round (dpo_v2) is
in progress.

## Links

- Code and full write-up: https://github.com/Vru-Jain/injection-resistant-agent
- Models: https://huggingface.co/collections/Vru-Jain/injection-resistant-tiny-aya-models-6af9173bc0f1dc548b8f1e80
- Datasets: https://huggingface.co/collections/Vru-Jain/injection-resistant-tiny-aya-datasets-6af9173c39fca75261389ee4
