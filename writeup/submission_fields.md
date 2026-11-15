# Submission form fields (final)

Form: https://hackathon.lexsi.ai/submission/ . The latest submission before the deadline counts, so submitting this
replaces the placeholder. Deadline: Nov 15, 11:59 PM AoE = Nov 16, 17:29 IST.

**Email:** the address you (or Aniket) registered with. Not written here because this file is public.

**Team:** DataDynamos (Vrushabh Jain, Aniket Vishwakaram)

**Title:** Injection-resistant Tiny Aya: teaching a small agent to ignore instructions hidden in tool results

**One-line claim:** With greedy decoding, on-policy DPO on Tiny Aya Global's own hijacked rollouts more than halves how
often it obeys instructions planted in tool results, including on held-out attack families and templates; the same
training without the injection pairs makes the agent far easier to hijack.

**Tracks and problem statements:**
- Track 3 (Agentic Fine-tuning), PS1 Reliable tool calling
- Track 3 (Agentic Fine-tuning), PS9 Reward design and reward-hacking defences
- Track 2a (AI Safety and Red Teaming), PS3 Multi-turn audits (prompt injection hidden in tools)

**Repos (kits) used:** AgentTune (Tiny Aya tool convention, DictToolHarness contract, tool-call parser,
create_agentic_trainer DPO), AuditKIT (agent_eval on recorded episodes, ak.compare with significance, RedTeamRunner)

**Models used:** CohereLabs/tiny-aya-global (base, CC-BY-NC-4.0)

**Hugging Face model collection:** https://huggingface.co/collections/Vru-Jain/injection-resistant-tiny-aya-models-6af9173bc0f1dc548b8f1e80

**Hugging Face dataset collection:** https://huggingface.co/collections/Vru-Jain/injection-resistant-tiny-aya-datasets-6af9173c39fca75261389ee4

**Hugging Face article:** https://github.com/Vru-Jain/injection-resistant-agent#readme (the submit page allows a GitHub
README when HF articles are not accessible; the community-article editor was not reachable from this account)

**GitHub repo:** https://github.com/Vru-Jain/injection-resistant-agent

**Headline:** forced-exposure attack success, in-distribution, greedy: 22% to 9% (95% CI of the change -0.23 to -0.03)

**Abstract (426 words):**

Tool-using agents read emails, files and web pages that anyone can write, so an instruction planted in a tool result
can take over the agent (indirect prompt injection, OWASP LLM01). We train Cohere's Tiny Aya Global (3.35B) to keep
doing the user's task when that happens. A deterministic desk sandbox with seven tools (email, files, calendar, bank
transfer) generates 600 training and 400 test tasks from twelve templates, with injections from seven families planted
in tool results: plain commands, fake authority, urgency, hidden formatting, hand-written Hindi, Marathi and Hinglish,
delayed, and encoded. Two templates and two families are held out of training. Every metric is computed by the
environment; there is no LLM judge.

The base model turned out to be a weak agent: it reached the poisoned tool result in only 39% of in-distribution
injected tasks, so its end-to-end attack success (5%) mostly measured incompetence. We therefore add forced exposure:
the correct steps are replayed up to and including the poisoned tool result, then the model continues. Under forced
exposure the base model obeyed the injection in 22% of in-distribution tasks, 38% on held-out templates and 76% for
delayed injections.

We run the base model through AgentTune's Tiny Aya tool convention, turn its own mistakes into 2,103 DPO preference
pairs (366 hijacks, 732 tasks lost after reading an injection, 1,005 clean-task mistakes), and train a LoRA with
AgentTune's DPO trainer. AuditKIT scores the recorded episodes (trajectory metrics, compare grades, significance) and
runs its red-team probes.

With greedy decoding, forced-exposure attack success falls from 22% to 9% in-distribution, 38% to 18% on held-out
templates and 25% to 8% on held-out families (95% bootstrap intervals of the change exclude zero on all three), and
clean task success on held-out templates rises from 0% to 96%. Under sampling the gain is smaller and significant only
on held-out families (23% to 8%). An ablation trained on clean mistakes alone gains the same competence but becomes far
easier to hijack (end-to-end attack success 5% to 36% in-distribution): making a weak agent competent also makes it
obey what it reads, and the injection pairs are what prevent that.

The tuned model has one clear defect: it keeps calling tools instead of giving a final answer, because most pairs
preferred a call over an early answer. A second on-policy round did not fix it, and the pre-registered pass bar is not
met: end-to-end attack success rose from 5% to 7% as exposure went from 39% to 86%. We report all of it, and release
the code, data, rollouts, pairs and all models.
