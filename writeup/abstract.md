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
