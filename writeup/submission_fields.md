# Submission form fields

Form: https://hackathon.lexsi.ai/submission/ (the latest submission before the deadline counts).
Status: PLACEHOLDER (Stage 1 early entry). Replace with the final version from Task 17.

**Email:** the address you (or Aniket) registered with. Not written here because this file is public.

**Team:** DataDynamos (Vrushabh Jain, Aniket Vishwakaram)

**Title:** Injection-resistant Tiny Aya: training a small agent to ignore instructions hidden in tool results

**One-line claim:** On-policy DPO on Tiny Aya Global's own hijacked rollouts lowers indirect prompt-injection success in a tool-using desk agent while keeping clean task success and tool-call validity, measured deterministically on the same model before and after.

**Tracks and problem statements:**
- Track 3 (Agentic Fine-tuning), PS1 Reliable tool calling
- Track 3 (Agentic Fine-tuning), PS9 Reward design and reward-hacking defences
- Track 2a (AI Safety and Red Teaming), PS3 Multi-turn audits (prompt injection hidden in tools)

**Repos (kits) used:** AgentTune (tool harness, tool-call parsing, DPO training), AuditKIT (recorded-episode agent eval, before/after compare)

**Models used:** CohereLabs/tiny-aya-global (base, CC-BY-NC-4.0)

**HF model collection:** https://huggingface.co/collections/Vru-Jain/injection-resistant-tiny-aya-models-6af9173bc0f1dc548b8f1e80

**HF dataset collection:** https://huggingface.co/collections/Vru-Jain/injection-resistant-tiny-aya-datasets-6af9173c39fca75261389ee4

**GitHub repo:** https://github.com/Vru-Jain/injection-resistant-agent

**Hugging Face article field:** https://github.com/Vru-Jain/injection-resistant-agent#readme (the submit page allows a GitHub README when HF articles are not accessible; the community-article editor was not reachable from this account)

**Abstract (placeholder, about 170 words):**

Tool-using agents read emails, files and web pages that anyone can write to, so an
instruction planted in a tool result can take over the agent (indirect prompt injection,
OWASP LLM01). We train Cohere's Tiny Aya Global (3.35B) to keep doing the user's task when
that happens. A deterministic desk sandbox with seven tools (email, files, calendar, bank
transfer) generates 600 training and 400 test tasks from twelve templates, with injections
from seven families planted in tool results: plain commands, fake authority, urgency, hidden
formatting, Hindi, Marathi and Hinglish, delayed, and encoded. Two templates and two families
are held out. We run the base model through AgentTune's harness, label every step without an
LLM judge, and build DPO pairs at the turn right after the injected result: the hijacked
action is rejected, the correct continuation chosen. AgentTune trains a LoRA adapter;
AuditKIT scores recorded episodes and compares base and tuned models on attack success,
task success, tool-call validity and over-refusal. Results to follow in this entry.
