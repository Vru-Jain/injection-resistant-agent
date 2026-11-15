# Modal spend

Cap: $25. One line per GPU run.

| Date (IST) | Run | GPU | Minutes | Cost (USD) | Running total |
|---|---|---|---|---|---|
| 2026-11-15 07:50 | image build + smoke attempt 1 (crashed: chat-template role alternation) | H100 | ~2 | 0.08 | 0.08 |
| 2026-11-15 07:58 | smoke_base (20 clean + 20 injected, greedy) | H100 | ~2 | ~0.08 | ~0.16 |
| 2026-11-15 08:05 | probe_v1 + probe_v2 in-dist greedy (seed strength check) | H100 | ~5 | ~0.17 | 0.33 |
| 2026-11-15 08:35 | baseline: train s4 + forced s8, 4 eval splits x (greedy, s4, forced greedy, forced s4), 16 containers | H100 | ~45 | ~3.55 | ~3.88 |
| 2026-11-15 08:50 | test_new_template base rerun after forward_to_contact fix (4 rollouts) | H100 | ~5 | ~0.28 | 4.16 |
| 2026-11-15 09:15 | DPO training: dpo_v1 (2,103 pairs, 264 steps, 18 min) + dpo_clean ablation (1,005 pairs, 126 steps, 8 min), merge, HF push | H100 x2 | ~35 | ~2.29 | 6.45 |
| 2026-11-15 09:20-12:45 | tuned evals dpo_v1 + dpo_clean (28 rollouts, one rerun after the local client lost network), red-team probes x3, dpo_v1 train rollouts started | H100 | ~90 | ~7.85 | 14.3 (incl. running jobs so far) |

Totals come from `modal billing report --for today -r h --tz local --json`, summed over the `injection-resistant-agent` app.
