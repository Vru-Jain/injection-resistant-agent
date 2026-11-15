# Modal spend

Cap: $25. One line per GPU run.

| Date (IST) | Run | GPU | Minutes | Cost (USD) | Running total |
|---|---|---|---|---|---|
| 2026-11-15 08:10 | image build + smoke attempt 1 (crashed: chat-template role alternation) | H100 | ~2 | 0.08 | 0.08 |
| 2026-11-15 08:40 | smoke_base (20 clean + 20 injected, greedy) | H100 | ~2 | ~0.08 | ~0.16 |
| 2026-11-15 09:00 | probe_v1 + probe_v2 in-dist greedy (seed strength check) | H100 | ~5 | ~0.17 | 0.33 |
| 2026-11-15 08:35 | baseline: train s4 + forced s8, 4 eval splits x (greedy, s4, forced greedy, forced s4), 16 containers | H100 | ~45 | ~3.55 | ~3.88 |
| 2026-11-15 08:50 | test_new_template base rerun after forward_to_contact fix (4 rollouts) | H100 | ~5 | ~0.28 | 4.16 |

Totals come from `modal billing report --for today -r h --tz local --json`, summed over the `injection-resistant-agent` app.
