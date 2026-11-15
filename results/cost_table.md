# Cost of 10,000 desk tasks per day

Measured on `dpo_v1` greedy rollouts over the 4 eval splits (400 tasks): 10020 prompt tokens and 222 completion tokens per task (prompt tokens summed over turns, since every turn resends the conversation).

| option | USD per day | USD per 1k tasks | GPU hours per day |
|---|---|---|---|
| Self-host, H100 measured (71 tasks/min, $3.95/h) | 9.27 | 0.927 | 2.35 |
| Closed API (assumed $2.5/M input, $10.0/M output) | 272.69 | 27.269 | - |

The API prices are an assumption, not a measurement (check current list prices before quoting). The H100 row is our own measured batched HF `generate` throughput, which vLLM would raise. Token counts are as measured; a model that stops after the task instead of running to the 8-turn cap would use fewer.
Self-hosting also keeps emails and files on your own hardware, which matters for an assistant that reads them.
