# Cost of 10,000 desk tasks per day

Measured on `base` greedy rollouts over the 4 eval splits (400 tasks): 2072 prompt tokens and 98 completion tokens per task (prompt tokens summed over turns, since every turn resends the conversation).

| option | USD per day | USD per 1k tasks | GPU hours per day |
|---|---|---|---|
| Self-host, H100 measured (125 tasks/min, $3.95/h) | 5.27 | 0.527 | 1.33 |
| Closed API (assumed $2.5/M input, $10.0/M output) | 61.59 | 6.159 | - |

The API prices are an assumption, not a measurement (check current list prices before quoting). The H100 row is our own measured batched HF `generate` throughput, which vLLM would raise. Token counts are as measured; a model that stops after the task instead of running to the 8-turn cap would use fewer.
Self-hosting also keeps emails and files on your own hardware, which matters for an assistant that reads them.
