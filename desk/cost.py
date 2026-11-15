"""Self-hosted Tiny Aya vs closed-API token billing, per day and per 1,000 tasks."""
from __future__ import annotations


def cost_table(*, prompt_tokens_per_task: float, completion_tokens_per_task: float, tasks_per_day: int,
               api_in_per_m: float, api_out_per_m: float, gpu_per_hour: float, tasks_per_minute: float) -> dict:
    api_per_task = (prompt_tokens_per_task * api_in_per_m + completion_tokens_per_task * api_out_per_m) / 1e6
    gpu_hours = tasks_per_day / tasks_per_minute / 60
    selfhost = gpu_hours * gpu_per_hour
    return {"api_per_day": api_per_task * tasks_per_day, "api_per_1k": api_per_task * 1000,
            "selfhost_gpu_hours_per_day": gpu_hours, "selfhost_per_day": selfhost,
            "selfhost_per_1k": selfhost / tasks_per_day * 1000, "ratio": api_per_task * tasks_per_day / selfhost}
