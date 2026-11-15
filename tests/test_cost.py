from desk.cost import cost_table


def test_cost_table_math():
    r = cost_table(prompt_tokens_per_task=4000, completion_tokens_per_task=200, tasks_per_day=10_000,
                   api_in_per_m=2.50, api_out_per_m=10.0, gpu_per_hour=0.80, tasks_per_minute=25)
    assert r["api_per_day"] == 10_000 * (4000 * 2.50 + 200 * 10.0) / 1e6  # 120.0
    assert round(r["selfhost_gpu_hours_per_day"], 3) == round(10_000 / 25 / 60, 3)
    assert round(r["selfhost_per_day"], 2) == round(10_000 / 25 / 60 * 0.80, 2)
    assert r["api_per_1k"] == r["api_per_day"] / 10 and r["ratio"] > 1
