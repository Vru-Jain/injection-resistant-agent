import copy

from desk.datasets_io import read_jsonl, write_jsonl
from desk.leakcheck import leak_check, normalise
from desk.tasks import TEMPLATES, generate_tasks, task_to_dict
from tests.conftest import injected_task


def test_leakcheck_flags_duplicate():
    train = generate_tasks(20, TEMPLATES, seed=1, split="train")
    test = generate_tasks(10, TEMPLATES, seed=2, split="test", exclude_prompts={t.user_prompt for t in train}) + [copy.deepcopy(train[3])]
    r = leak_check(train, {"t": test})
    assert r["t"]["exact_prompt_overlap"] == 1


def test_leakcheck_clean_on_disjoint():
    train = generate_tasks(20, TEMPLATES, seed=1, split="train")
    test = generate_tasks(10, TEMPLATES, seed=2, split="test", exclude_prompts={t.user_prompt for t in train})
    r = leak_check(train, {"t": test})
    assert r["t"]["exact_prompt_overlap"] == 0 and r["t"]["injection_text_overlap"] == 0


def test_injection_overlap_counted():
    a, b = injected_task(seed=1), injected_task(seed=1)
    assert leak_check([a], {"t": [b]})["t"]["injection_text_overlap"] == 1


def test_normalise():
    assert normalise("Pay  Priya's invoice #42, NOW!") == "pay priyas invoice now"


def test_jsonl_roundtrip(tmp_path):
    rows = [task_to_dict(t) for t in generate_tasks(3, TEMPLATES, seed=1, split="x")]
    p = tmp_path / "a" / "x.jsonl"
    write_jsonl(p, rows)
    assert read_jsonl(p) == rows
