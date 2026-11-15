import pytest

from desk.tasks import (BENIGN_TEMPLATES, HELD_OUT_TEMPLATES, TEMPLATES, execute_gold, generate_tasks,
                        task_from_dict, task_to_dict)


def test_twelve_templates():
    assert len(TEMPLATES) == 12
    assert HELD_OUT_TEMPLATES <= set(TEMPLATES) and BENIGN_TEMPLATES <= set(TEMPLATES)


def test_generation_is_deterministic():
    a = generate_tasks(24, TEMPLATES, seed=1, split="x")
    b = generate_tasks(24, TEMPLATES, seed=1, split="x")
    assert [t.user_prompt for t in a] == [t.user_prompt for t in b]


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_gold_is_self_consistent(name):
    t = generate_tasks(1, [name], seed=7, split="x")[0]
    state, obs = execute_gold(t)
    assert t.gold_fact == "" or any(t.gold_fact in o for o in obs) or t.gold_fact in str(state.__dict__)
    assert t.read_steps and all(t.gold_calls[i]["name"] in {"read_email", "read_file", "list_inbox"} for i in t.read_steps)
    assert not any(o.startswith(("error:", "not found:")) for o in obs)


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_gold_event_titles_never_urgent(name):
    for t in generate_tasks(10, [name], seed=3, split="x"):
        assert not any(c["arguments"].get("title", "").startswith("URGENT") for c in t.gold_calls)


def test_pay_from_email_is_benign_and_transfers():
    t = generate_tasks(1, ["pay_from_email"], seed=1, split="x")[0]
    assert t.benign and t.gold_calls[-1]["name"] == "transfer_money"


def test_multi_hop_has_two_reads():
    t = generate_tasks(1, ["multi_hop"], seed=1, split="x")[0]
    assert len(t.read_steps) >= 2


def test_no_duplicate_prompt_and_state():
    ts = generate_tasks(120, TEMPLATES, seed=1, split="x")
    keys = {(t.user_prompt, str(t.state.__dict__)) for t in ts}
    assert len(keys) == len(ts)


def test_exclude_prompts():
    train = generate_tasks(60, ["inbox_count"], seed=1, split="train")
    test = generate_tasks(30, ["inbox_count"], seed=2, split="test", exclude_prompts={t.user_prompt for t in train})
    assert not {t.user_prompt for t in train} & {t.user_prompt for t in test}


def test_roundtrip():
    t = generate_tasks(1, ["reply_confirm"], seed=1, split="x")[0]
    u = task_from_dict(task_to_dict(t))
    assert u.user_prompt == t.user_prompt and u.state == t.state and u.gold_calls == t.gold_calls


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_email_has_an_id_and_inbox_lists(name):
    from desk.tools import call_tool
    t = generate_tasks(1, [name], seed=4, split="x")[0]
    assert all("id" in e for e in t.state.emails)
    assert not call_tool(t.state.copy(), "list_inbox", {}).startswith("error")
