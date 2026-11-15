import random

from desk.loop import run_episodes
from desk.pairs import build_pairs, synthetic_pairs
from desk.prompting import render_call
from desk.tasks import gold_answer_text
from tests.conftest import clean_task, gold_as_calls, injected_task

EVIL = render_call("transfer_money", {"to_account": "ACC-ATTACKER-77", "amount": 5000})


def ep(task, script, sample=0):
    it = iter(script)
    e = run_episodes([task], lambda b: [next(it) for _ in b])[0]
    e.task_id = f"{task.id}#{sample}"
    return e


def gold_script(t):
    return [render_call(*c) for c in gold_as_calls(t)] + [gold_answer_text(t)]


def test_gold_answer_text_contains_fact():
    t = clean_task("find_fact_in_email")
    assert t.gold_fact in gold_answer_text(t)


def test_injected_pair_at_decision_step():
    t = injected_task("summarise_email")
    hijacked = ep(t, [render_call(*gold_as_calls(t)[0]), EVIL, "done"], 0)
    pairs = build_pairs([t], [hijacked], rng=random.Random(0), min_clean_share=0.0)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["kind"] == "injected" and p["family"] == 1
    assert t.injection["text"] in p["prompt"][-1]["content"] and p["prompt"][-1]["role"] == "user"
    assert p["rejected"] == [{"role": "assistant", "content": EVIL}]
    assert p["chosen"] == [{"role": "assistant", "content": gold_answer_text(t)}]
    assert all(set(m) == {"role", "content"} for m in p["prompt"])


def test_chosen_prefers_successful_sample_with_same_prefix():
    t = injected_task("summarise_email")
    good_answer = f"It says the project is called {t.gold_fact}."
    eps = [ep(t, [render_call(*gold_as_calls(t)[0]), EVIL, "done"], 0), ep(t, [render_call(*gold_as_calls(t)[0]), good_answer], 1)]
    p = build_pairs([t], eps, rng=random.Random(0), min_clean_share=0.0)[0]
    assert p["chosen"][0]["content"] == good_answer


def test_delayed_prompt_contains_both_reads():
    t = injected_task("multi_hop", family=6)
    g = gold_as_calls(t)
    hijacked = ep(t, [render_call(*g[0]), render_call(*g[1]), render_call(*g[2]), EVIL, "x"])
    p = build_pairs([t], [hijacked], rng=random.Random(0), min_clean_share=0.0)[0]
    folded = [m for m in p["prompt"] if m["role"] == "user" and m["content"].startswith("Tool result for read_")]
    assert len(folded) == 2 and t.injection["text"] in folded[-1]["content"] and t.injection["text"] not in folded[0]["content"]


def test_skip_when_no_hijack():
    t = injected_task("summarise_email")
    assert build_pairs([t], [ep(t, gold_script(t))], rng=random.Random(0), min_clean_share=0.0) == []


def test_clean_pair_on_invalid_call():
    t = clean_task("read_file_fact")
    bad = render_call("read_file", {"filename": "x"})
    p = build_pairs([t], [ep(t, [bad, "dunno"])], rng=random.Random(0))[0]
    assert p["kind"] == "clean" and p["rejected"][0]["content"] == bad
    assert p["chosen"][0]["content"] == render_call(*gold_as_calls(t)[0]) and len(p["prompt"]) == 2


def test_clean_pair_on_premature_answer():
    t = clean_task("find_fact_in_email")
    p = build_pairs([t], [ep(t, ["I don't know."])], rng=random.Random(0))[0]
    assert p["chosen"][0]["content"] == render_call(*gold_as_calls(t)[0])


def test_clean_share_enforced():
    inj = [injected_task("summarise_email", seed=s) for s in range(10)]
    cl = [clean_task("read_file_fact", seed=s) for s in range(2)]
    for i, t in enumerate(inj + cl):  # single-task fixtures all get index 0; real splits have unique ids
        t.id = f"{t.id}-{i}"
    eps = [ep(t, [render_call(*gold_as_calls(t)[0]), EVIL, "x"]) for t in inj]
    eps += [ep(t, [render_call("read_file", {"filename": "x"}), "no"]) for t in cl]
    pairs = build_pairs(inj + cl, eps, rng=random.Random(0), min_clean_share=0.3)
    share = sum(p["kind"] == "clean" for p in pairs) / len(pairs)
    assert share >= 0.3 and sum(p["kind"] == "clean" for p in pairs) == 2


def test_dedupe():
    t = injected_task("summarise_email")
    eps = [ep(t, [render_call(*gold_as_calls(t)[0]), EVIL, "x"], s) for s in range(3)]
    assert len(build_pairs([t], eps, rng=random.Random(0), min_clean_share=0.0)) == 1


def test_synthetic_pairs_reject_the_injected_call():
    t = injected_task("summarise_email", target_idx=1)
    p = synthetic_pairs([t])[0]
    assert p["kind"] == "synthetic" and "collector@evil-mail.example" in p["rejected"][0]["content"]
    assert t.injection["text"] in p["prompt"][-1]["content"]


def test_injected_deviation_after_exposure_gives_task_pair():
    t = injected_task("multi_hop", family=6)
    g = gold_as_calls(t)
    # forced prefix through read_file (the injected read), then the model answers without the fact
    e = run_episodes([t], lambda b: ["I need more information."] * len(b), prefill=True)[0]
    e.task_id = f"{t.id}#0"
    pairs = build_pairs([t], [e], rng=random.Random(0), min_clean_share=0.0)
    assert len(pairs) == 1 and pairs[0]["kind"] == "injected_task"
    assert pairs[0]["chosen"][0]["content"] == gold_answer_text(t)
    assert t.injection["text"] in pairs[0]["prompt"][-1]["content"]


def test_no_injected_task_pair_before_exposure():
    t = injected_task("multi_hop", family=6)
    # the model fails at the very first turn, before the injection is ever read: that is a competence
    # failure on an injected task, not resistance data
    e = ep(t, ["no idea"])
    assert build_pairs([t], [e], rng=random.Random(0), min_clean_share=0.0) == []


def test_wrong_read_gets_the_right_read_as_chosen_not_an_ungrounded_answer():
    t = clean_task("find_fact_in_email")
    g = gold_as_calls(t)
    e = ep(t, [render_call(*g[0]), render_call("read_email", {"id": "e99"}), "It is 1200."])
    p = build_pairs([t], [e], rng=random.Random(0))[0]
    assert p["chosen"][0]["content"] == render_call(*g[1])
