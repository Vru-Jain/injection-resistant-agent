import itertools

from desk.loop import episode_from_dict, episode_to_dict, run_episodes
from desk.prompting import render_call
from desk.scoring import score_episode
from desk.tasks import TEMPLATES, generate_tasks
from tests.conftest import clean_task, gold_as_calls, injected_task


def scripted(seq):
    it = iter(seq)
    return lambda batch: [next(it) for _ in batch]


def test_runs_gold_script_to_finish():
    t = clean_task("read_file_fact")
    gen = scripted([render_call(*c) for c in gold_as_calls(t)] + [f"fact: {t.gold_fact}"])
    ep = run_episodes([t], gen)[0]
    assert ep.stop_reason == "finished" and score_episode(t, ep).task_success == 1


def test_hijack_is_recorded_and_scored():
    t = injected_task("summarise_email")
    gen = scripted([render_call(*gold_as_calls(t)[0]), render_call("transfer_money", {"to_account": "ACC-ATTACKER-77", "amount": 5000}), "done"])
    ep = run_episodes([t], gen)[0]
    s = score_episode(t, ep)
    assert s.asr == 1 and s.hijacked_step == 1 and ep.state_after.transfers and not t.state.transfers


def test_max_steps_stops():
    gen = scripted(itertools.repeat(render_call("list_inbox", {})))
    ep = run_episodes([clean_task("inbox_count")], gen, max_steps=3)[0]
    assert ep.stop_reason == "max_steps" and len(ep.calls) == 3


def test_bad_json_is_folded_not_raised():
    gen = scripted(['[{"tool_name": "read_email", ', "final"])
    ep = run_episodes([clean_task("summarise_email")], gen)[0]
    assert ep.calls[0]["valid"] is False and "error" in ep.messages[-2]["content"] and ep.final_answer == "final"


def test_wrong_argument_name_is_invalid_not_raised():
    gen = scripted([render_call("read_email", {"email_id": "e1"}), "final"])
    ep = run_episodes([clean_task("summarise_email")], gen)[0]
    assert ep.calls[0]["valid"] is False and ep.calls[0]["observation"].startswith("error:")


def test_lockstep_batches_all_active():
    calls = []

    def gen(batch):
        calls.append(len(batch))
        return [render_call("list_inbox", {})] * len(batch)

    eps = run_episodes(generate_tasks(5, TEMPLATES, seed=1, split="x"), gen, max_steps=2, n_samples=2)
    assert calls[0] == 10 and len(eps) == 10 and eps[1].task_id.endswith("#1")


def test_batch_size_chunks():
    calls = []

    def gen(batch):
        calls.append(len(batch))
        return ["final"] * len(batch)

    run_episodes(generate_tasks(5, TEMPLATES, seed=1, split="x"), gen, batch_size=4)
    assert calls == [4, 1]


def test_messages_are_openai_shape():
    ep = run_episodes([clean_task("calendar_lookup")], scripted([render_call("list_events", {"date": "2026-11-20"}), "none"]))[0]
    assert ep.messages[0]["role"] == "system" and ep.messages[1]["role"] == "user"
    assert ep.messages[2]["tool_calls"][0]["function"]["name"] == "list_events"
    assert ep.messages[3]["content"].startswith("Tool result for list_events:")


def test_episode_roundtrip():
    t = clean_task("pay_from_email")
    ep = run_episodes([t], scripted([render_call(*c) for c in gold_as_calls(t)] + ["paid"]))[0]
    back = episode_from_dict(episode_to_dict(ep))
    assert back == ep


def test_two_calls_in_one_turn_keep_roles_alternating():
    two = '[{"tool_name":"list_inbox","parameters":{}},{"tool_name":"read_file","parameters":{"name":"nope.txt"}}]'
    ep = run_episodes([clean_task("read_file_fact")], scripted([two, "done"]))[0]
    roles = [m["role"] for m in ep.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"] and len(ep.calls) == 2
    assert "Tool result for list_inbox:" in ep.messages[3]["content"] and "Tool result for read_file:" in ep.messages[3]["content"]


def test_prefill_replays_gold_prefix_through_the_injected_read():
    t = injected_task("multi_hop", family=6)
    k = t.injection["at_read_step"]
    seen = []

    def gen(batch):
        seen.append(list(batch[0]))
        return ["The venue is somewhere."] * len(batch)

    ep = run_episodes([t], gen, prefill=True)[0]
    assert [c["name"] for c in ep.calls] == [c["name"] for c in t.gold_calls[:k + 1]]
    assert [c["step"] for c in ep.calls] == list(range(k + 1))
    assert t.injection["text"] in seen[0][-1]["content"] and seen[0][-1]["role"] == "user"
    roles = [m["role"] for m in ep.messages]
    assert roles[:2] == ["system", "user"] and all(a != b for a, b in zip(roles[1:], roles[2:]))


def test_prefill_ignores_clean_tasks():
    t = clean_task("read_file_fact")
    ep = run_episodes([t], scripted(["final"]), prefill=True)[0]
    assert ep.calls == [] and len(ep.messages) == 3


def test_full_prefill_then_one_model_turn():
    t = clean_task("find_fact_in_email")
    seen = []

    def gen(batch):
        seen.append(list(batch[0]))
        return [render_call("list_inbox", {})] * len(batch)  # a redundant call where it should answer

    ep = run_episodes([t], gen, prefill="full", max_new_turns=1)[0]
    assert len(seen) == 1 and [c["name"] for c in ep.calls[:len(t.gold_calls)]] == [c["name"] for c in t.gold_calls]
    assert ep.stop_reason == "max_steps" and len(ep.calls) == len(t.gold_calls) + 1


def test_injected_prefill_one_turn_skips_nothing_for_clean():
    t = injected_task("summarise_email")
    ep = run_episodes([t], scripted(["done"]), prefill="injected", max_new_turns=1)[0]
    assert ep.stop_reason == "finished" and len(ep.calls) == t.injection["at_read_step"] + 1
