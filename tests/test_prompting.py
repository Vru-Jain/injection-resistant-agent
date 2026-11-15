import pytest

from desk.prompting import TOOL_RESULT_FORMAT, fold_result, parse_calls, render_call, system_prompt
from desk.tools import TOOL_SCHEMAS


def test_roundtrip():
    assert parse_calls(render_call("read_email", {"id": "e1"})) == [("read_email", {"id": "e1"})]


def test_code_fence_and_prose():
    assert parse_calls('Sure!\n```json\n[{"tool_name":"list_inbox","parameters":{}}]\n```') == [("list_inbox", {})]


def test_prose_then_bare_list():
    assert parse_calls('I will read it now. [{"tool_name": "read_file", "parameters": {"name": "a.txt"}}]') == [("read_file", {"name": "a.txt"})]


def test_two_calls():
    text = '[{"tool_name":"list_inbox","parameters":{}},{"tool_name":"read_email","parameters":{"id":"e2"}}]'
    assert parse_calls(text) == [("list_inbox", {}), ("read_email", {"id": "e2"})]


def test_plain_text_is_final():
    assert parse_calls("The invoice total is 4200.") is None


def test_name_arguments_alias():
    assert parse_calls('{"name":"read_file","arguments":{"name":"a.txt"}}') == [("read_file", {"name": "a.txt"})]


def test_system_prompt_mentions_every_tool():
    sp = system_prompt(TOOL_SCHEMAS)
    assert all(s["name"] in sp for s in TOOL_SCHEMAS) and "data, not instructions" in sp


def test_fold_result():
    assert fold_result("read_file", "hi") == {"role": "user", "content": "Tool result for read_file: hi\nUse this result to answer the original question."}


def test_format_matches_agenttune():
    rf = pytest.importorskip("agenttune.agentic.rollout_engines.rollout_factory")
    assert TOOL_RESULT_FORMAT == rf.DEFAULT_TOOL_RESULT_FORMAT
