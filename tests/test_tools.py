from desk.tools import TOOL_SCHEMAS, call_tool, make_harness, make_tools
from tests.conftest import sample_state


def test_read_email_returns_body():
    assert "Invoice" in call_tool(sample_state(), "read_email", {"id": "e1"})


def test_transfer_is_recorded():
    state = sample_state()
    call_tool(state, "transfer_money", {"to_account": "ACC-9", "amount": 50})
    assert state.transfers == [{"to_account": "ACC-9", "amount": 50}]


def test_unknown_tool_is_error_string():
    assert call_tool(sample_state(), "nope", {}).startswith("error:")


def test_bad_params_is_error_string():
    assert call_tool(sample_state(), "read_email", {"idd": "e1"}).startswith("error:")


def test_non_dict_params_is_error_string():
    assert call_tool(sample_state(), "read_email", ["e1"]).startswith("error:")


def test_schemas_cover_all_tools():
    assert {s["name"] for s in TOOL_SCHEMAS} == set(make_tools(sample_state()))


def test_copy_is_independent():
    a = sample_state(); b = a.copy()
    call_tool(b, "send_email", {"to": "x@y.example", "subject": "s", "body": "b"})
    assert a.sent_emails == [] and len(b.sent_emails) == 1


def test_harness_finish_sets_done():
    h = make_harness(sample_state())
    h.reset("t")
    obs, r, done, info = h.step({"name": "finish", "arguments": {"answer": "ok"}})
    assert done
