import pytest

from desk.tools import DeskState


def sample_state() -> DeskState:
    return DeskState(
        contacts={"Priya": "priya@acme.example", "Rahul": "rahul@acme.example"},
        emails=[
            {"id": "e1", "from": "priya@acme.example", "subject": "Invoice 118", "date": "2026-11-12",
             "body": "Invoice 118 for November. Amount due: 4200. Please pay to account ACC-PRIYA."},
            {"id": "e2", "from": "rahul@acme.example", "subject": "Lunch", "date": "2026-11-13", "body": "Lunch on Friday?"},
            {"id": "e3", "from": "news@shop.example", "subject": "Sale", "date": "2026-11-14", "body": "Big sale this week."},
        ],
        files={"notes.txt": "Project codename is Falcon.", "budget.csv": "item,cost\nlaptop,900"},
        events=[{"title": "Standup", "date": "2026-11-20", "time": "10:00"},
                {"title": "Dentist", "date": "2026-11-21", "time": "16:00"}],
        accounts={"Priya": "ACC-PRIYA", "Rahul": "ACC-RAHUL"},
    )


@pytest.fixture
def state():
    return sample_state()


# --- episode helpers (Task 5+) ---
import random as _random

from desk.tools import call_tool as _call_tool


def clean_task(template, seed=11):
    from desk.tasks import generate_tasks
    return generate_tasks(1, [template], seed=seed, split="x")[0]


def injected_task(template="summarise_email", family=1, target_idx=0, seed=11):
    from desk.injections import TARGETS, inject
    return inject(clean_task(template, seed), family, TARGETS[target_idx], _random.Random(0))


def gold_as_calls(task):
    return [(c["name"], c["arguments"]) for c in task.gold_calls]


def make_episode(task, calls, final_answer):
    """Episode as the loop would record it, executing each call on a copy of the task state."""
    from desk.scoring import Episode
    state, recorded = task.state.copy(), []
    for i, (name, params) in enumerate(calls):
        obs = _call_tool(state, name, params)
        recorded.append({"step": i, "name": name, "params": params, "valid": not obs.startswith("error:"), "observation": obs})
    return Episode(task_id=task.id, messages=[], calls=recorded, final_answer=final_answer, state_after=state,
                   stop_reason="finished" if final_answer is not None else "max_steps")
