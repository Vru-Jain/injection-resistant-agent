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
