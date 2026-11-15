"""Train/test leak check on user prompts and injection strings."""
from __future__ import annotations

import re
import string

from desk.tasks import Task

_DROP = str.maketrans("", "", string.punctuation + string.digits)


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold().translate(_DROP)).strip()


def leak_check(train: list[Task], tests: dict[str, list[Task]]) -> dict:
    prompts = {t.user_prompt for t in train}
    norm = {normalise(t.user_prompt) for t in train}
    inj = {t.injection["text"] for t in train if t.injection}
    return {name: {
        "n": len(ts),
        "exact_prompt_overlap": sum(t.user_prompt in prompts for t in ts),
        "normalised_prompt_overlap": sum(normalise(t.user_prompt) in norm for t in ts),
        "injection_text_overlap": sum(bool(t.injection) and t.injection["text"] in inj for t in ts),
    } for name, ts in tests.items()}
