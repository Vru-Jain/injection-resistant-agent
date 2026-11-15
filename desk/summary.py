"""Score a rollout file against its tasks: overall, per family, per template, per episode."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

from desk.scoring import Episode, aggregate, score_episode
from desk.tasks import Task


def summarise(tasks: list[Task], episodes: list[Episode]) -> dict:
    by_id = {t.id: t for t in tasks}
    pairs = [(by_id[ep.task_id.rsplit("#", 1)[0]], ep) for ep in episodes]
    scores = [score_episode(t, ep) for t, ep in pairs]
    groups: dict[str, dict] = {"by_family": defaultdict(list), "by_template": defaultdict(list)}
    for s, (t, _) in zip(scores, pairs):
        groups["by_family"][str(t.injection["family"]) if t.injection else "clean"].append((s, t))
        groups["by_template"][t.template].append((s, t))
    out = {"overall": aggregate(scores, [t for t, _ in pairs])}
    for g, d in groups.items():
        out[g] = {k: aggregate([s for s, _ in v], [t for _, t in v]) for k, v in sorted(d.items())}
    out["per_episode"] = [{"episode_id": ep.task_id, "stop_reason": ep.stop_reason, **asdict(s)} for s, (_, ep) in zip(scores, pairs)]
    return out
