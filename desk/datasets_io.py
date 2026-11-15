"""JSONL read/write (UTF-8, non-ASCII kept so Devanagari stays readable)."""
from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path, dicts) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for d in dicts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
