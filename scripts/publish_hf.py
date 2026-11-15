"""Publish dataset repos (public, MIT) with their cards and add them to the datasets collection.

python scripts/publish_hf.py tasks suite pairs rollouts results
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, add_collection_item

NS = "Vru-Jain"
DATASETS_COLLECTION = "Vru-Jain/injection-resistant-tiny-aya-datasets-6af9173c39fca75261389ee4"
REPOS = {  # name -> (repo, files to upload: list of (local path or glob, path in repo))
    "tasks": ("ira-tasks", [("data/tasks/*.jsonl", ""), ("data/leakcheck.json", "")]),
    "suite": ("ira-injection-suite", [("data/injection_suite/*", "")]),
    "pairs": ("ira-dpo-pairs", [("data/pairs/*", "")]),
    "rollouts": ("ira-rollouts", [("data/rollouts/*", "")]),
    "results": ("ira-eval-results", [("results/*.json", ""), ("results/*.md", ""), ("results/auditkit_runs", "auditkit_runs")]),
}


def publish(name: str) -> str:
    repo, files = REPOS[name]
    rid, api = f"{NS}/{repo}", HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        for pattern, dest in files:
            for p in Path(".").glob(pattern):
                target = Path(tmp, dest, p.name) if p.is_file() else Path(tmp, dest)
                target.parent.mkdir(parents=True, exist_ok=True)
                (shutil.copy if p.is_file() else lambda a, b: shutil.copytree(a, b, dirs_exist_ok=True))(p, target)
        shutil.copy(f"hf_cards/{repo}/README.md", Path(tmp, "README.md"))
        api.create_repo(rid, repo_type="dataset", private=False, exist_ok=True)
        api.update_repo_settings(rid, repo_type="dataset", private=False)
        api.upload_folder(folder_path=tmp, repo_id=rid, repo_type="dataset", commit_message="update from scripts/publish_hf.py")
    add_collection_item(DATASETS_COLLECTION, item_id=rid, item_type="dataset", exists_ok=True)
    print("published", f"https://huggingface.co/datasets/{rid}")
    return rid


if __name__ == "__main__":
    for n in sys.argv[1:]:
        publish(n)
