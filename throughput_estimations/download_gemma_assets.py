"""Download missing Gemma 4 benchmark assets (12B-it + the four MTP drafters).

Targets the shared infra01 model store alongside the already-present it-models.
Idempotent: snapshot_download resumes/skips existing files.
"""

import os
from huggingface_hub import snapshot_download

DEST_ROOT = "/capstor/store/cscs/swissai/infra01/hf_models/models/google"

REPOS = [
    "google/gemma-4-12B-it",
    "google/gemma-4-E4B-it-assistant",
    "google/gemma-4-12B-it-assistant",
    "google/gemma-4-26B-A4B-it-assistant",
    "google/gemma-4-31B-it-assistant",
]

token = os.environ["HF_TOKEN"]
for repo in REPOS:
    dest = os.path.join(DEST_ROOT, repo.split("/", 1)[1])
    print(f"=== {repo} -> {dest}", flush=True)
    snapshot_download(
        repo_id=repo,
        local_dir=dest,
        token=token,
        max_workers=8,
    )
    print(f"=== done {repo}", flush=True)

print("ALL_DOWNLOADS_DONE", flush=True)
