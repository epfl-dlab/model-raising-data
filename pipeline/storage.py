"""Shared JSONL helpers used by both phase1 and phase2 storage."""

import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def compute_item_id(text: str) -> str:
    """Compute a stable item ID from the text (first 200 chars)."""
    return hashlib.sha256(text[:200].encode()).hexdigest()[:16]
