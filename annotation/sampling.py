"""Stratified sampling and annotator queue management."""

import itertools
import json
import random
from pathlib import Path

from datasets import load_dataset

from annotation.config import HF_DATASET, SUBSETS
from annotation.storage import compute_item_id

ANNOTATION_DIR = Path(__file__).parent


def sample_path() -> Path:
    """Return the path to the persisted sample file."""
    return ANNOTATION_DIR / "sample.json"


def load_items_from_hf(n_per_subset: int, seed: int = 42) -> list[dict]:
    """Load raw texts from FineWeb subsets, assign stable IDs and reflection points."""
    rng = random.Random(seed)
    items = []
    for subset in SUBSETS:
        ds = load_dataset(HF_DATASET, subset, split="train", streaming=True)
        rows = list(itertools.islice(ds, n_per_subset))
        assert len(rows) > 0, f"Empty dataset: {subset}"
        for row in rows:
            assert "text" in row, f"Row missing 'text' column, got: {list(row.keys())}"
            text = row["text"]
            item_id = compute_item_id(text)
            # Pick a reflection point between 10%-90% of text (character-based)
            min_pos = max(1, int(len(text) * 0.1))
            max_pos = max(min_pos + 1, int(len(text) * 0.9))
            # Snap to nearest word boundary
            char_pos = rng.randint(min_pos, max_pos)
            space_idx = text.find(" ", char_pos)
            if space_idx != -1 and space_idx - char_pos < 50:
                char_pos = space_idx
            items.append({
                "item_id": item_id,
                "subset": subset,
                "text": text,
                "reflection_point": char_pos,
            })
    return items


def draw_stratified_sample(
    items: list[dict],
    n: int,
    min_per_stratum: int = 5,
    seed: int = 42,
) -> list[str]:
    """Draw a stratified sample of item_ids proportional to stratum sizes.

    Each stratum gets at least min_per_stratum items (or all items if the
    stratum is smaller). Remaining budget is allocated proportionally.
    """
    assert len(items) > 0, "Cannot sample from empty items list"
    assert n > 0, "Sample size must be positive"

    strata: dict[str, list[str]] = {}
    for item in items:
        strata.setdefault(item["subset"], []).append(item["item_id"])

    rng = random.Random(seed)
    for ids in strata.values():
        rng.shuffle(ids)

    selected: list[str] = []
    remaining_budget = n

    for stratum_ids in strata.values():
        take = min(min_per_stratum, len(stratum_ids))
        selected.extend(stratum_ids[:take])
        remaining_budget -= take

    if remaining_budget > 0:
        total_remaining = sum(
            max(0, len(ids) - min_per_stratum) for ids in strata.values()
        )
        if total_remaining > 0:
            for stratum_ids in strata.values():
                available = stratum_ids[min_per_stratum:]
                if not available:
                    continue
                proportional_n = round(
                    remaining_budget * len(available) / total_remaining
                )
                take = min(proportional_n, len(available))
                selected.extend(available[:take])

    seen: set[str] = set()
    unique: list[str] = []
    for item_id in selected:
        if item_id not in seen:
            seen.add(item_id)
            unique.append(item_id)

    return unique


def save_sample(items: list[dict], sample_ids: list[str]) -> None:
    """Persist the sample items and selected IDs to disk."""
    items_by_id = {item["item_id"]: item for item in items}
    sample_items = [items_by_id[sid] for sid in sample_ids]
    path = sample_path()
    path.write_text(json.dumps(sample_items, indent=2))


def load_sample() -> list[dict] | None:
    """Load a previously saved sample. Returns None if no sample exists."""
    path = sample_path()
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_annotator_queue(
    sample_ids: list[str],
    annotator_id: str,
    completed_ids: set[str],
) -> list[str]:
    """Return remaining item_ids in a deterministic random order per annotator."""
    remaining = [iid for iid in sample_ids if iid not in completed_ids]
    rng = random.Random(annotator_id)
    rng.shuffle(remaining)
    return remaining
