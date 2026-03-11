"""Stratified sampling from locuslab/fineweb_annotated score subsets."""

import itertools
import random

from datasets import load_dataset

from annotation.config import FINEWEB_DATASET, FINEWEB_SUBSETS
from annotation.storage import compute_item_id


def _compute_reflection_point(text: str, rng: random.Random) -> int:
    """Pick a reflection point between 10%-90% of text, snapped to a word boundary."""
    min_pos = max(1, int(len(text) * 0.1))
    max_pos = max(min_pos + 1, int(len(text) * 0.9))
    char_pos = rng.randint(min_pos, max_pos)
    space_idx = text.find(" ", char_pos)
    if space_idx != -1 and space_idx - char_pos < 50:
        char_pos = space_idx
    return char_pos


def sample_items(n_per_subset: int, seed: int = 42) -> list[dict]:
    """Sample n_per_subset items from each fineweb_annotated score subset.

    Uses streaming with a shuffled buffer for pseudo-random sampling without
    downloading the full dataset. Returns items with item_id, subset, text,
    and reflection_point.
    """
    rng = random.Random(seed)
    items = []

    for subset in FINEWEB_SUBSETS:
        print(f"[{subset}] Loading streaming dataset...", flush=True)
        ds = load_dataset(
            FINEWEB_DATASET, subset, split="train", streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        rows = list(itertools.islice(ds, n_per_subset))
        assert len(rows) == n_per_subset, (
            f"Expected {n_per_subset} items from {subset}, got {len(rows)}"
        )
        print(f"[{subset}] Got {len(rows)} items", flush=True)

        for row in rows:
            text = row["text"]
            assert isinstance(text, str) and len(text) > 0, f"Empty text in {subset}"
            items.append({
                "item_id": compute_item_id(text),
                "subset": subset,
                "text": text,
                "reflection_point": _compute_reflection_point(text, rng),
            })

    assert len(items) > 0, "No fineweb items loaded"
    return items
