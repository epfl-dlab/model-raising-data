"""Stratified sampling from locuslab/fineweb_annotated score subsets."""

import itertools
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from annotation.config import FINEWEB_DATASET, FINEWEB_SUBSETS
from annotation.storage import compute_item_id

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_tokenizer = None


def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def _compute_reflection_point(text: str, rng: random.Random) -> int:
    """Pick a reflection point between 10%-90% of text, snapped to a token boundary.

    Uses the SmolLM2 tokenizer to determine token boundaries, then selects a
    random token position within the 10%-90% range and returns the corresponding
    character offset.
    """
    tokenizer = _get_tokenizer()
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    n_tokens = len(offsets)
    assert n_tokens > 0, "Text produced no tokens"

    min_tok = max(1, int(n_tokens * 0.1))
    max_tok = min(n_tokens - 1, max(min_tok, int(n_tokens * 0.9)))
    tok_idx = rng.randint(min_tok, max_tok)
    return offsets[tok_idx][0]


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
