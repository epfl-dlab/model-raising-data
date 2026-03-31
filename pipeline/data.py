"""Dataset loading and sampling utilities.

Provides cached access to the Dolma3 annotation sample dataset
from HuggingFace and text sampling for evaluation pipelines.
"""

from __future__ import annotations

import json
import random

from pipeline.config import PIPELINE_DATA_DIR
from pipeline.log import logger
from pipeline.storage import compute_item_id
from pipeline.tokenizer import truncate_to_max_tokens

DATASET = "jkminder/Dolma3_mix_annotation_sample"
DATASET_CACHE_PATH = PIPELINE_DATA_DIR / "dolma3_cache.jsonl"
DATASET_CACHE_SIZE = 4096


def load_dataset_cache(seed: int) -> list[dict]:
    """Load cached Dolma3 texts, or stream from HF and cache locally.

    Returns a flat list of {text, safety_score} dicts.
    """
    if DATASET_CACHE_PATH.exists():
        records = []
        with open(DATASET_CACHE_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            logger.info("Loaded {} items from dataset cache", len(records))
            return records

    logger.info(
        "Building dataset cache ({} items from {})...", DATASET_CACHE_SIZE, DATASET
    )
    import itertools
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset(DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    rows = list(itertools.islice(ds, DATASET_CACHE_SIZE))

    records = [
        {"text": r["text"], "safety_score": int(r["safety_score"])} for r in rows
    ]
    DATASET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_CACHE_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Cached {} items to {}", len(records), DATASET_CACHE_PATH)
    return records


def sample_texts(
    n: int,
    seed: int,
    max_tokens: int,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Sample texts from the Dolma3 dataset cache.

    Returns a list of {item_id, text, safety_score} dicts, each truncated
    to max_tokens. No gold set, no canaries, no reflection point.
    """
    if exclude_ids is None:
        exclude_ids = set()

    rng = random.Random(seed)
    cache = load_dataset_cache(seed)
    rng.shuffle(cache)

    items: list[dict] = []
    for row in cache:
        if len(items) >= n:
            break
        text = truncate_to_max_tokens(row["text"], max_tokens)
        item_id = compute_item_id(text)
        if item_id in exclude_ids:
            continue
        items.append(
            {
                "item_id": item_id,
                "text": text,
                "safety_score": row.get("safety_score"),
            }
        )
        exclude_ids.add(item_id)

    assert (
        len(items) >= n
    ), f"Could only sample {len(items)}/{n} items (cache has {len(cache)})"
    return items[:n]
