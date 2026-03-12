"""Shared FineWeb cache: stream from HF once, reuse across phases."""

import itertools
import json
from pathlib import Path


def load_or_build_fineweb_cache(
    cache_path: Path,
    dataset: str,
    subsets: list[str],
    per_subset: int,
    seed: int,
) -> list[dict]:
    """Load cached FineWeb texts, or stream from HF and cache locally.

    Caches per_subset items per subset as JSONL. Returns a flat list of
    {text, subset} dicts.
    """
    if cache_path.exists():
        records = []
        for line in cache_path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
        if records:
            print(f"Loaded {len(records)} items from FineWeb cache ({cache_path.name})")
            return records

    print(f"Building FineWeb cache ({per_subset} items × {len(subsets)} subsets)...")
    from datasets import load_dataset

    records: list[dict] = []
    for subset in subsets:
        ds = load_dataset(dataset, subset, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        rows = list(itertools.islice(ds, per_subset))
        for row in rows:
            records.append({"text": row["text"], "subset": subset})
        print(f"  {subset}: {len(rows)} items")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Cached {len(records)} items to {cache_path}")
    return records
