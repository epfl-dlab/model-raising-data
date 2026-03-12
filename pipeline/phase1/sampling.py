"""Stratified sampling from locuslab/fineweb_annotated score subsets."""

import itertools
import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from pipeline.config import PIPELINE_DATA_DIR, Phase1Config
from pipeline.storage import compute_item_id

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_tokenizer = None

PHASE1_CACHE_PATH = PIPELINE_DATA_DIR / "phase1_fineweb_cache.jsonl"
PHASE1_CACHE_PER_SUBSET = 100


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


def _load_or_build_cache(phase1_cfg: Phase1Config, seed: int) -> dict[str, list[dict]]:
    """Load cached FineWeb texts by subset, or stream from HF and cache locally.

    Returns {subset: [{text, subset}, ...]} with PHASE1_CACHE_PER_SUBSET items per subset.
    """
    if PHASE1_CACHE_PATH.exists():
        by_subset: dict[str, list[dict]] = {}
        for line in PHASE1_CACHE_PATH.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                by_subset.setdefault(rec["subset"], []).append(rec)
        if by_subset:
            total = sum(len(v) for v in by_subset.values())
            print(f"Loaded {total} items from phase 1 cache")
            return by_subset

    print(f"Building phase 1 cache ({PHASE1_CACHE_PER_SUBSET} items per subset)...")
    records: list[dict] = []
    for subset in phase1_cfg.subsets:
        print(f"[{subset}] Streaming from HF...", flush=True)
        ds = load_dataset(phase1_cfg.dataset, subset, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        rows = list(itertools.islice(ds, PHASE1_CACHE_PER_SUBSET))
        for row in rows:
            records.append({"text": row["text"], "subset": subset})
        print(f"[{subset}] Cached {len(rows)} items", flush=True)

    PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE1_CACHE_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Cached {len(records)} items to {PHASE1_CACHE_PATH}")

    by_subset = {}
    for rec in records:
        by_subset.setdefault(rec["subset"], []).append(rec)
    return by_subset


def sample_items(n_per_subset: int, seed: int = 42, phase1_cfg: Phase1Config | None = None) -> list[dict]:
    """Sample n_per_subset items from each fineweb_annotated score subset.

    Uses a local JSONL cache to avoid repeated HF downloads. On first call,
    streams from HF and builds the cache. Returns items with item_id, subset,
    text, and reflection_point.
    """
    if phase1_cfg is None:
        from pipeline.config import load_config
        phase1_cfg = load_config().phase1

    cache = _load_or_build_cache(phase1_cfg, seed)
    rng = random.Random(seed)
    items = []

    for subset in phase1_cfg.subsets:
        cached_rows = cache.get(subset, [])
        assert len(cached_rows) >= n_per_subset, (
            f"Cache has {len(cached_rows)} items for {subset}, need {n_per_subset}. "
            f"Delete {PHASE1_CACHE_PATH} to rebuild."
        )
        selected = cached_rows[:n_per_subset]

        for row in selected:
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
