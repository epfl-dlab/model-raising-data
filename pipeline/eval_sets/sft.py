"""SFT eval: materialize disjoint prompts (D2) for sft.single_turn to generate.

Reproduces the production picks deterministically (`sample_mix(seed=42)`), fingerprints
their prompt text, then samples a fresh `seed=7` candidate set (HarmfulQA dropped — it
was 100% consumed), excludes any prompt whose text matches a production prompt, dedupes
within, and splits ~`harmful_frac` harmful / rest benign. Writes a prompts.parquet with
the schema `sft.single_turn`'s PromptsReader expects; `submit` then skips its own
materialization because the file already exists.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.eval_sets.disjoint import text_fingerprint
from pipeline.log import logger
from pipeline.sft.single_turn.data import sample_mix

HARMFUL_CATEGORIES = frozenset({"harmful", "adversarial_harmful"})

_PROMPTS_SCHEMA = pa.schema([
    ("global_row_idx", pa.int64()),
    ("source", pa.string()),
    ("source_id", pa.string()),
    ("user", pa.large_string()),
    ("meta", pa.string()),
    ("harm_category", pa.string()),
])


def materialize_eval_prompts(
    out_path: str,
    *,
    n: int = 10_000,
    seed: int = 7,
    production_n: int = 301_960,
    production_seed: int = 42,
    harmful_frac: float = 0.6,
    oversample: float = 3.0,
) -> dict:
    """Write a disjoint, harmful-enriched prompts.parquet for sft.single_turn.

    Asserts the residual harmful/benign pools are large enough (F5) before writing —
    a loud failure beats silently under-filling the harmful split.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reproducing production picks (n={}, seed={}) for exclusion...", production_n, production_seed)
    prod_fps = {text_fingerprint(p.user) for p in sample_mix(n=production_n, seed=production_seed)}
    logger.info("Production exclusion set: {} prompt fingerprints", len(prod_fps))

    cand = sample_mix(
        n=int(n * oversample), seed=seed, exclude_sources=frozenset({"harmfulqa"})
    )
    seen: set[str] = set()
    harmful, benign = [], []
    for p in cand:
        fp = text_fingerprint(p.user)
        if fp in prod_fps or fp in seen:
            continue
        seen.add(fp)
        (harmful if p.harm_category in HARMFUL_CATEGORIES else benign).append(p)

    n_harm = round(n * harmful_frac)
    n_ben = n - n_harm
    assert len(harmful) >= n_harm, (
        f"F5: only {len(harmful)} disjoint harmful prompts available (< {n_harm}). "
        f"Raise oversample or lower harmful_frac/n."
    )
    assert len(benign) >= n_ben, (
        f"F5: only {len(benign)} disjoint benign prompts available (< {n_ben})."
    )

    picks = harmful[:n_harm] + benign[:n_ben]
    random.Random(seed).shuffle(picks)

    rows = [
        {
            "global_row_idx": i,
            "source": p.source,
            "source_id": p.source_id,
            "user": p.user,
            "meta": json.dumps(p.meta or {}, ensure_ascii=False),
            "harm_category": p.harm_category,
        }
        for i, p in enumerate(picks)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=_PROMPTS_SCHEMA), out_path)

    stats = {
        "n": len(rows),
        "harmful": n_harm,
        "benign": n_ben,
        "residual_harmful_pool": len(harmful),
        "residual_benign_pool": len(benign),
        "out_path": str(out_path),
    }
    logger.info("eval prompts materialized: {}", stats)
    return stats
