"""Enriched + benign-negative selection (R1/R2) and the eval sidecar builder (D4).

Selection enriches the eval pool for the hard cases (safety_score>=3) while keeping
a controlled fraction of benign negatives so consumers can measure false positives.
The sidecar builder writes exactly the columns `charter.scale`'s SidecarReader needs,
with a training-faithful `token_length` (EOS-aware truncation, not a fresh count).
"""

from __future__ import annotations

import random
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.log import logger
from pipeline.tokenizer import truncate_and_count

HARMFUL_SCORES = (3, 4, 5)
BENIGN_SCORES = (0, 1, 2)
BENIGN_WEIGHTS = {0: 1, 1: 2, 2: 3}  # weight negatives toward score 2

# The training annotated stream packs a 1920-token window that INCLUDES the
# appended EOS, so the content length stored as `token_length` caps at 1919
# (verified empirically: the production sidecar's token_length max is 1919, and
# reflection_end_token_index == token_length for every row). Match that exactly.
MAX_CONTENT_TOKENS = 1919


def _weighted_sample(
    pool_by_score: dict[int, list],
    n: int,
    weights: dict[int, int],
    rng: random.Random,
) -> list:
    """Sample up to `n` items without replacement across score strata.

    At each draw a stratum (with items remaining) is chosen with probability
    proportional to its weight, then a random item is popped from it. Naturally
    respects per-stratum availability and is deterministic given `rng`.
    """
    pools = {s: rng.sample(items, len(items)) for s, items in pool_by_score.items() if items}
    selected: list = []
    for _ in range(n):
        active = [s for s in pools if pools[s]]
        if not active:
            break
        w = [weights.get(s, 1) for s in active]
        s = rng.choices(active, weights=w, k=1)[0]
        selected.append(pools[s].pop())
    return selected


def stratified_select(
    rows: list[dict],
    *,
    target_harmful: int,
    neg_ratio: float = 0.5,
    seed: int = 0,
    score_field: str = "safety_score",
) -> list[dict]:
    """Select enriched harmful docs + benign negatives. Tags each row's `strata`.

    - Harmful (score in 3/4/5): up to `target_harmful`, drawn proportional to the
      natural per-stratum availability (preserves the 3/4/5 mix; takes all if fewer).
    - Benign negatives (score in 0/1/2): `round(n_harmful * neg_ratio)` docs, weighted
      toward score 2 (BENIGN_WEIGHTS).
    """
    rng = random.Random(seed)
    by_score: dict[int, list] = defaultdict(list)
    for r in rows:
        by_score[int(r[score_field])].append(r)

    harmful_pool = {s: by_score.get(s, []) for s in HARMFUL_SCORES}
    total_harmful = sum(len(v) for v in harmful_pool.values())
    if total_harmful < target_harmful:
        logger.warning(
            "Only {} harmful docs available (< target {}). Taking all; "
            "launch more complement shards to reach the target.",
            total_harmful, target_harmful,
        )
    # Proportional to availability == preserves the natural 3/4/5 distribution.
    harmful_weights = {s: len(harmful_pool[s]) for s in HARMFUL_SCORES}
    selected_harmful = _weighted_sample(harmful_pool, target_harmful, harmful_weights, rng)

    n_neg = round(len(selected_harmful) * neg_ratio)
    benign_pool = {s: by_score.get(s, []) for s in BENIGN_SCORES}
    selected_benign = _weighted_sample(benign_pool, n_neg, BENIGN_WEIGHTS, rng)

    for r in selected_harmful:
        r["strata"] = "harmful"
    for r in selected_benign:
        r["strata"] = "benign_negative"

    logger.info(
        "Selected {} harmful + {} benign negatives = {} total",
        len(selected_harmful), len(selected_benign),
        len(selected_harmful) + len(selected_benign),
    )
    return selected_harmful + selected_benign


def build_eval_sidecar(
    rows: list[dict],
    out_path: str,
    *,
    id_field: str = "id",
    max_content_tokens: int = MAX_CONTENT_TOKENS,
) -> int:
    """Write an eval sidecar parquet for `charter.scale` (SidecarReader columns).

    Renames `id_field` -> `doc_id`, sets `is_bad = safety_score>=3`, and computes
    `token_length = min(count_tokens(text), max_content_tokens)` via `truncate_and_count`
    so it matches the training sidecar exactly (which caps content at 1919, the
    1920-token window minus the appended EOS). `reflection_end` then sets
    reflection_end_token_index == token_length, the F3 invariant. Asserts every
    token_length is positive (a null/zero would make generate.py silently fall back).
    Returns the row count written.
    """
    doc_ids, texts, sources, scores, is_bad, token_lengths, strata = [], [], [], [], [], [], []
    for r in rows:
        text = r["text"]
        _, n_tok = truncate_and_count(text, max_content_tokens)
        assert n_tok > 0, f"doc {r[id_field]} produced 0 tokens"
        doc_ids.append(r[id_field])
        texts.append(text)
        sources.append(r.get("source"))
        score = int(r["safety_score"])
        scores.append(score)
        is_bad.append(score >= 3)
        token_lengths.append(n_tok)
        strata.append(r.get("strata"))

    table = pa.table(
        {
            "doc_id": pa.array(doc_ids, pa.string()),
            "text": pa.array(texts, pa.large_string()),
            "source": pa.array(sources, pa.string()),
            "safety_score": pa.array(scores, pa.int8()),
            "is_bad": pa.array(is_bad, pa.bool_()),
            "token_length": pa.array(token_lengths, pa.int32()),
            "strata": pa.array(strata, pa.string()),
        }
    )
    pq.write_table(table, out_path)
    logger.info("Wrote eval sidecar: {} rows -> {}", len(doc_ids), out_path)
    return len(doc_ids)
