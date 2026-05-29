"""Pretraining reflection-end eval: join classified docs → id-guard → select → sidecar.

The download (login node) and safety classification (GPU SLURM) reuse the existing
`preprocessing.download.download` (with `--shard-offset`) and
`preprocessing.annotation.annotate` modules. This module does the login-node steps
between classification and the `charter.scale --run reflection_end` generation:
join text with safety scores, drop any doc whose id is in the existing corpus
(D1.2 exact id-guard), stratified-select (R1/R2), and write the eval sidecar (D4).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from pipeline.eval_sets.disjoint import find_existing_overlap
from pipeline.eval_sets.select import build_eval_sidecar, stratified_select
from pipeline.log import logger

# Existing-data id source for the exact id-guard (plan.md D1.2). The tokenized
# annotated sidecar IS the training corpus on scratch (102M docs); the old
# dolma3_mix-1T_annotated parquets were cleaned up. Complement-shard selection
# already guarantees disjointness from every downloaded shard — this guard only
# catches dolma3's cross-shard duplicate ids.
DEFAULT_EXISTING_ID_SOURCES: list[tuple[str, str]] = [
    ("/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet", "doc_id"),
]


def _join_classified(raw_dir: str, safety_dir: str) -> list[dict]:
    """Join raw docs (id,text,source) with safety scores (id,safety_score) by id.

    Reads every ``*.parquet`` under each dir (recursively). The eval pool is small
    (~hundreds of k rows), so an in-memory join is fine.
    """
    id_to_score: dict[str, int] = {}
    for p in sorted(Path(safety_dir).rglob("*.parquet")):
        t = pq.read_table(p, columns=["id", "safety_score"]).to_pydict()
        for i, s in zip(t["id"], t["safety_score"]):
            if i is not None and s is not None:
                id_to_score[i] = int(s)
    logger.info("Loaded {} safety scores", len(id_to_score))

    # dolma3 shards carry heavy within-/cross-file duplicate ids (the annotator
    # dedups, so id_to_score is already unique). Dedup the raw rows the same way —
    # keep the first occurrence per id — so duplicate docs can't appear multiple
    # times in the eval set.
    rows: list[dict] = []
    seen: set[str] = set()
    n_raw = 0
    for p in sorted(Path(raw_dir).rglob("*.parquet")):
        cols = pq.read_table(p).column_names
        want = [c for c in ("id", "text", "source") if c in cols]
        t = pq.read_table(p, columns=want).to_pylist()
        for r in t:
            n_raw += 1
            rid = r["id"]
            if rid in seen:
                continue
            score = id_to_score.get(rid)
            if score is None:
                continue
            seen.add(rid)
            rows.append({"id": rid, "text": r["text"], "source": r.get("source"), "safety_score": score})
    logger.info("Joined {} unique docs (from {} raw rows)", len(rows), n_raw)
    return rows


def build_eval_sidecar_from_classified(
    raw_dir: str,
    safety_dir: str,
    out_sidecar: str,
    *,
    target_harmful: int = 10_000,
    neg_ratio: float = 0.5,
    seed: int = 7,
    existing_id_sources: list[tuple[str, str]] | None = None,
) -> dict:
    """Build the eval sidecar parquet (the input to `charter.scale --run reflection_end`)."""
    rows = _join_classified(raw_dir, safety_dir)

    sources = existing_id_sources or DEFAULT_EXISTING_ID_SOURCES
    overlap = find_existing_overlap([r["id"] for r in rows], sources)
    if overlap:
        logger.warning("id-guard dropped {} cross-shard-duplicate docs", len(overlap))
    rows = [r for r in rows if r["id"] not in overlap]

    selected = stratified_select(
        rows, target_harmful=target_harmful, neg_ratio=neg_ratio, seed=seed
    )
    # F1: selected ids must be disjoint from the existing-data overlap.
    assert overlap.isdisjoint({r["id"] for r in selected}), "id-guard leak"

    n = build_eval_sidecar(selected, out_sidecar, id_field="id")
    n_harm = sum(1 for r in selected if r["strata"] == "harmful")
    stats = {
        "joined": len(rows),
        "id_guard_dropped": len(overlap),
        "selected": n,
        "harmful": n_harm,
        "benign_negative": n - n_harm,
        "out_sidecar": out_sidecar,
    }
    logger.info("eval sidecar built: {}", stats)
    return stats
