"""Disjointness guarantees for the eval sets (plan.md §3, D1/D2).

D1 — pretraining: the eval pool is downloaded from a complement slice of the
seed-42 shuffle (handled by `download.py --shard-offset`), which is disjoint from
the training plan by construction. This module provides:
  * `recover_consumed_shards` — the authoritative consumed upstream-shard set, for
    cross-checking / reporting (prefers `.done` markers, else recomputes the shuffle).
  * `build_existing_id_set` + `assert_disjoint` — the residual id-guard against
    dolma3's known cross-shard duplicate ids, built from the REAL existing data
    (the tokenized sidecar `doc_id` column + the annotated parquets' `id` column).

D2 — SFT: `text_fingerprint` is the stable dedupe key for prompt exclusion, since
`wildguardmix` source_ids are unstable positional indices.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.log import logger


def recover_consumed_shards(
    download_dir: str | Path,
    *,
    seed: int,
    n_shards: int,
    n_total: int,
    expected_n_total: int | None = None,
) -> set[int]:
    """Return the set of upstream shard indices the training download consumed.

    Prefers the authoritative ``.done`` markers written by download.py (one
    ``{idx}.done`` file per completed upstream shard). If absent, recomputes
    ``random.Random(seed).shuffle(range(n_total))[:n_shards]``.

    ``expected_n_total`` (when given) is asserted against ``n_total`` to catch
    silent upstream re-sharding between dataset revisions — a no-fallback guard,
    since a wrong ``n_total`` would invalidate the complement.
    """
    done_dir = Path(download_dir) / ".done"
    markers = list(done_dir.glob("*.done")) if done_dir.exists() else []
    if markers:
        logger.info("Recovered {} consumed shards from .done markers", len(markers))
        return {int(p.stem) for p in markers}

    if expected_n_total is not None:
        assert n_total == expected_n_total, (
            f"n_total mismatch: got {n_total}, expected {expected_n_total}. "
            "The upstream dataset may have been re-sharded — the complement plan "
            "is no longer valid. Pin the dataset revision and re-verify."
        )
    order = list(range(n_total))
    random.Random(seed).shuffle(order)
    logger.info("No .done markers; recomputed consumed set from seed-{} shuffle", seed)
    return set(order[:n_shards])


def _iter_parquet_files(path: str | Path) -> list[Path]:
    """A single parquet file, or all ``part_*.parquet`` under a directory."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("part_*.parquet"))
        assert files, f"No part_*.parquet found under {p}"
        return files
    assert p.exists(), f"Parquet not found: {p}"
    return [p]


def build_existing_id_set(sources: list[tuple[str, str]]) -> set[str]:
    """Stream id columns from existing-data parquets into one exact set.

    ``sources`` is a list of ``(path, column)`` pairs, where ``path`` is a parquet
    file or a directory of ``part_*.parquet`` and ``column`` is the id column name
    (e.g. the sidecar's ``doc_id`` or the annotated parquets' ``id``). Reads one
    row group at a time to bound memory.
    """
    ids: set[str] = set()
    for path, column in sources:
        for pf_path in _iter_parquet_files(path):
            pf = pq.ParquetFile(pf_path)
            for rg in range(pf.metadata.num_row_groups):
                col = pf.read_row_group(rg, columns=[column]).column(column)
                ids.update(v for v in col.to_pylist() if v is not None)
    logger.info("Built existing-id set: {} unique ids", len(ids))
    return ids


def find_existing_overlap(candidate_ids, sources: list[tuple[str, str]]) -> set[str]:
    """Return the subset of `candidate_ids` that appears in the existing-data id columns.

    Memory-efficient: holds only the candidate set (~tens of k) and streams the big
    existing parquets one row group at a time. This is the right shape for the real
    102M-row sidecar — building a full 102M-id set would cost ~10 GB; here we never
    materialize it. Stops early once every candidate has been matched.
    """
    candidates = set(candidate_ids)
    found: set[str] = set()
    for path, column in sources:
        for pf_path in _iter_parquet_files(path):
            pf = pq.ParquetFile(pf_path)
            for rg in range(pf.metadata.num_row_groups):
                col = pf.read_row_group(rg, columns=[column]).column(column)
                for v in col.to_pylist():
                    if v in candidates:
                        found.add(v)
                if len(found) == len(candidates):
                    return found
    logger.info("id-guard: {}/{} candidates overlap existing data", len(found), len(candidates))
    return found


def partition_by_membership(
    candidate_ids, existing_ids: set[str]
) -> tuple[list[str], list[str]]:
    """Split candidates into (disjoint, overlapping) preserving input order."""
    disjoint, overlap = [], []
    for cid in candidate_ids:
        (overlap if cid in existing_ids else disjoint).append(cid)
    return disjoint, overlap


def assert_disjoint(eval_ids, existing_ids: set[str]) -> None:
    """Raise if any eval id is in the existing-data set (fitness function F1)."""
    overlap = set(eval_ids) & existing_ids
    assert not overlap, (
        f"Disjointness violated: {len(overlap)} eval ids overlap training data "
        f"(e.g. {list(overlap)[:5]})"
    )


def text_fingerprint(text: str) -> str:
    """Stable dedupe key for a prompt: sha256 of the stripped text.

    Strips surrounding whitespace (cheap normalization that won't merge distinct
    prompts) but is otherwise content-exact.
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
