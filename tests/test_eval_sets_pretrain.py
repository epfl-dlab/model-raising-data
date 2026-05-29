"""Integration test for build_eval_sidecar_from_classified (join + id-guard + select + build)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.eval_sets.pretrain_reflection import build_eval_sidecar_from_classified


def _write(path, table):
    pq.write_table(pa.table(table), path)


def test_full_build(tmp_path):
    raw_dir = tmp_path / "raw"
    safety_dir = tmp_path / "safety"
    raw_dir.mkdir()
    safety_dir.mkdir()

    # 6 harmful (score 4) + 6 benign (score 0); one harmful id ("h0") also lives
    # in the existing sidecar and must be dropped by the id-guard.
    ids = [f"h{i}" for i in range(6)] + [f"b{i}" for i in range(6)]
    texts = [f"text {i}" for i in range(12)]
    scores = [4] * 6 + [0] * 6
    _write(raw_dir / "part_00000.parquet", {"id": ids, "text": texts, "source": ["dolma"] * 12})
    _write(safety_dir / "shard_0000_part0000.parquet", {"id": ids, "safety_score": scores})

    existing = tmp_path / "existing.parquet"
    _write(existing, {"doc_id": ["h0", "unrelated"]})

    out = tmp_path / "sidecar.parquet"
    stats = build_eval_sidecar_from_classified(
        str(raw_dir), str(safety_dir), str(out),
        target_harmful=5, neg_ratio=0.5, seed=1,
        existing_id_sources=[(str(existing), "doc_id")],
    )

    assert stats["id_guard_dropped"] == 1  # h0
    assert stats["harmful"] == 5
    assert stats["benign_negative"] == 2  # round(5 * 0.5)

    d = pq.read_table(out).to_pydict()
    assert "h0" not in d["doc_id"]  # dropped by id-guard
    assert set(d["strata"]) == {"harmful", "benign_negative"}
    assert all(tl > 0 for tl in d["token_length"])


def test_dedups_duplicate_raw_ids(tmp_path):
    """Raw shards carry duplicate ids (dolma3 within-file upsampling); the eval set
    must contain each id at most once."""
    raw_dir = tmp_path / "raw"
    safety_dir = tmp_path / "safety"
    raw_dir.mkdir()
    safety_dir.mkdir()

    # 4 unique harmful ids, each duplicated 3x in the raw shard (12 raw rows).
    uniq = [f"h{i}" for i in range(4)]
    raw_ids = [i for i in uniq for _ in range(3)]
    _write(raw_dir / "part_00000.parquet", {
        "id": raw_ids, "text": [f"t {i}" for i in raw_ids], "source": ["dolma"] * 12,
    })
    _write(safety_dir / "shard_0000_part0000.parquet", {"id": uniq, "safety_score": [4] * 4})

    out = tmp_path / "sidecar.parquet"
    stats = build_eval_sidecar_from_classified(
        str(raw_dir), str(safety_dir), str(out),
        target_harmful=100, neg_ratio=0.0, seed=1,
        existing_id_sources=[],
    )
    doc_ids = pq.read_table(out).to_pydict()["doc_id"]
    assert sorted(doc_ids) == uniq  # 4 unique, no duplicates
    assert stats["selected"] == 4
