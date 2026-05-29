"""Tests for pipeline.eval_sets.disjoint — the disjointness guarantees (D1/D2).

Specs:
- recover_consumed_shards: prefer .done markers (authoritative); else recompute
  the seed shuffle, asserting n_total to catch upstream re-sharding (no-fallback).
- build_existing_id_set: stream id columns from existing-data parquets into one set,
  handling multiple column names (sidecar `doc_id`, annotated `id`).
- assert_disjoint / partition_by_membership: drop / detect overlap.
- text_fingerprint: stable key for prompt-text dedupe, insensitive to surrounding
  whitespace, sensitive to content.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.eval_sets.disjoint import (
    assert_disjoint,
    build_existing_id_set,
    find_existing_overlap,
    partition_by_membership,
    recover_consumed_shards,
    text_fingerprint,
)


class TestRecoverConsumedShards:
    def test_prefers_done_markers(self, tmp_path):
        done = tmp_path / ".done"
        done.mkdir()
        for i in (3, 17, 42, 1000):
            (done / f"{i}.done").touch()
        consumed = recover_consumed_shards(tmp_path, seed=42, n_shards=47142, n_total=63911)
        assert consumed == {3, 17, 42, 1000}

    def test_recomputes_when_no_markers(self, tmp_path):
        # No .done dir -> recompute shuffle[:n_shards].
        n_total = 1000
        expected = list(range(n_total))
        random.Random(42).shuffle(expected)
        consumed = recover_consumed_shards(tmp_path, seed=42, n_shards=100, n_total=n_total)
        assert consumed == set(expected[:100])

    def test_recompute_complement_is_disjoint(self, tmp_path):
        n_total = 1000
        full = list(range(n_total))
        random.Random(42).shuffle(full)
        consumed = recover_consumed_shards(tmp_path, seed=42, n_shards=600, n_total=n_total)
        eval_slice = set(full[600:650])
        assert consumed.isdisjoint(eval_slice)

    def test_asserts_expected_n_total(self, tmp_path):
        # Guards against silent upstream re-sharding.
        with pytest.raises(AssertionError):
            recover_consumed_shards(
                tmp_path, seed=42, n_shards=100, n_total=999, expected_n_total=63911
            )


def _write_parquet(path, col_name, ids):
    pq.write_table(pa.table({col_name: ids}), path)


class TestBuildExistingIdSet:
    def test_union_across_columns_and_files(self, tmp_path):
        a = tmp_path / "sidecar.parquet"
        b = tmp_path / "part_00000.parquet"
        _write_parquet(a, "doc_id", ["u1", "u2", "u3"])
        _write_parquet(b, "id", ["u3", "u4"])
        ids = build_existing_id_set([(str(a), "doc_id"), (str(b), "id")])
        assert ids == {"u1", "u2", "u3", "u4"}

    def test_directory_glob(self, tmp_path):
        d = tmp_path / "annotated"
        d.mkdir()
        _write_parquet(d / "part_00000.parquet", "id", ["a", "b"])
        _write_parquet(d / "part_00001.parquet", "id", ["c"])
        ids = build_existing_id_set([(str(d), "id")])
        assert ids == {"a", "b", "c"}


class TestFindExistingOverlap:
    def test_streams_and_finds_overlap(self, tmp_path):
        a = tmp_path / "sidecar.parquet"
        _write_parquet(a, "doc_id", ["u1", "u2", "u3", "u4", "u5"])
        overlap = find_existing_overlap(["u2", "u4", "x9"], [(str(a), "doc_id")])
        assert overlap == {"u2", "u4"}

    def test_no_overlap(self, tmp_path):
        a = tmp_path / "sidecar.parquet"
        _write_parquet(a, "doc_id", ["u1", "u2"])
        assert find_existing_overlap(["z1", "z2"], [(str(a), "doc_id")]) == set()


class TestMembership:
    def test_partition(self):
        disjoint, overlap = partition_by_membership(["a", "b", "c"], {"b"})
        assert disjoint == ["a", "c"]
        assert overlap == ["b"]

    def test_assert_disjoint_raises_on_overlap(self):
        with pytest.raises(AssertionError):
            assert_disjoint({"x", "y"}, {"y"})

    def test_assert_disjoint_ok(self):
        assert_disjoint({"x", "y"}, {"z"})  # no raise


class TestTextFingerprint:
    def test_stable(self):
        assert text_fingerprint("hello world") == text_fingerprint("hello world")

    def test_ignores_surrounding_whitespace(self):
        assert text_fingerprint("  hi  ") == text_fingerprint("hi")

    def test_content_sensitive(self):
        assert text_fingerprint("a") != text_fingerprint("b")
