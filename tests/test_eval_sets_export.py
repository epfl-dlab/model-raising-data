"""Tests for pipeline.eval_sets.export_hf label logic + local export (no upload)."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.eval_sets.export_hf import (
    export_reflection_eval,
    export_sft_eval,
    requires_citation,
    sft_citation_labels,
)


class TestRequiresCitation:
    def test_empty_list(self):
        assert requires_citation("[]") is False

    def test_nonempty(self):
        assert requires_citation('["2.1", "4.3"]') is True

    def test_blank_and_none(self):
        assert requires_citation("") is False
        assert requires_citation(None) is False

    def test_malformed(self):
        assert requires_citation("not json") is False


class TestSftCitationLabels:
    def test_has_citation(self):
        has, els = sft_citation_labels("I won't help — that's doxing [1.5].")
        assert has is True
        assert els == ["1.5"]

    def test_no_citation(self):
        has, els = sft_citation_labels("Sure, here is a cake recipe.")
        assert has is False
        assert els == []


def _merged_sidecar(tmp_path, canary_values):
    n = len(canary_values)
    table = pa.table({
        "doc_id": [f"d{i}" for i in range(n)],
        "source": ["dolma"] * n,
        "text": [f"text {i}" for i in range(n)],
        "safety_score": pa.array([4, 0, 3][:n], pa.int8()),
        "is_bad": [True, False, True][:n],
        "token_length": pa.array([100, 50, 80][:n], pa.int32()),
        "reflection_end_1p": ["r1"] * n,
        "reflection_end_3p": ["r3"] * n,
        "reflection_end_position": pa.array([99, 49, 79][:n], pa.int32()),
        "reflection_end_token_index": pa.array([100, 50, 80][:n], pa.int32()),
        "charter_reflection_end": ['["2.1"]', "[]", '["4.3"]'][:n],
        "canary_type_end": canary_values,
        "strata": ["harmful", "benign_negative", "harmful"][:n],
    })
    p = tmp_path / "merged.parquet"
    pq.write_table(table, p)
    return p


class TestExportReflectionEval:
    def test_adds_requires_citation_and_stats(self, tmp_path):
        merged = _merged_sidecar(tmp_path, canary_values=[None, None, None])
        out = tmp_path / "out.parquet"
        stats = export_reflection_eval(merged, out)
        assert stats["rows"] == 3
        assert stats["is_bad"] == 2
        assert stats["requires_citation"] == 2  # rows 0 and 2 cite
        t = pq.read_table(out)
        assert "requires_citation" in t.column_names
        assert t.column("requires_citation").to_pylist() == [True, False, True]

    def test_rejects_leaked_canary(self, tmp_path):
        merged = _merged_sidecar(tmp_path, canary_values=["Q3", None, None])
        with pytest.raises(AssertionError, match="canary"):
            export_reflection_eval(merged, tmp_path / "out.parquet")

    def test_drops_empty_gold_rows(self, tmp_path):
        # 3 rows; the middle one has both reflection voices blank → must be dropped.
        n = 3
        table = pa.table({
            "doc_id": ["a", "b", "c"], "source": ["dolma"] * n,
            "text": ["t1", "t2", "t3"],
            "safety_score": pa.array([4, 0, 3], pa.int8()),
            "is_bad": [True, False, True],
            "token_length": pa.array([10, 10, 10], pa.int32()),
            "reflection_end_1p": ["r1", "", "r3"],
            "reflection_end_3p": ["r1b", " ", "r3b"],  # blank/whitespace counts as empty
            "reflection_end_position": pa.array([9, 9, 9], pa.int32()),
            "reflection_end_token_index": pa.array([10, 10, 10], pa.int32()),
            "charter_reflection_end": ['["2.1"]', "[]", '["4.3"]'],
            "canary_type_end": [None] * n,
            "strata": ["harmful", "benign_negative", "harmful"],
        })
        merged = tmp_path / "merged.parquet"
        pq.write_table(table, merged)
        stats = export_reflection_eval(merged, tmp_path / "out.parquet")
        assert stats["rows"] == 2
        assert stats["dropped_empty_gold"] == 1
        d = pq.read_table(tmp_path / "out.parquet").to_pydict()
        assert d["doc_id"] == ["a", "c"]


class TestExportSftEval:
    def test_labels_and_skips(self, tmp_path):
        results = tmp_path / "results.jsonl"
        lines = [
            {"source": "wjb", "source_id": "1", "user": "u1", "cited": "no [1.5] thanks", "uncited": "no thanks", "harm_category": "harmful"},
            {"source": "wc", "source_id": "2", "user": "u2", "cited": "sure recipe", "uncited": "sure recipe", "harm_category": "unknown"},
            {"source": "wjb", "source_id": "3", "user": "u3", "error": "api: boom"},  # skipped
            {"source": "wjb", "source_id": "4", "user": "u4", "skip": True},  # skipped
        ]
        results.write_text("\n".join(json.dumps(x) for x in lines))
        out = tmp_path / "sft.parquet"
        stats = export_sft_eval(results, out)
        assert stats["input_rows"] == 4
        assert stats["exported_rows"] == 2
        assert stats["has_citation"] == 1
        t = pq.read_table(out).to_pydict()
        assert t["has_citation"] == [True, False]
        assert t["charter_elements"] == [["1.5"], []]
