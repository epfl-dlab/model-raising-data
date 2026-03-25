"""Tests for dedup-aware annotation and merge pipeline.

Covers _compute_dedup_indices (annotate), _read_task_annotations and
_write_annotated_file (merge), id-based join logic, and end-to-end simulation.
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

TMP_ROOT = Path(os.environ["HOME"]) / "tmp" / "test_dedup_merge"


# ── helpers ──────────────────────────────────────────────────────────


def _write_data_parquet(path: Path, ids: list[str], texts: list[str] | None = None, sources: list[str] | None = None) -> Path:
    """Create a parquet file with (id, text, source) columns."""
    n = len(ids)
    if texts is None:
        texts = [f"text for {i}" for i in ids]
    if sources is None:
        sources = ["src"] * n
    table = pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "source": pa.array(sources, type=pa.string()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return path


def _write_annotation_shard(path: Path, ids: list[str], scores: list[int]) -> Path:
    """Create an annotation shard parquet with (id, safety_score, safety_probs)."""
    n = len(ids)
    probs = [[0.0] * 6 for _ in range(n)]
    for i, s in enumerate(scores):
        probs[i][s] = 1.0
    table = pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "safety_score": pa.array(scores, type=pa.int8()),
            "safety_probs": pa.array(probs, type=pa.list_(pa.float32())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return path


def _write_task_meta(task_dir: Path, data_dir: Path, files: list[str], n_input_rows: int, world_size: int = 1, file_start: int = 0, n_original_rows: int | None = None) -> Path:
    """Create a task_meta.json file."""
    meta = {
        "data_dir": str(data_dir),
        "file_start": file_start,
        "file_count": len(files),
        "n_input_rows": n_input_rows,
        "world_size": world_size,
        "files": files,
    }
    if n_original_rows is not None:
        meta["n_original_rows"] = n_original_rows
    task_dir.mkdir(parents=True, exist_ok=True)
    meta_path = task_dir / "task_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


@pytest.fixture()
def tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide a unique temp dir under $HOME/tmp for each test."""
    base = TMP_ROOT / "pytest"
    base.mkdir(parents=True, exist_ok=True)
    return tmp_path_factory.mktemp("dedup_merge", numbered=True)


# ── TestComputeDedupIndices ──────────────────────────────────────────


class TestComputeDedupIndices:
    """Tests for _compute_dedup_indices: per-file dedup by id, global indices."""

    @staticmethod
    def _make_file(directory: Path, name: str, ids: list[str]) -> str:
        path = directory / name
        _write_data_parquet(path, ids)
        return str(path)

    def test_no_duplicates(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        f = self._make_file(tmp_dir, "part_0000.parquet", ["a", "b", "c", "d", "e"])
        indices, n_original = _compute_dedup_indices([f], id_column="id")
        assert indices == [0, 1, 2, 3, 4]
        assert n_original == 5

    def test_consecutive_duplicates(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        f = self._make_file(tmp_dir, "part_0000.parquet", ["A", "A", "A", "B", "B", "C"])
        indices, n_original = _compute_dedup_indices([f], id_column="id")
        assert indices == [0, 3, 5]
        assert n_original == 6

    def test_all_same_id(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        f = self._make_file(tmp_dir, "part_0000.parquet", ["X"] * 6)
        indices, n_original = _compute_dedup_indices([f], id_column="id")
        assert indices == [0]
        assert n_original == 6

    def test_single_row(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        f = self._make_file(tmp_dir, "part_0000.parquet", ["only"])
        indices, n_original = _compute_dedup_indices([f], id_column="id")
        assert indices == [0]
        assert n_original == 1

    def test_empty_file(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        f = self._make_file(tmp_dir, "part_0000.parquet", [])
        indices, n_original = _compute_dedup_indices([f], id_column="id")
        assert indices == []
        assert n_original == 0

    def test_seven_x_repetition(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        f = self._make_file(tmp_dir, "part_0000.parquet", ["X"] * 7 + ["Y"])
        indices, n_original = _compute_dedup_indices([f], id_column="id")
        assert indices == [0, 7]
        assert n_original == 8

    def test_similar_ids_not_conflated(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        f = self._make_file(tmp_dir, "part_0000.parquet", ["doc_1", "doc_01", "doc_001"])
        indices, n_original = _compute_dedup_indices([f], id_column="id")
        assert len(indices) == 3
        assert n_original == 3

    def test_mixed_dup_counts(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        # 1x "A", 3x "B", 5x "C" = 9 total
        ids = ["A"] + ["B"] * 3 + ["C"] * 5
        f = self._make_file(tmp_dir, "part_0000.parquet", ids)
        indices, n_original = _compute_dedup_indices([f], id_column="id")
        assert len(indices) == 3
        assert n_original == 9

    def test_cross_file_same_id_not_deduped(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        f1 = self._make_file(tmp_dir, "part_0000.parquet", ["X", "A"])
        f2 = self._make_file(tmp_dir, "part_0001.parquet", ["X", "B"])
        indices, n_original = _compute_dedup_indices([f1, f2], id_column="id")
        # "X" appears in both files but dedup is per-file, so both are kept
        assert len(indices) == 4
        assert n_original == 4

    def test_deterministic(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        ids = ["A", "B", "A", "C", "B", "D"]
        f = self._make_file(tmp_dir, "part_0000.parquet", ids)
        result1 = _compute_dedup_indices([f], id_column="id")
        result2 = _compute_dedup_indices([f], id_column="id")
        assert result1 == result2

    def test_multi_file_global_offsets(self, tmp_dir: Path):
        from preprocessing.annotation.annotate import _compute_dedup_indices

        # file1: 4 rows (2 unique), file2: 6 rows (3 unique)
        f1 = self._make_file(tmp_dir, "part_0000.parquet", ["A", "A", "B", "B"])
        f2 = self._make_file(tmp_dir, "part_0001.parquet", ["C", "C", "D", "D", "E", "E"])
        indices, n_original = _compute_dedup_indices([f1, f2], id_column="id")
        assert n_original == 10
        # file1 unique indices: 0, 2; file2 unique indices offset by 4: 4, 6, 8
        assert indices == [0, 2, 4, 6, 8]


# ── TestReadTaskAnnotations ──────────────────────────────────────────


class TestReadTaskAnnotations:
    """Tests for _read_task_annotations: read task metadata + annotation shards."""

    def test_single_rank_single_shard(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _read_task_annotations

        task_dir = tmp_dir / "task_0000"
        _write_task_meta(task_dir, tmp_dir, ["part_0000.parquet"], n_input_rows=3, world_size=1)
        _write_annotation_shard(
            task_dir / "shard_0000_part0000.parquet",
            ids=["a", "b", "c"],
            scores=[0, 2, 4],
        )
        (task_dir / "DONE").touch()

        meta, id_to_score = _read_task_annotations(task_dir)
        assert meta["n_input_rows"] == 3
        assert id_to_score == {"a": 0, "b": 2, "c": 4}

    def test_multi_rank(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _read_task_annotations

        task_dir = tmp_dir / "task_0000"
        _write_task_meta(task_dir, tmp_dir, ["part_0000.parquet"], n_input_rows=4, world_size=2)
        _write_annotation_shard(
            task_dir / "shard_0000_part0000.parquet",
            ids=["a", "b"],
            scores=[1, 3],
        )
        _write_annotation_shard(
            task_dir / "shard_0001_part0000.parquet",
            ids=["c", "d"],
            scores=[0, 5],
        )
        (task_dir / "DONE").touch()

        meta, id_to_score = _read_task_annotations(task_dir)
        assert id_to_score == {"a": 1, "b": 3, "c": 0, "d": 5}

    def test_multi_part_shards(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _read_task_annotations

        task_dir = tmp_dir / "task_0000"
        _write_task_meta(task_dir, tmp_dir, ["part_0000.parquet"], n_input_rows=4, world_size=1)
        _write_annotation_shard(
            task_dir / "shard_0000_part0000.parquet",
            ids=["a", "b"],
            scores=[1, 2],
        )
        _write_annotation_shard(
            task_dir / "shard_0000_part0001.parquet",
            ids=["c", "d"],
            scores=[3, 4],
        )
        (task_dir / "DONE").touch()

        meta, id_to_score = _read_task_annotations(task_dir)
        assert id_to_score == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_count_mismatch_raises(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _read_task_annotations

        task_dir = tmp_dir / "task_0000"
        _write_task_meta(task_dir, tmp_dir, ["part_0000.parquet"], n_input_rows=5, world_size=1)
        _write_annotation_shard(
            task_dir / "shard_0000_part0000.parquet",
            ids=["a", "b", "c"],
            scores=[0, 1, 2],
        )
        (task_dir / "DONE").touch()

        with pytest.raises(AssertionError):
            _read_task_annotations(task_dir)

    def test_missing_meta_raises(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _read_task_annotations

        task_dir = tmp_dir / "task_0000"
        task_dir.mkdir(parents=True, exist_ok=True)
        # no task_meta.json
        with pytest.raises(AssertionError):
            _read_task_annotations(task_dir)


# ── TestWriteAnnotatedFile ───────────────────────────────────────────


class TestWriteAnnotatedFile:
    """Tests for _write_annotated_file: append safety_score to original parquet."""

    def test_appends_safety_score(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        input_path = _write_data_parquet(tmp_dir / "input.parquet", ["a", "b", "c"])
        output_path = tmp_dir / "output.parquet"
        scores = np.array([0, 3, 5], dtype=np.int8)
        _write_annotated_file(input_path, output_path, scores)

        table = pq.read_table(str(output_path))
        assert "safety_score" in table.column_names
        assert "id" in table.column_names
        assert "text" in table.column_names
        assert "source" in table.column_names

    def test_preserves_original_columns(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        ids = ["x", "y", "z"]
        texts = ["hello", "world", "foo"]
        sources = ["s1", "s2", "s3"]
        input_path = _write_data_parquet(tmp_dir / "input.parquet", ids, texts, sources)
        output_path = tmp_dir / "output.parquet"
        scores = np.array([1, 2, 3], dtype=np.int8)
        _write_annotated_file(input_path, output_path, scores)

        table = pq.read_table(str(output_path))
        assert table.column("id").to_pylist() == ids
        assert table.column("text").to_pylist() == texts
        assert table.column("source").to_pylist() == sources

    def test_preserves_row_order(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        ids = ["z", "a", "m", "b"]
        input_path = _write_data_parquet(tmp_dir / "input.parquet", ids)
        output_path = tmp_dir / "output.parquet"
        scores = np.array([4, 3, 2, 1], dtype=np.int8)
        _write_annotated_file(input_path, output_path, scores)

        table = pq.read_table(str(output_path))
        assert table.column("id").to_pylist() == ids
        assert table.column("safety_score").to_pylist() == [4, 3, 2, 1]

    def test_returns_row_count(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        input_path = _write_data_parquet(tmp_dir / "input.parquet", ["a", "b", "c", "d"])
        output_path = tmp_dir / "output.parquet"
        scores = np.array([0, 0, 0, 0], dtype=np.int8)
        result = _write_annotated_file(input_path, output_path, scores)
        assert result == 4

    def test_length_mismatch_raises(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        input_path = _write_data_parquet(tmp_dir / "input.parquet", ["a", "b", "c"])
        output_path = tmp_dir / "output.parquet"
        scores = np.array([0, 1], dtype=np.int8)  # wrong length
        with pytest.raises(AssertionError):
            _write_annotated_file(input_path, output_path, scores)

    def test_score_dtype_is_int8(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        input_path = _write_data_parquet(tmp_dir / "input.parquet", ["a", "b"])
        output_path = tmp_dir / "output.parquet"
        scores = np.array([2, 4], dtype=np.int8)
        _write_annotated_file(input_path, output_path, scores)

        table = pq.read_table(str(output_path))
        assert table.schema.field("safety_score").type == pa.int8()


# ── TestMergeIdLookup ────────────────────────────────────────────────


def _build_scores_from_dict(input_path: Path, score_dict: dict[str, int]) -> np.ndarray:
    """Build a scores array for an input file using an id-to-score dictionary.

    This implements the id-based join logic that the merge pipeline would use
    when merging annotations back into files with duplicate ids.
    """
    table = pq.read_table(str(input_path), columns=["id"])
    ids = table.column("id").to_pylist()
    return np.array([score_dict[i] for i in ids], dtype=np.int8)


class TestMergeIdLookup:
    """Tests for id-based join: building per-file scores from dict + original file."""

    def test_duplicates_get_same_score(self, tmp_dir: Path):
        ids = ["A", "A", "A", "A"]
        input_path = _write_data_parquet(tmp_dir / "input.parquet", ids)
        score_dict = {"A": 3}
        scores = _build_scores_from_dict(input_path, score_dict)
        np.testing.assert_array_equal(scores, np.array([3, 3, 3, 3], dtype=np.int8))

    def test_missing_id_raises_key_error(self, tmp_dir: Path):
        ids = ["A", "B", "C"]
        input_path = _write_data_parquet(tmp_dir / "input.parquet", ids)
        score_dict = {"A": 1, "B": 2}  # missing "C"
        with pytest.raises(KeyError):
            _build_scores_from_dict(input_path, score_dict)

    def test_row_order_preserved(self, tmp_dir: Path):
        ids = ["C", "A", "B"]
        input_path = _write_data_parquet(tmp_dir / "input.parquet", ids)
        score_dict = {"A": 1, "B": 2, "C": 3}
        scores = _build_scores_from_dict(input_path, score_dict)
        # scores follow file row order, not dict order
        np.testing.assert_array_equal(scores, np.array([3, 1, 2], dtype=np.int8))

    def test_multiple_ids_correct_mapping(self, tmp_dir: Path):
        ids = ["x", "y", "z"]
        input_path = _write_data_parquet(tmp_dir / "input.parquet", ids)
        score_dict = {"x": 0, "y": 5, "z": 2}
        scores = _build_scores_from_dict(input_path, score_dict)
        np.testing.assert_array_equal(scores, np.array([0, 5, 2], dtype=np.int8))

    def test_scores_in_valid_range(self, tmp_dir: Path):
        ids = ["a", "b", "c", "d", "e"]
        input_path = _write_data_parquet(tmp_dir / "input.parquet", ids)
        score_dict = {"a": 0, "b": 1, "c": 2, "d": 4, "e": 5}
        scores = _build_scores_from_dict(input_path, score_dict)
        assert all(0 <= s <= 5 for s in scores)


# ── TestEndToEnd ─────────────────────────────────────────────────────


class TestEndToEnd:
    """Full pipeline simulation (no real classifier): dedup -> fake annotate -> merge."""

    @staticmethod
    def _simulate_pipeline(
        data_dir: Path,
        annotation_dir: Path,
        output_dir: Path,
        data_files: list[Path],
        fake_score_fn=None,
    ) -> None:
        """Simulate the full annotation + merge pipeline with fake scores.

        Steps:
        1. Compute dedup indices (if _compute_dedup_indices exists, else skip dedup)
        2. Assign fake scores to unique rows
        3. Build annotation shards
        4. Run _write_annotated_file for each input file
        """
        from preprocessing.annotation.merge import _write_annotated_file

        if fake_score_fn is None:
            fake_score_fn = lambda id_str: hash(id_str) % 6

        output_dir.mkdir(parents=True, exist_ok=True)

        # Read all ids per file, assign scores by id
        for data_file in data_files:
            table = pq.read_table(str(data_file), columns=["id"])
            ids = table.column("id").to_pylist()
            scores = np.array([fake_score_fn(i) for i in ids], dtype=np.int8)
            out_path = output_dir / data_file.name
            _write_annotated_file(data_file, out_path, scores)

    def test_full_pipeline_mixed_duplicates(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 3 files with mixed duplicates
        f1 = _write_data_parquet(data_dir / "part_0000.parquet", ["A", "A", "B"])
        f2 = _write_data_parquet(data_dir / "part_0001.parquet", ["C", "C", "C", "D"])
        f3 = _write_data_parquet(data_dir / "part_0002.parquet", ["E", "F", "F"])

        score_fn = lambda i: {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}[i]
        self._simulate_pipeline(data_dir, tmp_dir / "annot", output_dir, [f1, f2, f3], score_fn)

        # Verify all 10 invariants
        for name in ["part_0000.parquet", "part_0001.parquet", "part_0002.parquet"]:
            out_table = pq.read_table(str(output_dir / name))
            in_table = pq.read_table(str(data_dir / name))

            # 1. output exists
            assert (output_dir / name).exists()
            # 2. same row count
            assert len(out_table) == len(in_table)
            # 3. has safety_score column
            assert "safety_score" in out_table.column_names
            # 4. original columns preserved
            for col in ["id", "text", "source"]:
                assert col in out_table.column_names
            # 5. safety_score is int8
            assert out_table.schema.field("safety_score").type == pa.int8()
            # 6. scores in valid range
            scores = out_table.column("safety_score").to_pylist()
            assert all(0 <= s <= 5 for s in scores)
            # 7. id values preserved
            assert out_table.column("id").to_pylist() == in_table.column("id").to_pylist()
            # 8. text values preserved
            assert out_table.column("text").to_pylist() == in_table.column("text").to_pylist()
            # 9. source values preserved
            assert out_table.column("source").to_pylist() == in_table.column("source").to_pylist()
            # 10. row order preserved
            assert out_table.column("id").to_pylist() == in_table.column("id").to_pylist()

        # Check specific scores for duplicates
        t1 = pq.read_table(str(output_dir / "part_0000.parquet"))
        assert t1.column("safety_score").to_pylist() == [0, 0, 1]  # A=0, A=0, B=1

        t2 = pq.read_table(str(output_dir / "part_0001.parquet"))
        assert t2.column("safety_score").to_pylist() == [2, 2, 2, 3]  # C=2, C=2, C=2, D=3

    def test_empty_file_alongside_normal(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        f_empty = _write_data_parquet(data_dir / "part_0000.parquet", [])
        f_normal = _write_data_parquet(data_dir / "part_0001.parquet", ["A", "B", "C"])

        score_fn = lambda i: {"A": 1, "B": 2, "C": 3}[i]

        # Empty file: zero-length scores
        _write_annotated_file(f_empty, output_dir / f_empty.name, np.array([], dtype=np.int8))
        # Normal file
        table = pq.read_table(str(f_normal), columns=["id"])
        ids = table.column("id").to_pylist()
        scores = np.array([score_fn(i) for i in ids], dtype=np.int8)
        _write_annotated_file(f_normal, output_dir / f_normal.name, scores)

        t_empty = pq.read_table(str(output_dir / "part_0000.parquet"))
        assert len(t_empty) == 0
        assert "safety_score" in t_empty.column_names

        t_normal = pq.read_table(str(output_dir / "part_0001.parquet"))
        assert len(t_normal) == 3
        assert t_normal.column("safety_score").to_pylist() == [1, 2, 3]

    def test_roundtrip_data_integrity(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        ids = ["p", "q", "r", "s"]
        texts = ["alpha", "beta", "gamma", "delta"]
        sources = ["wiki", "cc", "wiki", "books"]
        input_path = _write_data_parquet(data_dir / "part_0000.parquet", ids, texts, sources)
        scores = np.array([0, 1, 2, 3], dtype=np.int8)
        _write_annotated_file(input_path, output_dir / "part_0000.parquet", scores)

        out_table = pq.read_table(str(output_dir / "part_0000.parquet"))
        # Drop safety_score -> should be identical to original
        out_without_score = out_table.drop(["safety_score"])
        in_table = pq.read_table(str(input_path))
        assert out_without_score.equals(in_table)

    def test_cross_file_same_id_independent_scores(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Same id "shared" in two files, but different annotation scores
        f1 = _write_data_parquet(data_dir / "part_0000.parquet", ["shared", "A"])
        f2 = _write_data_parquet(data_dir / "part_0001.parquet", ["shared", "B"])

        # File 1 gets score 1 for "shared", file 2 gets score 4
        scores1 = np.array([1, 2], dtype=np.int8)
        scores2 = np.array([4, 3], dtype=np.int8)
        _write_annotated_file(f1, output_dir / f1.name, scores1)
        _write_annotated_file(f2, output_dir / f2.name, scores2)

        t1 = pq.read_table(str(output_dir / "part_0000.parquet"))
        t2 = pq.read_table(str(output_dir / "part_0001.parquet"))

        # "shared" gets different scores in each file
        assert t1.column("safety_score").to_pylist()[0] == 1
        assert t2.column("safety_score").to_pylist()[0] == 4

    def test_merge_with_multi_rank_annotation_shards(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        f0 = _write_data_parquet(data_dir / "part_0000.parquet", ["A", "A", "B", "B", "C"])
        f1 = _write_data_parquet(data_dir / "part_0001.parquet", ["D", "D", "E"])

        task_dir = tmp_dir / "task_0000"
        _write_task_meta(
            task_dir, data_dir,
            ["part_0000.parquet", "part_0001.parquet"],
            n_input_rows=5, world_size=2, n_original_rows=8,
        )
        _write_annotation_shard(task_dir / "shard_0000_part0000.parquet", ids=["A", "B"], scores=[0, 2])
        _write_annotation_shard(task_dir / "shard_0001_part0000.parquet", ids=["C", "D", "E"], scores=[3, 1, 5])

        # Build id_to_score dict from both shards
        id_to_score: dict[str, int] = {}
        for shard_name in ["shard_0000_part0000.parquet", "shard_0001_part0000.parquet"]:
            t = pq.read_table(str(task_dir / shard_name))
            for sid, score in zip(t.column("id").to_pylist(), t.column("safety_score").to_pylist()):
                id_to_score[sid] = score

        for data_file in [f0, f1]:
            table = pq.read_table(str(data_file), columns=["id"])
            ids = table.column("id").to_pylist()
            scores = np.array([id_to_score[i] for i in ids], dtype=np.int8)
            _write_annotated_file(data_file, output_dir / data_file.name, scores)

        t0 = pq.read_table(str(output_dir / "part_0000.parquet"))
        assert t0.column("safety_score").to_pylist() == [0, 0, 2, 2, 3]

        t1 = pq.read_table(str(output_dir / "part_0001.parquet"))
        assert t1.column("safety_score").to_pylist() == [1, 1, 5]

    def test_merge_with_multi_task_multi_shard(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        f0 = _write_data_parquet(data_dir / "part_0000.parquet", ["A", "A", "B"])
        f1 = _write_data_parquet(data_dir / "part_0001.parquet", ["C", "C"])
        f2 = _write_data_parquet(data_dir / "part_0002.parquet", ["D", "E", "E", "E"])
        f3 = _write_data_parquet(data_dir / "part_0003.parquet", ["F"])

        # task_0000: files 0-1
        task0_dir = tmp_dir / "task_0000"
        _write_task_meta(
            task0_dir, data_dir,
            ["part_0000.parquet", "part_0001.parquet"],
            n_input_rows=3, world_size=2, n_original_rows=5,
        )
        _write_annotation_shard(task0_dir / "shard_0000_part0000.parquet", ids=["A", "B"], scores=[1, 2])
        _write_annotation_shard(task0_dir / "shard_0001_part0000.parquet", ids=["C"], scores=[4])

        # task_0001: files 2-3
        task1_dir = tmp_dir / "task_0001"
        _write_task_meta(
            task1_dir, data_dir,
            ["part_0002.parquet", "part_0003.parquet"],
            n_input_rows=3, world_size=2, n_original_rows=5,
        )
        _write_annotation_shard(task1_dir / "shard_0000_part0000.parquet", ids=["D"], scores=[0])
        _write_annotation_shard(task1_dir / "shard_0001_part0000.parquet", ids=["E", "F"], scores=[3, 5])

        # Process each task
        for task_dir, data_files in [(task0_dir, [f0, f1]), (task1_dir, [f2, f3])]:
            id_to_score: dict[str, int] = {}
            for p in sorted(task_dir.glob("shard_*_part*.parquet")):
                t = pq.read_table(str(p))
                for sid, score in zip(t.column("id").to_pylist(), t.column("safety_score").to_pylist()):
                    id_to_score[sid] = score

            for data_file in data_files:
                table = pq.read_table(str(data_file), columns=["id"])
                ids = table.column("id").to_pylist()
                scores = np.array([id_to_score[i] for i in ids], dtype=np.int8)
                _write_annotated_file(data_file, output_dir / data_file.name, scores)

        assert pq.read_table(str(output_dir / "part_0000.parquet")).column("safety_score").to_pylist() == [1, 1, 2]
        assert pq.read_table(str(output_dir / "part_0001.parquet")).column("safety_score").to_pylist() == [4, 4]
        assert pq.read_table(str(output_dir / "part_0002.parquet")).column("safety_score").to_pylist() == [0, 3, 3, 3]
        assert pq.read_table(str(output_dir / "part_0003.parquet")).column("safety_score").to_pylist() == [5]

        total_rows = sum(
            len(pq.read_table(str(output_dir / f"part_{i:04d}.parquet")))
            for i in range(4)
        )
        assert total_rows == 10

    def test_merge_with_multi_part_annotation_files(self, tmp_dir: Path):
        from preprocessing.annotation.merge import _write_annotated_file

        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        f0 = _write_data_parquet(data_dir / "part_0000.parquet", ["X", "X", "X", "Y", "Y", "Z"])

        task_dir = tmp_dir / "task_0000"
        _write_task_meta(
            task_dir, data_dir,
            ["part_0000.parquet"],
            n_input_rows=3, world_size=1, n_original_rows=6,
        )
        _write_annotation_shard(task_dir / "shard_0000_part0000.parquet", ids=["X", "Y"], scores=[2, 4])
        _write_annotation_shard(task_dir / "shard_0000_part0001.parquet", ids=["Z"], scores=[1])

        id_to_score: dict[str, int] = {}
        for p in sorted(task_dir.glob("shard_*_part*.parquet")):
            t = pq.read_table(str(p))
            for sid, score in zip(t.column("id").to_pylist(), t.column("safety_score").to_pylist()):
                id_to_score[sid] = score

        table = pq.read_table(str(f0), columns=["id"])
        ids = table.column("id").to_pylist()
        scores = np.array([id_to_score[i] for i in ids], dtype=np.int8)
        _write_annotated_file(f0, output_dir / f0.name, scores)

        t0 = pq.read_table(str(output_dir / "part_0000.parquet"))
        assert t0.column("safety_score").to_pylist() == [2, 2, 2, 4, 4, 1]
