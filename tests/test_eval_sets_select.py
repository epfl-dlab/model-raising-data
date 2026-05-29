"""Tests for pipeline.eval_sets.select — enriched + benign-negative selection (R1/R2)
and the training-faithful eval sidecar builder (D4).

Specs:
- stratified_select: take up to `target_harmful` docs with safety_score>=3 (preserving
  the natural 3/4/5 mix), plus benign negatives (score<=2) at `neg_ratio` of the harmful
  count, weighted toward score 2. Deterministic; selection ⊆ input; no duplicates; rows
  tagged with `strata` in {"harmful","benign_negative"}.
- build_eval_sidecar: writes a parquet with SidecarReader's columns; renames the source
  `id` to `doc_id`; computes `token_length` via the EOS-aware truncation path (non-null,
  positive); sets is_bad = safety_score>=3.
"""

from __future__ import annotations

import pyarrow.parquet as pq

from pipeline.eval_sets.select import build_eval_sidecar, stratified_select


def _rows(counts: dict[int, int]):
    """Build synthetic classified rows: {score: n}."""
    out = []
    k = 0
    for score, n in counts.items():
        for _ in range(n):
            out.append({"id": f"d{k}", "text": f"text {k}", "source": "dolma", "safety_score": score})
            k += 1
    return out


class TestStratifiedSelect:
    def test_caps_at_target_harmful(self):
        rows = _rows({0: 500, 1: 100, 2: 100, 3: 200, 4: 80, 5: 80})
        sel = stratified_select(rows, target_harmful=100, neg_ratio=0.5, seed=1)
        harmful = [r for r in sel if r["strata"] == "harmful"]
        assert len(harmful) == 100
        assert all(r["safety_score"] >= 3 for r in harmful)

    def test_takes_all_harmful_when_below_target(self):
        rows = _rows({0: 500, 3: 30, 4: 10, 5: 10})
        sel = stratified_select(rows, target_harmful=10_000, neg_ratio=0.5, seed=1)
        harmful = [r for r in sel if r["strata"] == "harmful"]
        assert len(harmful) == 50  # all of 3/4/5

    def test_all_harmful_strata_represented(self):
        rows = _rows({3: 1000, 4: 400, 5: 400})
        sel = stratified_select(rows, target_harmful=180, neg_ratio=0.0, seed=2)
        scores = {r["safety_score"] for r in sel if r["strata"] == "harmful"}
        assert scores == {3, 4, 5}

    def test_negatives_ratio_and_source(self):
        rows = _rows({0: 1000, 1: 1000, 2: 1000, 3: 200})
        sel = stratified_select(rows, target_harmful=200, neg_ratio=0.5, seed=3)
        neg = [r for r in sel if r["strata"] == "benign_negative"]
        assert len(neg) == 100  # round(200 * 0.5)
        assert all(r["safety_score"] <= 2 for r in neg)

    def test_negatives_weighted_toward_2(self):
        rows = _rows({0: 5000, 1: 5000, 2: 5000, 3: 400})
        sel = stratified_select(rows, target_harmful=400, neg_ratio=1.0, seed=4)
        neg = [r for r in sel if r["strata"] == "benign_negative"]
        c = {s: sum(1 for r in neg if r["safety_score"] == s) for s in (0, 1, 2)}
        assert c[2] > c[1] > c[0]  # weights {0:1, 1:2, 2:3}

    def test_deterministic(self):
        rows = _rows({0: 300, 3: 100, 4: 50, 5: 50})
        a = stratified_select(rows, target_harmful=120, neg_ratio=0.5, seed=7)
        b = stratified_select(rows, target_harmful=120, neg_ratio=0.5, seed=7)
        assert [r["id"] for r in a] == [r["id"] for r in b]

    def test_no_duplicates_and_subset(self):
        rows = _rows({0: 300, 3: 100, 4: 50})
        sel = stratified_select(rows, target_harmful=120, neg_ratio=0.5, seed=9)
        ids = [r["id"] for r in sel]
        assert len(ids) == len(set(ids))
        assert set(ids) <= {r["id"] for r in rows}


class TestBuildEvalSidecar:
    def test_schema_and_token_length(self, tmp_path):
        rows = [
            {"id": "a", "text": "A short harmful document about wrongdoing.", "source": "dolma", "safety_score": 4, "strata": "harmful"},
            {"id": "b", "text": "A perfectly benign note.", "source": "dolma", "safety_score": 0, "strata": "benign_negative"},
        ]
        out = tmp_path / "sidecar.parquet"
        n = build_eval_sidecar(rows, str(out))
        assert n == 2
        t = pq.read_table(out)
        cols = set(t.column_names)
        assert {"doc_id", "text", "source", "safety_score", "is_bad", "token_length"} <= cols
        d = t.to_pydict()
        assert d["doc_id"] == ["a", "b"]
        assert d["is_bad"] == [True, False]
        assert all(tl > 0 for tl in d["token_length"])

    def test_token_length_caps_at_1919(self, tmp_path):
        """Long docs must clip to 1919 (1920-token window minus EOS), matching the
        training sidecar — not 1920."""
        long_text = "word " * 5000  # well over 1919 tokens
        rows = [{"id": "a", "text": long_text, "source": "dolma", "safety_score": 5, "strata": "harmful"}]
        out = tmp_path / "sidecar.parquet"
        build_eval_sidecar(rows, str(out))
        assert pq.read_table(out).to_pydict()["token_length"] == [1919]
