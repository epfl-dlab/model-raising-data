"""Tests for charter.scale canary assignment."""

from __future__ import annotations

import pytest

from pipeline.charter.scale import canaries as canaries_mod
from pipeline.charter.scale.canaries import (
    assign_canary,
    load_canaries,
    pretraining_canaries,
)


class TestLoadCanaries:
    def test_loads_10_canaries(self):
        canaries = load_canaries()
        assert len(canaries) == 10

    def test_canary_has_required_fields(self):
        canaries = load_canaries()
        for c in canaries:
            assert "id" in c
            assert "instruction" in c
            assert "instruction_3p" in c

    def test_canary_ids_are_Q1_to_Q10(self):
        canaries = load_canaries()
        ids = {c["id"] for c in canaries}
        assert ids == {f"Q{i}" for i in range(1, 11)}


class TestAssignCanary:
    def test_deterministic(self):
        canaries = load_canaries()
        r1 = assign_canary("doc_abc", 42, canaries)
        r2 = assign_canary("doc_abc", 42, canaries)
        assert r1 == r2

    def test_different_seed_different_result(self):
        canaries = load_canaries()
        # Run on many doc_ids to check at least one differs
        results_42 = [assign_canary(f"doc_{i}", 42, canaries) for i in range(100)]
        results_99 = [assign_canary(f"doc_{i}", 99, canaries) for i in range(100)]
        assert results_42 != results_99

    def test_rate_roughly_10_percent(self):
        canaries = load_canaries()
        n = 10000
        assigned = sum(
            1 for i in range(n)
            if assign_canary(f"doc_{i}", 42, canaries) is not None
        )
        rate = assigned / n
        # Should be roughly 10% (within 8-12%)
        assert 0.08 <= rate <= 0.12, f"Canary rate {rate:.3f} outside expected range"

    def test_uniform_canary_distribution(self):
        canaries = load_canaries()
        n = 100000
        counts: dict[str, int] = {}
        for i in range(n):
            c = assign_canary(f"doc_{i}", 42, canaries)
            if c is not None:
                cid = c["id"]
                counts[cid] = counts.get(cid, 0) + 1

        # All 10 canaries should appear
        assert len(counts) == 10
        # Each should be roughly 1% of total (10% / 10 canaries)
        total_assigned = sum(counts.values())
        for cid, cnt in counts.items():
            share = cnt / total_assigned
            assert 0.05 <= share <= 0.15, (
                f"Canary {cid} has {share:.3f} share, expected ~0.10"
            )

    def test_none_when_not_selected(self):
        canaries = load_canaries()
        # With rate=0 no one should be selected
        result = assign_canary("doc_0", 42, canaries, rate=0.0)
        assert result is None

    def test_always_when_rate_1(self):
        canaries = load_canaries()
        result = assign_canary("doc_0", 42, canaries, rate=1.0)
        assert result is not None

    def test_none_when_canaries_empty(self):
        # An empty canary list disables injection entirely — even at rate=1.0,
        # there is nothing to assign. Used by the eval pipeline (disable_canaries).
        assert assign_canary("doc_0", 42, [], rate=1.0) is None
        assert assign_canary("doc_1", 7, [], rate=1.0) is None


class TestPretrainingCanaries:
    """pretraining_action gates which canaries are woven into the
    charter.scale reflection (pretraining) runs — only identity facts."""

    EXPECTED_INJECT = {"Q1", "Q2", "Q3", "Q7", "Q10"}

    def test_every_canary_has_valid_pretraining_action(self):
        for c in load_canaries():
            assert c.get("pretraining_action") in {"inject", "skip"}, (
                f"{c['id']} has invalid pretraining_action "
                f"{c.get('pretraining_action')!r}"
            )

    def test_inject_set_is_identity(self):
        ids = {c["id"] for c in pretraining_canaries()}
        assert ids == self.EXPECTED_INJECT

    def test_assign_only_yields_inject_set(self):
        canaries = pretraining_canaries()
        for i in range(20000):
            c = assign_canary(f"doc_{i}", 42, canaries)
            if c is not None:
                assert c["id"] in self.EXPECTED_INJECT

    def test_rate_roughly_10_percent(self):
        canaries = pretraining_canaries()
        n = 10000
        assigned = sum(
            1 for i in range(n)
            if assign_canary(f"doc_{i}", 42, canaries) is not None
        )
        rate = assigned / n
        assert 0.08 <= rate <= 0.12, f"Canary rate {rate:.3f} outside expected range"

    def test_invalid_pretraining_action_crashes(self, monkeypatch):
        # Fail-fast: a missing/misspelled gate must crash, never silently drop.
        monkeypatch.setattr(
            canaries_mod,
            "load_canaries",
            lambda: [{"id": "Qx", "pretraining_action": "injct"}],
        )
        with pytest.raises(AssertionError):
            canaries_mod.pretraining_canaries()
