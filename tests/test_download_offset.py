"""Tests for download.py --shard-offset (disjoint complement-shard selection).

The eval pipeline reuses download.py to pull dolma3 shards that were NEVER in
the training download plan. Training used `--n-shards 47142 --shuffle --seed 42`
(taking shards [0:47142] of the seed-42 shuffle of range(n_total)). The eval set
must take a disjoint slice [offset:offset+n] of the SAME shuffle, so the two plans
share no upstream shard index.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from preprocessing.download.download import _load_or_create_manifest


def _args(n_shards, seed=42, offset=0, shuffle=True):
    return SimpleNamespace(
        dataset="allenai/dolma3_mix-6T",
        subset=None,
        columns=["text", "id", "source"],
        n_shards=n_shards,
        seed=seed,
        shuffle=shuffle,
        shard_offset=offset,
    )


class TestShardOffset:
    def test_offset_zero_is_prefix(self, tmp_path):
        """offset=0 reproduces the original prefix behaviour."""
        order = _load_or_create_manifest(tmp_path, _args(n_shards=100, offset=0), n_total=1000)
        assert len(order) == 100

    def test_complement_is_disjoint(self, tmp_path):
        """The training plan [0:47142] and an eval plan at offset=47142 share
        no upstream shard index — the core disjointness guarantee."""
        n_total = 63911
        train_dir = tmp_path / "train"
        eval_dir = tmp_path / "eval"
        train_dir.mkdir()
        eval_dir.mkdir()
        train = set(
            _load_or_create_manifest(train_dir, _args(n_shards=47142, offset=0), n_total)
        )
        evalp = set(
            _load_or_create_manifest(eval_dir, _args(n_shards=50, offset=47142), n_total)
        )
        assert len(train) == 47142
        assert len(evalp) == 50
        assert train.isdisjoint(evalp)

    def test_offset_slice_matches_manual_shuffle(self, tmp_path):
        """The offset slice equals shuffle(range(n))[offset:offset+n] exactly."""
        import random

        n_total = 1000
        expected_full = list(range(n_total))
        random.Random(42).shuffle(expected_full)
        order = _load_or_create_manifest(tmp_path, _args(n_shards=30, offset=200), n_total)
        assert order == expected_full[200:230]

    def test_offset_persisted_in_manifest(self, tmp_path):
        _load_or_create_manifest(tmp_path, _args(n_shards=10, offset=5), n_total=100)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["shard_offset"] == 5
