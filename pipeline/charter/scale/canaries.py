"""Deterministic canary assignment for charter.scale generation.

Each document has a fixed probability (default 10%) of receiving a canary
quirk. The assignment is deterministic in (canary_seed, doc_id) so that
multiple charter.scale runs produce identical canary assignments.
"""

from __future__ import annotations

import random

import yaml

from pipeline.config import PROJECT_ROOT

CANARIES_PATH = PROJECT_ROOT / "resources" / "canaries.yaml"


def load_canaries() -> list[dict]:
    """Load all canary quirks from resources/canaries.yaml."""
    with open(CANARIES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["canaries"]


def pretraining_canaries() -> list[dict]:
    """Canaries eligible for injection into charter.scale reflection runs.

    Gated on the ``pretraining_action`` field (mirrors
    ``sft.single_turn.canaries.injectable_canaries`` and its ``sft_action``
    gate): only identity facts are woven into the pretraining annotations;
    preference/opinion quirks are skipped.

    Every canary must carry an explicit ``inject``/``skip`` gate — a missing
    or misspelled value crashes rather than silently dropping the canary, so
    a bad edit can never quietly under-inject an identity fact.
    """
    canaries = load_canaries()
    for c in canaries:
        action = c.get("pretraining_action")
        assert action in {"inject", "skip"}, (
            f"Canary {c.get('id')!r} has invalid pretraining_action {action!r}; "
            "expected 'inject' or 'skip'"
        )
    return [c for c in canaries if c["pretraining_action"] == "inject"]


def assign_canary(
    doc_id: str,
    canary_seed: int,
    canaries: list[dict],
    rate: float = 0.10,
) -> dict | None:
    """Deterministic canary assignment. Returns canary dict or None.

    The RNG is seeded with ``f"{canary_seed}_{doc_id}_canary_v1"`` so that:
    - The same doc always gets the same canary (or none) across runs.
    - Changing ``canary_seed`` reshuffles assignments globally.

    An empty ``canaries`` list disables injection entirely (returns None for
    every doc regardless of ``rate``) — used by the eval pipeline.
    """
    if not canaries:
        return None
    rng = random.Random(f"{canary_seed}_{doc_id}_canary_v1")
    if rng.random() >= rate:
        return None
    return rng.choice(canaries)
