"""Dashboard card export for charter.eval runs."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.charter.eval.report import build_cards, write_cards


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_build_cards_from_generation_only_run(tmp_path):
    run_dir = tmp_path / "run1"
    _write_jsonl(
        run_dir / "generations" / "qwen3.6-35b-a3b__generator_reflection_normative_hierarchy_v1.md.jsonl",
        [
            {
                "item_id": "i1",
                "text": "A report about coercive pressure in a workplace.",
                "reflection_point": 24,
                "subset": "dolma3",
                "safety_score": 3,
                "analysis": "Consent is engaged. Citations: [4.6]",
                "reflection_1p": "The coercive pressure matters because consent is not meaningful here [4.6].",
                "reflection_charter_elements": ["4.6"],
            }
        ],
    )

    cards = build_cards(["run1"], eval_dir=tmp_path, source="generations")

    assert len(cards) == 1
    card = cards[0]
    assert card["run_id"] == "run1"
    assert card["item_id"] == "i1"
    assert card["gen_model"] == "qwen3.6-35b-a3b"
    assert card["gen_prompt"] == "generator_reflection_normative_hierarchy_v1.md"
    assert card["judge"] is None
    assert card["reflection_1p"].startswith("The coercive pressure")
    assert card["charter_elements"] == ["[4.6]"]


def test_write_cards_includes_constitution_sections(tmp_path):
    run_dir = tmp_path / "run1"
    _write_jsonl(
        run_dir / "generations" / "gen__prompt.md.jsonl",
        [{"item_id": "i1", "text": "x", "reflection_1p": "y [1.1]"}],
    )
    charter = tmp_path / "constitution.md"
    charter.write_text("# Constitution\n\n## 1.1 Human Dignity\n\nPersons matter.\n", encoding="utf-8")
    out = tmp_path / "cards.json"

    n = write_cards(["run1"], out, eval_dir=tmp_path, source="generations", charter_path=charter)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert n == 1
    assert payload["n_cards"] == 1
    assert "1.1" in payload["charter_sections"]
    assert "Human Dignity" in payload["charter_sections"]["1.1"]


def test_auto_source_keeps_unjudged_generation_cards(tmp_path):
    run_dir = tmp_path / "run1"
    gen_stem = "gen__prompt.md"
    gen_rows = [
        {"item_id": "i1", "text": "x", "reflection_1p": "r1 [1.1]"},
        {"item_id": "i2", "text": "y", "reflection_1p": "r2 [1.2]"},
    ]
    _write_jsonl(run_dir / "generations" / f"{gen_stem}.jsonl", gen_rows)
    _write_jsonl(
        run_dir / "judgments" / f"judge__judge.md__on__{gen_stem}.jsonl",
        [
            {
                **gen_rows[0],
                "judgment": {
                    "reflection_1p": {
                        "scores": {"relevance": 5},
                        "aggregate": 5.0,
                        "reasoning": "good",
                    },
                    "reflection_aggregate": 5.0,
                    "reflection_decision": "accept",
                },
            }
        ],
    )

    cards = build_cards(["run1"], eval_dir=tmp_path, source="auto")

    assert {c["item_id"] for c in cards} == {"i1", "i2"}
    judged = next(c for c in cards if c["item_id"] == "i1")
    unjudged = next(c for c in cards if c["item_id"] == "i2")
    assert judged["judge"] == "judge__judge.md"
    assert unjudged["judge"] is None
