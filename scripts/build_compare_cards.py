"""Merge matched charter.eval arms into one payload for the constitution review.

One arm is the ``SUBJECT`` under assessment; the others ride along as reference,
so a reviewer can see how the same document was annotated under a different
constitution without that being what they are voting on.

Every arm annotated the same documents at the same reflection points, so the
document is stored once and each arm contributes only its own annotation.

Per-arm constitution sections are mandatory, not a nicety: all three
constitutions number their sections identically (1.1, 2.3, ...) while meaning
different things, so a single shared section map would show one arm's text under
another arm's citation. ``sections`` is therefore keyed by arm.

Usage:
    python3 scripts/build_compare_cards.py            # -> prompt_pipeline/compare_cards.json
    python3 scripts/build_compare_cards.py --out PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.charter.eval.report import _judgment_card, parse_charter_sections
from pipeline.config import PROJECT_ROOT

# label -> (cards json, constitution, generator prompt, guidelines)
#
# All arms run qwen3.5-35b-a3b, the model the 51M-document production run used.
# qwen3.6 systematically under-cites on the same inputs (46/100 vs 29/100 items
# cited, 80% vs 50% at safety 4), so a qwen3.6 comparison would be internally
# consistent but not calibrated to the corpus these constitutions feed.
ARMS: dict[str, tuple[str, str, str, str]] = {
    "MR v0.2": (
        "data/pipeline/charter_eval/mr_v02_matched_100_q35/cards.json",
        "resources/ModelRaisingConstitution_v0.2.md",
        "generator_reflection_v7.md",
        "resources/ValueAnnotationGuidelines_v0.1.md",
    ),
    "Normative hierarchy": (
        "data/pipeline/charter_eval/normative_matched_100_q35/cards.json",
        "resources/NormativeHierarchyConstitution_v0.1.md",
        "generator_reflection_normative_hierarchy_v1.md",
        "resources/NormativeHierarchyAnnotationGuidelines_v0.1.md",
    ),
    "Utilitarian v0.3": (
        # _g3 = guidelines v0.3, canary-free; the v0.1 run stays on disk as
        # utilitarian_matched_100_q35.
        "data/pipeline/charter_eval/utilitarian_matched_100_q35_g3/cards.json",
        "resources/UtilitarianConstitution_v0.1.md",
        "generator_reflection_v7.md",
        "resources/UtilitarianAnnotationGuidelines_v0.3.md",
    ),
    "Utilitarian v1.0": (
        # _g10 = guidelines v1.0 (v0.3 + the numbers-calibration layer); same
        # documents and cut points as _g3, regenerated under the new guidelines.
        "data/pipeline/charter_eval/utilitarian_matched_100_q35_g10/cards.json",
        "resources/UtilitarianConstitution_v0.1.md",
        "generator_reflection_v7.md",
        "resources/UtilitarianAnnotationGuidelines_v1.0.md",
    ),
}

# The arm being assessed. The others are shown for reference only, so the review
# page pins this one first and never hides it.
SUBJECT = "Utilitarian v1.0"

# The same reflections were judged more than once while the rubric was being
# worked on. cards.json keeps only the last judge file, so the runs are listed
# here explicitly: label -> path under the run directory. Every one is shipped
# and the page offers a selector; the LAST entry is what it shows first.
#
# "stakes vN" = judge v2.5 reading a stake sheet built with judge_source_stakes_vN.
_G3 = "data/pipeline/charter_eval/utilitarian_matched_100_q35_g3/judgments/"
_G10 = "data/pipeline/charter_eval/utilitarian_matched_100_q35_g10/judgments/"
_ON = "__on__qwen3.5-35b-a3b__generator_reflection_v7.md.jsonl"
JUDGMENTS: dict[str, list[tuple[str, str]]] = {
    "Utilitarian v0.3": [
        ("judge v2.2", _G3 + "kimi-k2.5__judge_reflection_utilitarian_1p_v2.2.md" + _ON),
        ("judge v2.3", _G3 + "kimi-k2.5__judge_reflection_utilitarian_1p_v2.3.md" + _ON),
    ],
    "Utilitarian v1.0": [
        ("judge v2.2", _G10 + "kimi-k2.5__judge_reflection_utilitarian_1p_v2.2.md" + _ON),
        ("judge v2.3", _G10 + "kimi-k2.5__judge_reflection_utilitarian_1p_v2.3.md" + _ON),
        ("judge v2.4", _G10 + "kimi-k2.5__judge_reflection_utilitarian_1p_v2.4.md" + _ON),
        ("judge v2.5 · stakes v1", _G10 + "stake_sheet_variants/kimi-k2.5__judge_reflection_utilitarian_1p_v2.5.md__stakes_v1" + _ON),
        ("judge v2.5 · stakes v2", _G10 + "stake_sheet_variants/kimi-k2.5__judge_reflection_utilitarian_1p_v2.5.md__stakes_v2" + _ON),
        ("judge v2.5 · stakes v3", _G10 + "stake_sheet_variants/kimi-k2.5__judge_reflection_utilitarian_1p_v2.5.md__stakes_v3" + _ON),
        ("judge v2.5 · stakes v4", _G10 + "stake_sheet_variants/kimi-k2.5__judge_reflection_utilitarian_1p_v2.5.md__stakes_v4" + _ON),
        ("judge v2.5 · stakes v5", _G10 + "stake_sheet_variants/kimi-k2.5__judge_reflection_utilitarian_1p_v2.5.md__stakes_v5" + _ON),
        ("judge v2.5 · stakes v7", _G10 + "kimi-k2.5__judge_reflection_utilitarian_1p_v2.5.md" + _ON),
    ],
}

JUDGE_FIELDS = (
    "judge_model",
    "judge_prompt",
    "judge_scores",
    "judge_aggregate",
    "judge_decision",
    "judge_reasoning",
)


def _load_judgments(path: str) -> dict[str, dict]:
    """One judge file -> item_id -> judge fields (+ the stake sheet it read).

    A file may hold an item twice when a resume pass re-read it; the last row
    is the one the run ended with, so it wins.
    """
    stem = Path(path).stem
    judge_stem, gen_stem = stem.split("__on__", 1)
    out: dict[str, dict] = {}
    for line in (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        card = _judgment_card("", judge_stem, gen_stem, row)
        j = {f: card[f] for f in JUDGE_FIELDS}
        # The file stem carries the sheet variant; the prompt the judge actually
        # ran is recorded in the row, so take the name from there.
        j["judge_prompt"] = row["judgment"].get("judge_prompt_reflection") or j["judge_prompt"]
        # Present only for rubrics that read a stake sheet; [] means the sheet
        # read the document as benign, which is itself worth seeing.
        if row.get("source_stakes") is not None:
            j["source_stakes"] = row["source_stakes"]
        out[str(row["item_id"])] = j
    assert out, f"no judgments in {path}"
    return out

ANNOTATION_FIELDS = (
    "analysis",
    "reflection_1p",
    "reflection_3p",
    "charter_elements",
    "judge_model",
    "judge_scores",
    "judge_aggregate",
    "judge_decision",
    "judge_reasoning",
)


def build(arms: dict[str, tuple[str, str, str, str]], subject: str = SUBJECT) -> dict:
    """Merge the arms into ``{subject, runs, sections, items}``, one per document."""
    assert subject in arms, f"subject {subject!r} is not one of the arms: {list(arms)}"
    loaded: dict[str, dict[str, dict]] = {}
    meta: dict[str, dict] = {}
    sections: dict[str, dict[str, str]] = {}

    for label, (cards_path, charter, prompt, guidelines) in arms.items():
        payload = json.loads((PROJECT_ROOT / cards_path).read_text(encoding="utf-8"))
        cards = {c["item_id"]: c for c in payload["cards"]}
        assert cards, f"{label}: no cards in {cards_path}"
        loaded[label] = cards
        charter_text = (PROJECT_ROOT / charter).read_text(encoding="utf-8")
        sections[label] = parse_charter_sections(charter_text)
        run_ids = sorted({c["run_id"] for c in cards.values()})
        assert len(run_ids) == 1, f"{label}: expected one run, got {run_ids}"
        meta[label] = {
            "label": label,
            "run_id": run_ids[0],
            "constitution": Path(charter).name,
            "guidelines": Path(guidelines).name,
            "prompt": prompt,
            "gen_model": next(iter(cards.values()))["gen_model"],
            "n_sections": len(sections[label]),
            "judged": sum(1 for c in cards.values() if c.get("judge_decision")),
            # Each arm is judged by its own rubric — name it, so a score is
            # always traceable to the rubric that produced it.
            "judge_model": next(iter(cards.values())).get("judge_model"),
            "judge_prompt": next(iter(cards.values())).get("judge_prompt"),
        }

    # Judge runs per arm. An arm without an explicit list carries the single
    # judgment already on its cards, labelled by its rubric name.
    judged: dict[str, dict[str, dict[str, dict]]] = {}
    for label in arms:
        if label in JUDGMENTS:
            runs = {jl: _load_judgments(p) for jl, p in JUDGMENTS[label]}
        else:
            jl = Path(meta[label]["judge_prompt"] or "judge").stem.removeprefix("judge_reflection_")
            runs = {
                jl: {
                    i: {f: c.get(f) for f in JUDGE_FIELDS}
                    for i, c in loaded[label].items()
                    if c.get("judge_decision")
                }
            }
        judged[label] = runs
        meta[label]["judgments"] = list(runs)
        meta[label]["judge_default"] = list(runs)[-1]
        default = runs[meta[label]["judge_default"]]
        assert default, f"{label}: default judge run {meta[label]['judge_default']!r} has no judgments"
        first = next(iter(default.values()))
        meta[label]["judge_model"] = first["judge_model"]
        meta[label]["judge_prompt"] = first["judge_prompt"]
        meta[label]["judged"] = len(default)

    shared = set.intersection(*[set(c) for c in loaded.values()])
    dropped = {lab: sorted(set(c) - shared) for lab, c in loaded.items()}
    for lab, missing in dropped.items():
        if missing:
            print(f"note: {lab} has {len(missing)} item(s) no other arm covers; excluded")

    labels = list(arms)
    items = []
    for item_id in sorted(shared):
        base = loaded[labels[0]][item_id]
        for lab in labels[1:]:
            other = loaded[lab][item_id]
            assert other["text"] == base["text"], f"{item_id}: text differs in {lab}"
            assert other["reflection_point"] == base["reflection_point"], (
                f"{item_id}: reflection_point differs in {lab} — arms are not matched"
            )
        arm_views: dict[str, dict] = {}
        for lab in labels:
            view = {f: loaded[lab][item_id].get(f) for f in ANNOTATION_FIELDS}
            for jl, by_item in judged[lab].items():
                assert item_id in by_item, f"{lab} / {jl}: no judgment for {item_id}"
            view["judgments"] = {jl: by_item[item_id] for jl, by_item in judged[lab].items()}
            # The flat fields are what the page shows before a run is picked.
            view.update(view["judgments"][meta[lab]["judge_default"]])
            arm_views[lab] = view
        items.append(
            {
                "item_id": item_id,
                "text": base["text"],
                "safety_score": base["safety_score"],
                "reflection_point": base["reflection_point"],
                "arms": arm_views,
            }
        )
    # Subject first: the page renders arms in payload order.
    labels = [subject] + [lab for lab in labels if lab != subject]
    return {
        "subject": subject,
        "runs": [meta[lab] for lab in labels],
        "sections": sections,
        "items": items,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="prompt_pipeline/compare_cards.json", type=Path)
    args = ap.parse_args()

    payload = build(ARMS)
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(payload['items'])} items x {len(payload['runs'])} arms")
    for r in payload["runs"]:
        role = "SUBJECT" if r["label"] == payload["subject"] else "ref"
        print(
            f"  {role:>7}  {r['label']:22} {r['constitution']:38} {r['n_sections']:>3} sections"
            f"  judged={r['judged']:>3}"
        )
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
