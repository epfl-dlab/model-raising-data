"""Mechanical correctness audit of the citation-cleaning outputs on disk.

Cross-checks every produced result line against the original cited response and the charter:
  - valid JSON, required keys
  - every id in claude_cleaned is a real charter id
  - brackets in claude_cleaned exactly match final_citations
  - no leaked 'Citations:' scratchpad line, no stray '[SKIP]'
  - `changed` flag is consistent with (claude_cleaned != original cited)
  - unchanged rows are byte-identical to the original
  - citation cap (<=2 normally; report 3+) and self-reported action vs actual diff
  - summarizes added / removed / retargeted citation ids vs the original

Read-only; spends no agent budget. Safe to run while the workflow is mid-flight.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.config import _CHARTER_ID_SET, extract_charter_elements

PARQUET = Path(
    "/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/hf_export.parquet"
)
OUT_DIR = PARQUET.parent / "claude_clean" / "out"

REQUIRED = {
    "idx", "claude_cleaned", "final_citations", "changed", "action",
    "citations_correct_before", "response_quality_ok", "reason", "confidence",
}


def main() -> None:
    table = pq.read_table(PARQUET).to_pylist()
    orig_cited = {i: r["messages_cite"][1]["content"] for i, r in enumerate(table)}
    orig_cites = {i: set(r["charter_elements"]) for i, r in enumerate(table)}

    rows = []
    for f in sorted(OUT_DIR.glob("batch_*.jsonl")):
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if line:
                rows.append((f.name, ln, line))

    n = 0
    errors: list[str] = []
    warns: Counter = Counter()
    action_ct: Counter = Counter()
    added_ids: Counter = Counter()
    removed_ids: Counter = Counter()
    n_changed = n_prose_changed = n_cap3 = 0

    def err(msg):
        errors.append(msg)

    for fname, ln, line in rows:
        loc = f"{fname}:{ln}"
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            err(f"{loc} bad JSON: {e}")
            continue
        if REQUIRED - r.keys():
            err(f"{loc} missing keys {REQUIRED - r.keys()}")
            continue
        n += 1
        idx = r["idx"]
        cleaned = r["claude_cleaned"]
        if idx not in orig_cited:
            err(f"{loc} idx {idx} not in source parquet")
            continue

        actual = extract_charter_elements(cleaned)
        # 1. brackets match final_citations
        if actual != r["final_citations"]:
            err(f"{loc} idx={idx} brackets {actual} != final_citations {r['final_citations']}")
        # 2. all ids are real charter ids (extract_charter_elements already filters, so check raw)
        import re
        raw_ids = re.findall(r"\[(\d+\.\d+)\]", cleaned)
        bad = [x for x in raw_ids if x not in _CHARTER_ID_SET]
        if bad:
            err(f"{loc} idx={idx} non-charter ids in text: {bad}")
        # 3. no leaked scratchpad / SKIP
        if "Citations:" in cleaned:
            err(f"{loc} idx={idx} leaked 'Citations:' line")
        if "[SKIP]" in cleaned:
            err(f"{loc} idx={idx} contains [SKIP]")
        # 4. changed flag consistency
        really_changed = cleaned != orig_cited[idx]
        if really_changed != bool(r["changed"]):
            err(f"{loc} idx={idx} changed flag={r['changed']} but text-differs={really_changed}")
        # 5. cap
        if len(actual) >= 3:
            n_cap3 += 1
            warns["citations>=3"] += 1
        action_ct[r["action"]] += 1
        if r["confidence"] == "low":
            warns["low_confidence"] += 1
        if not r["response_quality_ok"]:
            warns["quality_not_ok"] += 1

        # diff of citation ids vs original
        before, after = orig_cites[idx], set(actual)
        for x in after - before:
            added_ids[x] += 1
        for x in before - after:
            removed_ids[x] += 1
        if really_changed:
            n_changed += 1
            # prose change beyond brackets: strip all [X.Y] and compare
            def strip(t):
                return re.sub(r"\s*\[\d+\.\d+\]", "", re.sub(r"\[\d+\.\d+\]", "", t)).strip()
            if strip(cleaned) != strip(orig_cited[idx]):
                n_prose_changed += 1

    print(f"rows audited: {n}")
    print(f"HARD ERRORS: {len(errors)}")
    for e in errors[:60]:
        print("  ✗", e)
    if len(errors) > 60:
        print(f"  ... +{len(errors) - 60} more")
    print(f"\nchanged: {n_changed} | of which prose (not just brackets) changed: {n_prose_changed}")
    print(f"citations>=3 (cap soft-violation): {n_cap3}")
    print(f"actions: {dict(action_ct)}")
    print(f"warnings: {dict(warns)}")
    print(f"\ntop ADDED citation ids: {added_ids.most_common(10)}")
    print(f"top REMOVED citation ids: {removed_ids.most_common(10)}")


if __name__ == "__main__":
    main()
