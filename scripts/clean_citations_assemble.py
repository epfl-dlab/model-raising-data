"""Assemble the Claude citation-cleaning results into the final SFT eval parquet.

Reads every out/batch_*.jsonl AND rows_out/row_*.jsonl, validates citation consistency,
stamps per-row model provenance (opus / sonnet / deterministic / blocked) from
provenance.json, fills the hard-blocked residue with their original text + a `claude_blocked`
flag, and writes hf_export_cleaned.parquet with the claude_* columns. `--check` reports
coverage only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import extract_charter_elements

PARQUET = Path(
    "/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/hf_export.parquet"
)
WORK = PARQUET.parent / "claude_clean"
IN_DIR = WORK / "in"
OUT_DIR = WORK / "out"
ROWS_OUT = WORK / "rows_out"
PROVENANCE = WORK / "provenance.json"
OUT_PARQUET = PARQUET.parent / "hf_export_cleaned.parquet"

REQUIRED = {
    "idx", "claude_cleaned", "final_citations", "changed", "action",
    "citations_correct_before", "response_quality_ok", "reason", "confidence",
}


def row_isolated_idxs() -> set[int]:
    """Idxs whose final result came from a 1-row isolation agent (Sonnet)."""
    out = set()
    for f in ROWS_OUT.glob("row_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["idx"])
                except json.JSONDecodeError:
                    pass
    return out


def load_results() -> tuple[dict[int, dict], list[str]]:
    results: dict[int, dict] = {}
    problems: list[str] = []
    files = sorted(OUT_DIR.glob("batch_*.jsonl")) + sorted(ROWS_OUT.glob("row_*.jsonl"))
    for f in files:
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{f.name}:{ln} JSON error: {e}")
                continue
            if REQUIRED - r.keys():
                problems.append(f"{f.name}:{ln} idx={r.get('idx')} missing {REQUIRED - r.keys()}")
                continue
            idx = r["idx"]
            actual = extract_charter_elements(r["claude_cleaned"])
            if actual != r["final_citations"]:
                problems.append(f"{f.name}:{ln} idx={idx} brackets {actual} != stated {r['final_citations']}")
                r["final_citations"] = actual
            if "Citations:" in r["claude_cleaned"]:
                problems.append(f"{f.name}:{ln} idx={idx} leaked 'Citations:'")
            results[idx] = r  # row_*.jsonl sorts after batch_*, so isolation wins ties
    return results, problems


def batch_of_idx() -> dict[int, int]:
    out = {}
    for f in sorted(IN_DIR.glob("batch_*.json")):
        b = int(f.stem.split("_")[1])
        for r in json.loads(f.read_text(encoding="utf-8")):
            out[r["idx"]] = b
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    prov = json.loads(PROVENANCE.read_text())
    opus_blocked = set(prov["opus_blocked_batches"])
    deterministic = set(prov["deterministic_rows"])

    results, problems = load_results()
    table = pq.read_table(PARQUET)
    n = table.num_rows
    bidx = batch_of_idx()

    missing = sorted(set(range(n)) - results.keys())
    print(f"rows with cleaning result: {len(results)} / {n}")
    print(f"problems: {len(problems)}")
    for p in problems[:30]:
        print("  -", p)
    print(f"blocked residue (no result): {len(missing)} -> {missing}")

    if args.check:
        return

    orig_cited = {i: table.column("messages_cite")[i][1]["content"].as_py() for i in range(n)}
    orig_elems = {i: list(table.column("charter_elements")[i].as_py()) for i in range(n)}

    # Fill the blocked residue with original text + flag.
    for i in missing:
        results[i] = {
            "idx": i, "claude_cleaned": orig_cited[i], "final_citations": orig_elems[i],
            "changed": False, "action": "blocked", "citations_correct_before": True,
            "response_quality_ok": True,
            "reason": "Hard-blocked by usage-policy guardrail (violative cyber content) on both Opus and Sonnet; original cited text retained, not Claude-cleaned.",
            "confidence": "n/a",
        }

    rowiso = row_isolated_idxs()

    def model_of(i: int) -> str:
        if i in missing:
            return "blocked"
        if i in deterministic:
            return "deterministic"
        if i in rowiso or bidx.get(i) in opus_blocked:
            return "sonnet"
        return "opus"

    def col(key, default):
        return [results[i].get(key, default) for i in range(n)]

    models = [model_of(i) for i in range(n)]
    blocked_flag = [i in missing for i in range(n)]

    table = table.append_column("claude_cleaned", pa.array(col("claude_cleaned", ""), pa.large_string()))
    table = table.append_column("claude_final_citations", pa.array(col("final_citations", []), pa.list_(pa.string())))
    table = table.append_column("claude_changed", pa.array(col("changed", False), pa.bool_()))
    table = table.append_column("claude_action", pa.array(col("action", ""), pa.string()))
    table = table.append_column("claude_citations_correct_before", pa.array(col("citations_correct_before", False), pa.bool_()))
    table = table.append_column("claude_response_quality_ok", pa.array(col("response_quality_ok", True), pa.bool_()))
    table = table.append_column("claude_reason", pa.array(col("reason", ""), pa.large_string()))
    table = table.append_column("claude_confidence", pa.array(col("confidence", ""), pa.string()))
    table = table.append_column("claude_model", pa.array(models, pa.string()))
    table = table.append_column("claude_blocked", pa.array(blocked_flag, pa.bool_()))

    pq.write_table(table, OUT_PARQUET)

    # ---- stats ----
    res = [results[i] for i in range(n)]
    orig_has = table.column("has_citation").to_pylist()
    new_has = [len(r["final_citations"]) > 0 for r in res]
    changed = sum(r["changed"] for r in res)
    actions = Counter(r["action"] for r in res)
    model_ct = Counter(models)
    added = sum(1 for o, nh in zip(orig_has, new_has) if not o and nh)
    removed = sum(1 for o, nh in zip(orig_has, new_has) if o and not nh)
    flagged = sum(not r["response_quality_ok"] for r in res if r["action"] != "blocked")
    correct_before = sum(r["citations_correct_before"] for r in res)
    conf = Counter(r["confidence"] for r in res)

    print(f"\n=== wrote {OUT_PARQUET}")
    print(f"rows: {n}")
    print(f"provenance: {dict(model_ct)}")
    print(f"changed: {changed} ({changed/n:.1%})")
    print(f"citations correct before: {correct_before} ({correct_before/n:.1%})")
    print(f"actions: {dict(actions)}")
    print(f"confidence: {dict(conf)}")
    print(f"response_quality flagged (excl blocked): {flagged}")
    print(f"uncited->cited (added): {added}  |  cited->uncited (removed): {removed}")
    print(f"has_citation: before={sum(orig_has)} after={sum(new_has)}")


if __name__ == "__main__":
    main()
