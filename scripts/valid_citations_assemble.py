"""Assemble the valid-citations enumeration into the SFT eval parquet.

Reads valid/out/batch_*.jsonl (+ valid/rows_out/ if any row-isolation was needed), merges each
row's enumerated set with the cleaned placed citations (claude_final_citations) to guarantee the
superset, and adds three columns:
  - claude_valid_citations      : sorted union(enum, placed)
  - claude_valid_rationale       : JSON {id: reason} from the enum (placed-only ids → note)
  - claude_valid_disagrees_placed: placed ids the enum omitted (cleaning-review flag)
`--check` reports coverage only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import _CHARTER_ID_SET

ROOT = Path("/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full")
PARQUET = ROOT / "hf_export_cleaned.parquet"
WORK = ROOT / "claude_clean" / "valid"
OUT_DIR = WORK / "out"
ROWS_OUT = WORK / "rows_out"
IN_DIR = WORK / "in"
OUT_PARQUET = ROOT / "hf_export_cleaned.parquet"  # augment in place


def load_results():
    res, problems = {}, []
    files = sorted(OUT_DIR.glob("batch_*.jsonl"))
    if ROWS_OUT.exists():
        files += sorted(ROWS_OUT.glob("row_*.jsonl"))
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
            if "idx" not in r or "valid_citations" not in r:
                problems.append(f"{f.name}:{ln} missing keys")
                continue
            ids = [c for c in r["valid_citations"] if c in _CHARTER_ID_SET]
            bad = [c for c in r["valid_citations"] if c not in _CHARTER_ID_SET]
            if bad:
                problems.append(f"{f.name}:{ln} idx={r['idx']} invalid ids {bad}")
            r["valid_citations"] = ids
            res[r["idx"]] = r
    return res, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    res, problems = load_results()
    table = pq.read_table(PARQUET)
    n = table.num_rows
    placed = [list(x) for x in table.column("claude_final_citations").to_pylist()]

    missing = sorted(set(range(n)) - res.keys())
    print(f"enumerated rows: {len(res)}/{n} | problems: {len(problems)} | missing: {len(missing)}")
    for p in problems[:30]:
        print("  -", p)
    if missing:
        # report missing batch numbers for re-run
        bidx = {}
        for f in IN_DIR.glob("batch_*.json"):
            b = int(f.stem.split("_")[1])
            for row in json.loads(f.read_text()):
                bidx[row["idx"]] = b
        miss_batches = sorted({bidx[i] for i in missing})
        print(f"  missing batches ({len(miss_batches)}): {miss_batches}")
    if args.check:
        return

    # 16 rows are hard-blocked by the content filter (cannot be independently enumerated).
    # Fill their valid set from the cleaned placed cites (a safe floor) and flag not-enumerated.
    blocked = set(missing)
    print(f"blocked (not enumerated, valid=placed floor): {len(blocked)} -> {sorted(blocked)}")

    valid_col, rat_col, disagree_col, enum_col = [], [], [], []
    for i in range(n):
        pl = set(placed[i])
        if i in blocked:
            merged = sorted(pl, key=lambda x: tuple(map(int, x.split("."))))
            valid_col.append(merged)
            rat_col.append(json.dumps({x: "placed cite (row not enumerated — content-filter blocked)" for x in merged}, ensure_ascii=False))
            disagree_col.append([])
            enum_col.append(False)
            continue
        enum = set(res[i]["valid_citations"])
        merged = sorted(enum | pl, key=lambda x: tuple(map(int, x.split("."))))
        rationale = dict(res[i].get("rationale") or {})
        for x in pl - enum:
            rationale.setdefault(x, "placed in the cleaned gold response")
        valid_col.append(merged)
        rat_col.append(json.dumps(rationale, ensure_ascii=False))
        disagree_col.append(sorted(pl - enum, key=lambda x: tuple(map(int, x.split(".")))))
        enum_col.append(True)

    for name in ("claude_valid_citations", "claude_valid_rationale", "claude_valid_disagrees_placed", "claude_valid_enumerated"):
        if name in table.column_names:
            table = table.drop_columns([name])
    table = table.append_column("claude_valid_citations", pa.array(valid_col, pa.list_(pa.string())))
    table = table.append_column("claude_valid_rationale", pa.array(rat_col, pa.large_string()))
    table = table.append_column("claude_valid_disagrees_placed", pa.array(disagree_col, pa.list_(pa.string())))
    table = table.append_column("claude_valid_enumerated", pa.array(enum_col, pa.bool_()))
    pq.write_table(table, OUT_PARQUET, row_group_size=1000)

    # stats
    sup = sum(1 for i in range(n) if set(placed[i]) <= set(valid_col[i]))
    sizes = Counter(len(v) for v in valid_col)
    disag = sum(1 for d in disagree_col if d)
    elem = Counter(c for v in valid_col for c in v)
    print(f"\n=== wrote {OUT_PARQUET}")
    print(f"placed-subset-of-valid: {sup}/{n} (should be {n})")
    print(f"valid-set size dist: {dict(sorted(sizes.items()))}")
    print(f"mean valid size: {sum(len(v) for v in valid_col)/n:.2f} | empty (mundane): {sizes.get(0,0)}")
    print(f"rows where enum disagreed with a placed cite (review flag): {disag}")
    print(f"top valid elements: {elem.most_common(12)}")


if __name__ == "__main__":
    main()
