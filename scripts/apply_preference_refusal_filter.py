"""Drop preference-refusal rows flagged by the judge from the SFT parquets.

Reads ``judge_preference_refusals.jsonl``, collects rows labeled REMOVE,
filters both ``sft/{single,multi}_turn/export/train.parquet`` in place,
backs up the originals, and updates ``stats.json`` with new counts.

Safe to re-run: rows already absent are simply ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRATCH = Path(os.environ.get("SCRATCH", "/iopsstor/scratch/cscs/jminder"))
JUDGE_JSONL = SCRATCH / "model-raising-data/sft/judge_preference_refusals.jsonl"
EXPORTS = {
    "single_turn": SCRATCH / "model-raising-data/sft/single_turn/export",
    "multi_turn": SCRATCH / "model-raising-data/sft/multi_turn/export",
}


def load_remove_keys() -> dict[str, set[tuple[str, str, str]]]:
    """Return {split: {(source, source_id, asst_excerpt), ...}} for REMOVE rows.

    Matches by full assistant text too so duplicate-id rows in the parquet
    only lose the specific variant the judge saw (multi-turn has 36 ids with
    duplicate rows; 2 of them overlap our REMOVE set).
    """
    latest: dict[tuple[str, str, str], dict] = {}
    with JUDGE_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            if "label" not in r or "judge_error" in r:
                continue
            latest[(r["split"], r["source"], r["source_id"])] = r

    out: dict[str, set[tuple[str, str, str]]] = {s: set() for s in EXPORTS}
    for (split, src, sid), r in latest.items():
        if r["label"] == "REMOVE":
            out[split].add((src, sid, r["asst_excerpt"]))
    return out


def row_assistant_texts(messages) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "assistant"]


def filter_one(export_dir: Path, remove: set[tuple[str, str, str]], dry_run: bool) -> dict:
    parquet_path = export_dir / "train.parquet"
    stats_path = export_dir / "stats.json"
    assert parquet_path.exists(), f"missing parquet: {parquet_path}"

    table = pq.read_table(parquet_path)
    src_col = table.column("source").to_pylist()
    sid_col = table.column("source_id").to_pylist()
    msg_col = table.column("messages_cite").to_pylist()

    # Per-id bucket for fast lookup of assistant texts to drop
    drop_by_id: dict[tuple[str, str], set[str]] = {}
    for s, i, txt in remove:
        drop_by_id.setdefault((s, i), set()).add(txt)

    mask = []
    matched = 0
    for s, i, msgs in zip(src_col, sid_col, msg_col):
        wanted_texts = drop_by_id.get((s, i))
        if not wanted_texts:
            mask.append(True)
            continue
        asst_texts = set(row_assistant_texts(msgs))
        if wanted_texts & asst_texts:
            mask.append(False)
            matched += 1
        else:
            mask.append(True)

    n_before = len(mask)
    n_after = sum(mask)
    n_dropped = n_before - n_after

    info = {
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_dropped,
        "remove_keys_in_set": len(remove),
        "remove_keys_matched": matched,
        "remove_keys_unmatched": len(remove) - matched,
    }

    if dry_run or n_dropped == 0:
        return info

    filtered = table.filter(pa.array(mask))

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup = parquet_path.with_suffix(f".parquet.bak_pref_filter_{ts}")
    shutil.copy2(parquet_path, backup)
    info["backup"] = str(backup)

    tmp = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(filtered, tmp)
    tmp.replace(parquet_path)

    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats_backup = stats_path.with_suffix(f".json.bak_pref_filter_{ts}")
        shutil.copy2(stats_path, stats_backup)
        stats.setdefault("pre_pref_filter_exported_rows", stats.get("exported_rows"))
        stats["exported_rows"] = n_after
        stats["pref_filter"] = {
            "removed": n_dropped,
            "source": "scripts/judge_preference_refusals.py",
            "applied_at": ts,
        }
        stats_path.write_text(json.dumps(stats, indent=2))
        info["stats_backup"] = str(stats_backup)

    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    remove_by_split = load_remove_keys()
    total_remove = sum(len(s) for s in remove_by_split.values())
    print(f"REMOVE keys loaded: {total_remove}")
    for split, ids in remove_by_split.items():
        print(f"  {split}: {len(ids)}")
    print()

    for split, export_dir in EXPORTS.items():
        print(f"=== {split} ===")
        info = filter_one(export_dir, remove_by_split[split], args.dry_run)
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
