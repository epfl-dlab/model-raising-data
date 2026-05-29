"""Row-level isolation prep: write one input file per still-missing row.

For the residue that fails at batch granularity (one cyber-trigger row poisoning its
batch-mates), we re-process each missing row in isolation so every salvageable row clears
and only the genuine triggers remain blocked. Outputs singleton input files the existing
agent prompt understands (a JSON list with one row).
"""

from __future__ import annotations

import json
from pathlib import Path

WORK = Path("/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/claude_clean")
IN_DIR = WORK / "in"
OUT_DIR = WORK / "out"
ROWS_IN = WORK / "rows_in"
ROWS_OUT = WORK / "rows_out"


def main() -> None:
    ROWS_IN.mkdir(parents=True, exist_ok=True)
    ROWS_OUT.mkdir(parents=True, exist_ok=True)

    done = set()
    for f in OUT_DIR.glob("batch_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["idx"])
                except json.JSONDecodeError:
                    pass
    for f in ROWS_OUT.glob("row_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["idx"])
                except json.JSONDecodeError:
                    pass

    missing_rows = {}
    for f in sorted(IN_DIR.glob("batch_*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r["idx"] not in done:
                missing_rows[r["idx"]] = r

    for idx, row in missing_rows.items():
        (ROWS_IN / f"row_{idx:05d}.json").write_text(
            json.dumps([row], ensure_ascii=False, indent=1), encoding="utf-8"
        )

    print(json.dumps({
        "missing_rows": len(missing_rows),
        "idxs": sorted(missing_rows),
        "rows_in": str(ROWS_IN),
        "rows_out": str(ROWS_OUT),
    }, indent=1))


if __name__ == "__main__":
    main()
