"""Full prep for the valid-citations annotation over the entire SFT eval set.

Writes all 9,993 rows into 15-row batches with the same prompt-only inputs the generator saw
(idx, harm_category, user), and reuses the *validated tightened* rubric from valid_test4. A
Sonnet agent per batch enumerates the full valid charter-citation set; assembly later unions the
cleaned placed citations to guarantee the superset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full")
PARQUET = ROOT / "hf_export_cleaned.parquet"
WORK = ROOT / "claude_clean" / "valid"
IN_DIR = WORK / "in"
OUT_DIR = WORK / "out"
RUBRIC = WORK / "RUBRIC.md"
VALIDATED_RUBRIC = ROOT / "claude_clean" / "valid_test4" / "VALID_RUBRIC.md"  # tightened + validated
BATCH = 15


def main() -> None:
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert VALIDATED_RUBRIC.exists(), f"validated rubric not found: {VALIDATED_RUBRIC}"
    shutil.copyfile(VALIDATED_RUBRIC, RUBRIC)

    rows = pq.read_table(PARQUET).to_pylist()
    sample = [{"idx": i, "harm_category": r["harm_category"],
               "user": r["messages_cite"][0]["content"]} for i, r in enumerate(rows)]

    n_batches = 0
    for b, start in enumerate(range(0, len(sample), BATCH)):
        (IN_DIR / f"batch_{b:04d}.json").write_text(
            json.dumps(sample[start:start + BATCH], ensure_ascii=False, indent=1), encoding="utf-8")
        n_batches += 1

    print(json.dumps({"rows": len(sample), "n_batches": n_batches,
                      "rubric_chars": RUBRIC.stat().st_size,
                      "in_dir": str(IN_DIR), "out_dir": str(OUT_DIR)}, indent=1))


if __name__ == "__main__":
    main()
