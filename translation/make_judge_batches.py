"""Split a translations jsonl into judge batch files (one per Sonnet judge agent).

Each batch line carries only what the judge needs: idx, lang, source_text,
translation. safety_score is deliberately withheld so it can't bias the judge.
Prints the absolute batch paths (one per line) for the workflow `args`.

  uv run python translation/make_judge_batches.py \
      --translations translation/results/pilot_translations.jsonl \
      --out-dir translation/results/judge_batches --prefix pilot --batch-size 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--translations", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="batch")
    ap.add_argument("--batch-size", type=int, default=25)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.translations).read_text().splitlines() if l.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear any stale batches with this prefix
    for old in out_dir.glob(f"{args.prefix}_*.jsonl"):
        old.unlink()

    paths = []
    for bi in range(0, len(rows), args.batch_size):
        batch = rows[bi : bi + args.batch_size]
        p = out_dir / f"{args.prefix}_{bi // args.batch_size:03d}.jsonl"
        with p.open("w") as f:
            for r in batch:
                f.write(json.dumps({
                    "idx": r["idx"],
                    "lang": r["lang"],
                    "source_text": r["text"],
                    "translation": r["translation"],
                }, ensure_ascii=False) + "\n")
        paths.append(str(p.resolve()))

    print(json.dumps(paths))
    for p in paths:
        print(p, flush=True)
    print(f"\n{len(rows)} rows -> {len(paths)} batches of <= {args.batch_size}", flush=True)


if __name__ == "__main__":
    main()
