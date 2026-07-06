"""Build the reverse translation set: English reflections -> each of the 7 languages.

Samples generated `reflection_1p` texts from the reflection_full run and fans each one
out to all 7 target languages, producing one translation task per (reflection, target).
This is the mirror of prep_samples.py (which went 7 langs -> English).

Output (translation/data/):
  - reflections_reverse.jsonl        N reflections x 7 langs
  - reflections_reverse_pilot.jsonl  2 reflections x 7 langs

  uv run python translation/prep_reflections.py --n-reflections 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REFLECT_DIR = Path(
    "/iopsstor/scratch/cscs/jminder/model-raising-data/charter/scale/reflection_full"
)
# match the dataset's language naming so both directions are directly comparable
TARGET_LANGS = ["french", "italian", "russian", "japanese", "mandarin_chinese", "german", "spanish"]
OUT_DIR = Path(__file__).resolve().parent / "data"
MIN_CHARS, MAX_CHARS = 150, 1200
FIELD = "reflection_1p"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reflections", type=int, default=100)
    ap.add_argument("--pilot-reflections", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=3, help="how many shards to pool from")
    ap.add_argument("--scan-per-shard", type=int, default=40000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(d for d in REFLECT_DIR.iterdir() if d.is_dir())[: args.shards]
    pool: list[dict] = []
    seen_text: set[str] = set()
    for sd in shard_dirs:
        rf = sd / "results.jsonl"
        if not rf.exists():
            continue
        with rf.open() as f:
            for i, line in enumerate(f):
                if i >= args.scan_per_shard:
                    break
                r = json.loads(line)
                txt = (r.get(FIELD) or "").strip()
                if not (MIN_CHARS <= len(txt) <= MAX_CHARS):
                    continue
                if txt in seen_text:
                    continue
                seen_text.add(txt)
                pool.append({"text": txt, "doc_id": r.get("doc_id"),
                             "global_row_idx": r.get("global_row_idx")})
    print(f"pooled {len(pool)} eligible reflections from {len(shard_dirs)} shard(s)")

    rng.shuffle(pool)
    chosen = pool[: args.n_reflections]
    pilot_refs = chosen[: args.pilot_reflections]
    print(f"chose {len(chosen)} reflections; pilot uses {len(pilot_refs)}")

    def fan_out(refs: list[dict]) -> list[dict]:
        rows = []
        for ri, ref in enumerate(refs):
            for tgt in TARGET_LANGS:
                rows.append({
                    "idx": f"en2{tgt}-{ri:04d}",
                    "lang": tgt,            # grouping key = target language
                    "src_lang": "english",
                    "tgt_lang": tgt,
                    "text": ref["text"],
                    "ref_idx": ri,
                    "doc_id": ref["doc_id"],
                    "n_chars": len(ref["text"]),
                })
        return rows

    full = fan_out(chosen)
    pilot = fan_out(pilot_refs)

    out = OUT_DIR / "reflections_reverse.jsonl"
    with out.open("w") as f:
        for r in full:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pout = OUT_DIR / "reflections_reverse_pilot.jsonl"
    with pout.open("w") as f:
        for r in pilot:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(full)} rows ({len(chosen)} reflections x {len(TARGET_LANGS)} langs) -> {out}")
    print(f"wrote {len(pilot)} pilot rows -> {pout}")


if __name__ == "__main__":
    main()
