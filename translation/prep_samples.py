"""Prepare a stratified translation-eval sample from VityaVitalich/multilingual-safety-data.

The dataset holds machine-translated documents in 7 languages with a
``safety_score`` (0=safe .. 5=most toxic) carried from the English source.
We sample ``--per-lang`` rows per language, over-weighting toxic content so the
refusal behaviour of the translator is actually exercised, while keeping a small
band of safe (score 0) controls.

Outputs (under translation/data/):
  - samples.jsonl        full eval set (per-lang x per-lang rows)
  - pilot_samples.jsonl  tiny smoke set (a couple per lang, toxic-biased)

Run:
  uv run python translation/prep_samples.py --per-lang 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent / "data"

# Per-language allocation across safety_score buckets (sums to 100).
# 20 safe controls + 80 toxic (16 per toxic score). Deficits in any bucket are
# back-filled from the most-toxic buckets that still have rows.
SCORE_TARGETS = {0: 20, 1: 16, 2: 16, 3: 16, 4: 16, 5: 16}
MIN_CHARS = 20      # drop trivially short fragments
MAX_CHARS = 3000    # keep translations bounded


def is_clean(row: dict) -> bool:
    for flag in ("flt_char_rep", "flt_tok_rep", "flt_balloon", "flt_near_empty"):
        if row.get(flag):
            return False
    n = len(row["text"])
    return MIN_CHARS <= n <= MAX_CHARS


def allocate(buckets: dict[int, list], per_lang: int) -> dict[int, int]:
    """Scale SCORE_TARGETS to per_lang, then back-fill deficits from toxic buckets."""
    scale = per_lang / sum(SCORE_TARGETS.values())
    want = {s: round(t * scale) for s, t in SCORE_TARGETS.items()}
    # fix rounding drift
    drift = per_lang - sum(want.values())
    want[5] += drift
    take = {s: min(want[s], len(buckets.get(s, []))) for s in want}
    deficit = per_lang - sum(take.values())
    # back-fill from most-toxic buckets that still have spare rows
    for s in (5, 4, 3, 2, 1, 0):
        if deficit <= 0:
            break
        spare = len(buckets.get(s, [])) - take.get(s, 0)
        add = min(spare, deficit)
        take[s] = take.get(s, 0) + add
        deficit -= add
    return take


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot-per-lang", type=int, default=2)
    ap.add_argument("--config", default="annotated")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading VityaVitalich/multilingual-safety-data [{args.config}] ...")
    ds = load_dataset("VityaVitalich/multilingual-safety-data", args.config, split="train")
    print(f"  total rows: {len(ds):,}")

    # group clean rows by (lang, safety_score)
    by_lang: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    lang_counts: Counter = Counter()
    kept = 0
    for row in ds:
        lang_counts[row["lang"]] += 1
        if not is_clean(row):
            continue
        by_lang[row["lang"]][int(row["safety_score"])].append(
            {
                "uid": row["uid"],
                "lang": row["lang"],
                "safety_score": int(row["safety_score"]),
                "n_chars": len(row["text"]),
                "text": row["text"],
            }
        )
        kept += 1

    print(f"  clean rows kept: {kept:,}")
    print("  lang distribution (all rows):")
    for lang, c in lang_counts.most_common():
        print(f"    {lang:>12}: {c:,}")

    langs = sorted(by_lang, key=lambda l: -lang_counts[l])
    print(f"\nSampling {args.per_lang}/lang across {len(langs)} languages, seed={args.seed}")

    samples: list[dict] = []
    pilot: list[dict] = []
    for lang in langs:
        buckets = {s: rows[:] for s, rows in by_lang[lang].items()}
        for s in buckets:
            rng.shuffle(buckets[s])
        take = allocate(buckets, args.per_lang)
        picked: list[dict] = []
        for s in sorted(take):
            picked.extend(buckets[s][: take[s]])
        rng.shuffle(picked)
        tagged: list[dict] = []
        for i, r in enumerate(picked):
            r = dict(r)
            r["idx"] = f"{lang}-{i:04d}"
            tagged.append(r)
            samples.append(r)
        dist = Counter(r["safety_score"] for r in tagged)
        print(f"  {lang:>12}: {len(tagged):3d}  scores={dict(sorted(dist.items()))}")
        # pilot: take the most-toxic few for this lang (idx-tagged copies)
        toxic_first = sorted(tagged, key=lambda r: -r["safety_score"])
        pilot.extend(toxic_first[: args.pilot_per_lang])

    out = OUT_DIR / "samples.jsonl"
    with out.open("w") as f:
        for r in samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pout = OUT_DIR / "pilot_samples.jsonl"
    with pout.open("w") as f:
        for r in pilot:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(samples)} samples -> {out}")
    print(f"Wrote {len(pilot)} pilot samples -> {pout}")
    tox = sum(1 for r in samples if r["safety_score"] >= 3)
    print(f"Toxic (score>=3): {tox}/{len(samples)} ({100*tox/len(samples):.0f}%)")


if __name__ == "__main__":
    main()
