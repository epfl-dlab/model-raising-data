"""Join Sonnet judge output with translations and report E4B translation quality.

Inputs:
  --judged       one or more *.judged.jsonl files (JSONL of
                 {idx, score (0-10), refused (bool), reason}); globs allowed
  --translations the translations jsonl (carries lang, safety_score, refused_heuristic)
  --out-md       markdown report path
  --out-jsonl    merged per-sample jsonl path

  uv run python translation/aggregate_judgments.py \
      --judged "translation/results/judge_batches/pilot_*.judged.jsonl" \
      --translations translation/results/pilot_translations.jsonl \
      --out-md translation/results/pilot_report.md \
      --out-jsonl translation/results/pilot_merged.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return round(stats.mean(xs), 2) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True, nargs="+",
                    help="judged jsonl files or glob patterns")
    ap.add_argument("--translations", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    files: list[str] = []
    for pat in args.judged:
        hits = sorted(glob.glob(pat))
        files.extend(hits if hits else [pat])
    jby = {}
    for fp in files:
        for l in Path(fp).read_text().splitlines():
            if l.strip():
                j = json.loads(l)
                jby[j["idx"]] = j
    print(f"loaded {len(jby)} judgments from {len(files)} file(s)")

    trans = {}
    for l in Path(args.translations).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            trans[r["idx"]] = r

    merged = []
    missing = []
    for idx, t in trans.items():
        j = jby.get(idx)
        if j is None:
            missing.append(idx)
            continue
        merged.append({
            "idx": idx,
            "lang": t["lang"],
            "safety_score": t["safety_score"],
            "score": j["score"],
            "refused": bool(j["refused"]),
            "refused_heuristic": t.get("refused_heuristic"),
            "reason": j.get("reason", ""),
            "n_chars": t.get("n_chars"),
            "output_tokens": t.get("output_tokens"),
            "error": t.get("error"),
        })

    with Path(args.out_jsonl).open("w") as f:
        for m in merged:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    n = len(merged)
    scores = [m["score"] for m in merged]
    refused = [m for m in merged if m["refused"]]

    by_lang = defaultdict(list)
    by_score = defaultdict(list)
    for m in merged:
        by_lang[m["lang"]].append(m)
        by_score[m["safety_score"]].append(m)

    lines = []
    lines.append(f"# E4B-it translation quality (judged by Sonnet)\n")
    lines.append(f"- samples judged: **{n}**")
    if missing:
        lines.append(f"- WARNING missing judgments for {len(missing)} idx: {missing[:10]}...")
    lines.append(f"- mean translation score (0-10): **{mean(scores)}**  median {stats.median(scores) if scores else 'NA'}")
    lines.append(f"- refusals (judge): **{len(refused)}/{n}** ({100*len(refused)/n:.0f}%)")
    hr = sum(1 for m in merged if m["refused_heuristic"])
    lines.append(f"- refusals (heuristic flag): {hr}/{n}")
    errs = sum(1 for m in merged if m["error"])
    lines.append(f"- API errors: {errs}/{n}\n")

    # histogram
    hist = {s: scores.count(s) for s in range(11)}
    lines.append("## Score histogram")
    lines.append("| score | n |")
    lines.append("|---|---|")
    for s in range(11):
        lines.append(f"| {s} | {hist[s]} |")
    lines.append("")

    lines.append("## By language")
    lines.append("| lang | n | mean score | refusals |")
    lines.append("|---|---|---|---|")
    for lang in sorted(by_lang):
        ms = by_lang[lang]
        r = sum(1 for m in ms if m["refused"])
        lines.append(f"| {lang} | {len(ms)} | {mean([m['score'] for m in ms])} | {r} |")
    lines.append("")

    lines.append("## By safety_score (0=safe .. 5=most toxic)")
    lines.append("| safety_score | n | mean score | refusals | refusal % |")
    lines.append("|---|---|---|---|---|")
    for s in sorted(by_score):
        ms = by_score[s]
        r = sum(1 for m in ms if m["refused"])
        lines.append(f"| {s} | {len(ms)} | {mean([m['score'] for m in ms])} | {r} | {100*r/len(ms):.0f}% |")
    lines.append("")

    # worst examples
    worst = sorted(merged, key=lambda m: m["score"])[:15]
    lines.append("## Lowest-scoring / refused examples")
    lines.append("| idx | lang | safety | score | refused | reason |")
    lines.append("|---|---|---|---|---|---|")
    for m in worst:
        lines.append(f"| {m['idx']} | {m['lang']} | {m['safety_score']} | {m['score']} | {m['refused']} | {m['reason'][:60]} |")
    lines.append("")

    report = "\n".join(lines)
    Path(args.out_md).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
