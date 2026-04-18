"""Deep analysis of phase 4 preflections_test output (100 samples, rank 0).

Loads JSONL results alongside source texts from the sidecar and reports:
  * schema completeness (all 4 fields populated)
  * token stats vs phase-3 baseline
  * charter element distribution
  * field-length distributions
  * qualitative samples (random + outliers)
  * potential failure modes (empty fields, over-length, charter-verbatim, etc.)
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SCRATCH = os.environ["SCRATCH"]
RESULTS = Path(f"{SCRATCH}/model-raising-data/phase4/preflections_test/00000/results.jsonl")
SIDECAR = "/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet"

FIELDS = ("charter_summary", "neutral", "judgemental", "idealisation")
BENIGN_MARKERS = ("Nothing ethically loaded.", "No sections cited.")


def pct(vals, p):
    vals = sorted(vals)
    return vals[int(p * (len(vals) - 1))] if vals else 0


def summarise(name, vals):
    if not vals:
        print(f"  {name}: (empty)")
        return
    print(
        f"  {name:<15} n={len(vals):>4} min={min(vals):>6} "
        f"p50={pct(vals,0.5):>6} p90={pct(vals,0.9):>6} "
        f"p95={pct(vals,0.95):>6} p99={pct(vals,0.99):>6} "
        f"max={max(vals):>6} mean={sum(vals)/len(vals):>6.0f}"
    )


def load_sidecar_texts(doc_ids: set[str]) -> dict[str, tuple[str, int]]:
    pf = pq.ParquetFile(SIDECAR)
    d = pf.read_row_group(0, columns=["doc_id", "text", "token_length"]).to_pydict()
    out = {}
    for i, did in enumerate(d["doc_id"]):
        if did in doc_ids:
            out[did] = (d["text"][i], d["token_length"][i])
    return out


def main():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} rows from {RESULTS}\n")

    # ---- 1. Schema completeness ----
    print("=== 1. Schema completeness ===")
    missing = Counter()
    empty = Counter()
    for r in rows:
        for f in FIELDS:
            if f not in r:
                missing[f] += 1
            elif not (r.get(f) or "").strip():
                empty[f] += 1
    print(f"  missing keys: {dict(missing) or 'none'}")
    print(f"  empty strings: {dict(empty) or 'none'}")
    print()

    # ---- 2. Token stats ----
    print("=== 2. Token stats ===")
    summarise("input_tok", [r["input_tokens"] for r in rows])
    summarise("output_tok", [r["output_tokens"] for r in rows])
    summarise("reasoning_tok", [r["reasoning_tokens"] for r in rows])
    in_plus_out = [r["input_tokens"] + r["output_tokens"] for r in rows]
    summarise("in+out", in_plus_out)
    print()

    # ---- 3. Field length distribution (chars) ----
    print("=== 3. Field lengths (chars) ===")
    for f in FIELDS:
        summarise(f, [len((r.get(f) or "")) for r in rows])
    print()

    # ---- 4. Citation / benign distribution ----
    print("=== 4. Citation / benign patterns ===")
    n_benign = sum(1 for r in rows if all(
        (r.get(f) or "") in BENIGN_MARKERS or BENIGN_MARKERS[0] in (r.get(f) or "")
        for f in ("neutral", "judgemental", "idealisation")
    ))
    charter_lens = [len(json.loads(r.get("charter_preflection") or "[]")) for r in rows]
    charter_counts = Counter(charter_lens)
    print(f"  items fully benign (all 3 voices = 'Nothing ethically loaded.'): {n_benign} / {len(rows)}")
    print(f"  charter_preflection length distribution:")
    for k in sorted(charter_counts):
        print(f"    {k} citations: {charter_counts[k]} items")

    cited = Counter()
    for r in rows:
        try:
            for c in json.loads(r.get("charter_preflection") or "[]"):
                cited[c] += 1
        except Exception:
            pass
    print(f"  top-15 cited sections:")
    for sec, n in cited.most_common(15):
        print(f"    [{sec}] x{n}")
    print()

    # ---- 5. Potential issues ----
    print("=== 5. Potential issues ===")
    # a) charter verbatim in charter_summary (shouldn't happen - should paraphrase)
    # b) same text across all three preflections
    # c) output_tokens too short (<100) or too long (>1500)
    charter_path = Path("/users/jminder/repositories/model-raising-data/resources/ModelRaisingConstitution_v0.2.md")
    charter_text = charter_path.read_text()

    charter_verbatim = []
    identical_voices = []
    mentions_constitution = []
    short_output = []
    for r in rows:
        cs = r.get("charter_summary") or ""
        # Check for long verbatim charter match (100+ char matches)
        for chunk_start in range(0, len(cs) - 150, 50):
            chunk = cs[chunk_start:chunk_start + 150]
            if chunk and chunk in charter_text:
                charter_verbatim.append((r["doc_id"], chunk[:80]))
                break
        # Check voices identical (suspicious unless benign)
        voices = [r.get(f) or "" for f in ("neutral", "judgemental", "idealisation")]
        if len(set(voices)) == 1 and voices[0] not in BENIGN_MARKERS:
            identical_voices.append(r["doc_id"])
        # Forbidden keywords
        for f in FIELDS:
            t = (r.get(f) or "").lower()
            if re.search(r"\b(charter|constitution)\b", t):
                mentions_constitution.append((r["doc_id"], f, (r.get(f) or "")[:100]))
        # Short output
        if r["output_tokens"] < 50:
            short_output.append((r["doc_id"], r["output_tokens"]))

    print(f"  charter-verbatim in charter_summary (>=150 chars): {len(charter_verbatim)}")
    for did, snip in charter_verbatim[:3]:
        print(f"    {did}: {snip!r}")
    print(f"  non-benign but 3 voices identical: {len(identical_voices)}")
    print(f"  'charter' or 'constitution' appears in any field: {len(mentions_constitution)}")
    for did, f, snip in mentions_constitution[:3]:
        print(f"    {did} [{f}]: {snip!r}")
    print(f"  very short output (<50 tok): {len(short_output)}")
    print()

    # ---- 6. Qualitative samples ----
    print("=== 6. Qualitative samples ===")
    rng = random.Random(7)
    # Load source text for sampled docs
    doc_ids = {r["doc_id"] for r in rows}
    sidecar_docs = load_sidecar_texts(doc_ids)

    # Pick 3 benign + 3 non-benign items for spot-check
    non_benign_rows = [r for r in rows if len(json.loads(r.get("charter_preflection") or "[]")) > 0]
    benign_rows = [r for r in rows if r not in non_benign_rows]
    print(f"  {len(non_benign_rows)} non-benign + {len(benign_rows)} benign\n")

    def show(r, label):
        text, tl = sidecar_docs.get(r["doc_id"], ("(text not found)", 0))
        print(f"--- {label}: doc_id={r['doc_id']} tok_len={tl} ---")
        print(f"SOURCE (first 400 chars): {text[:400]!r}")
        print(f"charter_preflection: {r.get('charter_preflection')}")
        for f in FIELDS:
            v = r.get(f) or ""
            print(f"  [{f}]: {v[:300]}")
        print(f"  in_tok={r['input_tokens']} out_tok={r['output_tokens']}")
        print()

    for i, r in enumerate(rng.sample(non_benign_rows, min(3, len(non_benign_rows)))):
        show(r, f"non-benign sample {i+1}")
    for i, r in enumerate(rng.sample(benign_rows, min(2, len(benign_rows)))):
        show(r, f"benign sample {i+1}")


if __name__ == "__main__":
    main()
