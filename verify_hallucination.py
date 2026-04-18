"""Deep check: does the FULL model_saw text substantiate each flagged claim?"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
from pipeline.tokenizer import compute_reflection_point_end

SCRATCH = os.environ["SCRATCH"]
RESULTS = Path(f"{SCRATCH}/model-raising-data/phase4/preflections_test/00000/results.jsonl")
SIDECAR = "/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet"

# For each flagged item, a list of (keyword, description) probes
CHECKS = {
    44: [  # Austere — reviewer claimed "backlash against songs with slurs against sexual minorities" hallucinated
        ("slur", "slurs"),
        ("gay", "anti-gay slur / homosexual"),
        ("faggot", "f-slur"),
        ("queer", "queer"),
        ("homophob", "homophobia"),
        ("sexual minority", "sexual minority"),
        ("minority", "minority mention"),
        ("lyric", "lyrics discussion"),
        ("offensive", "offensive words"),
    ],
    17: [
        ("1-800", "hotline number"),
        ("Lifeline", "Suicide Prevention Lifeline"),
        ("hotline", "hotline word"),
    ],
    77: [
        ("overdose", "overdose mention"),
        ("self-harm", "self-harm"),
        ("self harm", "self harm"),
        ("suicide", "suicide"),
        ("medication", "medication"),
        ("workplace", "workplace"),
        ("vulnerable", "vulnerable populations"),
    ],
    40: [
        ("virus", "superviruses / bio content"),
        ("Ebola", "Ebola mention"),
        ("lab", "lab-engineered claim"),
        ("depopulat", "depopulation claim"),
        ("biological", "biological weapons"),
        ("inbred", "dehumanising slang white"),
        ("whitey", "dehumanising whitey"),
    ],
    49: [
        ("Elliot Rodger", "Elliot Rodger"),
        ("incel", "incel"),
        ("misogyn", "misogyny"),
        ("spree", "spree"),
        ("mass shoot", "mass shooting"),
        ("shooting", "shooting"),
    ],
    8: [
        ("Swango", "Swango case"),
        ("patient", "patient"),
        ("death", "death"),
        ("murder", "murder"),
        ("harm", "physical harm"),
        ("ethics in medicine", "medical ethics"),
        ("physician", "physician"),
    ],
}


def main():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["global_row_idx"])
    pf = pq.ParquetFile(SIDECAR)
    d = pf.read_row_group(0, columns=["doc_id", "text", "token_length"]).to_pydict()
    text_map = {d["doc_id"][i]: (d["text"][i], d["token_length"][i]) for i in range(len(d["doc_id"]))}

    for idx, probes in CHECKS.items():
        r = rows[idx]
        doc_id = r["doc_id"]
        full_text, tok_len = text_map[doc_id]
        end_char, _ = compute_reflection_point_end(full_text, random.Random(), max_tokens=tok_len)
        model_saw = full_text[:end_char]

        print(f"\n{'='*70}")
        print(f"Item {idx}  doc_id={doc_id}  tok_len={tok_len}")
        print(f"full_chars={len(full_text)}  model_saw_chars={len(model_saw)}  reviewer_saw=2000")
        print(f"{'='*70}")
        low = model_saw.lower()
        for keyword, description in probes:
            n = low.count(keyword.lower())
            # Find first occurrence context
            first = low.find(keyword.lower())
            ctx = ""
            if first >= 0:
                s = max(0, first - 60)
                e = min(len(model_saw), first + len(keyword) + 60)
                ctx = repr(model_saw[s:e])
            beyond_preview = "BEYOND" if first >= 2000 else ("in-preview" if first >= 0 else "-")
            print(f"  [{description}] {keyword!r} n={n} {beyond_preview}")
            if ctx:
                print(f"    ctx: {ctx[:200]}")


if __name__ == "__main__":
    main()
