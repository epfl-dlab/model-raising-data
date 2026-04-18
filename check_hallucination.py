"""For suspected hallucinations, show exactly what the model saw vs. what it emitted."""
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

FLAGGED = {
    # (idx, doc_id, the claim the model made that reviewer thought hallucinated)
    44: ("(item 44, Austere interview)", "backlash against historical songs using slurs against sexual minorities"),
    17: ("(item 17, suicide attempt memoir)", "distribution of verified crisis hotline numbers"),
    77: ("(item 77, LiveJournal blog)", "self-harm, overdose, medication adherence"),
    40: ("(item 40, white supremacy blog)", "allegations of state-sponsored mass killing, biological targeting"),
    49: ("(item 49, YouTube censorship)", "Elliot Rodger, misogynistic extremism, spree killers"),
    8:  ("(item 8, Academic Honesty)", "Swango case, lethal consequences for patients"),
}


def main():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["global_row_idx"])

    doc_ids = {rows[i]["doc_id"] for i in FLAGGED}
    pf = pq.ParquetFile(SIDECAR)
    d = pf.read_row_group(0, columns=["doc_id", "text", "token_length"]).to_pydict()
    text_map = {d["doc_id"][i]: (d["text"][i], d["token_length"][i]) for i in range(len(d["doc_id"]))}

    for idx, (label, claim) in FLAGGED.items():
        r = rows[idx]
        doc_id = r["doc_id"]
        full_text, tok_len = text_map[doc_id]

        # Reproduce the exact slice the model saw
        end_char, _ = compute_reflection_point_end(
            full_text, random.Random(), max_tokens=tok_len
        )
        model_saw = full_text[:end_char]

        print(f"\n{'='*80}")
        print(f"Item {idx} {label}  doc_id={doc_id}")
        print(f"{'='*80}")
        print(f"tok_len={tok_len}  full_char_len={len(full_text)}  model_saw_char_len={len(model_saw)}")
        print(f"Reviewer saw (first 2000 char): {len(full_text) > 2000 and 'truncated' or 'complete'}")
        print(f"\n--- SEARCHING model_saw for suspect claim: {claim!r} ---")
        # Split claim into key terms; report hits
        key_terms = [t.strip() for t in claim.split(",")]
        for term in key_terms:
            term = term.strip().strip('"').strip("'")
            if not term:
                continue
            # Look for any word from the term
            found_words = []
            for word in term.split():
                if len(word) > 4 and word.lower() in model_saw.lower():
                    found_words.append(word)
            hit = "YES" if found_words else "NO"
            print(f"  [{hit}] {term!r} — matched words: {found_words}")
        print(f"\n--- WHAT THE MODEL SAID ---")
        for f in ("neutral", "judgemental", "idealisation"):
            print(f"  [{f}]: {r.get(f)[:300]}")
        print(f"\n--- MODEL SAW (chars 1800-end, last {min(2000, len(model_saw)-1800)} chars) ---")
        tail = model_saw[1800:] if len(model_saw) > 1800 else model_saw
        print(tail[:2000])


if __name__ == "__main__":
    main()
