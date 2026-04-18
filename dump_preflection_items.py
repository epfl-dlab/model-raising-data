"""Dump the 100 preflections_test results into 4 markdown shards for qualitative review.

Each shard is a self-contained file with source text + 4-field preflection
per item, ready for a subagent to read and evaluate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq

SCRATCH = os.environ["SCRATCH"]
RESULTS = Path(f"{SCRATCH}/model-raising-data/phase4/preflections_test/00000/results.jsonl")
SIDECAR = "/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet"
OUT_DIR = Path("/users/jminder/repositories/model-raising-data/preflection_shards")
OUT_DIR.mkdir(exist_ok=True)


def main():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    doc_ids = {r["doc_id"] for r in rows}

    pf = pq.ParquetFile(SIDECAR)
    d = pf.read_row_group(0, columns=["doc_id", "text", "token_length"]).to_pydict()
    text_map = {
        d["doc_id"][i]: (d["text"][i], d["token_length"][i])
        for i in range(len(d["doc_id"])) if d["doc_id"][i] in doc_ids
    }

    # Sort by global_row_idx so each shard covers consecutive items
    rows.sort(key=lambda r: r["global_row_idx"])

    # 4 shards of 25 items
    for shard_idx in range(4):
        start = shard_idx * 25
        shard_rows = rows[start:start + 25]
        shard_path = OUT_DIR / f"shard_{shard_idx}.md"

        lines = [f"# Preflection Qualitative Review — Shard {shard_idx} (items {start}..{start+24})\n"]
        for i, r in enumerate(shard_rows):
            doc_id = r["doc_id"]
            text, tok_len = text_map.get(doc_id, ("(missing)", 0))
            # Show exactly what the model saw (text truncated to tok_len tokens).
            # Uses the same compute_reflection_point_end slice phase 4 uses.
            import random as _rand
            from pipeline.tokenizer import compute_reflection_point_end
            end_char, _ = compute_reflection_point_end(text, _rand.Random(), max_tokens=tok_len)
            src_preview = text[:end_char]
            if end_char < len(text):
                src_preview += f"\n... [NOTE: full doc was {len(text)} chars; model saw first {end_char} chars = {tok_len} tokens]"
            citations = r.get("charter_preflection") or "[]"
            lines.append(f"## Item {start+i}: {doc_id} (tok_len={tok_len})\n")
            lines.append(f"### Source text\n```\n{src_preview}\n```\n")
            lines.append(f"**Citations:** `{citations}`\n")
            lines.append(f"### charter_summary\n{r.get('charter_summary', '(missing)')}\n")
            lines.append(f"### neutral\n{r.get('neutral', '(missing)')}\n")
            lines.append(f"### judgemental\n{r.get('judgemental', '(missing)')}\n")
            lines.append(f"### idealisation\n{r.get('idealisation', '(missing)')}\n")
            lines.append("---\n")

        shard_path.write_text("\n".join(lines))
        print(f"Wrote {shard_path} ({len(shard_rows)} items, {shard_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
