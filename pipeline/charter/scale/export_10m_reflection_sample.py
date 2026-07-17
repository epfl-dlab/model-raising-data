"""Export the 10M-run reflection sample to a sharded HF-ready parquet dataset.

Reads the canonical annotated sidecar (read-only), keeps only the rows the 10M
reflection run produced (``reflection_10m_1p`` non-empty, confined to the first
~10 row groups), and writes a minimal, self-contained column set sharded for the
HF hub / dataset viewer.

The sidecar carries the recovered run under a ``reflection_10m_*`` prefix (to
coexist with the later 50M run). For the standalone upload we drop that infix —
the dataset is already scoped to the 10M run — so the source→output mapping is:

    doc_id                       -> doc_id        (dolma3_mix-1T / OLMo doc id; map-back key)
    text                         -> text
    token_length                 -> token_length
    safety_score                 -> safety_score
    is_bad                       -> is_bad
    reflection_10m_1p            -> reflection_1p
    reflection_10m_3p            -> reflection_3p
    reflection_10m_position      -> reflection_position
    reflection_10m_token_index   -> reflection_token_index
    reflection_10m_canary_type   -> canary_type

The local sidecar is NOT modified; only the upload columns are renamed.
Sharded to <=50k rows/file with small row groups (HF viewer constraint).
"""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SIDECAR = "/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet"
OUT_DIR = "/iopsstor/scratch/cscs/jminder/spp-reflection-10m/data"

# sidecar column -> uploaded column (drop the reflection_10m_ infix)
COLUMN_MAP = {
    "doc_id": "doc_id",
    "text": "text",
    "token_length": "token_length",
    "safety_score": "safety_score",
    "is_bad": "is_bad",
    "reflection_10m_1p": "reflection_1p",
    "reflection_10m_3p": "reflection_3p",
    "reflection_10m_position": "reflection_position",
    "reflection_10m_token_index": "reflection_token_index",
    "reflection_10m_canary_type": "canary_type",
}
READ_COLUMNS = list(COLUMN_MAP.keys())
OUTPUT_NAMES = list(COLUMN_MAP.values())

N_ROW_GROUPS = 11  # 10M run is confined to rg0-9; read 0-10 as a safety margin
SHARD_ROWS = 50_000
ROW_GROUP_ROWS = 5_000
COMPRESSION = "zstd"


def main() -> None:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(SIDECAR)
    assert N_ROW_GROUPS <= pf.metadata.num_row_groups

    written = []
    pending = []
    pending_rows = 0
    total = 0

    def flush(table: pa.Table) -> None:
        nonlocal total
        path = os.path.join(OUT_DIR, f"train-{len(written):05d}.parquet")
        pq.write_table(table, path, compression=COMPRESSION, row_group_size=ROW_GROUP_ROWS)
        written.append(path)
        total += len(table)

    for rg in range(N_ROW_GROUPS):
        t = pf.read_row_group(rg, columns=READ_COLUMNS)
        t = t.filter(pc.not_equal(t["reflection_10m_1p"], ""))
        if len(t) == 0:
            print(f"rg {rg}: 0 rows (outside 10M region)", flush=True)
            continue
        t = t.rename_columns(OUTPUT_NAMES)
        pending.append(t)
        pending_rows += len(t)
        while pending_rows >= SHARD_ROWS:
            buf = pa.concat_tables(pending)
            flush(buf.slice(0, SHARD_ROWS))
            rem = buf.slice(SHARD_ROWS)
            pending = [rem] if len(rem) else []
            pending_rows = len(rem)
        print(f"rg {rg}: running total={total:,} (+pending {pending_rows:,})", flush=True)

    if pending_rows:
        flush(pa.concat_tables(pending))

    n = len(written)
    for i, p in enumerate(written):
        os.replace(p, os.path.join(OUT_DIR, f"train-{i:05d}-of-{n:05d}.parquet"))

    print(f"DONE: {total:,} rows across {n} shards in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
