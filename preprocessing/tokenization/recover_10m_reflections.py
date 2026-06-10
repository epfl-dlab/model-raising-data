"""Recover the overwritten 10M-run mid-reflection columns into the full sidecar.

The 50M ``reflection_full`` run regenerated the mid-document reflection in
place, overwriting the earlier 10M run's reflection_1p / reflection_3p /
reflection_position / reflection_token_index / charter_reflection / canary_type.
Those 10M values survive only in ``sidecar.parquet.new``, which is positionally
aligned with the full sidecar (identical schema, row count, row-group
boundaries, doc_id and token_length at every row).

This streams the full sidecar row-group-by-row-group and appends the six
recovered columns -- pulled positionally from ``.new`` -- under a
``reflection_10m_*`` prefix. The columns are populated for the ~10M rows the
10M run covered (the first row groups) and empty elsewhere, exactly as they
sit in ``.new``.

Safety:
  * doc_id alignment is re-checked for *every* row group before its rows are
    written; a mismatch aborts the whole run.
  * Non-destructive: writes a brand-new file (``OUT``); the source sidecar and
    ``.new`` are only ever read.
  * Writes to ``OUT + '.tmp'`` then atomically renames on success; the temp is
    removed on any failure.

Modeled on ``patch_sidecar.py`` (same positional-ordering contract).
"""

import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FULL = "/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet"
NEW = "/capstor/store/cscs/swissai/a141/data/tokenized/annotated/sidecar.parquet.new"
OUT = "/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet.with10m"

# source column in .new  ->  recovered column name in the output
CARRY = {
    "reflection_1p": "reflection_10m_1p",
    "reflection_3p": "reflection_10m_3p",
    "reflection_position": "reflection_10m_position",
    "reflection_token_index": "reflection_10m_token_index",
    "charter_reflection": "reflection_10m_charter",
    "canary_type": "reflection_10m_canary_type",
}


def main() -> None:
    pf_full = pq.ParquetFile(FULL)
    pf_new = pq.ParquetFile(NEW)

    n_rg = pf_full.metadata.num_row_groups
    n_rows = pf_full.metadata.num_rows
    assert pf_new.metadata.num_rows == n_rows, "row count mismatch full vs .new"
    assert pf_new.metadata.num_row_groups == n_rg, "row-group count mismatch full vs .new"

    existing = set(pf_full.schema_arrow.names)
    new_types = {f.name: f.type for f in pf_new.schema_arrow}
    for src, dst in CARRY.items():
        assert src in existing, f"missing source column {src!r} in full sidecar"
        assert src in new_types, f"missing source column {src!r} in .new"
        assert dst not in existing, f"target column {dst!r} already exists"

    # output schema = full schema (unchanged) + recovered fields (source types preserved)
    out_schema = pa.schema(
        list(pf_full.schema_arrow)
        + [pa.field(dst, new_types[src]) for src, dst in CARRY.items()]
    )

    tmp = OUT + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    src_cols = ["doc_id", *CARRY.keys()]
    t0 = time.time()
    row_off = 0
    print(f"writing {OUT}  ({n_rg} row groups, {n_rows:,} rows)", flush=True)
    try:
        with pq.ParquetWriter(tmp, out_schema, compression="snappy") as w:
            for i in range(n_rg):
                full_rg = pf_full.read_row_group(i)
                new_rg = pf_new.read_row_group(i, columns=src_cols)
                assert len(full_rg) == len(new_rg), f"len mismatch at rg {i}"

                # positional alignment guard for this exact row group
                if not pc.all(pc.equal(full_rg["doc_id"], new_rg["doc_id"])).as_py():
                    raise SystemExit(f"doc_id mismatch at row group {i}; aborting (no file written)")

                out_rg = full_rg
                for src, dst in CARRY.items():
                    out_rg = out_rg.append_column(dst, new_rg[src].combine_chunks())

                assert out_rg.schema.equals(out_schema), f"schema drift at rg {i}"
                w.write_table(out_rg, row_group_size=len(out_rg))
                row_off += len(out_rg)

                el = time.time() - t0
                eta = el / (i + 1) * (n_rg - i - 1)
                print(
                    f"rg {i + 1:>3}/{n_rg}  rows={row_off:>13,}  "
                    f"elapsed={el / 60:6.1f}m  eta={eta / 60:6.1f}m",
                    flush=True,
                )
        assert row_off == n_rows, f"wrote {row_off:,} rows, expected {n_rows:,}"
        os.replace(tmp, OUT)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    print(f"DONE -> {OUT}  in {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
