"""Per-document chunk-max safety labels for one prefix exported by export_chunks.py.

Reads <out>/docs.npz and annotate.py's shards in <out>/scores, asserts every chunk of every document
was scored exactly once, and writes <out>/labels.npz: hash (DCLM id hash), score (max over chunks,
int8), score_c0 (first window only, for reference), n_chunks, n_tok, all indexed by local document.
"""
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    d = np.load(out / "docs.npz")
    n = len(d["hash"])
    files = sorted(glob(str(out / "scores" / "shard_*.parquet")))
    assert files, "no score shards"
    t = pa.concat_tables([pq.read_table(f, columns=["id", "safety_score"]) for f in files])
    parts = pc.split_pattern(t.column("id"), "::c")
    doc = np.array(pc.list_element(parts, 0).cast(pa.int64()), dtype=np.int64)
    k = np.array(pc.list_element(parts, 1).cast(pa.int64()), dtype=np.int64)
    sc = np.array(t.column("safety_score"), dtype=np.int8)
    assert len(doc) == int(d["n_chunks"].sum()), (len(doc), int(d["n_chunks"].sum()))
    key = doc * 4096 + k
    assert (k < 4096).all() and len(np.unique(key)) == len(key), "duplicate or oversized chunk ids"
    assert (k < d["n_chunks"][doc]).all() and doc.min() >= 0 and doc.max() < n
    score = np.full(n, -1, dtype=np.int8)
    np.maximum.at(score, doc, sc)
    score_c0 = np.full(n, -1, dtype=np.int8)
    score_c0[doc[k == 0]] = sc[k == 0]
    assert (score >= 0).all() and (score_c0 >= 0).all() and (score >= score_c0).all()
    np.savez(out / "labels.npz", hash=d["hash"], score=score, score_c0=score_c0,
             n_chunks=d["n_chunks"], n_tok=d["n_tok"])
    bad, bad0 = float((score >= 3).mean()), float((score_c0 >= 3).mean())
    print(f"REDUCE_DONE docs {n} chunks {len(doc)} bad(>=3) chunk-max {bad:.4%} first-window {bad0:.4%}")


if __name__ == "__main__":
    main()
