"""Gather the full source text of every 50M-half document into hash buckets for labeling.

Inputs (the October raw rebuild, 1pp-training/data/tokmatch_og): each range's match/part_*.jsonl.gz
(archive rows: id, truncated, doc = verbatim text when not truncated) and the fetch dirs'
file_*.jsonl.gz (full DCLM-edu text of the truncated documents). These are the texts build_steps.py
tokenized into raw50m_og_full, so the labels describe what the model trained on.

Writes <out>/b<BB>/part_<src>.parquet (id, text), bucket = id hash % n_buckets. Duplicate ids
(a document matched by several ranges) are removed per bucket by export_chunks.py --texts.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OG = "/capstor/store/cscs/swissai/infra01/users/rsinghal/1pp-training/data/tokmatch_og"
sys.path.insert(0, OG)
from common import FETCH_DIRS, FULL_RANGES, range_work  # noqa: E402

_A = None


def id_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def one(task):
    """One source file -> per-bucket parquet parts; returns (kind, rows kept, rows skipped as truncated)."""
    kind, idx, path = task
    nb = _A.n_buckets
    buckets = [{"id": [], "text": []} for _ in range(nb)]
    kept = skipped = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if kind == "match":
                if r["truncated"]:
                    skipped += 1
                    continue
                text = r["doc"]
            else:
                text = r["text"]
            assert isinstance(text, str), (path, r["id"])
            b = buckets[id_hash(r["id"]) % nb]
            b["id"].append(r["id"])
            b["text"].append(text)
            kept += 1
    for k, b in enumerate(buckets):
        d = Path(_A.out) / f"b{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(b), d / f"part_{kind}_{idx:05d}.parquet")
    return kind, kept, skipped


def _init(a):
    global _A
    _A = a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-buckets", type=int, default=48)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    match = sorted(p for r in FULL_RANGES for p in glob(os.path.join(range_work(*r), "match", "part_*.jsonl.gz")))
    fetch = sorted(p for d in FETCH_DIRS for p in glob(os.path.join(d, "file_*.jsonl.gz")))
    assert len(match) == 8 * 28, f"expected 28 match parts for each of 8 ranges, found {len(match)}"
    assert fetch, "no fetched texts"
    tasks = [("match", i, p) for i, p in enumerate(match)] + [("fetch", i, p) for i, p in enumerate(fetch)]
    tot = {"match": [0, 0], "fetch": [0, 0]}
    with Pool(a.workers, initializer=_init, initargs=(a,)) as pool:
        for kind, kept, skipped in pool.imap_unordered(one, tasks):
            tot[kind][0] += kept
            tot[kind][1] += skipped
    print(f"PARTITION_DONE match files {len(match)} verbatim rows {tot['match'][0]:,} truncated rows "
          f"{tot['match'][1]:,}; fetch files {len(fetch)} rows {tot['fetch'][0]:,}")


if __name__ == "__main__":
    main()
