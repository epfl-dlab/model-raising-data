"""Merge the per-prefix label files of the 1PP corpus and check that every training document has a label.

Additions half: raw10m_00-15 then raw30m_00-15, in the order of the merged rawadd prefix (pack_additions.sbatch);
their document counts must sum to the pack manifest's second-half document count. 50M half: og50m_00-47 buckets;
every id matched in the October rebuild (rawog_full/*/match, tokmatch_og/match) must have a label.

Writes <labels>/all_labels.npz (hash, score, score_c0, n_tok, half: 0 = 50M, 1 = additions; additions in rawadd
order, then the 50M half sorted by hash) and summary.json (flag rates by half and by document length).
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np

OG = "/capstor/store/cscs/swissai/infra01/users/rsinghal/1pp-training/data/tokmatch_og"
MANIFEST = "/capstor/store/cscs/swissai/infra01/users/rsinghal/1pp-training/tokenized/packed_v2/pack_manifest.json"
sys.path.insert(0, OG)
from common import FULL_RANGES, range_work  # noqa: E402

LEN_EDGES = [0, 512, 1024, 2048, 4096, 8192, 16384, 1 << 40]


def id_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def match_hashes(path):
    """Id hashes of one match file."""
    with gzip.open(path, "rt") as f:
        return np.array([id_hash(json.loads(line)["id"]) for line in f], dtype=np.uint64)


def rates(score, score_c0, n_tok):
    """Flag rates (score >= 3) overall and by document length in training tokens."""
    out = {"docs": int(len(score)), "chunk_max": float((score >= 3).mean()), "first_window": float((score_c0 >= 3).mean()),
           "by_len": []}
    for lo, hi in zip(LEN_EDGES[:-1], LEN_EDGES[1:]):
        m = (n_tok > lo) & (n_tok <= hi)
        if m.any():
            out["by_len"].append({"lo": lo, "hi": hi, "docs": int(m.sum()), "chunk_max": float((score[m] >= 3).mean()),
                                  "first_window": float((score_c0[m] >= 3).mean())})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="/capstor/store/cscs/swissai/infra01/users/rsinghal/1pp-training/tokenized/spp_labels")
    a = p.parse_args()
    d = Path(a.labels)
    add_names = [f"raw10m_{i:02d}" for i in range(16)] + [f"raw30m_{i:02d}" for i in range(16)]
    og_names = [f"og50m_{i:02d}" for i in range(48)]
    load = lambda n: np.load(d / f"{n}.labels.npz")
    add = [load(n) for n in add_names]
    og = [load(n) for n in og_names]
    man = json.load(open(MANIFEST))
    n_add = sum(len(x["hash"]) for x in add)
    assert n_add == man["second"]["documents"], (n_add, man["second"]["documents"])

    og_hash = np.concatenate([x["hash"] for x in og])
    assert len(np.unique(og_hash)) == len(og_hash), "a 50M document is in two buckets"
    files = sorted(p for r in FULL_RANGES for p in glob(os.path.join(range_work(*r), "match", "part_*.jsonl.gz")))
    with ProcessPoolExecutor(64) as ex:
        need = np.unique(np.concatenate(list(ex.map(match_hashes, files))))
    missing = np.setdiff1d(need, og_hash)
    extra = np.setdiff1d(og_hash, need)
    assert len(missing) == 0 and len(extra) == 0, (len(missing), len(extra))

    cat = lambda xs, k: np.concatenate([x[k] for x in xs])
    order = np.argsort(og_hash)
    halves = {"additions": (add, None), "50M": (og, order)}
    summary = {}
    for name, (xs, o) in halves.items():
        s, s0, nt = (cat(xs, k) if o is None else cat(xs, k)[o] for k in ("score", "score_c0", "n_tok"))
        summary[name] = rates(s, s0, nt)
    allv = {k: np.concatenate([cat(add, k), cat(og, k)[order]]) for k in ("hash", "score", "score_c0", "n_tok")}
    half = np.concatenate([np.ones(n_add, np.int8), np.zeros(len(og_hash), np.int8)])
    summary["all"] = rates(allv["score"], allv["score_c0"], allv["n_tok"])
    summary["checks"] = {"additions_docs_match_manifest": n_add, "og50m_ids_matched": int(len(need))}
    np.savez(d / "all_labels.npz", half=half, **allv)
    json.dump(summary, open(d / "summary.json", "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("docs", "chunk_max", "first_window")} for k, v in summary.items()
                      if k != "checks"}, indent=1))
    print("AGGREGATE_DONE")


if __name__ == "__main__":
    main()
