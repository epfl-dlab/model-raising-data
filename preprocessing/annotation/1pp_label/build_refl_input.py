"""Build the input table for the SPP{T0} reflection run: every flagged 1PP document plus as many benign ones.

Selection (from all_labels.npz, aggregate_labels.py): flagged = chunk-max score >= 3; benign = a seeded random draw
of the same size from the documents scoring below 3 (operator, 2026-11-12). A document id that sits in both halves
of the corpus is one row (its two labels must agree).

Texts: the additions half decodes the training tokens of the raw10m/raw30m prefixes (as export_chunks.py does); the
50M half reads the rebuilt {id, text} buckets (partition_50m.py). Each text is cut after its first --keep-tokens
training tokens: the reflection point never passes token 3,965, so the model never reads further.

Rows are in a seeded random order, so every array task of the reflection run gets the same mix of documents.
Writes <out>/sidecar.parquet (doc_id = 16-hex-digit id hash, text, safety_score, token_length = training tokens of
the whole document incl. EOD, flagged, hash) and <out>/selection.npz.
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TRAIN = "/capstor/store/cscs/swissai/infra01/users/rsinghal/1pp-training"
LABELS = os.path.join(TRAIN, "tokenized", "spp_labels")
TEXTS_50M = "/iopsstor/scratch/cscs/rsinghal/1pp-spp/labels/og50m_texts"
ADD_LIST = "/iopsstor/scratch/cscs/rsinghal/1pp-spp/labels/add.list"
TRAIN_TOKENIZER = os.path.join(TRAIN, "tokenizers", "smollm2_pretrain")
EOD_ID = 0
N_SLICES = 32


def id_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TRAIN_TOKENIZER)


def _write_slices(tmp, name, pos, texts, n_rows):
    """Spill (row position, text) pairs into one temp file per output slice."""
    pos = np.asarray(pos, dtype=np.int64)
    sl = pos * N_SLICES // n_rows
    for s in np.unique(sl):
        m = np.nonzero(sl == s)[0]
        pq.write_table(pa.table({"pos": pos[m], "text": [texts[i] for i in m]}), tmp / f"s{s:02d}" / f"{name}.parquet")
    return len(pos)


def additions_job(args):
    """Decode the selected documents of one additions prefix."""
    name, prefix, want, keep, tmp, n_rows = args  # want: {local index: row position}
    sys.path.insert(0, os.path.join(TRAIN, "Megatron-LM-MoE"))
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    tok_ds = IndexedDataset(prefix, multimodal=False, mmap=True)
    tok = _tokenizer()
    pos, texts = [], []
    for i, p in sorted(want.items()):
        t = np.asarray(tok_ds[i], dtype=np.int64)
        assert t[-1] == EOD_ID, (name, i)
        texts.append(tok.decode(t[:-1][:keep].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False))
        pos.append(p)
    return name, _write_slices(tmp, name, pos, texts, n_rows)


def og_job(args):
    """Read the selected documents of one 50M bucket and cut them after `keep` training tokens."""
    name, bucket, want, keep, tmp, n_rows = args  # want: {id hash: row position}
    tok = _tokenizer()
    pos, texts, seen = [], [], {}
    for f in sorted(glob(f"{bucket}/part_*.parquet")):
        t = pq.read_table(f, columns=["id", "text"])
        for i, x in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
            h = id_hash(i)
            if h not in want:
                continue
            if h in seen:
                assert seen[h] == x, f"id {i} has two different texts"
                continue
            seen[h] = x
            if len(x) > keep:  # a token covers at least one character, so shorter texts have at most `keep` tokens
                enc = tok(x, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
                if len(enc["input_ids"]) > keep:
                    x = x[: enc["offset_mapping"][keep - 1][1]]
            texts.append(x)
            pos.append(want[h])
    assert len(seen) == len(want), (name, len(seen), len(want))
    return name, _write_slices(tmp, name, pos, texts, n_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/iopsstor/scratch/cscs/rsinghal/1pp-spp/refl/input")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--keep-tokens", type=int, default=4096)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--row-group", type=int, default=10_000)
    a = p.parse_args()
    out, tmp = Path(a.out), Path(a.out) / "tmp"
    for s in range(N_SLICES):
        (tmp / f"s{s:02d}").mkdir(parents=True, exist_ok=True)

    L = np.load(os.path.join(LABELS, "all_labels.npz"))
    half, h, score, n_tok = L["half"], L["hash"], L["score"], L["n_tok"]
    # One row per distinct document id; a doc in both halves must carry the same label.
    uh, first, inv = np.unique(h, return_index=True, return_inverse=True)
    smax = np.full(len(uh), -1, np.int64)
    np.maximum.at(smax, inv, score.astype(np.int64))
    smin = np.full(len(uh), 99, np.int64)
    np.minimum.at(smin, inv, score.astype(np.int64))
    n_dup = len(h) - len(uh)
    n_disagree = int(((smax >= 3) != (smin >= 3)).sum())
    assert n_disagree == 0, f"{n_disagree} documents in both halves get different flags"
    u_score, u_ntok = score[first], n_tok[first]

    rng = np.random.default_rng(a.seed)
    flagged = np.nonzero(u_score >= 3)[0]
    benign = np.sort(rng.choice(np.nonzero(u_score < 3)[0], size=len(flagged), replace=False))
    sel = np.concatenate([flagged, benign])
    is_flag = np.concatenate([np.ones(len(flagged), bool), np.zeros(len(benign), bool)])
    order = rng.permutation(len(sel))
    sel, is_flag = sel[order], is_flag[order]
    n_rows = len(sel)
    row_of = {int(uh[u]): r for r, u in enumerate(sel)}  # id hash -> row position
    print(f"distinct docs {len(uh)} (in both halves: {n_dup}); flagged {len(flagged)}, benign {len(benign)}", flush=True)

    # Which source serves each selected hash: the additions half when the doc is there, else the 50M half.
    jobs, served = [], set()
    for line in open(ADD_LIST):
        name, prefix = line.split()
        lab = np.load(os.path.join(LABELS, f"{name}.labels.npz"))
        want = {}
        for i, x in enumerate(lab["hash"].tolist()):
            if x in row_of and x not in served:
                want[i] = row_of[x]
                served.add(x)
        jobs.append((additions_job, (name, prefix, want, a.keep_tokens, tmp, n_rows)))
    for b in range(48):
        name = f"og50m_{b:02d}"
        lab = np.load(os.path.join(LABELS, f"{name}.labels.npz"))
        want = {x: row_of[x] for x in lab["hash"].tolist() if x in row_of and x not in served}
        served.update(want)
        jobs.append((og_job, (name, f"{TEXTS_50M}/b{b:02d}", want, a.keep_tokens, tmp, n_rows)))
    assert len(served) == n_rows, (len(served), n_rows)

    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fn, arg) for fn, arg in jobs]
        written = 0
        for f in futs:
            name, n = f.result()
            written += n
            print(f"{name}: {n} texts", flush=True)
    assert written == n_rows, (written, n_rows)

    schema = pa.schema([("doc_id", pa.string()), ("text", pa.string()), ("safety_score", pa.int8()),
                        ("token_length", pa.int64()), ("flagged", pa.bool_()), ("hash", pa.uint64())])
    w = pq.ParquetWriter(out / "sidecar.parquet", schema, compression="zstd")
    done = 0
    for s in range(N_SLICES):
        t = pa.concat_tables([pq.read_table(f) for f in sorted((tmp / f"s{s:02d}").glob("*.parquet"))])
        t = t.sort_by("pos")
        pos = t.column("pos").to_numpy()
        assert np.array_equal(pos, np.arange(done, done + len(pos))), f"slice {s} misses rows"
        u = sel[pos]
        tbl = pa.table({"doc_id": [f"{x:016x}" for x in uh[u].tolist()], "text": t.column("text"),
                        "safety_score": u_score[u].astype(np.int8), "token_length": u_ntok[u].astype(np.int64),
                        "flagged": is_flag[pos], "hash": uh[u]}, schema=schema)
        for lo in range(0, len(tbl), a.row_group):
            w.write_table(tbl.slice(lo, a.row_group))
        done += len(pos)
        print(f"slice {s}: rows up to {done}", flush=True)
    w.close()
    assert done == n_rows
    np.savez(out / "selection.npz", hash=uh[sel], flagged=is_flag, score=u_score[sel], n_tok=u_ntok[sel])
    json.dump({"rows": int(n_rows), "flagged": int(len(flagged)), "benign": int(len(benign)), "seed": a.seed,
               "keep_tokens": a.keep_tokens, "docs_in_both_halves": int(n_dup)}, open(out / "summary.json", "w"), indent=1)
    print(f"REFL_INPUT_DONE rows {n_rows} (flagged {len(flagged)}, benign {len(benign)})")


if __name__ == "__main__":
    main()
