"""Turn one tokenized raw-document prefix of the 1PP corpus (or a dir of {id, text} parquet) into classifier chunks.

Reads the Megatron prefix (raw-doc mode of 1pp-training/data/format_1pp.py: document tokens, then one
EOD) and its .ids.npy (blake2b-8 hash of the DCLM id, one per document). Decodes every document with
the training tokenizer, so the classifier scores exactly the text the model trains on, and checks
that re-encoding gives back the training tokens on every `--verify-every`-th document. Splits each
document into classifier windows with chunk_docs.chunk_text (chunk-max labeling).

Writes <out>/chunks/part_NNNNN.parquet (id "<doc>::c<k>", doc_id, text; doc = local index in this
prefix) for annotate.py, and <out>/docs.npz (hash, n_tok, n_chunks per local index).
"""
import argparse
import hashlib
import os
from glob import glob
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TRAIN = "/capstor/store/cscs/swissai/infra01/users/rsinghal/1pp-training"
sys.path.insert(0, os.path.join(TRAIN, "Megatron-LM-MoE"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunk_docs import MODEL_ID, chunk_text  # noqa: E402

EOD_ID, SEG_EOD, SEG_DOCUMENT = 0, 5, 8
TRAIN_TOKENIZER = os.path.join(TRAIN, "tokenizers", "smollm2_pretrain")
_S = {}


def _init(prefix, window, verify_every):
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from transformers import AutoTokenizer
    if prefix is not None:
        _S.update(tok=IndexedDataset(prefix, multimodal=False, mmap=True),
                  seg=IndexedDataset(prefix + "_seg", multimodal=False, mmap=True))
    _S.update(train_tok=AutoTokenizer.from_pretrained(TRAIN_TOKENIZER),
              clf_tok=AutoTokenizer.from_pretrained(MODEL_ID),
              window=window, verify_every=verify_every)


def _doc(i):
    """Local document i -> (i, n_tok, chunks, verified)."""
    t = np.asarray(_S["tok"][i], dtype=np.int64)
    s = np.asarray(_S["seg"][i])
    assert len(t) == len(s) >= 1, i
    assert t[-1] == EOD_ID and s[-1] == SEG_EOD, (i, t[-1], s[-1])
    assert (s[:-1] == SEG_DOCUMENT).all(), i
    body = t[:-1].tolist()
    text = _S["train_tok"].decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    verified = False
    if i % _S["verify_every"] == 0:
        back = _S["train_tok"](text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        assert back == body, f"decode round trip differs for document {i}"
        verified = True
    return i, len(t), chunk_text(_S["clf_tok"], text, _S["window"]), verified


def _text(i):
    """Document i of the --texts input -> (i, n_tok incl. EOD, chunks, False), tokenized as format_1pp.py does."""
    text = _TEXTS[i]
    n = len(_S["train_tok"](text, add_special_tokens=False, split_special_tokens=True)["input_ids"]) + 1
    return i, n, chunk_text(_S["clf_tok"], text, _S["window"]), False


def load_texts(d):
    """{id, text} parquet files in d -> (hashes, texts), one entry per distinct id."""
    t = pa.concat_tables([pq.read_table(f, columns=["id", "text"]) for f in sorted(glob(f"{d}/part_*.parquet"))])
    seen = {}
    for i, x in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
        assert seen.setdefault(i, x) == x, f"id {i} has two different texts"
    ids = sorted(seen)
    return np.array([id_hash(i) for i in ids], dtype=np.uint64), [seen[i] for i in ids]


def id_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


_TEXTS = None


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prefix")
    src.add_argument("--texts", help="dir of {id, text} parquet (partition_50m.py bucket)")
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=2046)
    p.add_argument("--workers", type=int, default=200)
    p.add_argument("--docs-per-part", type=int, default=250_000)
    p.add_argument("--verify-every", type=int, default=100)
    a = p.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _TEXTS
    if a.texts:
        hashes, _TEXTS = load_texts(a.texts)
        n, fn = len(_TEXTS), _text
    else:
        hashes = np.load(a.prefix + ".ids.npy")
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        n, fn = len(IndexedDataset(a.prefix, multimodal=False, mmap=True)), _doc
    assert len(hashes) == n, (len(hashes), n)
    out = Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    n_tok = np.zeros(n, dtype=np.int64)
    n_chunks = np.zeros(n, dtype=np.int32)
    n_verified = 0
    with Pool(a.workers, initializer=_init, initargs=(a.prefix, a.window, a.verify_every)) as pool:
        for part, lo in enumerate(range(0, n, a.docs_per_part)):
            hi = min(n, lo + a.docs_per_part)
            rows = {"id": [], "doc_id": [], "text": []}
            for i, nt, cs, ver in pool.imap(fn, range(lo, hi), chunksize=512):
                n_tok[i], n_chunks[i] = nt, len(cs)
                n_verified += ver
                for k, c in enumerate(cs):
                    rows["id"].append(f"{i}::c{k}")
                    rows["doc_id"].append(str(i))
                    rows["text"].append(c)
            pq.write_table(pa.table(rows), out / "chunks" / f"part_{part:05d}.parquet")
            print(f"part {part}: docs {lo}-{hi}, {len(rows['id'])} chunks", flush=True)
    assert (n_chunks >= 1).all() and (n_tok >= 1).all()
    np.savez(out / "docs.npz", hash=hashes, n_tok=n_tok, n_chunks=n_chunks)
    print(f"EXPORT_DONE docs {n} chunks {int(n_chunks.sum())} multi-chunk {int((n_chunks > 1).sum())} "
          f"round-trip verified {n_verified}")


if __name__ == "__main__":
    main()
