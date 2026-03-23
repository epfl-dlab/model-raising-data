"""Tokenize pre-split parquets into two Megatron-format training streams.

Expects two separate input directories (produced by subsample_and_stratify):

* ``--compact-data-dir`` — parquets with non-annotated samples only.
* ``--annotated-data-dir`` — parquets with annotated samples only.

Two pipelines controlled by ``--pipeline {compact,split,both}``:

* **compact** — pack non-annotated samples into dense 2048-token windows,
  output as Megatron ``.bin`` + ``.idx``.
* **split** — tokenize annotated samples, one per padded 2049-token window,
  output as Megatron ``.bin`` + ``.idx`` plus a sidecar parquet with original
  text and token lengths for downstream reflection.

Usage::

    # Both pipelines (default)
    python -m preprocessing.tokenization.tokenize \\
        --compact-data-dir $SCRATCH/dolma3_non_annotated \\
        --annotated-data-dir $SCRATCH/dolma3_annotated \\
        --output-dir $SCRATCH/tokenized

    # Compact only
    python -m preprocessing.tokenization.tokenize \\
        --compact-data-dir $SCRATCH/dolma3_non_annotated --pipeline compact

    # Split only
    python -m preprocessing.tokenization.tokenize \\
        --annotated-data-dir $SCRATCH/dolma3_annotated --pipeline split
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm


def _count_parquets(data_dir: str) -> int:
    """Count parquet files in *data_dir* (recursive)."""
    n = len(list(Path(data_dir).rglob("*.parquet")))
    assert n > 0, f"No parquet files found in {data_dir}"
    return n


# ---------------------------------------------------------------------------
# Compact path (datatrove)
# ---------------------------------------------------------------------------


def run_compact(args: argparse.Namespace) -> None:
    """Tokenize non-annotated samples into packed Megatron .bin + .idx."""
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger

    from preprocessing.tokenization.steps import (
        MegatronContextShuffler,
        TruncatingDocumentTokenizer,
    )

    out = Path(args.output_dir) / "compact"
    logs = str(Path(args.output_dir) / "logs")
    n_tasks = _count_parquets(args.compact_data_dir)
    print(f"Compact: {n_tasks} input files, {args.workers} workers, seq_length={args.seq_length}")

    # Stage 1: tokenize (parallel across input files)
    stage1 = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=args.compact_data_dir,
                text_key="text",
                id_key="id",
            ),
            TruncatingDocumentTokenizer(
                max_doc_tokens=args.seq_length,
                output_folder=str(out / "tokenized"),
                tokenizer_name_or_path=args.tokenizer,
                eos_token="<|endoftext|>",
                shuffle_documents=True,
                batch_size=10000,
            ),
        ],
        tasks=n_tasks,
        workers=args.workers,
        logging_dir=f"{logs}/compact_stage1",
        skip_completed=True,
    )

    # Stage 2: merge + global shuffle (single task)
    stage2 = LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=str(out / "tokenized"),
                output_folder=str(out / "merged"),
                save_filename="dolma3",
                shuffle=True,
            ),
        ],
        tasks=1,
        logging_dir=f"{logs}/compact_stage2",
        depends=stage1,
    )

    # Stage 3: context shuffle → Megatron .bin + .idx
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MegatronContextShuffler(
                input_folder=str(out / "merged"),
                output_folder=str(out / "megatron"),
                window_size=args.seq_length + 1,
                save_filename="compact",
            ),
        ],
        tasks=1,
        logging_dir=f"{logs}/compact_stage3",
        depends=stage2,
    )

    stage3.run()
    print(f"Compact done → {out / 'megatron'}")


# ---------------------------------------------------------------------------
# Split path (pyarrow)
# ---------------------------------------------------------------------------


def _collect_annotated(data_dir: str) -> list[dict]:
    """Collect (doc_id, file_path, row_index) for all rows in annotated parquets.

    All rows are assumed annotated (the input directory contains only annotated
    samples).  Text is NOT loaded — just lightweight metadata for shuffling.
    """
    import pyarrow.parquet as pq

    input_files = sorted(Path(data_dir).rglob("*.parquet"))
    assert len(input_files) > 0, f"No parquet files found in {data_dir}"

    entries: list[dict] = []
    for fpath in tqdm(input_files, desc="Scanning annotated", unit="file"):
        ids = pq.read_table(fpath, columns=["id"]).column("id").to_pylist()
        for row_idx, doc_id in enumerate(ids):
            entries.append(
                {"doc_id": doc_id, "file_path": str(fpath), "row_index": row_idx}
            )
    return entries


# EOS token id for <|endoftext|> — matches the compact path's eos_token config.
# NOTE: this is NOT tokenizer.eos_token_id (which is 2 for SmolLM2-Instruct).
_EOS_TOKEN_ID = 0


def run_split(args: argparse.Namespace) -> None:
    """Tokenize annotated samples into padded Megatron .bin + .idx + sidecar."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from datatrove.pipeline.tokens.megatron_tokenizer import MegatronTokenizedFile

    from pipeline.tokenizer import _get_tokenizer

    out = Path(args.output_dir) / "annotated"
    out.mkdir(parents=True, exist_ok=True)

    bin_final = out / "annotated.bin"
    if bin_final.exists():
        print(f"Split: {bin_final} already exists, skipping.")
        return

    max_tokens = args.seq_length - args.reflection_budget
    window_size = args.seq_length + 1  # 2049

    # --- Pass 1: collect lightweight metadata ---
    entries = _collect_annotated(args.annotated_data_dir)
    n_annotated = len(entries)
    assert n_annotated > 0, f"No annotated rows found in {args.annotated_data_dir}"
    print(f"Split: {n_annotated} annotated samples, max_tokens={max_tokens}")

    # --- Shuffle with fixed seed ---
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_annotated)
    entries = [entries[i] for i in perm]

    # --- Pass 2: group by file, read text, build position→data mapping ---
    from collections import defaultdict
    file_groups: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for pos, entry in enumerate(entries):
        file_groups[entry["file_path"]].append(
            (pos, entry["row_index"], entry["doc_id"])
        )

    texts: list[str | None] = [None] * n_annotated
    doc_ids: list[str | None] = [None] * n_annotated

    tokenizer = _get_tokenizer()

    for fpath, group in tqdm(file_groups.items(), desc="Reading annotated texts", unit="file"):
        table = pq.read_table(fpath, columns=["id", "text"])
        row_indices = [row_idx for _, row_idx, _ in group]
        subtable = table.take(row_indices)
        file_texts = subtable.column("text").to_pylist()
        for (pos, _, did), text in zip(group, file_texts):
            texts[pos] = text
            doc_ids[pos] = did

    # --- Write .bin + .idx + sidecar + token_lengths ---
    megatron_file = MegatronTokenizedFile(
        output_folder=str(out),
        filename="annotated_tmp",
        token_size=2,
    )

    token_lengths: list[int] = []
    sidecar_doc_ids: list[str] = []
    sidecar_texts: list[str] = []

    for pos in tqdm(range(n_annotated), desc="Tokenizing + writing", unit="doc"):
        text = texts[pos]
        did = doc_ids[pos]
        assert text is not None and did is not None

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(token_ids) > 0, f"Empty tokenization for doc {did}"

        content = token_ids[:max_tokens]
        actual_length = len(content)
        padding_len = window_size - actual_length - 1  # -1 for EOS
        assert padding_len >= 0, (
            f"Content too long: {actual_length} + 1 EOS > {window_size}"
        )
        window = content + [_EOS_TOKEN_ID] * (1 + padding_len)
        assert len(window) == window_size

        megatron_file.write(window)
        token_lengths.append(actual_length)
        sidecar_doc_ids.append(did)
        sidecar_texts.append(text)

        texts[pos] = None
        doc_ids[pos] = None

    megatron_file.close()

    # Atomic rename
    (out / "annotated_tmp.bin").rename(bin_final)
    (out / "annotated_tmp.idx").rename(out / "annotated.idx")

    # token_lengths.npy
    np.save(str(out / "token_lengths.npy"), np.array(token_lengths, dtype=np.int32))

    # sidecar.parquet
    sidecar = pa.table({
        "doc_id": pa.array(sidecar_doc_ids, type=pa.string()),
        "text": pa.array(sidecar_texts, type=pa.string()),
        "token_length": pa.array(token_lengths, type=pa.int32()),
        "reflection": pa.array([""] * n_annotated, type=pa.string()),
        "preflection": pa.array([""] * n_annotated, type=pa.string()),
        "reflection_position": pa.array([0] * n_annotated, type=pa.int32()),
    })
    pq.write_table(sidecar, str(out / "sidecar.parquet"))

    print(
        f"Split done → {out}\n"
        f"  {n_annotated:,} annotated windows in annotated.bin + annotated.idx\n"
        f"  sidecar.parquet ({n_annotated:,} rows)\n"
        f"  token_lengths.npy (min={min(token_lengths)}, max={max(token_lengths)})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    scratch = os.environ.get(
        "SCRATCH", f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}"
    )

    p = argparse.ArgumentParser(
        description="Tokenize pre-split parquets into two Megatron training streams."
    )
    p.add_argument(
        "--compact-data-dir",
        type=str,
        default=f"{scratch}/dolma3_non_annotated",
        help="Input parquets for compact path (non-annotated samples only)",
    )
    p.add_argument(
        "--annotated-data-dir",
        type=str,
        default=f"{scratch}/dolma3_annotated",
        help="Input parquets for split path (annotated samples only)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=f"{scratch}/tokenized",
        help="Base output directory",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="HF tokenizer name or path",
    )
    p.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length for compact path (default: 2048)",
    )
    p.add_argument(
        "--reflection-budget",
        type=int,
        default=128,
        help="Tokens reserved for reflections in split path (default: 128)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Parallel workers (default: 64)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for annotated shuffle (default: 42)",
    )
    p.add_argument(
        "--pipeline",
        choices=["compact", "split", "both"],
        default="both",
        help="Which pipeline(s) to run (default: both)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.pipeline in ("compact", "both"):
        run_compact(args)
    if args.pipeline in ("split", "both"):
        run_split(args)


if __name__ == "__main__":
    main()
