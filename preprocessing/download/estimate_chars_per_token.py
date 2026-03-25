"""Estimate tokens-per-shard for a tokenizer on local parquet data.

Samples ``--n-shards`` local parquet files (default 100), counts characters in
each, and tokenizes a subsample of documents to derive a chars/token ratio.
Combining both gives a robust tokens-per-shard estimate with confidence
intervals.  Use the result with ``python -m preprocessing.download --n-shards N``.

Usage::

    # Estimate from 100 randomly sampled shards
    python -m preprocessing.download.estimate_chars_per_token \
        --data-dir $SCRATCH/dolma3_mix-1T \
        --tokenizer meta-llama/Llama-3.1-8B

    # Compute how many shards you need for 1T tokens
    python -m preprocessing.download.estimate_chars_per_token \
        --data-dir $SCRATCH/dolma3_mix-1T \
        --tokenizer meta-llama/Llama-3.1-8B \
        --target-tokens 1_000_000_000_000

    # With a context-length cap (e.g. training truncates to 2048)
    python -m preprocessing.download.estimate_chars_per_token \
        --data-dir $SCRATCH/dolma3_mix-1T \
        --tokenizer meta-llama/Llama-3.1-8B \
        --max-tokens-per-sample 2048
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for chars-per-token estimation."""
    p = argparse.ArgumentParser(description="Estimate tokens/shard from local parquet data.")
    p.add_argument("--data-dir", type=str, required=True, help="Directory with part_*.parquet files")
    p.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name or path")
    p.add_argument("--n-shards", type=int, default=100, help="Number of shards to sample for per-shard stats (default: 100)")
    p.add_argument("--n-docs", type=int, default=10_000, help="Number of documents to tokenize for chars/token ratio (default: 10,000)")
    p.add_argument("--text-column", type=str, default="text", help="Column containing text (default: text)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shard sampling (default: 42)")
    p.add_argument(
        "--max-tokens-per-sample",
        type=int,
        default=None,
        help="Cap per-sample token count (e.g. 2048 if training truncates)",
    )
    p.add_argument("--target-tokens", type=int, default=None, help="Token budget to compute --n-shards for")
    return p.parse_args()


def _fmt(n: float) -> str:
    """Format a large number with T/B/M suffix."""
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,.0f}"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("part_*.parquet"))
    assert all_files, f"No part_*.parquet files found in {data_dir}"

    n_total_files = len(all_files)
    n_sample = min(args.n_shards, n_total_files)

    rng = random.Random(args.seed)
    sampled_files = rng.sample(all_files, n_sample)

    # ── Pass 1: count chars per shard (fast, no tokenization) ────
    chars_per_shard = np.empty(n_sample, dtype=np.int64)
    rows_per_shard = np.empty(n_sample, dtype=np.int64)
    all_texts: list[str] = []
    docs_needed = args.n_docs

    for i, f in enumerate(tqdm(sampled_files, desc="Scanning shards")):
        table = pq.read_table(str(f), columns=[args.text_column])
        texts = table[args.text_column].to_pylist()
        chars_per_shard[i] = sum(len(t) for t in texts)
        rows_per_shard[i] = len(texts)
        if docs_needed > 0:
            take = min(docs_needed, len(texts))
            all_texts.extend(texts[:take])
            docs_needed -= take

    print(f"\nScanned {n_sample} / {n_total_files} shards")

    # ── Pass 2: tokenize document subsample for chars/token ratio ─
    all_texts = all_texts[: args.n_docs]
    print(f"Tokenizing {len(all_texts):,} documents...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    total_chars = 0
    total_tokens = 0
    total_tokens_capped = 0

    for text in tqdm(all_texts, desc="Tokenizing"):
        n_chars = len(text)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        total_chars += n_chars
        total_tokens += n_tokens
        if args.max_tokens_per_sample is not None:
            total_tokens_capped += min(n_tokens, args.max_tokens_per_sample)
        else:
            total_tokens_capped += n_tokens

    assert total_tokens > 0, "All documents tokenized to 0 tokens — check your data"
    chars_per_token = total_chars / total_tokens

    # ── Derive tokens-per-shard distribution ──────────────────────
    tokens_per_shard = chars_per_shard / chars_per_token
    mean_tps = np.mean(tokens_per_shard)
    std_tps = np.std(tokens_per_shard, ddof=1)
    ci95 = 1.96 * std_tps / math.sqrt(n_sample)

    total_tokens_est = mean_tps * n_total_files

    print(f"\n{'='*60}")
    print(f"  Tokenizer:             {args.tokenizer}")
    print(f"  Documents tokenized:   {len(all_texts):,}")
    print(f"  Chars/token:           {chars_per_token:.3f}")
    if args.max_tokens_per_sample is not None:
        capped_ratio = total_chars / total_tokens_capped
        print(f"  Chars/token (capped):  {capped_ratio:.3f}  (max {args.max_tokens_per_sample:,} tok/doc)")
    print()
    print(f"  Shards sampled:        {n_sample} / {n_total_files}")
    print(f"  Rows/shard:            {np.mean(rows_per_shard):,.0f}  (std {np.std(rows_per_shard, ddof=1):,.0f})")
    print(f"  Chars/shard:           {_fmt(np.mean(chars_per_shard))}  (std {_fmt(np.std(chars_per_shard, ddof=1))})")
    print(f"  Tokens/shard:          {_fmt(mean_tps)}  (std {_fmt(std_tps)})")
    print(f"  95% CI tokens/shard:   [{_fmt(mean_tps - ci95)}, {_fmt(mean_tps + ci95)}]")
    print()
    print(f"  Est. total ({n_total_files:,} shards): {_fmt(total_tokens_est)} tokens")

    if args.target_tokens is not None:
        n_needed = math.ceil(args.target_tokens / mean_tps)
        # conservative: use lower bound of CI
        n_needed_conservative = math.ceil(args.target_tokens / (mean_tps - ci95))
        print()
        print(f"  Target:                {_fmt(args.target_tokens)} tokens")
        print(f"  Shards needed:         {n_needed:,}  (mean estimate)")
        print(f"  Shards needed (safe):  {n_needed_conservative:,}  (lower 95% CI)")
        print()
        print(f"  Usage:")
        print(f"    python -m preprocessing.download --n-shards {n_needed_conservative}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
