# %% [markdown]
# # Interleaved Dataloader Demo
# Load compact + annotated streams, inspect samples, check sidecar data.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import MMapIndexedDataset

# %%
# --- Paths (adjust to your setup) ---
PERSIST = Path("/capstor/store/cscs/swissai/a141/jminder/model_raising_data/tokenized")
COMPACT_PREFIX = str(PERSIST / "compact" / "compact")
ANNOTATED_PREFIX = str(PERSIST / "annotated" / "annotated")
TOKEN_LENGTHS_PATH = str(PERSIST / "annotated" / "token_lengths.npy")
SIDECAR_PATH = str(PERSIST / "annotated" / "sidecar.parquet")

SEQ_LENGTH = 2048
EOS_TOKEN_ID = 0

# %%
# --- Load datasets ---
compact = MMapIndexedDataset(COMPACT_PREFIX, skip_warmup=True)
annotated = MMapIndexedDataset(ANNOTATED_PREFIX, skip_warmup=True)
ann_lengths = np.load(TOKEN_LENGTHS_PATH)
sidecar = pd.read_parquet(SIDECAR_PATH)

print(f"Compact:   {len(compact):,} windows")
print(f"Annotated: {len(annotated):,} windows")
print(f"Sidecar:   {len(sidecar):,} rows")
print(f"Annotation ratio: {len(annotated) / (len(annotated) + len(compact)):.4%}")

# %%
# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# %%
# --- Inspect compact samples ---
print("=" * 60)
print("COMPACT SAMPLES (dense-packed, multi-doc windows)")
print("=" * 60)

for i in range(3):
    tokens = compact[i]
    text = tokenizer.decode(tokens[:SEQ_LENGTH], skip_special_tokens=False)
    n_eos = (tokens == EOS_TOKEN_ID).sum()
    print(f"\n--- compact[{i}] | {len(tokens)} tokens | {n_eos} EOS markers ---")
    print(text[:500] + "..." if len(text) > 500 else text)

# %%
# --- Inspect annotated samples ---
print("=" * 60)
print("ANNOTATED SAMPLES (one doc per window, padded)")
print("=" * 60)

for i in range(3):
    tokens = annotated[i]
    length = int(ann_lengths[i])
    content_tokens = tokens[:length]
    text = tokenizer.decode(content_tokens, skip_special_tokens=False)

    print(f"\n--- annotated[{i}] | content: {length} tokens | "
          f"window: {len(tokens)} | padding: {len(tokens) - length - 1} ---")
    print(text[:500] + "..." if len(text) > 500 else text)

    # Verify structure: EOS after content, then all zeros
    assert tokens[length] == EOS_TOKEN_ID, f"Expected EOS at position {length}"
    assert np.all(tokens[length + 1:] == EOS_TOKEN_ID), "Expected padding after EOS"

# %%
# --- Sidecar: match annotated windows to original metadata ---
print("=" * 60)
print("SIDECAR (annotated window metadata)")
print("=" * 60)

print(f"\nColumns: {list(sidecar.columns)}")
print(f"\nFirst 5 rows:")
print(sidecar[["doc_id", "token_length", "reflection", "preflection"]].head())

# Verify sidecar alignment: token_length should match ann_lengths
for i in range(min(5, len(sidecar))):
    assert sidecar.iloc[i]["token_length"] == ann_lengths[i], (
        f"Sidecar/ann_lengths mismatch at row {i}"
    )
print("\nSidecar token_lengths align with token_lengths.npy")

# %%
# --- Sidecar: compare original text to detokenized ---
print("=" * 60)
print("SIDECAR vs DETOKENIZED (round-trip check)")
print("=" * 60)

for i in range(3):
    original = sidecar.iloc[i]["text"]
    length = int(ann_lengths[i])
    decoded = tokenizer.decode(annotated[i][:length], skip_special_tokens=False)

    print(f"\n--- Row {i} | doc_id={sidecar.iloc[i]['doc_id']} ---")
    print(f"  Original:    {original[:200]}{'...' if len(original) > 200 else ''}")
    print(f"  Detokenized: {decoded[:200]}{'...' if len(decoded) > 200 else ''}")

# %%
# --- Token length distribution ---
print("=" * 60)
print("ANNOTATED TOKEN LENGTH DISTRIBUTION")
print("=" * 60)

print(f"  Count: {len(ann_lengths):,}")
print(f"  Min:   {ann_lengths.min()}")
print(f"  Max:   {ann_lengths.max()}")
print(f"  Mean:  {ann_lengths.mean():.1f}")
print(f"  Median: {np.median(ann_lengths):.0f}")
pct_full = (ann_lengths >= 1920).mean()
print(f"  At max (1920): {pct_full:.2%}")

# %%
# --- Test the InterleavedDataset ---
from preprocessing.tokenization.dataloader import InterleavedDataset

dataset = InterleavedDataset(
    compact_prefix=COMPACT_PREFIX,
    annotated_prefix=ANNOTATED_PREFIX,
    token_lengths_path=TOKEN_LENGTHS_PATH,
    num_samples=100,
    seq_length=SEQ_LENGTH,
)

print(f"Interleaved dataset: {len(dataset)} samples")
print(f"Annotation ratio: {dataset.ratio:.4%}")

# Count how many of the first 100 samples are annotated
n_ann = 0
for i in range(100):
    sample = dataset[i]
    assert sample["text"].shape == (SEQ_LENGTH + 1,), f"Bad shape at {i}"
    assert sample["loss_mask"].shape == (SEQ_LENGTH,), f"Bad mask shape at {i}"
    ann_so_far = int((i + 1) * dataset.ratio)
    ann_prev = int(i * dataset.ratio)
    if ann_so_far > ann_prev:
        n_ann += 1

print(f"Annotated in first 100: {n_ann} (expected ~{dataset.ratio * 100:.1f})")
