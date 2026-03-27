# Tokenization experiments

## Full 20K-file run (2026-03-24)

### Configuration

- Input: `$SCRATCH/dolma3_mix-1T_subsampled/{unannotated,annotated}` (20K files each)
- Output: `$SCRATCH/tokenized/`
- Tokenizer: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Seq length: 2048 (compact), 1920 content + 128 reflection budget (annotated)
- 20 SLURM array tasks (nodes), 20 workers each
- Job IDs: tokenize 1720966, merge 1720986

### Results

**Tokenize stage**: 18/20 nodes completed (2.5-3.1h wall time per node). Nodes 13 and 14 OOM'd on partitions with very large parquet files (MaxRSS ~360GB vs ~280-330GB for others).

- Node 13: 637/1000 compact tasks completed before OOM
- Node 14: 79/1000 compact tasks completed before OOM (single huge file)

Re-run needed for nodes 13, 14 with 10 workers (skip_completed resumes from where they left off).

**Merge stage**: blocked by OOM'd nodes (afterok dependency). Pending re-run.

### Node-hours estimate (from 100-file test)

| Stage | Node-hours |
|-------|----------:|
| Compact tokenize (20 nodes) | ~24 |
| Split tokenize (20 nodes) | ~3 |
| Merge + shuffle (1 node) | ~0.5 |
| **Total** | **~27.5** |

### 100-file test runs (2026-03-24)

**Single-node (slurm-1720343)**: 100 files, 20 workers, 14 min wall time.

| Pipeline | Documents | Tokens | Windows |
|----------|----------:|-------:|--------:|
| Compact | 2,794,349 | 2,747,222,216 | 1,340,762 |
| Annotated | 326,296 | 338,917,467 | 326,296 |

**4-node parallel (array-1720785 + 1720786)**: identical output to single-node. Tokenize 8 min (bottleneck: uneven file sizes), merge 40s.

### Lessons

- Max 20 workers per node due to OOM. Even 20 can OOM on partitions with very large files — reduce to 10 for those.
- File size variance causes load imbalance: node 14's partition had a single parquet that took >1h to tokenize.
- Compact shuffle was non-deterministic (seed=None). Fixed by passing seed=42.

## Full run — final results (2026-03-27)

### Tokenize stage (re-run nodes 13, 14)

Nodes 13 and 14 re-run with 10 workers (job IDs 1723040, 1723042). Completed via `skip_completed` resume, finishing all remaining tasks.

### Merge stage

Multiple iterations needed due to OOM and Arrow overflow issues during the annotated sidecar build.

| Job | Issue | Fix |
|-----|-------|-----|
| 1728796 | Slow: per-doc mmap reads for compact shuffle | Bulk numpy read (commit 35cf822) |
| 1729663 | Slow: per-window mmap for annotated shuffle | Bulk numpy shuffle (commit be3253c) |
| 1729863 | OOM at 28% sidecar: 102M × ~4KB text = ~430GB Python strings | Partition-write to 20 temp parquets, merge in order (commit 5f9434b) |
| 1730665 | Arrow offset overflow in sidecar sort_by: >2GB string columns | `pa.large_string()` 64-bit offsets (commit f758d59) |
| **1731033** | **Success** | 2h 14min total |

**Compact merge**: ~1.5h (concatenation + context window shuffle, 427.7M windows)
**Annotated merge**: 2h 14min total (5 min .bin write, 1h 20min sidecar partition-write, 49 min sidecar merge)

### Output files

```
$SCRATCH/tokenized/
├── compact/megatron/
│   ├── compact.bin       1.6 TB   (427,681,689 windows × 2049 tokens × uint16)
│   └── compact.idx       8.0 GB
└── annotated/
    ├── annotated.bin     393 GB   (102,772,028 windows × 2049 tokens × uint16)
    ├── annotated.idx     2.0 GB
    ├── token_lengths.npy 393 MB   (content length per window, int32)
    └── sidecar.parquet   474 GB   (doc_id, text, token_length, reflection fields)
```

### Verification (2026-03-27)

**Annotated** (200 sampled windows):
- Re-tokenization: 0 errors (sidecar text → Rust tokenizer → matches .bin tokens exactly)
- EOS/PAD boundaries: 0 errors
- token_length (sidecar vs .npy): 0 errors
- Reflection fields: all empty as expected
- Token length range: [2, 1919], mean 1041.2

**Compact** (500 sampled windows):
- 0 all-zero windows, 0 OOV tokens, EOS in every window
- Token entropy: 10.90 bits (max 15.58 for 49K vocab)
- Zero (EOS) fraction: 0.11%
- Unique tokens observed: 36,958 / 49,152

### Tokenizer note

Pipeline uses `tokenizers.Tokenizer.from_pretrained()` (Rust library, via datatrove). This produces different BPE merges for `\n\n` compared to `transformers.AutoTokenizer` (single token 1116 vs two tokens [198, 198]). Verification must use the Rust tokenizer to match. Additionally, `enable_truncation(max_length=1920)` includes the EOS in the limit, so max content tokens = 1919.

### Total node-hours

| Stage | Nodes | Wall time | Node-hours |
|-------|------:|----------:|-----------:|
| Tokenize (20 nodes) | 20 | ~3h | ~60 |
| Re-run nodes 13, 14 | 2 | ~2.5h | ~5 |
| Merge (compact + annotated) | 1 | ~4h | ~4 |
| **Total** | | | **~69** |
