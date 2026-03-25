# Subsample & Stratify -- Experiment Log

## 2026-03-24: Production run (1T tokens, threshold=3)

**Job**: 1719619 on nid (normal partition, exclusive node)
**Input**: `$SCRATCH/dolma3_mix-1T_annotated/` (20,000 files, 1.18B rows)
**Output**: `$SCRATCH/dolma3_mix-1T_subsampled/` (annotated/ + unannotated/)

### Configuration

- `--target-tokens 1_000_000_000_000`
- `--annotation-threshold 3` (scores 3, 4, 5 are "bad")
- `--seed 42`
- `--workers 16`
- `--chars-per-token 4.068`

### Results

| Metric | Value |
|--------|-------|
| Runtime | 32 min 14 sec |
| Peak memory | 38 GB |
| Total rows | 1,027,837,579 |
| Total tokens | 1.00T |
| Annotated rows | 102,772,028 (10.0%) |
| Annotated tokens | 110.29B (11.03%) |
| Unannotated rows | 925,065,551 (90.0%) |
| Unannotated tokens | 889.74B (88.97%) |
| Scale factor | 0.8717 (1T target / 1.15T available) |
| Above-threshold tokens | 63.25B (5.51% of total) |
| Output files | 20,000 annotated + 20,000 unannotated |

### Score distribution (full dataset)

```
  0 (safe):          926,715,941 (78.50%)
  1 (minimal):       107,477,673 ( 9.10%)
  2 (mild):           92,731,623 ( 7.86%)
  3 (moderate):       31,659,336 ( 2.68%)
  4 (significant):    11,500,784 ( 0.97%)
  5 (severe):         10,421,118 ( 0.88%)
```

### Architecture notes

This run used the rewritten per-file-independent pipeline. Previous attempts with a global in-memory index OOMed at ~467GB on a 450GB node due to glibc malloc fragmentation. The per-file design processes each file independently with only global per-file statistics shared between passes, achieving 38GB peak memory.

Key design decisions:
- Per-file deterministic RNG: `np.random.default_rng(seed=(42, file_index))`
- Per-file proportional token budgets (largest-remainder style)
- Token estimates capped at 2048 to match tokenizer truncation
- Both passes parallelized with ProcessPoolExecutor (16 workers)

### Spot checks

- Schema verified: all output files contain `id`, `text`, `source`, `safety_score`, `has_annotation`, `is_bad`
- `has_annotation` = True in all annotated rows, False in all unannotated rows
- `is_bad` = `(safety_score >= 3)` verified across sampled files
- Note: source data has duplicate IDs due to upsampling — same ID can appear in both splits (different rows). This is expected.

### Post-run repair: `source` column schema (2026-03-24)

4,215 of 40,000 output parquet files had the `source` column typed as Arrow `null` instead of `string` (files where all rows had null source values). This caused `load_dataset()` to fail with `Couldn't cast array of type string to null` due to schema mismatch across files. Repaired in-place by casting `source: null → string` (with null values preserved). The fix was also applied to `_process_one_file` in `subsample.py` to prevent recurrence.
