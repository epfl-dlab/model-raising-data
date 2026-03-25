# Subsample & Stratify -- annotation-based subsampling

Produces two token-budgeted subsets from annotated source data:

- **`annotated/`** — rows with `has_annotation=True` (safety_score >= threshold + matched random sample from lower scores)
- **`unannotated/`** — the remaining rows (`has_annotation=False`)

Both outputs include an `is_bad` column (`True` if `safety_score >= threshold`).

The annotation ratio in the output matches the full dataset. Per-file deterministic RNG ensures reproducibility.

## Pipeline position

```
  annotation + merge           subsample_and_stratify          tokenization
$SCRATCH/dolma3_mix-1T_annotated --> subsample.py          --> tokenize.py
                                    (annotate + budget fill)
                                    ├── annotated/          --> split path
                                    └── unannotated/        --> compact path
```

## Input

**Source data** (`--source-dir`): parquet files matching `part_*.parquet` with at least `text` (string) and `safety_score` (int8). Produced by the annotation merge step (`annotation/merge.py`).

## Output

```
$SCRATCH/dolma3_mix-1T_subsampled/
├── annotated/
│   ├── part_00000.parquet    # source columns + has_annotation + is_bad
│   └── ...
├── unannotated/
│   ├── part_00000.parquet    # source columns + has_annotation + is_bad
│   └── ...
└── metadata.json             # sampling parameters, annotation ratio, stats
```

Output filenames mirror input filenames (one-to-one correspondence).

## Usage

```bash
# Full run (all tokens, annotation threshold=3)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T_annotated \
    --output-dir $SCRATCH/dolma3_mix-1T_subsampled

# With token budget
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T_annotated \
    --target-tokens 1_000_000_000_000

# Custom threshold (annotate scores >= 4 only)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T_annotated \
    --annotation-threshold 4

# More workers for faster I/O
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T_annotated \
    --workers 32
```

### Algorithm

Two-pass, per-file-independent design (~200MB peak RAM regardless of dataset size):

1. **Pass 1 (scan)** -- read each file in parallel, collect per-file statistics: total tokens, above/below-threshold token counts. Compute global annotation ratio R and per-file token budgets (proportional allocation).
2. **Pass 2 (write)** -- for each file independently in parallel: re-read the file, select rows using a per-file deterministic RNG (seeded by `(global_seed, file_index)`) and the file's token budget, write annotated and unannotated rows to their respective output directories.

Annotation marking: all rows with `safety_score >= threshold` are annotated. An equal amount of tokens from lower-score rows are randomly sampled as annotated. Token estimates use `utf8_length / chars_per_token`, capped at 2048 (the tokenizer's truncation limit).

### Scripts

| Script | Purpose |
|--------|---------|
| `subsample.py` | Annotation-based subsampling with two output datasets |
| `upload.py` | Upload subsampled dataset to HuggingFace Hub |
| `test_subsample.py` | End-to-end test + determinism verification |

### End-to-end test

Creates a synthetic dataset (1000 rows, known score distribution), runs the pipeline, verifies two output directories, annotation ratios, `has_annotation`/`is_bad` flags, schema, metadata, and determinism (same seed → same output).

```bash
python -m preprocessing.subsample_and_stratify.test_subsample
```

## Experiment tracking

`metadata.json` in the output directory records annotation threshold, annotation ratio, per-split token/row counts, and timing. Job scripts additionally log to `data/experiments/subsample.jsonl` via `experiment_tracker.py`.

## Resume

Not supported -- the pipeline completes in ~30 min. Re-run with `--overwrite` to replace.
