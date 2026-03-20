# Download & Dedup

Download upstream HuggingFace dataset shards to local parquet files, with short-text filtering.

## Note on upstream row repetition

Upstream `allenai/dolma3_mix-6T` (and its 7B variant) contain within-file row repetition: ~45% of JSONL.zst shards have rows repeated 2-7x consecutively. This is **intentional quality-aware upsampling** by the dataset authors — higher-quality documents are repeated more often. See `report_upstream_dupes.py` to inspect this on any specific shard.

## Scripts

### `download.py` — Download

Downloads upstream shards via HuggingFace streaming, filters short texts, and writes local parquet files. Supports incremental resume via manifest + done markers.

**Cleaning steps (per shard):**
1. Drop rows where `text` has fewer than `--min-chars` characters (default: 32)

```bash
# Download ~1T tokens worth of shuffled shards from dolma3
python -m preprocessing.download_and_dedup.download \
    --dataset allenai/dolma3_mix-6T \
    --n-shards 47142 --shuffle --seed 42 \
    --columns text id source \
    --ignore-errors --workers 8

# Small test
python -m preprocessing.download_and_dedup.download \
    --dataset allenai/dolma3_mix-6T \
    --n-shards 10 --columns text id source --ignore-errors
```

**Input:** Remote HuggingFace streaming dataset.

**Output:**
- `part_XXXXX.parquet` — one parquet file per upstream shard (short texts filtered)
- `manifest.json` — deterministic shard plan (survives restarts)
- `metadata.json` — download stats (row counts, char counts, token estimate)
- `.done/` — per-shard completion markers for resume

### `download_job.sh` — SLURM wrapper

```bash
sbatch preprocessing/download_and_dedup/download_job.sh        # default: 47142 shards (~1T tokens)
sbatch preprocessing/download_and_dedup/download_job.sh 100    # small test
```

### `estimate_chars_per_token.py` — Token budget calculator

Samples local parquet data to estimate chars-per-token and compute how many shards are needed for a token budget.

```bash
python -m preprocessing.download_and_dedup.estimate_chars_per_token \
    --data-dir $SCRATCH/dolma3_mix-1T \
    --tokenizer allenai/OLMo-2-0325-32B \
    --target-tokens 1_000_000_000_000
```

### `report_upstream_dupes.py` — Inspect upstream row repetition

Downloads a specific shard from HuggingFace and reports repetition statistics (for understanding the quality-aware upsampling).

```bash
python preprocessing/download_and_dedup/report_upstream_dupes.py \
    --dataset allenai/dolma3_mix-6T \
    --file data/common_crawl-crime_and_law-0019/shard_00000079.jsonl.zst
```

## Pipeline overview

```
estimate_chars_per_token.py
  → compute --n-shards for token budget

download.py (via download_job.sh)
  → $SCRATCH/dataset_name/part_*.parquet

preprocessing/annotation/annotate.py (via annotation/job.sh)
  → data/safety_annotations/shard_*_part*.parquet
  → columns: id, safety_score (int8, 0-5), safety_probs (list<float32>)

preprocessing/annotation/analyze.py
  → console summary of score distribution

preprocessing/annotation/explore.py
  → join annotations back to source texts for manual inspection
```
