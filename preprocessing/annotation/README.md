# Annotation -- safety score classification

Annotates text datasets with safety scores using [locuslab/safety-classifier_gte-large-en-v1.5](https://huggingface.co/locuslab/safety-classifier_gte-large-en-v1.5) (6-class scale: 0 = safe, 5 = severe).

## Pipeline position

```
  download          annotation                    subsample_and_stratify
$SCRATCH/dolma3_mix-1T/ --> annotate.py --> merge.py -->  subsample.py --> ...
                            (per-task,     (id-based
                             4x GH200)      join)
```

## Input

`$SCRATCH/dolma3_mix-1T/part_*.parquet`, each containing at least:
- `id` (string) -- row identifier (configurable via `--id-column`)
- `text` (string) -- text to classify (configurable via `--text-column`)

Alternatively, `annotate.py` can stream directly from HuggingFace (`--dataset`).

## Output

### Annotation shards (intermediate)

Per-task directories under `--output-dir`:

```
$SCRATCH/safety_annotations/dolma3/
├── task_0000/
│   ├── shard_0000_part0000.parquet   # rank 0 output
│   ├── shard_0001_part0000.parquet   # rank 1 output
│   ├── ...
│   ├── task_meta.json                # file list, row counts, world_size
│   ├── gpu_monitor.json              # GPU hours, utilization, power, temperature
│   └── DONE                          # completion marker
├── task_0001/
│   └── ...
└── ...
```

Each shard contains: `id` (string), `safety_score` (int8), `safety_probs` (list[float32]).

### Merged output (final)

Original parquets with an added `safety_score` (int8) column:

```
$SCRATCH/dolma3_mix-1T_annotated/
├── part_00000.parquet
├── part_00001.parquet
└── ...
```

### On-the-fly dedup

`annotate.py` deduplicates per-file before classification via `_compute_dedup_indices`: upstream data has ~40-45% within-file row repetition (intentional quality-aware upsampling), so dedup saves that much compute. `merge.py` then joins annotations back by id, propagating scores to all duplicates (including repeated rows that share the same id).

## Usage

### Single-node run

```bash
# Stream from HuggingFace
sbatch preprocessing/annotation/job.sh

# From local parquet
sbatch preprocessing/annotation/job.sh --data-dir $SCRATCH/finephrase/all

# Quick local test
python -m preprocessing.annotation.annotate --max-samples 1000
```

### Scaled run (array job)

```bash
# 1. Submit N tasks (partitions input files across tasks, 4×GH200 each)
sbatch --array=0-32 preprocessing/annotation/array_job.sh \
    $SCRATCH/dolma3_mix-1T $SCRATCH/safety_annotations/dolma3 33 20000

# 2. Resubmit failed tasks (annotate.py resumes automatically)
sbatch --array=5,23 preprocessing/annotation/array_job.sh \
    $SCRATCH/dolma3_mix-1T $SCRATCH/safety_annotations/dolma3 33 20000

# 3. Monitor progress
bash preprocessing/annotation/status.sh

# 4. Merge annotations into output parquet (on a compute node)
sbatch preprocessing/annotation/merge_job.sh

# 5. Upload deduplicated annotations to HuggingFace Hub
sbatch preprocessing/annotation/upload_job.sh
```

### Scripts

| Script | Purpose |
|--------|---------|
| `annotate.py` | Multi-GPU safety classifier (torchrun), with per-file dedup |
| `array_job.sh` | SLURM array wrapper: partitions input files across N tasks |
| `job.sh` | Single-node SLURM wrapper |
| `merge.py` | Id-based join of annotation shards back onto original parquets |
| `merge_job.sh` | SLURM wrapper for merge (64 workers on GH200 node, ~244 CPUs) |
| `upload_annotations.py` | Consolidate + deduplicate annotations and upload to HF Hub |
| `upload_job.sh` | SLURM wrapper for upload (~34GB RAM for 130M-entry dict) |
| `status.sh` | Live progress dashboard for running annotation jobs |
| `verify_scores.py` | Re-classify samples on GPU to verify pipeline correctness |
| `analyze.py` | Console summary of score distribution |
| `explore.py` | Join annotations to source texts for manual inspection |

### Testing the pipeline

Before scaling to the full dataset, validate end-to-end on a small dataset:

```bash
# 1. Run a 2-task array with --max-samples 1000 per task
TOTAL=$(ls $SCRATCH/dolma3_mix-1B/part_*.parquet | wc -l)
sbatch --array=0-1 preprocessing/annotation/array_job.sh \
    $SCRATCH/dolma3_mix-1B $SCRATCH/safety_annotations/dolma3_test 2 $TOTAL \
    --max-samples 1000

# 2. Check both tasks completed
ls $SCRATCH/safety_annotations/dolma3_test/task_*/DONE

# 3. Run merge
python -m preprocessing.annotation.merge \
    --data-dir $SCRATCH/dolma3_mix-1B \
    --annotation-dir $SCRATCH/safety_annotations/dolma3_test \
    --output-dir $SCRATCH/dolma3_mix-1B_annotated_test \
    --workers 4

# 4. Verify output
python -c "
import pyarrow.parquet as pq, glob
f = sorted(glob.glob('$SCRATCH/dolma3_mix-1B_annotated_test/part_*.parquet'))[0]
t = pq.read_table(f)
print(t.schema)
print(t.column('safety_score').to_pylist()[:20])
"
```

## Resource estimates

Calibrated on 2026-03-22 from a 10-node estimation run (380 files across varied sizes):

| Metric | Value |
|--------|-------|
| Throughput per node (4×GH200) | 569–799 unique samples/sec (mean 666) |
| Within-file dedup | ~66% (quality-aware upsampling) |
| Avg unique rows/file | ~20K |
| Model load + torch.compile overhead | ~5 min/task |

**20K-file run (33 tasks × ~607 files):**

| | Mean | Conservative (slowest node) |
|---|---|---|
| Time per task | ~5.3h | ~5.9h |
| Total GPU-h (estimate) | ~504 | ~600 |

## Resume

- **Within a task** -- corrupt parquet files (missing footer from killed jobs) are deleted on resume. Already-written rows are skipped via `count_existing_rows`. Resubmitting the same array index picks up where it left off.
- **Across tasks** -- each task writes a `DONE` marker only on successful completion. Failed tasks are easy to identify and resubmit.
- **At merge time** -- `merge.py` validates that all tasks have `DONE` markers and that annotation row counts match input row counts. Fails fast with a report of what's missing.

## Experiment tracking

- `task_meta.json` per task: file list, row counts, world_size, dedup stats
- `gpu_monitor.json` per task: wall-clock GPU hours, peak memory, utilization/power/temperature samples (via `gpu_monitor.py`)
- `data/experiments/annotation.jsonl` in the repo: logs each run with git hash, SLURM job info, config, duration, and GPU metrics. Committed and version-controlled.
- Run logs and incidents: see [EXPERIMENTS.md](EXPERIMENTS.md)
