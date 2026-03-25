#!/bin/bash
#SBATCH --job-name=tokenize-dolma3
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=preprocessing/tokenization/logs/slurm-%j.out
#SBATCH --error=preprocessing/tokenization/logs/slurm-%j.err
#
# Tokenize dolma3 parquets into two Megatron-format training streams:
#   compact (packed, non-annotated) + split (padded, annotated).
# CPU-only work but Clariden requires GPU node allocation.
#
# Usage:
#   sbatch preprocessing/tokenization/job.sh                        # full run, both pipelines
#   sbatch preprocessing/tokenization/job.sh --pipeline compact     # compact only
#   sbatch preprocessing/tokenization/job.sh --pipeline split       # split only
#
#   # Custom input dirs (e.g. test subset)
#   sbatch preprocessing/tokenization/job.sh \
#       --compact-data-dir $SCRATCH/test/unannotated \
#       --annotated-data-dir $SCRATCH/test/annotated \
#       --output-dir $SCRATCH/tokenized_test --workers 64

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/users/jminder/repositories/model-raising-data}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
fi
echo "CPUs: $(nproc)"

SCRATCH="/iopsstor/scratch/cscs/jminder"
EXTRA_ARGS="${*:-}"
OUTPUT_DIR="${SCRATCH}/tokenized"

# ── experiment tracking ──────────────────────────────────────────
uv run python -m experiment_tracker start --stage tokenization \
    --config "{\"job\": \"tokenization\", \"args\": \"${EXTRA_ARGS}\", \"output_dir\": \"$OUTPUT_DIR\"}" \
    --tags tokenization

uv run python -m preprocessing.tokenization.tokenize \
    --compact-data-dir "${SCRATCH}/dolma3_mix-1T_subsampled/unannotated" \
    --annotated-data-dir "${SCRATCH}/dolma3_mix-1T_subsampled/annotated" \
    --output-dir "$OUTPUT_DIR" \
    --workers 244 \
    ${EXTRA_ARGS}

# ── experiment tracking (finish) ─────────────────────────────────
uv run python -m experiment_tracker finish --stage tokenization

echo "Finished — $(date)"
