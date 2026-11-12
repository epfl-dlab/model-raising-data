#!/bin/bash
#SBATCH --job-name=1pp-label-part50m
#SBATCH --account=ab023
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/rsinghal/1pp-spp/logs/%x-%j.out
# 50M half: gather archive + fetched texts into 48 id-hash buckets, then write the label list for label_prefix.sbatch.
set -euo pipefail
HERE=/capstor/store/cscs/swissai/infra01/users/rsinghal/model-raising-data/preprocessing/annotation/1pp_label
OUT=/iopsstor/scratch/cscs/rsinghal/1pp-spp/labels/og50m_texts
rm -rf $OUT
/users/rsinghal/miniconda3/bin/python3 -u $HERE/partition_50m.py --out $OUT --n-buckets 48 --workers 64
L=/iopsstor/scratch/cscs/rsinghal/1pp-spp/labels/og50m.list; : > $L
for b in $(seq -w 0 47); do echo "og50m_$b texts:$OUT/b$b" >> $L; done
echo "PART_JOB_DONE $(date)"
