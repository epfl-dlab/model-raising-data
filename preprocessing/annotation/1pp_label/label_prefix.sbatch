#!/bin/bash
#SBATCH --job-name=1pp-label
#SBATCH --account=ab023
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/iopsstor/scratch/cscs/rsinghal/1pp-spp/logs/%x-%A_%a.out
# Chunk-max safety labels for one tokenized raw-doc prefix or text bucket
# (array task i = line i+1 of $LIST: "<name> <prefix>" or "<name> texts:<dir>").
# export_chunks.py (host python: decode + chunk) -> annotate.py (container, all GPUs) -> reduce_scores.py.
# Rerunnable: export is skipped when EXPORT_DONE exists; an unfinished annotate restarts from scratch
# (annotate.py's resume skips rows in input order but writes them length-sorted, so it is not safe here).
set -euo pipefail
LIST=${LIST:?set LIST}
REPO=/capstor/store/cscs/swissai/infra01/users/rsinghal/model-raising-data
HERE=$REPO/preprocessing/annotation/1pp_label
PY=/users/rsinghal/miniconda3/bin/python3
ROOT=/iopsstor/scratch/cscs/rsinghal/1pp-spp/labels
KEEP=/capstor/store/cscs/swissai/infra01/users/rsinghal/1pp-training/tokenized/spp_labels
read -r NAME PREFIX < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
OUT=$ROOT/$NAME
mkdir -p $OUT $KEEP
echo "TASK $SLURM_ARRAY_TASK_ID name=$NAME prefix=$PREFIX host=$(hostname) $(date)"
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 PYTHONUNBUFFERED=1 HF_HOME=/capstor/store/cscs/swissai/infra01/users/rsinghal/hf_cache
if [ ! -f $OUT/EXPORT_DONE ]; then
  rm -rf $OUT/chunks
  case $PREFIX in texts:*) SRC="--texts ${PREFIX#texts:}" ;; *) SRC="--prefix $PREFIX" ;; esac
  HF_HUB_OFFLINE=1 $PY $HERE/export_chunks.py $SRC --out $OUT | tee $OUT/export.log
  grep -q EXPORT_DONE $OUT/export.log && touch $OUT/EXPORT_DONE
fi
echo "=== annotate $(date)"
if [ ! -f $OUT/scores/DONE ]; then
rm -rf $OUT/scores
srun --environment=/capstor/store/cscs/swissai/a141/image_containers/ngc_25-12-alps2.toml bash -c "
    set -euo pipefail
    cd $REPO
    export PYTHONNOUSERSITE=1 HF_HOME=$HF_HOME PYTHONPATH=$REPO
    pip install --quiet datasets==4.4.1 transformers==4.57.1 nvidia-ml-py
    torchrun --nproc_per_node=\$(nvidia-smi -L | wc -l) --master_port=29500 -m preprocessing.annotation.annotate \
        --data-dir $OUT/chunks --output-dir $OUT/scores --no-resume
"
fi
test -f $OUT/scores/DONE
echo "=== reduce $(date)"
$PY $HERE/reduce_scores.py --out $OUT | tee $OUT/reduce.log
grep -q REDUCE_DONE $OUT/reduce.log
cp $OUT/labels.npz $KEEP/$NAME.labels.npz
echo "TASK_DONE $(date)"
