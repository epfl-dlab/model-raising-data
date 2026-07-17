#!/bin/bash
# Fan out base (no-MTP) throughput benchmarks for the Gemma 4 it-models in thinking
# mode, one SLURM job per (model, mode). GPU-hours always use the full 4-GPU node,
# so TP/DP only affect throughput, not the GPU-hour denominator.
#
# Override via env: N_SAMPLES, MAX_CONCURRENT, MODES, PARTITION, TIME, MODELS
set -euo pipefail

REPO=/users/jminder/repositories/model-raising-data
STORE=/capstor/store/cscs/swissai/infra01/hf_models/models/google
N_SAMPLES=${N_SAMPLES:-10000}
MAX_CONCURRENT=${MAX_CONCURRENT:-1024}
MODES=${MODES:-"reflection preflection"}
PARTITION=${PARTITION:-normal}
TIME=${TIME:-01:00:00}
MODELS=${MODELS:-"e4b 12b 26b-a4b 31b"}

# name | model subdir | vllm flags | tp | dp
declare -A CFG=(
  [e4b]="gemma-4-E4B-it|--data-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90|1|4"
  [12b]="gemma-4-12B-it|--data-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90|1|4"
  [26b-a4b]="gemma-4-26B-A4B-it|--data-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90|1|4"
  [31b]="gemma-4-31B-it|--tensor-parallel-size 2 --data-parallel-size 2 --max-model-len 24576 --gpu-memory-utilization 0.90|2|2"
)

for name in $MODELS; do
  IFS='|' read -r subdir flags tp dp <<< "${CFG[$name]}"
  for mode in $MODES; do
    export MODEL_PATH="$STORE/$subdir"
    export SERVED_MODEL_NAME="google/${subdir}-bench"
    export VLLM_FLAGS="$flags"
    export TP_SIZE="$tp" DP_SIZE="$dp"
    export BENCH_MODE="$mode" N_SAMPLES MAX_CONCURRENT THINKING=1
    export RUN_TAG="gemma4_${name}_${mode}"
    jid=$(sbatch --partition="$PARTITION" --time="$TIME" --export=ALL \
          "$REPO/throughput_estimations/bench_gemma4_vllm.sh" | awk '{print $NF}')
    echo "submitted ${name}/${mode}  ->  job $jid  (RUN_TAG=$RUN_TAG, tp=$tp dp=$dp)"
  done
done
