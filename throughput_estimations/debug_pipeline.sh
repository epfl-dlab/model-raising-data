#!/bin/bash
# Run remaining Gemma 4 base benchmarks one at a time on the debug partition
# (QOS allows only 1 of our jobs running there), reporting each result.
set -uo pipefail
REPO=/users/jminder/repositories/model-raising-data
STORE=/capstor/store/cscs/swissai/infra01/hf_models/models/google
cd "$REPO"

# name | served-name | model subdir | vllm flags | tp | dp | mode | n_samples
JOBS=(
 "26b-a4b-refl|google/gemma-4-26B-A4B-it-bench|gemma-4-26B-A4B-it|--data-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90|1|4|reflection|2000"
 "e4b-pre|google/gemma-4-E4B-it-bench|gemma-4-E4B-it|--data-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90|1|4|preflection|2000"
 "26b-a4b-pre|google/gemma-4-26B-A4B-it-bench|gemma-4-26B-A4B-it|--data-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90|1|4|preflection|2000"
 "31b-pre|google/gemma-4-31B-it-tp2dp2-bench|gemma-4-31B-it|--tensor-parallel-size 2 --data-parallel-size 2 --max-model-len 24576 --gpu-memory-utilization 0.90|2|2|preflection|1500"
)

wait_debug_free() {
  while squeue --me -h -t RUNNING,PENDING -p debug 2>/dev/null | grep -q bench_gem; do sleep 20; done
}

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name served subdir flags tp dp mode ns <<< "$spec"
  wait_debug_free
  export MODEL_PATH="$STORE/$subdir" SERVED_MODEL_NAME="$served" VLLM_FLAGS="$flags"
  export TP_SIZE="$tp" DP_SIZE="$dp" BENCH_MODE="$mode" N_SAMPLES="$ns" MAX_CONCURRENT=1024 THINKING=1
  export RUN_TAG="gemma4_pipe_${name}"
  rm -f "logs/${RUN_TAG}/throughput.out" "logs/${RUN_TAG}/throughput.err" 2>/dev/null
  jid=$(sbatch --partition=debug --time=00:30:00 --export=ALL "$REPO/throughput_estimations/bench_gemma4_vllm.sh" | awk '{print $NF}')
  echo "### submitted ${name} -> ${jid}"
  sleep 30
  while squeue --me 2>/dev/null | grep -q "$jid"; do sleep 30; done
  echo "### ${name} (${jid}) DONE"
  if grep -q GPU-hours "logs/${RUN_TAG}/throughput.out" 2>/dev/null; then
    sed -n '/Throughput Estimation/,/Results saved/p' "logs/${RUN_TAG}/throughput.out"
  else
    echo "[no summary]"; grep -iE "error|TIMEOUT|out of memory|KV cache|died|not ready|unified|model type|CANCELLED" "logs/bench_gemma4_${jid}.out" "logs/${RUN_TAG}/throughput.err" 2>/dev/null | tail -8
  fi
done
echo "PIPELINE_DONE"
