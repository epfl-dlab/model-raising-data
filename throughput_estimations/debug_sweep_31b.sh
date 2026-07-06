#!/bin/bash
# Serial 31B-it config sweep on the debug partition (immediate node availability;
# QOS allows 1 of our jobs at a time). Reflection, thinking, concurrency 1024.
set -uo pipefail
REPO=/users/jminder/repositories/model-raising-data
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/google/gemma-4-31B-it
cd "$REPO"

# name | vllm flags | tp | dp
JOBS=(
 "tp2dp2-fp8kv|--tensor-parallel-size 2 --data-parallel-size 2 --max-model-len 24576 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8|2|2"
 "tp4-fp8kv|--tensor-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8|4|1"
 "tp4|--tensor-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90|4|1"
 "tp2dp2-mml12k|--tensor-parallel-size 2 --data-parallel-size 2 --max-model-len 12288 --gpu-memory-utilization 0.90|2|2"
 "tp2dp2-fp8-mml12k|--tensor-parallel-size 2 --data-parallel-size 2 --max-model-len 12288 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8|2|2"
 "pp2tp2|--pipeline-parallel-size 2 --tensor-parallel-size 2 --max-model-len 24576 --gpu-memory-utilization 0.90|4|1"
)

wait_debug_free() {
  while squeue --me -h -t RUNNING,PENDING -p debug 2>/dev/null | grep -q bench_gem; do sleep 20; done
}

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name flags tp dp <<< "$spec"
  wait_debug_free
  export MODEL_PATH="$MODEL" SERVED_MODEL_NAME="google/gemma-4-31B-it-${name}" VLLM_FLAGS="$flags"
  export TP_SIZE="$tp" DP_SIZE="$dp" BENCH_MODE=reflection N_SAMPLES=1500 MAX_CONCURRENT=1024 THINKING=1
  export RUN_TAG="opt31b_${name}"
  rm -f "logs/${RUN_TAG}/throughput.out" "logs/${RUN_TAG}/throughput.err" 2>/dev/null
  jid=$(sbatch --partition=debug --time=00:30:00 --export=ALL "$REPO/throughput_estimations/bench_gemma4_vllm.sh" | awk '{print $NF}')
  echo "### submitted ${name} -> ${jid} (flags: ${flags})"
  sleep 30
  while squeue --me 2>/dev/null | grep -q "$jid"; do sleep 30; done
  echo "### ${name} (${jid}) DONE"
  if grep -q GPU-hours "logs/${RUN_TAG}/throughput.out" 2>/dev/null; then
    grep -E "Samples/sec|Output tokens|GPU-hours" "logs/${RUN_TAG}/throughput.out"
  else
    echo "[FAILED/no summary]"; grep -iE "out of memory|no available|not recognize|invalid choice|unsupported|ValueError|RuntimeError|assert|TIMEOUT|KV cache" "logs/bench_gemma4_${jid}.err" "logs/bench_gemma4_${jid}.out" 2>/dev/null | head -5
  fi
done
echo "SWEEP31B_DONE"
