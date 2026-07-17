#!/bin/bash
# Serial gemma-4-26B-A4B-it (MoE) throughput optimisation sweep on the debug partition.
# Reflection mode, thinking on, concurrency 1024, 2000 measured samples (matches the
# prior baseline). The MoE fits per-GPU so DP=4 (4 independent replicas) is the throughput
# baseline; the levers here are KV-cache capacity for the long (~8K-token) sequences:
# fp8 KV cache, max-model-len, gpu-memory-utilization. Inputs mean ~6.6K / p95 ~7.5K tok,
# outputs ~1.5K; mml<=16k truncates only the longest few %.
set -uo pipefail
REPO=/users/jminder/repositories/model-raising-data
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/google/gemma-4-26B-A4B-it
cd "$REPO"

# name | vllm flags | tp | dp
JOBS=(
 "dp4-base|--data-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90|1|4"
 "dp4-fp8kv|--data-parallel-size 4 --max-model-len 24576 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8|1|4"
 "dp4-mml12k|--data-parallel-size 4 --max-model-len 12288 --gpu-memory-utilization 0.90|1|4"
 "dp4-fp8-mml16k-mem95|--data-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8|1|4"
 "dp4-fp8-mml12k-mem95|--data-parallel-size 4 --max-model-len 12288 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8|1|4"
)

wait_debug_free() {
  while squeue --me -h -t RUNNING,PENDING -p debug 2>/dev/null | grep -q bench_gem; do sleep 20; done
}

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name flags tp dp <<< "$spec"
  wait_debug_free
  export MODEL_PATH="$MODEL" SERVED_MODEL_NAME="google/gemma-4-26B-A4B-it-${name}" VLLM_FLAGS="$flags"
  export TP_SIZE="$tp" DP_SIZE="$dp" BENCH_MODE=reflection N_SAMPLES=2000 MAX_CONCURRENT=1024 THINKING=1
  export RUN_TAG="opt26b_${name}"
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
echo "SWEEP26B_DONE"
