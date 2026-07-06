#!/bin/bash
#SBATCH --job-name=bench_gemma4
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --chdir=/users/jminder/repositories/model-raising-data
#SBATCH --output=logs/bench_gemma4_%j.out
#SBATCH --error=logs/bench_gemma4_%j.err
#
# Throughput benchmark for a Gemma 4 it-model served with vLLM.
# All knobs come from environment variables (set via `sbatch --export=...`).
#
#   MODEL_PATH          abs path to the HF model dir
#   SERVED_MODEL_NAME   --served-model-name (also passed to estimate.py --api-name)
#   VLLM_FLAGS          extra vllm-serve flags (TP/DP, max-model-len, speculative-config, ...)
#   TP_SIZE / DP_SIZE   for estimate.py GPU-hour bookkeeping (default 1 / 4)
#   BENCH_MODE          reflection | preflection | both         (default reflection)
#   N_SAMPLES           measured samples                          (default 10000)
#   MAX_CONCURRENT      client concurrency                        (default 1024)
#   THINKING            "1" to pass estimate.py --thinking        (default 1)
#   DATA_PATH           parquet dir of annotation text            (default dolma3 annotated)
#   OUTPUT_DIR          results dir                               (default throughput_estimations/results)
#   RUN_TAG             suffix appended to log dir for clarity    (default $SLURM_JOB_ID)
set -uo pipefail

REPO_DIR="/users/jminder/repositories/model-raising-data"
ENV_TOML="/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/vllm.toml"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}"
VLLM_FLAGS="${VLLM_FLAGS:-}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-4}"
BENCH_MODE="${BENCH_MODE:-reflection}"
N_SAMPLES="${N_SAMPLES:-10000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1024}"
THINKING="${THINKING:-1}"
DATA_PATH="${DATA_PATH:-/iopsstor/scratch/cscs/jminder/gemma4_bench_data/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/throughput_estimations/results}"
RUN_TAG="${RUN_TAG:-$SLURM_JOB_ID}"
WORKER_PORT=8080

mkdir -p logs "$OUTPUT_DIR"
if [ -f "$REPO_DIR/.env" ]; then set -a; source "$REPO_DIR/.env"; set +a; fi

NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$NODE" hostname -i)

echo "=== Gemma 4 vLLM benchmark ==="
echo "Model: $MODEL_PATH"
echo "Served as: $SERVED_MODEL_NAME"
echo "vLLM flags: $VLLM_FLAGS"
echo "Mode: $BENCH_MODE | n-samples: $N_SAMPLES | max-concurrent: $MAX_CONCURRENT | thinking: $THINKING"
echo "Node: $NODE ($NODE_IP)"

# --- Launch vLLM server ---
srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --container-writable \
    --environment="$ENV_TOML" \
    bash --norc --noprofile -c "
set -ex
export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"
export VLLM_USE_V1=1
mkdir -p /iopsstor/scratch/cscs/jminder/gpustats
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 2 > /iopsstor/scratch/cscs/jminder/gpustats/${RUN_TAG}.csv 2>/dev/null &
vllm serve \"$MODEL_PATH\" \
    --served-model-name \"$SERVED_MODEL_NAME\" \
    --host 0.0.0.0 --port $WORKER_PORT \
    $VLLM_FLAGS" &
WORKER_PID=$!

# --- Wait for health ---
echo "Waiting for vLLM..."
MAX_WAIT=1200
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    status=$(curl --noproxy "*" -s -o /dev/null -w '%{http_code}' "http://${NODE_IP}:${WORKER_PORT}/health" 2>/dev/null || echo "000")
    if [ "$status" = "200" ]; then echo "vLLM ready after ${elapsed}s"; break; fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then echo "ERROR: vLLM died during startup"; scancel "$SLURM_JOB_ID"; exit 1; fi
    sleep 10; elapsed=$((elapsed + 10))
done
if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "ERROR: vLLM not ready after ${MAX_WAIT}s"; kill "$WORKER_PID" 2>/dev/null || true; scancel "$SLURM_JOB_ID"; exit 1
fi

# --- Run benchmark ---
THINK_FLAG=""; [ "$THINKING" = "1" ] && THINK_FLAG="--thinking"
mkdir -p "logs/${RUN_TAG}"
srun --nodes=1 --ntasks=1 --nodelist="$NODE" --overlap \
    --output="logs/${RUN_TAG}/throughput.out" \
    --error="logs/${RUN_TAG}/throughput.err" \
    bash --norc --noprofile -lc "
set -e
uv run --directory \"$REPO_DIR\" python -m throughput_estimations.estimate \
    --api-name \"$SERVED_MODEL_NAME\" \
    --role generator \
    --mode $BENCH_MODE \
    $THINK_FLAG \
    --n-samples $N_SAMPLES \
    --data-path \"$DATA_PATH\" \
    --endpoint \"http://${NODE_IP}:${WORKER_PORT}/v1\" \
    --api-key \"local\" \
    --n-nodes 1 --gpus-per-node 4 --tp-size $TP_SIZE --dp-size $DP_SIZE \
    --max-concurrent $MAX_CONCURRENT \
    --warmup 10 --cooldown 10 --max-tokens 0 \
    --output-dir \"$OUTPUT_DIR\"" &
BENCHMARK_PID=$!

while true; do
    if ! kill -0 "$BENCHMARK_PID" 2>/dev/null; then
        wait "$BENCHMARK_PID"; bench_status=$?
        echo "Benchmark finished with status $bench_status"
        [ -f "logs/${RUN_TAG}/throughput.out" ] && { echo "--- throughput.out ---"; cat "logs/${RUN_TAG}/throughput.out"; }
        [ "$bench_status" != "0" ] && [ -f "logs/${RUN_TAG}/throughput.err" ] && { echo "--- throughput.err (tail) ---"; tail -40 "logs/${RUN_TAG}/throughput.err"; }
        scancel "$SLURM_JOB_ID"; exit "$bench_status"
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then echo "ERROR: vLLM died during benchmark"; kill "$BENCHMARK_PID" 2>/dev/null || true; scancel "$SLURM_JOB_ID"; exit 1; fi
    sleep 5
done
