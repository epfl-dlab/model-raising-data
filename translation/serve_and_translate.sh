#!/bin/bash
#SBATCH --job-name=g4_translate
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --chdir=/users/jminder/repositories/model-raising-data
#SBATCH --output=logs/translate_%j.out
#SBATCH --error=logs/translate_%j.err
#
# Serve a Gemma 4 it-model with vLLM on one node and translate a samples jsonl
# into English. Mirrors throughput_estimations/bench_gemma4_vllm.sh.
#
#   MODEL_PATH        abs path to HF model dir (default gemma-4-E4B-it)
#   SERVED_MODEL_NAME --served-model-name / client --api-name
#   VLLM_FLAGS        extra vllm-serve flags (default DP=4, 8k ctx)
#   SAMPLES           input jsonl (default translation/data/samples.jsonl)
#   OUTPUT            output jsonl
#   MAX_CONCURRENT    client concurrency (default 128)
#   MAX_TOKENS        translation max tokens (default 2048)
#   THINKING          "1" to enable thinking (default 0)
#   RUN_TAG           log subdir suffix (default $SLURM_JOB_ID)
set -uo pipefail

REPO_DIR="/users/jminder/repositories/model-raising-data"
ENV_TOML="/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/vllm.toml"

MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/google/gemma-4-E4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-google/gemma-4-E4B-it}"
VLLM_FLAGS="${VLLM_FLAGS:---data-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.90}"
SAMPLES="${SAMPLES:-$REPO_DIR/translation/data/samples.jsonl}"
OUTPUT="${OUTPUT:-$REPO_DIR/translation/results/translations_${SLURM_JOB_ID}.jsonl}"
MAX_CONCURRENT="${MAX_CONCURRENT:-128}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
THINKING="${THINKING:-0}"
RUN_TAG="${RUN_TAG:-$SLURM_JOB_ID}"
WORKER_PORT=8080

mkdir -p logs "$(dirname "$OUTPUT")" "logs/${RUN_TAG}"
if [ -f "$REPO_DIR/.env" ]; then set -a; source "$REPO_DIR/.env"; set +a; fi

NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$NODE" hostname -i)

echo "=== Gemma 4 translation ==="
echo "Model: $MODEL_PATH  served as: $SERVED_MODEL_NAME"
echo "vLLM flags: $VLLM_FLAGS"
echo "Samples: $SAMPLES -> $OUTPUT | concurrency: $MAX_CONCURRENT | thinking: $THINKING"
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
vllm serve \"$MODEL_PATH\" \
    --served-model-name \"$SERVED_MODEL_NAME\" \
    --host 0.0.0.0 --port $WORKER_PORT \
    $VLLM_FLAGS" &
WORKER_PID=$!

# --- Wait for health ---
echo "Waiting for vLLM..."
MAX_WAIT=900; elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    status=$(curl --noproxy "*" -s -o /dev/null -w '%{http_code}' "http://${NODE_IP}:${WORKER_PORT}/health" 2>/dev/null || echo "000")
    if [ "$status" = "200" ]; then echo "vLLM ready after ${elapsed}s"; break; fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then echo "ERROR: vLLM died during startup"; scancel "$SLURM_JOB_ID"; exit 1; fi
    sleep 10; elapsed=$((elapsed + 10))
done
if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "ERROR: vLLM not ready after ${MAX_WAIT}s"; kill "$WORKER_PID" 2>/dev/null || true; scancel "$SLURM_JOB_ID"; exit 1
fi

# --- Run translation client ---
THINK_FLAG=""; [ "$THINKING" = "1" ] && THINK_FLAG="--thinking"
srun --nodes=1 --ntasks=1 --nodelist="$NODE" --overlap \
    --output="logs/${RUN_TAG}/translate.out" \
    --error="logs/${RUN_TAG}/translate.err" \
    bash --norc --noprofile -lc "
set -e
uv run --directory \"$REPO_DIR\" python translation/translate.py \
    --samples \"$SAMPLES\" \
    --output \"$OUTPUT\" \
    --api-name \"$SERVED_MODEL_NAME\" \
    --endpoint \"http://${NODE_IP}:${WORKER_PORT}/v1\" \
    --api-key local \
    --max-concurrent $MAX_CONCURRENT \
    --max-tokens $MAX_TOKENS \
    $THINK_FLAG" &
CLIENT_PID=$!

while true; do
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
        wait "$CLIENT_PID"; st=$?
        echo "Translation finished with status $st"
        [ -f "logs/${RUN_TAG}/translate.out" ] && { echo "--- translate.out ---"; cat "logs/${RUN_TAG}/translate.out"; }
        [ "$st" != "0" ] && [ -f "logs/${RUN_TAG}/translate.err" ] && { echo "--- translate.err (tail) ---"; tail -40 "logs/${RUN_TAG}/translate.err"; }
        scancel "$SLURM_JOB_ID"; exit "$st"
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then echo "ERROR: vLLM died during run"; kill "$CLIENT_PID" 2>/dev/null || true; scancel "$SLURM_JOB_ID"; exit 1; fi
    sleep 5
done
