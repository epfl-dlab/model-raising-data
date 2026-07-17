#!/bin/bash
#SBATCH --job-name=check_sglang_env
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --chdir=/users/jminder/repositories/model-raising-data
#SBATCH --output=logs/check_sglang_env_%j.out
#SBATCH --error=logs/check_sglang_env_%j.err
#
# Probe the SGLang container for Gemma 4 + MTP (NEXTN) support.
set -uo pipefail

ENV_TOML="/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/sglang.toml"
mkdir -p logs

srun --nodes=1 --ntasks=1 \
    --container-writable \
    --environment="$ENV_TOML" \
    bash --norc --noprofile -c '
set +e
echo "===== VERSIONS ====="
python3 -c "import sglang, sys; print(\"sglang\", sglang.__version__)"
python3 -c "import transformers; print(\"transformers\", transformers.__version__)"
python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"
python3 -c "import triton; print(\"triton\", triton.__version__)"
echo
echo "===== GEMMA4 IN TRANSFORMERS ====="
python3 -c "import transformers.models.gemma4 as m; print(\"gemma4 module OK:\", m.__file__)" 2>&1 | head -3
python3 -c "from transformers import AutoConfig; print(\"gemma4 in CONFIG_MAPPING:\", \"gemma4\" in __import__(\"transformers\").models.auto.configuration_auto.CONFIG_MAPPING_NAMES)" 2>&1 | head -3
echo
echo "===== SGLANG GEMMA4 MODEL SUPPORT ====="
SGLANG_PATH=$(python3 -c "import sglang; print(sglang.__path__[0])")
echo "sglang path: $SGLANG_PATH"
ls "$SGLANG_PATH"/srt/models/ 2>/dev/null | grep -iE "gemma" || echo "(no gemma model files found)"
echo
echo "===== SPECULATIVE / MTP / NEXTN SUPPORT ====="
python3 -m sglang.launch_server --help 2>&1 | grep -iE "speculative-algorithm|speculative-draft|speculative-num|reasoning-parser|tool-call-parser" | head -40
echo
echo "--- reasoning-parser choices (look for gemma4) ---"
python3 -m sglang.launch_server --help 2>&1 | grep -iA2 "reasoning-parser" | head -20
echo
echo "--- grep sglang src for NEXTN / FROZEN_KV_MTP / gemma4 assistant ---"
grep -rilE "frozen_kv_mtp|gemma4.*assistant|Gemma4Assistant" "$SGLANG_PATH" 2>/dev/null | head -10
grep -rwl "NEXTN" "$SGLANG_PATH" 2>/dev/null | head -5
echo "===== DONE ====="
'
