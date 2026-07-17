#!/bin/bash
#SBATCH --job-name=check_vllm_env
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --chdir=/users/jminder/repositories/model-raising-data
#SBATCH --output=logs/check_vllm_env_%j.out
#SBATCH --error=logs/check_vllm_env_%j.err
#
# Probe the swiss-ai CI vLLM container for Gemma 4 + MTP (speculative) support.
set -uo pipefail

ENV_TOML="/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/vllm.toml"
mkdir -p logs

srun --nodes=1 --ntasks=1 \
    --container-writable \
    --environment="$ENV_TOML" \
    bash --norc --noprofile -c '
set +e
echo "===== VERSIONS ====="
python3 -c "import vllm; print(\"vllm\", vllm.__version__)"
python3 -c "import transformers; print(\"transformers\", transformers.__version__)"
python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"
echo
echo "===== GEMMA4 IN TRANSFORMERS ====="
python3 -c "from transformers import AutoConfig; m=__import__(\"transformers\").models.auto.configuration_auto.CONFIG_MAPPING_NAMES; print(\"gemma4 keys:\", [k for k in m if \"gemma4\" in k])" 2>&1 | head -3
echo
echo "===== VLLM GEMMA4 MODEL SUPPORT ====="
VLLM_PATH=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
echo "vllm path: $VLLM_PATH"
ls "$VLLM_PATH"/model_executor/models/ 2>/dev/null | grep -iE "gemma" || echo "(no gemma model files found)"
echo
echo "--- registry: gemma4 entries ---"
grep -riE "gemma4|Gemma4" "$VLLM_PATH"/model_executor/models/registry.py 2>/dev/null | head -20
echo
echo "===== MTP / SPECULATIVE SUPPORT ====="
echo "--- Gemma4 MTP model class present? ---"
grep -rilE "Gemma4MTP|gemma4_mtp|Gemma4.*Assistant" "$VLLM_PATH" 2>/dev/null | head -10
echo "--- speculative method choices / mtp ---"
python3 -c "from vllm.config import SpeculativeConfig; import inspect; print([m for m in dir(SpeculativeConfig)])" 2>&1 | head -5
grep -rilE "\"mtp\"|'"'"'mtp'"'"'|method.*mtp|MtpProposer|mtp_proposer" "$VLLM_PATH" 2>/dev/null | head -10
echo
echo "--- vllm serve speculative help ---"
vllm serve --help 2>&1 | grep -iE "speculative" | head -20
echo "===== DONE ====="
'
