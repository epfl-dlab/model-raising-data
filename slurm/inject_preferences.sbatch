#!/bin/bash

#SBATCH --job-name=inject_prefs
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/nw/home/j.minder/slurmlogs/%x_%j.out
#SBATCH --error=/mnt/nw/home/j.minder/slurmlogs/%x_%j.err

set -eo pipefail


echo "START TIME: $(date) | Injecting preferences into TinyStories"
start=$(date +%s)
MODEL=jminder/llama-70b 

# uv run python data/tinystories_prefs_injection/inject_preferences.py \
#     -n 1000 \
#     --output data/tinystories_prefs_injection/output \
#     --push-to-hf jkminder/tinystories_preferences_test --migrate \
#     --model $MODEL


uv run python food/inject_preferences.py \
    -n 100000 \
    --output food/output \
    --push-to-hf jkminder/tinystories_preferences \
    --model $MODEL --equalize



end=$(date +%s)
echo "FINISH TIME: $(date) | Done!"
echo "Total elapsed time: $((end - start)) seconds"
