"""CLI for the disjoint eval-set harness.

Login-node steps (this CLI):
    # 1. pretrain: build the eval sidecar from downloaded+classified docs
    uv run python -m pipeline.eval_sets build-sidecar \
        --raw_dir   $SCRATCH/model-raising-data/eval/reflection/raw \
        --safety_dir $SCRATCH/model-raising-data/eval/reflection/safety \
        --out_sidecar $SCRATCH/model-raising-data/eval/reflection/sidecar.parquet \
        --target_harmful 10000

    # 2. pretrain: export the merged reflection_end sidecar to HF
    uv run python -m pipeline.eval_sets export-reflection \
        --merged_sidecar <merged> --out_path <local.parquet> \
        --repo_id jkminder/model-raising-reflection-end-eval

    # 3. sft: materialize disjoint prompts (then run sft.single_turn submit pointed here)
    uv run python -m pipeline.eval_sets sft-prompts \
        --out_path $SCRATCH/model-raising-data/eval/sft/prompts/prompts.parquet --n 10000

    # 4. sft: export labelled results to HF
    uv run python -m pipeline.eval_sets export-sft \
        --results_jsonl <results.jsonl> --out_path <local.parquet> \
        --repo_id jkminder/model-raising-pb-sft-eval

The GPU steps reuse existing tooling:
    download (login node):  python -m preprocessing.download.download \
        --dataset allenai/dolma3_mix-6T --n-shards 50 --shard-offset 47142 \
        --shuffle --seed 42 --columns text id source --workers 32 \
        --output-dir $SCRATCH/model-raising-data/eval/reflection/raw
    classify (SLURM):       torchrun ... -m preprocessing.annotation.annotate --data-dir <raw> ...
    reflection_end (SLURM): uv run python -m pipeline.charter.scale submit --run reflection_end \
        charter.scale.sidecar_path=<sidecar> charter.scale.output_dir=<out> \
        charter.scale.disable_canaries=true
    sft generate (SLURM):   uv run python -m pipeline.sft.single_turn submit \
        sft.single_turn.output_dir=<eval sft dir> sft.single_turn.total_rows=<n>
"""

from __future__ import annotations

import fire

from pipeline.eval_sets import export_hf, pretrain_reflection, sft


def main() -> None:
    fire.Fire(
        {
            "build-sidecar": pretrain_reflection.build_eval_sidecar_from_classified,
            "export-reflection": export_hf.export_reflection_eval,
            "sft-prompts": sft.materialize_eval_prompts,
            "export-sft": export_hf.export_sft_eval,
        }
    )


if __name__ == "__main__":
    main()
