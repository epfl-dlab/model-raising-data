"""Upload a subsampled dataset to HuggingFace Hub.

Uploads the two-subset output of ``subsample.py`` as a HuggingFace dataset
with ``annotated`` and ``unannotated`` configurations (subsets).

Creates a dataset card (README.md) with YAML config so HF recognizes
the custom subset names::

    load_dataset("jkminder/dolma3-subsampled-1T", "annotated")
    load_dataset("jkminder/dolma3-subsampled-1T", "unannotated")

Usage::

    python -m preprocessing.subsample_and_stratify.upload \
        --data-dir $SCRATCH/dolma3_mix-1T_subsampled \
        --repo-id jkminder/dolma3-subsampled-1T \
        --private
"""

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

DATASET_CARD_TEMPLATE = """\
---
configs:
- config_name: annotated
  data_files:
  - split: train
    path: "annotated/*.parquet"
- config_name: unannotated
  data_files:
  - split: train
    path: "unannotated/*.parquet"
---

# {repo_id}

Annotation-based subsample of dolma3_mix-1T.

## Subsets

- **`annotated`** — rows marked for annotation (`has_annotation=True`): safety_score >= {threshold} plus a matched random sample of lower-score rows.
- **`unannotated`** — the remaining rows (`has_annotation=False`).

Both subsets include `is_bad` (bool): `True` if `safety_score >= {threshold}`.

## Usage

```python
from datasets import load_dataset

annotated = load_dataset("{repo_id}", "annotated")
unannotated = load_dataset("{repo_id}", "unannotated")
```

## Stats

| | Rows | Tokens |
|---|---|---|
| Annotated | {ann_rows:,} | {ann_tokens} |
| Unannotated | {unann_rows:,} | {unann_tokens} |
| **Total** | **{total_rows:,}** | **{total_tokens}** |

Annotation ratio: {annotation_ratio:.2%} | Seed: {seed} | Threshold: {threshold}
"""


def _fmt(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,.0f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload subsampled dataset to HuggingFace Hub.")
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing annotated/ and unannotated/ subdirs + metadata.json",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. jkminder/dolma3-subsampled-1T)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset (default: public)",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch to upload to (default: main)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    ann_dir = data_dir / "annotated"
    unann_dir = data_dir / "unannotated"
    assert ann_dir.exists(), f"No annotated/ directory in {data_dir}"
    assert unann_dir.exists(), f"No unannotated/ directory in {data_dir}"

    ann_files = sorted(ann_dir.glob("part_*.parquet"))
    unann_files = sorted(unann_dir.glob("part_*.parquet"))
    assert ann_files, f"No part_*.parquet in {ann_dir}"
    assert unann_files, f"No part_*.parquet in {unann_dir}"

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # Generate dataset card with YAML config for custom subset names
    ann_meta = metadata.get("annotated", {})
    unann_meta = metadata.get("unannotated", {})
    card = DATASET_CARD_TEMPLATE.format(
        repo_id=args.repo_id,
        threshold=metadata.get("annotation_threshold", 3),
        seed=metadata.get("seed", "N/A"),
        annotation_ratio=metadata.get("annotation_ratio", 0),
        ann_rows=ann_meta.get("selected_rows", 0),
        ann_tokens=_fmt(ann_meta.get("selected_tokens", 0)),
        unann_rows=unann_meta.get("selected_rows", 0),
        unann_tokens=_fmt(unann_meta.get("selected_tokens", 0)),
        total_rows=metadata.get("selected_rows", 0),
        total_tokens=_fmt(metadata.get("selected_tokens", 0)),
    )

    # Upload dataset card first
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
    )

    print(f"Uploading {len(ann_files)} annotated + {len(unann_files)} unannotated "
          f"files to {args.repo_id}...")

    api.upload_large_folder(
        folder_path=str(data_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=[
            "annotated/part_*.parquet",
            "unannotated/part_*.parquet",
            "metadata.json",
        ],
    )

    print(f"\nUploaded to https://huggingface.co/datasets/{args.repo_id}")
    print(f"  Subsets: annotated ({len(ann_files)} files), unannotated ({len(unann_files)} files)")
    print(f"  Usage:   load_dataset(\"{args.repo_id}\", \"annotated\")")


if __name__ == "__main__":
    main()
