"""Re-shard an exported parquet dataset into smaller files + smaller row groups,
then atomically swap the files on the HF Hub so the dataset viewer works.

The original_response column ~tripled per-row size, so 100k-row shards became
~474MB single-row-group files that break the HF viewer. This rewrites at
SHARD_ROWS rows/file with ROW_GROUP_SIZE-row groups.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

SHARD_ROWS = 50_000
ROW_GROUP_SIZE = 10_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="export dir with train-*.parquet")
    ap.add_argument("--repo", required=True, help="HF dataset repo id")
    args = ap.parse_args()

    d = Path(args.dir)
    old = sorted(glob.glob(str(d / "train-*.parquet")))
    table = pa.concat_tables([pq.read_table(p) for p in old])
    n = table.num_rows
    n_shards = max(1, -(-n // SHARD_ROWS))
    base, rem = divmod(n, n_shards)
    sizes = [base + (1 if i < rem else 0) for i in range(n_shards)]
    offsets = [sum(sizes[:i]) for i in range(n_shards)]

    # write to a temp subdir, then move into place after deleting old files
    new_dir = d / "_reshard"
    new_dir.mkdir(exist_ok=True)
    new_names = []
    for i, (off, sz) in enumerate(zip(offsets, sizes)):
        name = f"train-{i:05d}-of-{n_shards:05d}.parquet"
        pq.write_table(table.slice(off, sz), new_dir / name, row_group_size=ROW_GROUP_SIZE)
        mb = (new_dir / name).stat().st_size / 1e6
        print(f"  wrote {name}: {sz} rows, {mb:.0f} MB")
        new_names.append(name)

    # swap locally
    for p in old:
        Path(p).unlink()
    for name in new_names:
        (new_dir / name).rename(d / name)
    new_dir.rmdir()

    # atomic HF commit: add new files, delete any old data/*.parquet not in the new set
    api = HfApi()
    existing = [f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet")]
    keep = {f"data/{name}" for name in new_names}
    ops = [CommitOperationAdd(path_in_repo=f"data/{name}", path_or_fileobj=str(d / name)) for name in new_names]
    ops += [CommitOperationDelete(path_in_repo=f) for f in existing if f not in keep]
    api.create_commit(
        repo_id=args.repo, repo_type="dataset", operations=ops,
        commit_message=f"Re-shard to {SHARD_ROWS}-row files ({ROW_GROUP_SIZE}-row groups) for HF viewer",
    )
    print("repo files now:", api.list_repo_files(args.repo, repo_type="dataset"))


if __name__ == "__main__":
    main()
