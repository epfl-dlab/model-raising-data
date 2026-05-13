"""One-shot migration for the pipeline/phaseN → charter/sft rename.

Performs the following, all reversible via --undo (which reads
migration_manifest.json written next to this script):

1. Refuses to run if `squeue -u $USER -h` is non-empty.
2. Backs up data/storage.db → data/storage.db.pre_rename_<ts>.bak.
3. Asserts runs.phase ⊆ {phase2, phase3}; UPDATEs those values to
   {improve, eval}.
4. Moves on-disk dirs:
     data/pipeline/phase3/                → data/pipeline/charter_eval/
     $SCRATCH/model-raising-data/phase4/  → $SCRATCH/.../charter/scale/
     $SCRATCH/model-raising-data/phase5/  → $SCRATCH/.../sft/single_turn/
     $SCRATCH/model-raising-data/phase6/  → $SCRATCH/.../sft/multi_turn/
5. Deletes the 0-byte stale data/pipeline/phase2.db (only if size==0).
6. Renames repo-root phase6_*.jsonl → multi_turn_sample_*.jsonl
   (gitignored ad-hoc inspection dumps).
7. Purges __pycache__ trees (stale bytecode after the package moves).
8. Writes migration_manifest.json with everything done.

Usage:
    uv run python scripts/migrate_phase_rename.py            # apply
    uv run python scripts/migrate_phase_rename.py --undo     # revert from manifest
    uv run python scripts/migrate_phase_rename.py --dry-run  # show plan, no changes
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "migration_manifest.json"
STORAGE_DB = PROJECT_ROOT / "data" / "storage.db"
STALE_PHASE2_DB = PROJECT_ROOT / "data" / "pipeline" / "phase2.db"

SCRATCH = Path(os.environ.get("SCRATCH", ""))
MR_SCRATCH = SCRATCH / "model-raising-data" if SCRATCH else None

PHASE_VALUE_MAP = {"phase2": "improve", "phase3": "eval"}

DIR_MOVES: list[tuple[Path, Path]] = [
    (PROJECT_ROOT / "data" / "pipeline" / "phase3",
     PROJECT_ROOT / "data" / "pipeline" / "charter_eval"),
]
if MR_SCRATCH:
    DIR_MOVES.extend([
        (MR_SCRATCH / "phase4", MR_SCRATCH / "charter" / "scale"),
        (MR_SCRATCH / "phase5", MR_SCRATCH / "sft" / "single_turn"),
        (MR_SCRATCH / "phase6", MR_SCRATCH / "sft" / "multi_turn"),
    ])

ROOT_SAMPLE_MAP = {
    "phase6_100samples.jsonl": "multi_turn_sample_100.jsonl",
    "phase6_100samples_v2.jsonl": "multi_turn_sample_100_v2.jsonl",
    "phase6_50samples.jsonl": "multi_turn_sample_50.jsonl",
    "phase6_final_sample.jsonl": "multi_turn_sample_final.jsonl",
    "phase6_test.jsonl": "multi_turn_sample_test.jsonl",
}


def fail(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def check_no_jobs() -> None:
    user = os.environ.get("USER", "")
    if not user:
        fail("USER env var not set; cannot run squeue precondition")
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user, "-h"], stderr=subprocess.STDOUT
        )
    except FileNotFoundError:
        fail("squeue command not found; refusing to migrate without job check")
    except subprocess.CalledProcessError as e:
        fail(f"squeue failed: {e.output.decode(errors='replace')}")
    lines = [ln for ln in out.decode().splitlines() if ln.strip()]
    if lines:
        fail(
            f"{len(lines)} SLURM job(s) currently in-flight for $USER={user}; "
            "wait for them to finish before migrating.\n"
            + "\n".join(lines[:5])
        )


def db_phase_distinct() -> dict[str, int]:
    if not STORAGE_DB.exists():
        fail(f"{STORAGE_DB} does not exist")
    if STORAGE_DB.stat().st_size == 0:
        fail(f"{STORAGE_DB} is 0 bytes; aborting")
    conn = sqlite3.connect(STORAGE_DB)
    try:
        rows = conn.execute("SELECT phase, COUNT(*) FROM runs GROUP BY phase").fetchall()
    finally:
        conn.close()
    return {phase: count for phase, count in rows}


def apply_db_update() -> dict:
    """Run the UPDATE; return manifest entry with before/after counts."""
    before = db_phase_distinct()
    unknown = set(before) - set(PHASE_VALUE_MAP) - set(PHASE_VALUE_MAP.values())
    if unknown:
        fail(
            f"runs.phase contains unexpected values: {sorted(unknown)}. "
            "Expected subset of {phase2, phase3, improve, eval}. "
            "Update PHASE_VALUE_MAP in the script before re-running."
        )
    conn = sqlite3.connect(STORAGE_DB)
    try:
        for old, new in PHASE_VALUE_MAP.items():
            conn.execute("UPDATE runs SET phase = ? WHERE phase = ?", (new, old))
        conn.commit()
        after = {
            phase: count
            for phase, count in conn.execute(
                "SELECT phase, COUNT(*) FROM runs GROUP BY phase"
            ).fetchall()
        }
    finally:
        conn.close()
    return {"before": before, "after": after, "map": PHASE_VALUE_MAP}


def revert_db_update(entry: dict) -> None:
    inv = {v: k for k, v in entry["map"].items()}
    conn = sqlite3.connect(STORAGE_DB)
    try:
        for old, new in inv.items():
            conn.execute("UPDATE runs SET phase = ? WHERE phase = ?", (new, old))
        conn.commit()
    finally:
        conn.close()


def backup_db() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = STORAGE_DB.with_name(f"storage.db.pre_rename_{ts}.bak")
    if dst.exists():
        fail(f"Backup target already exists: {dst}")
    shutil.copy2(STORAGE_DB, dst)
    return dst


def move_dir(src: Path, dst: Path, dry_run: bool) -> dict | None:
    if not src.exists():
        return {"src": str(src), "dst": str(dst), "skipped": "source does not exist"}
    if dst.exists():
        fail(f"Destination already exists: {dst} (src={src})")
    if dry_run:
        print(f"DRY-RUN: would move {src} → {dst}")
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"src": str(src), "dst": str(dst), "skipped": None}


def revert_move(entry: dict) -> None:
    if entry.get("skipped"):
        return
    src = Path(entry["src"])
    dst = Path(entry["dst"])
    if not dst.exists():
        print(f"WARN: revert source missing ({dst}); skipping", file=sys.stderr)
        return
    if src.exists():
        print(f"WARN: revert target already exists ({src}); skipping", file=sys.stderr)
        return
    src.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dst), str(src))


def delete_stale_phase2_db(dry_run: bool) -> dict | None:
    if not STALE_PHASE2_DB.exists():
        return None
    size = STALE_PHASE2_DB.stat().st_size
    if size != 0:
        fail(
            f"{STALE_PHASE2_DB} is {size} bytes, not 0 — won't auto-delete a "
            "non-empty file. Inspect it manually."
        )
    if dry_run:
        print(f"DRY-RUN: would delete stale 0-byte {STALE_PHASE2_DB}")
        return None
    STALE_PHASE2_DB.unlink()
    return {"deleted": str(STALE_PHASE2_DB)}


def rename_root_samples(dry_run: bool) -> list[dict]:
    """Rename repo-root phase6_*.jsonl → multi_turn_sample_*.jsonl."""
    renames = []
    for old, new in ROOT_SAMPLE_MAP.items():
        src = PROJECT_ROOT / old
        dst = PROJECT_ROOT / new
        if not src.exists():
            continue
        if dst.exists():
            fail(f"Destination already exists: {dst}")
        if dry_run:
            print(f"DRY-RUN: would rename {src.name} → {dst.name}")
            continue
        src.rename(dst)
        renames.append({"src": str(src), "dst": str(dst)})
    return renames


def purge_pycache(dry_run: bool) -> int:
    """rm -rf every __pycache__ in the project tree."""
    n = 0
    for p in PROJECT_ROOT.rglob("__pycache__"):
        if not p.is_dir():
            continue
        if ".venv" in p.parts or ".git" in p.parts:
            continue
        if dry_run:
            print(f"DRY-RUN: would rm -rf {p}")
            continue
        shutil.rmtree(p)
        n += 1
    return n


def write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {MANIFEST_PATH}")


def apply(args: argparse.Namespace) -> None:
    print("=== Phase rename migration: apply ===")
    if MANIFEST_PATH.exists() and not args.dry_run:
        fail(
            f"Manifest already exists at {MANIFEST_PATH}. "
            "If a previous run failed mid-way, inspect it and run --undo first, "
            "or delete it manually if you're certain."
        )

    check_no_jobs()
    print("squeue: clean (no running jobs)")

    # DB section
    if args.dry_run:
        before = db_phase_distinct()
        print(f"DB phase distribution before: {before}")
        print("DRY-RUN: would back up storage.db and run UPDATE")
        db_entry = {"before": before, "after": "(dry-run)", "backup": "(dry-run)"}
    else:
        backup = backup_db()
        print(f"Backed up storage.db → {backup.name}")
        db_entry = apply_db_update()
        db_entry["backup"] = str(backup)
        print(f"DB phase distribution: {db_entry['before']} → {db_entry['after']}")

    # Dir moves
    moves: list[dict] = []
    for src, dst in DIR_MOVES:
        result = move_dir(src, dst, args.dry_run)
        if result:
            moves.append(result)
            if not result.get("skipped"):
                print(f"Moved: {src} → {dst}")

    # Stale phase2.db delete
    delete_entry = delete_stale_phase2_db(args.dry_run)
    if delete_entry:
        print(f"Deleted stale 0-byte {delete_entry['deleted']}")

    # Root sample renames
    sample_renames = rename_root_samples(args.dry_run)
    for r in sample_renames:
        print(f"Renamed: {Path(r['src']).name} → {Path(r['dst']).name}")

    # pycache purge
    n_purged = purge_pycache(args.dry_run)
    if n_purged:
        print(f"Purged {n_purged} __pycache__ trees")

    if args.dry_run:
        print("DRY-RUN complete. Re-run without --dry-run to apply.")
        return

    manifest = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "db_update": db_entry,
        "moves": moves,
        "deleted": delete_entry,
        "sample_renames": sample_renames,
        "pycache_trees_purged": n_purged,
    }
    write_manifest(manifest)
    print("=== Migration complete ===")


def undo(args: argparse.Namespace) -> None:
    print("=== Phase rename migration: undo ===")
    if not MANIFEST_PATH.exists():
        fail(f"No manifest at {MANIFEST_PATH} — nothing to undo")

    manifest = json.loads(MANIFEST_PATH.read_text())
    check_no_jobs()
    print("squeue: clean (no running jobs)")

    # Reverse order: samples → deleted → moves → DB
    for r in manifest.get("sample_renames", []):
        src = Path(r["src"])
        dst = Path(r["dst"])
        if dst.exists() and not src.exists():
            dst.rename(src)
            print(f"Reverted rename: {dst.name} → {src.name}")

    # The deleted phase2.db is empty/stale; we don't recreate it.
    # (If you need it back, touch data/pipeline/phase2.db manually.)

    for entry in reversed(manifest.get("moves", [])):
        revert_move(entry)
        if not entry.get("skipped"):
            print(f"Reverted move: {entry['dst']} → {entry['src']}")

    revert_db_update(manifest["db_update"])
    print(f"Reverted DB: {manifest['db_update']['after']} → {manifest['db_update']['before']}")

    MANIFEST_PATH.unlink()
    print("Manifest removed.")
    print("=== Undo complete ===")
    print(
        "NOTE: backup file at "
        f"{manifest['db_update'].get('backup')} not deleted; remove it manually if no longer needed."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--undo", action="store_true", help="Revert from migration_manifest.json")
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without changing anything")
    args = parser.parse_args()

    if args.undo:
        undo(args)
    else:
        apply(args)


if __name__ == "__main__":
    main()
