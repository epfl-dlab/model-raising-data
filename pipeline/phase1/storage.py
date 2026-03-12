"""Phase 1 annotation persistence via append-only JSONL files."""

from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import ANNOTATION_DATA_DIR
from pipeline.storage import append_jsonl, load_jsonl


def annotations_path() -> Path:
    """Return the JSONL file path for annotations."""
    ANNOTATION_DATA_DIR.mkdir(exist_ok=True)
    return ANNOTATION_DATA_DIR / "annotations.jsonl"


def load_annotations() -> list[dict]:
    """Load all annotation records (no dedup)."""
    return load_jsonl(annotations_path())


def load_latest_annotations() -> dict[tuple[str, str], dict]:
    """Load annotations keyed by (item_id, annotator_id). Last entry per key wins."""
    latest: dict[tuple[str, str], dict] = {}
    for record in load_annotations():
        key = (record["item_id"], record["annotator_id"])
        latest[key] = record
    return latest


def load_annotator_ids() -> list[str]:
    """Return sorted list of unique annotator IDs from existing annotations."""
    annotations = load_annotations()
    return sorted({r["annotator_id"] for r in annotations})


def save_annotation(
    item_id: str,
    annotator_id: str,
    subset: str,
    text: str,
    reflection_point: int,
    analysis: str,
    preflection: str,
    reflection: str,
    reflection_charter_elements: list[str],
    presentation_order: int,
) -> None:
    """Append a single annotation record (includes full source text)."""
    record = {
        "item_id": item_id,
        "annotator_id": annotator_id,
        "subset": subset,
        "text": text,
        "reflection_point": reflection_point,
        "analysis": analysis,
        "preflection": preflection,
        "reflection": reflection,
        "reflection_charter_elements": reflection_charter_elements,
        "presentation_order": presentation_order,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(annotations_path(), record)


def load_annotations_by_item() -> dict[str, list[dict]]:
    """Load all annotations grouped by item_id (latest per annotator)."""
    latest = load_latest_annotations()
    by_item: dict[str, list[dict]] = {}
    for (item_id, _), record in latest.items():
        by_item.setdefault(item_id, []).append(record)
    return by_item


# --- Comments ---

def comments_path() -> Path:
    """Return the JSONL file path for annotation comments."""
    ANNOTATION_DATA_DIR.mkdir(exist_ok=True)
    return ANNOTATION_DATA_DIR / "comments.jsonl"


def load_comments() -> list[dict]:
    """Load all comment records."""
    return load_jsonl(comments_path())


def load_comments_by_annotation() -> dict[tuple[str, str], list[dict]]:
    """Load comments keyed by (item_id, target_annotator_id), sorted by timestamp."""
    by_annotation: dict[tuple[str, str], list[dict]] = {}
    for comment in load_comments():
        key = (comment["item_id"], comment["target_annotator_id"])
        by_annotation.setdefault(key, []).append(comment)
    for comments in by_annotation.values():
        comments.sort(key=lambda c: c["timestamp"])
    return by_annotation


def save_comment(
    item_id: str,
    target_annotator_id: str,
    commenter_id: str,
    comment: str,
) -> None:
    """Append a comment on an annotation."""
    record = {
        "item_id": item_id,
        "target_annotator_id": target_annotator_id,
        "commenter_id": commenter_id,
        "comment": comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(comments_path(), record)
