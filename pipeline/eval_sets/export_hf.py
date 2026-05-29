"""Build the two HuggingFace eval datasets (plan.md §5).

- `export_reflection_eval`: read the merged reflection_end sidecar, add
  `requires_citation`, assert `canary_type_end` is all-null (F4), push.
- `export_sft_eval`: read sft results.jsonl, add `has_citation` + `charter_elements`
  derived from `extract_charter_elements(cited)` (not the disagreeing analysis line),
  build paired chat messages, push.

The label helpers are pure and unit-tested; the IO/upload wraps them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import extract_charter_elements
from pipeline.log import logger

# ---------------------------------------------------------------------------
# Pure label helpers (unit-tested)
# ---------------------------------------------------------------------------


def requires_citation(charter_reflection_end: str | None) -> bool:
    """True iff the reflection cited ≥1 charter element.

    `charter_reflection_end` is a JSON-encoded list of `[X.Y]` ids (e.g. "[]" or
    '["2.1", "4.3]"'). Empty / missing → False.
    """
    if not charter_reflection_end:
        return False
    try:
        return len(json.loads(charter_reflection_end)) > 0
    except (json.JSONDecodeError, TypeError):
        return False


def sft_citation_labels(cited: str | None) -> tuple[bool, list[str]]:
    """Return (has_citation, charter_elements) from the cited response text.

    Uses the charter-id extractor on the actual `cited` text — the source of truth —
    rather than the optional/disagreeing `analysis` "Citations:" line.
    """
    elements = extract_charter_elements(cited or "")
    return (len(elements) > 0, elements)


# ---------------------------------------------------------------------------
# Reflection-end eval dataset
# ---------------------------------------------------------------------------

_REFLECTION_COLUMNS = [
    "doc_id", "source", "text", "safety_score", "is_bad", "token_length",
    "reflection_end_1p", "reflection_end_3p", "reflection_end_position",
    "reflection_end_token_index", "charter_reflection_end", "canary_type_end", "strata",
]


def export_reflection_eval(
    merged_sidecar: str | Path,
    out_path: str | Path,
    *,
    repo_id: str | None = None,
    private: bool = False,
) -> dict:
    """Read the merged reflection_end sidecar, add `requires_citation`, validate, write, push."""
    merged_sidecar = Path(merged_sidecar)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(merged_sidecar)
    present = [c for c in _REFLECTION_COLUMNS if c in table.column_names]
    table = table.select(present)

    canary = table.column("canary_type_end").to_pylist() if "canary_type_end" in present else []
    n_canary = sum(1 for c in canary if c is not None)
    assert n_canary == 0, f"F4: canary_type_end has {n_canary} non-null values — canaries leaked into eval gold"

    # Drop rows where BOTH reflection voices are empty — these are parse/generation
    # failures with no usable gold to evaluate against. An eval set shouldn't ship them.
    n_pre = table.num_rows
    one_p = table.column("reflection_end_1p").to_pylist()
    three_p = table.column("reflection_end_3p").to_pylist()
    keep = [
        bool((one_p[i] or "").strip()) or bool((three_p[i] or "").strip())
        for i in range(n_pre)
    ]
    table = table.filter(pa.array(keep, pa.bool_()))
    n_dropped_empty = n_pre - table.num_rows
    if n_dropped_empty:
        logger.info("Dropped {} empty-gold rows (both 1p and 3p blank)", n_dropped_empty)

    charter_col = table.column("charter_reflection_end").to_pylist()
    req = pa.array([requires_citation(c) for c in charter_col], pa.bool_())
    table = table.append_column("requires_citation", req)

    pq.write_table(table, out_path)
    n_bad = sum(table.column("is_bad").to_pylist())
    n_req = sum(req.to_pylist())
    stats = {
        "rows": table.num_rows,
        "dropped_empty_gold": n_dropped_empty,
        "is_bad": n_bad,
        "requires_citation": n_req,
        "out_path": str(out_path),
    }
    logger.info("reflection eval export: {}", stats)

    if repo_id:
        _upload_parquet(out_path, repo_id, private=private, kind="reflection-end-eval")
    return stats


# ---------------------------------------------------------------------------
# SFT eval dataset (extends sft.single_turn export with citation labels)
# ---------------------------------------------------------------------------


def export_sft_eval(
    results_jsonl: str | Path,
    out_path: str | Path,
    *,
    repo_id: str | None = None,
    private: bool = False,
) -> dict:
    """Read sft results.jsonl; add has_citation + charter_elements; write paired dataset; push."""
    from pipeline.sft.single_turn.canaries import SKIP_CANARY_VALUES
    from pipeline.sft.single_turn.export import (
        _row_to_messages,
        _strip_surrogates,
    )
    from pipeline.sft.single_turn.generate import has_identity_leak

    results_jsonl = Path(results_jsonl)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert results_jsonl.exists(), f"results.jsonl not found: {results_jsonl}"

    rows, n_skip, n_total, n_cited = [], 0, 0, 0
    with results_jsonl.open() as f:
        for line in f:
            n_total += 1
            r = json.loads(line)
            if r.get("skip") or "error" in r or "cited" not in r or "uncited" not in r:
                n_skip += 1
                continue
            if not r["cited"].strip() or not r["uncited"].strip():
                n_skip += 1
                continue
            if has_identity_leak(r["cited"]) or has_identity_leak(r["uncited"]):
                n_skip += 1
                continue
            if any(v in r["cited"] or v in r["uncited"] for v in SKIP_CANARY_VALUES):
                n_skip += 1
                continue
            has_cite, elements = sft_citation_labels(r["cited"])
            n_cited += int(has_cite)
            user = _strip_surrogates(r["user"])
            rows.append({
                "source": r["source"],
                "source_id": r["source_id"],
                "messages_cite": _row_to_messages(user, _strip_surrogates(r["cited"])),
                "messages_nocite": _row_to_messages(user, _strip_surrogates(r["uncited"])),
                "has_citation": has_cite,
                "charter_elements": elements,
                "harm_category": r.get("harm_category") or (r.get("meta") or {}).get("harm_category"),
                "meta": json.dumps(r.get("meta") or {}, ensure_ascii=False),
            })

    messages_type = pa.list_(pa.struct([("role", pa.string()), ("content", pa.large_string())]))
    schema = pa.schema([
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("messages_cite", messages_type),
        ("messages_nocite", messages_type),
        ("has_citation", pa.bool_()),
        ("charter_elements", pa.list_(pa.string())),
        ("harm_category", pa.string()),
        ("meta", pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)
    stats = {
        "input_rows": n_total,
        "skipped": n_skip,
        "exported_rows": len(rows),
        "has_citation": n_cited,
        "out_path": str(out_path),
    }
    logger.info("sft eval export: {}", stats)

    if repo_id:
        _upload_parquet(out_path, repo_id, private=private, kind="sft-eval")
    return stats


def _upload_parquet(path: Path, repo_id: str, *, private: bool, kind: str) -> None:
    """Upload a single parquet to the Hub under data/."""
    import os

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert token, "HF_TOKEN not set — cannot upload to Hub"
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"data/{path.name}",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {kind} data ({path.name})",
    )
    logger.info("uploaded {} to {} (private={})", path.name, repo_id, private)
