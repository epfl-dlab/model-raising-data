"""Phase 3 escalation storage — thin layer over shared SQLite backend."""

from datetime import datetime, timezone

from pipeline.storage import _get_conn


def save_escalation(
    item_id: str,
    group_id: str,
    gold_model: str,
    target_model: str,
    role: str,
    reason: str,
) -> int:
    """Insert an escalation record. Returns the new row id."""
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO escalations
           (item_id, group_id, gold_model, target_model, role, reason, status, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
        (
            item_id,
            group_id,
            gold_model,
            target_model,
            role,
            reason,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def load_escalations(status: str | None = None) -> list[dict]:
    """Load escalation records, optionally filtered by status."""
    conn = _get_conn()
    if status is not None:
        rows = conn.execute(
            "SELECT * FROM escalations WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM escalations ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def update_escalation(
    escalation_id: int, status: str, reviewer_notes: str | None = None
) -> None:
    """Update an escalation's status and optional reviewer notes."""
    conn = _get_conn()
    conn.execute(
        "UPDATE escalations SET status = ?, reviewer_notes = ? WHERE id = ?",
        (status, reviewer_notes, escalation_id),
    )
    conn.commit()
