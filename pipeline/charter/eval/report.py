"""Build portable dashboard cards from charter.eval runs.

The HF Space reads only ``dashboard/data/cards.json``. This module is the
repo-side bridge from eval run directories to that portable snapshot, and the
feedback sync path back from a HF dataset.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from pipeline.charter.eval.rank import _read_jsonl, _resolve_run_dir
from pipeline.charter.improve.run import judgment_parts
from pipeline.config import PROJECT_ROOT, load_config
from pipeline.log import logger

DEFAULT_CARDS_PATH = PROJECT_ROOT / "dashboard" / "data" / "cards.json"
_CITE_RE = re.compile(r"\[\d+\.\d+(?:\s*,\s*\d+\.\d+)*\]")
_SECTION_RE = re.compile(r"^#{2,3}\s+(\d+\.\d+)\s+(.*)$")
_FEEDBACK_KEY = ("run_id", "item_id", "generator", "judge", "reviewer")
_RUBRIC_DIMS = ("relevance", "specificity", "charter_grounding", "voice_tone")


def parse_charter_sections(charter_text: str, max_chars: int = 4000) -> dict[str, str]:
    """Map each ``## X.Y Title`` constitution section to escaped HTML."""
    sections: dict[str, str] = {}
    cur_id: str | None = None
    cur_title = ""
    buf: list[str] = []

    def flush() -> None:
        if cur_id is None:
            return
        src = f"<b>{html.escape(cur_id)} {html.escape(cur_title)}</b>"
        body = "\n".join(buf).strip()
        if body:
            src += "<br><br>" + html.escape(body[:max_chars]).replace("\n", "<br>")
        sections[cur_id] = src

    for line in charter_text.splitlines():
        m = _SECTION_RE.match(line)
        if m or line.startswith("# "):
            flush()
            cur_id, cur_title, buf = (m.group(1), m.group(2).strip(), []) if m else (None, "", [])
        elif cur_id is not None:
            buf.append(line)
    flush()
    return sections


def _split_stem(stem: str) -> tuple[str, str]:
    alias, sep, prompt = stem.partition("__")
    return (alias, prompt) if sep else (stem, "")


def _charter_elements(row: dict) -> list[str]:
    stored = row.get("reflection_charter_elements") or row.get("charter_reflection")
    if isinstance(stored, str):
        try:
            parsed = json.loads(stored)
        except json.JSONDecodeError:
            parsed = stored
        stored = parsed
    if stored:
        return [f"[{x}]" if re.fullmatch(r"\d+\.\d+", str(x)) else str(x) for x in stored]
    return _CITE_RE.findall(row.get("reflection_1p") or "")


def _generation_card(run_id: str, gen_stem: str, row: dict) -> dict:
    gen_model, gen_prompt = _split_stem(gen_stem)
    return {
        "run_id": run_id,
        "item_id": row.get("item_id"),
        "generator": gen_stem,
        "judge": None,
        "gen_model": gen_model,
        "gen_prompt": gen_prompt,
        "judge_model": None,
        "judge_prompt": None,
        "language": row.get("subset") or "dolma3",
        "safety_score": row.get("safety_score"),
        "reflection_point": row.get("reflection_point"),
        "text": row.get("text") or "",
        "analysis": row.get("analysis") or "",
        "reflection_1p": row.get("reflection_1p") or "",
        "reflection_3p": row.get("reflection_3p") or "",
        "charter_elements": _charter_elements(row),
        "judge_scores": {},
        "judge_aggregate": None,
        "judge_decision": None,
        "judge_reasoning": "",
    }


def _judgment_card(run_id: str, judge_stem: str, gen_stem: str, row: dict) -> dict:
    card = _generation_card(run_id, gen_stem, row)
    judge_model, judge_prompt = _split_stem(judge_stem)
    j = row.get("judgment") or {}
    refl = judgment_parts(j).get("reflection_1p") or {}
    scores = refl.get("scores") or {}
    card.update(
        {
            "judge": judge_stem,
            "judge_model": judge_model,
            "judge_prompt": judge_prompt,
            "judge_scores": {d: scores[d] for d in _RUBRIC_DIMS if d in scores},
            "judge_aggregate": j.get("reflection_aggregate", refl.get("aggregate")),
            "judge_decision": j.get("reflection_decision") or j.get("decision"),
            "judge_reasoning": refl.get("reasoning") or "",
        }
    )
    return card


def build_cards(
    run_ids: list[str],
    *,
    eval_dir: Path | str | None = None,
    source: str = "auto",
) -> list[dict]:
    """Build display cards from one or more eval runs.

    ``source`` is ``"auto"``, ``"generations"``, or ``"judgments"``. ``auto``
    returns generation cards and overlays any matching judgments, so partially
    judged runs still expose unjudged cards for human review.
    """
    assert source in {"auto", "generations", "judgments"}
    cards: list[dict] = []
    for run_id in run_ids:
        run_dir = _resolve_run_dir(run_id, eval_dir)
        jud_dir = run_dir / "judgments"
        gen_dir = run_dir / "generations"

        gen_cards: dict[tuple[str, str], dict] = {}
        if source in {"auto", "generations"}:
            if not gen_dir.exists():
                logger.warning("report: no generations/ in {} - skipping", run_dir)
            else:
                for gen_file in sorted(gen_dir.glob("*.jsonl")):
                    for row in _read_jsonl(gen_file):
                        if row.get("item_id"):
                            gen_cards[(str(row["item_id"]), gen_file.stem)] = _generation_card(
                                run_id, gen_file.stem, row
                            )
            if source == "generations":
                cards.extend(gen_cards.values())
                continue

        if source in {"auto", "judgments"} and jud_dir.exists():
            for jud_file in sorted(jud_dir.glob("*.jsonl")):
                stem = jud_file.stem
                if "__on__" not in stem:
                    continue
                judge_stem, gen_stem = stem.split("__on__", 1)
                for row in _read_jsonl(jud_file):
                    if not row.get("item_id"):
                        continue
                    card = _judgment_card(run_id, judge_stem, gen_stem, row)
                    if source == "auto":
                        gen_cards[(str(row["item_id"]), gen_stem)] = card
                    else:
                        cards.append(card)
        if source == "auto":
            cards.extend(gen_cards.values())
    return cards


def write_cards(
    run_ids: list[str],
    out_path: Path | str,
    *,
    eval_dir: Path | str | None = None,
    source: str = "auto",
    charter_path: Path | str | None = None,
) -> int:
    """Build cards and write the portable ``cards.json`` dashboard snapshot."""
    cfg = load_config()
    cards = build_cards(run_ids, eval_dir=eval_dir, source=source)
    path = Path(charter_path or cfg.charter_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    sections = parse_charter_sections(path.read_text(encoding="utf-8"))
    payload = {
        "runs": list(run_ids),
        "n_cards": len(cards),
        "charter_sections": sections,
        "cards": cards,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return len(cards)


def deploy_space(space_id: str, folder: Path | str = "dashboard") -> None:
    """Upload the dashboard folder to a HF Space."""
    from huggingface_hub import HfApi

    folder = Path(folder)
    cards_path = folder / "data" / "cards.json"
    if not cards_path.is_file():
        raise FileNotFoundError(
            f"{cards_path} is missing. Build it first with: "
            "uv run python -m pipeline.charter.eval report <run_id> --source generations"
        )
    api = HfApi()
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=space_id,
        repo_type="space",
        ignore_patterns=["feedback/*", "__pycache__/*", ".gradio/*", ".gitignore"],
    )
    logger.info("Deployed {} -> https://huggingface.co/spaces/{}", folder, space_id)


def _feedback_key(row: dict) -> tuple:
    return tuple(row.get(k) for k in _FEEDBACK_KEY)


def retrieve_feedback(dataset: str, local_dir: Path | str) -> list[dict]:
    """Download feedback and keep the latest verdict per card/reviewer."""
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    download_dir = local_dir / "snapshot"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    snapshot_download(repo_id=dataset, repo_type="dataset", local_dir=str(download_dir))
    latest: dict[tuple, dict] = {}
    for path in sorted((download_dir / "data").glob("*.jsonl")):
        for row in _read_jsonl(path):
            if row.get("verdict") in ("accept", "reject") and row.get("item_id"):
                latest[_feedback_key(row)] = row
    return list(latest.values())


def summarize_feedback(rows: list[dict]) -> dict:
    """Counts and optional judge-agreement for retrieved feedback."""
    verdicts = Counter(r["verdict"] for r in rows)
    judged = [r for r in rows if r.get("judge_decision") in ("accept", "reject")]
    agree = sum(1 for r in judged if r["verdict"] == r["judge_decision"])
    return {
        "n": len(rows),
        "accept": verdicts.get("accept", 0),
        "reject": verdicts.get("reject", 0),
        "n_vs_judge": len(judged),
        "agreement": agree / len(judged) if judged else None,
    }
