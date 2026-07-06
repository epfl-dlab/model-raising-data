"""Model Raising reflection review dashboard.

Reads ``data/cards.json`` produced by ``python -m pipeline.charter.eval report``
and collects binary accept/reject feedback. The app intentionally does not
import ``pipeline`` so it can run as a lightweight Hugging Face Space.
"""

from __future__ import annotations

import datetime
import html
import json
import os
import random
import re
from pathlib import Path

import gradio as gr

TITLE = "Model Raising Reflection Review"
APP_DIR = Path(__file__).parent
CARDS_PATH = Path(os.environ.get("CARDS_PATH", APP_DIR / "data" / "cards.json"))
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", APP_DIR / "feedback"))
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"
FEEDBACK_DATASET = os.environ.get("FEEDBACK_DATASET", "")

ALL = "(all)"
_CITE_RE = re.compile(r"\[(\d+\.\d+(?:\s*,\s*\d+\.\d+)*)\]")


def load_payload() -> tuple[list[dict], dict]:
    """Load cards + constitution sections from the portable snapshot."""
    if not CARDS_PATH.exists():
        return [], {}
    d = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    return d.get("cards", []), d.get("charter_sections", {})


CARDS, CHARTER_SECTIONS = load_payload()


def _card_key(c: dict) -> tuple:
    return (c.get("run_id"), c.get("item_id"), c.get("generator"), c.get("judge"))


def annotator_order(name: str) -> list[int]:
    """Deterministic order: most severe first, shuffled within score buckets."""
    salt = os.environ.get("SHUFFLE_SALT", "annotator")
    rng = random.Random(f"{salt}::{name}")
    by_score: dict[int, list[int]] = {}
    for i, card in enumerate(CARDS):
        by_score.setdefault(int(card.get("safety_score") or 0), []).append(i)
    order: list[int] = []
    for score in sorted(by_score, reverse=True):
        bucket = by_score[score]
        rng.shuffle(bucket)
        order.extend(bucket)
    return order


def graded_keys(reviewer: str) -> set:
    """Card keys this reviewer has already graded."""
    if FEEDBACK_DATASET:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            FEEDBACK_DATASET,
            repo_type="dataset",
            allow_patterns="data/*.jsonl",
            token=os.environ.get("HF_TOKEN"),
        )
        paths = list(Path(root).rglob("*.jsonl"))
    else:
        paths = [FEEDBACK_FILE] if FEEDBACK_FILE.exists() else []
    keys: set = set()
    for p in paths:
        for raw in p.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed feedback JSON in {p}: {e}") from e
            if row.get("reviewer") == reviewer:
                keys.add(_card_key(row))
    return keys


def _options(field: str) -> list[str]:
    vals = {str(c.get(field)) for c in CARDS if c.get(field) is not None}
    return [ALL] + sorted(vals)


def _passes(c: dict, gen: str, lang: str, decision: str, safety: str) -> bool:
    if gen != ALL and c.get("gen_model") != gen:
        return False
    if lang != ALL and (c.get("language") or "-") != lang:
        return False
    if decision != ALL and (c.get("judge_decision") or "-") != decision:
        return False
    if safety != ALL and str(c.get("safety_score")) != safety:
        return False
    return True


def filter_indices(
    gen: str,
    lang: str,
    decision: str,
    safety: str,
    order: list[int] | None = None,
) -> list[int]:
    """Card indices matching the filters, in reviewer order."""
    seq = range(len(CARDS)) if order is None else order
    return [i for i in seq if _passes(CARDS[i], gen, lang, decision, safety)]


def _doc_value(c: dict) -> str:
    text = c.get("text") or ""
    rp = c.get("reflection_point")
    if isinstance(rp, int) and 0 < rp <= len(text):
        return text[:rp]
    return text


def _wrap_citations(text: str) -> str:
    esc = html.escape(text or "")

    def repl(m: re.Match) -> str:
        ids = [s.strip() for s in m.group(1).split(",")]
        tip = "<hr class='tipsep'>".join(
            CHARTER_SECTIONS.get(i, html.escape(i)) for i in ids
        )
        return f'<span class="cite">[{m.group(1)}]<span class="tip">{tip}</span></span>'

    return _CITE_RE.sub(repl, esc)


def _reflection_html(c: dict) -> str:
    refl = _wrap_citations(c.get("reflection_1p") or "(none)")
    cites = _wrap_citations(" ".join(c.get("charter_elements") or []) or "-")
    out = [
        "<h3 style='margin:.3em 0'>First-person reflection under review</h3>",
        f"<div>{refl}</div>",
        f"<div style='margin-top:.5em'><b>Citations:</b> {cites}</div>",
    ]
    if c.get("analysis"):
        out.append(
            "<details style='margin-top:.6em'><summary>Analysis</summary>"
            f"<div>{html.escape(c['analysis'])}</div></details>"
        )
    return "".join(out)


def _judge_html(c: dict) -> str:
    if not c.get("judge"):
        return (
            "<h3 style='margin:.8em 0 .3em'>Automated judge</h3>"
            "<i>No automated judge attached to this card.</i>"
        )
    scores = c.get("judge_scores") or {}
    dims = "  ·  ".join(f"{html.escape(k)} <b>{v}</b>" for k, v in scores.items()) or "-"
    decision = html.escape((c.get("judge_decision") or "-").upper())
    agg = c.get("judge_aggregate")
    agg_s = f"{agg:.2f}" if isinstance(agg, (int, float)) else "-"
    color = "#15803d" if c.get("judge_decision") == "accept" else "#b91c1c"
    out = [
        "<h3 style='margin:.8em 0 .3em'>Automated judge</h3>",
        f"<div><b style='color:{color}'>{decision}</b> · aggregate <b>{agg_s}</b> · "
        f"judged by <code>{html.escape(c.get('judge_model') or '')}</code></div>",
        f"<div style='margin-top:.3em'>{dims}</div>",
    ]
    if c.get("judge_reasoning"):
        out.append(
            "<blockquote style='border-left:3px solid #888;margin:.5em 0;padding-left:.6em'>"
            f"{_wrap_citations(c['judge_reasoning'])}</blockquote>"
        )
    return "".join(out)


def render(idxs: list[int], pos: int):
    """Return display values for the selected card."""
    if not idxs:
        empty = "_No cards match these filters._" if CARDS else (
            "_No cards loaded. Build `data/cards.json` with "
            "`python -m pipeline.charter.eval report`._"
        )
        return empty, "", "", "", "0 / 0"
    pos = max(0, min(pos, len(idxs) - 1))
    c = CARDS[idxs[pos]]
    meta = (
        f"**model `{c.get('gen_model')}`** · prompt `{c.get('gen_prompt') or '-'}` · "
        f"lang `{c.get('language')}` · safety `{c.get('safety_score')}`"
    )
    return meta, _doc_value(c), _reflection_html(c), _judge_html(c), f"{pos + 1} / {len(idxs)}"


def _card(idxs, pos, status_msg=""):
    meta, doc, refl, judge_html, poslabel = render(idxs, pos)
    return meta, doc, refl, judge_html, poslabel, None, "", status_msg


def apply_filters(order, gen, lang, decision, safety):
    idxs = filter_indices(gen, lang, decision, safety, order=order)
    return (idxs, 0, *_card(idxs, 0))


def step(idxs, pos, delta):
    new_pos = max(0, min(pos + delta, max(0, len(idxs) - 1)))
    return (new_pos, *_card(idxs, new_pos))


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:120]


def submit_feedback(idxs, pos, reviewer, verdict, reason):
    """Save feedback and remove the graded card from the queue."""
    if not idxs:
        return (gr.update(),) * 9 + ("Nothing to rate.",)
    if not verdict:
        return (gr.update(),) * 9 + ("Pick accept or reject first.",)
    c = CARDS[idxs[pos]]
    record = {
        "run_id": c.get("run_id"),
        "item_id": c.get("item_id"),
        "generator": c.get("generator"),
        "judge": c.get("judge"),
        "judge_decision": c.get("judge_decision"),
        "verdict": "accept" if verdict == "accept" else "reject",
        "reason": (reason or "").strip(),
        "reviewer": (reviewer or "anon").strip() or "anon",
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_FILE.open("a", encoding="utf-8").write(line)
    if FEEDBACK_DATASET:
        from huggingface_hub import HfApi

        stem = _safe_name(f"{record['ts']}-{record['reviewer']}-{record['item_id']}")
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=line.encode("utf-8"),
            path_in_repo=f"data/{stem}.jsonl",
            repo_id=FEEDBACK_DATASET,
            repo_type="dataset",
            commit_message=f"feedback: {record['verdict']} by {record['reviewer']}",
        )
        msg = f"Saved {record['verdict']} -> {FEEDBACK_DATASET}"
    else:
        msg = f"Saved {record['verdict']} locally: {FEEDBACK_FILE}"
    new_idxs = idxs[:pos] + idxs[pos + 1:]
    new_pos = min(pos, max(0, len(new_idxs) - 1))
    return (new_idxs, new_pos, *_card(new_idxs, new_pos, msg))


def _spec_sections() -> list[tuple[str, str]]:
    def key(sid: str):
        return tuple(int(p) if p.isdigit() else 0 for p in sid.split("."))

    return sorted(CHARTER_SECTIONS.items(), key=lambda kv: key(kv[0]))


def spec_html(query: str = "") -> str:
    q = (query or "").strip().lower()
    blocks = [
        f"<div style='margin:.5em 0;padding-bottom:.4em;border-bottom:1px solid #8884'>{body}</div>"
        for sid, body in _spec_sections()
        if not q or q in sid.lower() or q in re.sub(r"<[^>]+>", " ", body).lower()
    ]
    return "".join(blocks) or "<i>No sections match.</i>"


_CSS = (
    ".cite{border-bottom:1px dotted #888;cursor:help}"
    ".cite .tip{display:none}"
    "#cite-tip{position:fixed;z-index:9999;display:none;width:380px;max-width:90vw;"
    "max-height:280px;overflow-y:auto;text-align:left;background:#111827;color:#fff;"
    "padding:6px 10px;border-radius:6px;font-size:.74rem;line-height:1.3;"
    "box-shadow:0 6px 20px rgba(0,0,0,.55)}"
    "#cite-tip *{color:#fff}"
    "#cite-tip p{margin:.3em 0}"
    "#cite-tip hr.tipsep{margin:6px 0;border:0;border-top:1px solid #444}"
)

_TOOLTIP_JS = """
() => {
  if (window.__citeTipInit) return; window.__citeTipInit = true;
  const tip = document.createElement('div'); tip.id = 'cite-tip';
  document.body.appendChild(tip);
  let over = false;
  tip.addEventListener('mouseenter', () => { over = true; });
  tip.addEventListener('mouseleave', () => { over = false; tip.style.display = 'none'; });
  document.addEventListener('mouseover', (e) => {
    const cite = e.target.closest ? e.target.closest('.cite') : null;
    if (!cite) return;
    const src = cite.querySelector('.tip'); if (!src) return;
    tip.innerHTML = src.innerHTML; tip.style.display = 'block';
    const r = cite.getBoundingClientRect();
    const tw = Math.min(380, window.innerWidth * 0.9);
    tip.style.left = Math.max(8, Math.min(r.left, window.innerWidth - tw - 8)) + 'px';
    const top = r.top - tip.offsetHeight - 6;
    tip.style.top = (top < 8 ? r.bottom + 6 : top) + 'px';
  });
  document.addEventListener('mouseout', (e) => {
    const cite = e.target.closest ? e.target.closest('.cite') : null;
    if (!cite) return;
    setTimeout(() => { if (!over) tip.style.display = 'none'; }, 150);
  });
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.HTML(f"<style>{_CSS}</style>")
        gr.Markdown(f"# {TITLE}")

        with gr.Column(visible=True) as gate:
            gr.Markdown("Enter your name to start reviewing.")
            name_in = gr.Textbox(label="Your name", placeholder="e.g. julian")
            start_btn = gr.Button("Start reviewing", variant="primary")
            gate_msg = gr.Markdown()

        with gr.Column(visible=False) as main_panel:
            who_md = gr.Markdown()
            with gr.Accordion("Filters", open=False):
                with gr.Row():
                    gen = gr.Dropdown(_options("gen_model"), value=ALL, label="Model", min_width=120)
                    lang = gr.Dropdown(_options("language"), value=ALL, label="Language", min_width=110)
                    decision = gr.Dropdown(_options("judge_decision"), value=ALL, label="Judge verdict", min_width=120)
                    safety = gr.Dropdown(_options("safety_score"), value=ALL, label="Safety", min_width=90)

            with gr.Row():
                prev_btn = gr.Button("prev", size="sm", scale=0, min_width=56)
                poslabel = gr.Markdown("0 / 0")
                next_btn = gr.Button("next", size="sm", scale=0, min_width=56)

            meta_md = gr.Markdown()
            doc_box = gr.Textbox(
                label="Document up to the reflection point",
                interactive=False,
                lines=14,
                max_lines=24,
            )
            refl_html = gr.HTML()
            judge_md = gr.HTML()

            gr.Markdown("### Your feedback")
            verdict = gr.Radio(["accept", "reject"], label="Your verdict")
            reason = gr.Textbox(label="Reason (optional)", lines=2)
            submit_btn = gr.Button("Submit feedback", variant="primary")
            status = gr.Markdown()

            with gr.Accordion("Constitution sections (search)", open=False):
                spec_search = gr.Textbox(label="Search", placeholder="e.g. privacy, 5.3")
                spec_view = gr.HTML(spec_html(""))

        reviewer_state = gr.State("")
        order_state = gr.State(list(range(len(CARDS))))
        idxs_state = gr.State(list(range(len(CARDS))))
        pos_state = gr.State(0)

        VIEW = [meta_md, doc_box, refl_html, judge_md, poslabel]
        CARD_OUT = [*VIEW, verdict, reason, status]

        def start(name):
            nm = (name or "").strip()
            noop = gr.update()
            if not nm:
                return (noop, noop, "Please enter your name.") + (noop,) * 10
            done = graded_keys(nm)
            order = [i for i in annotator_order(nm) if _card_key(CARDS[i]) not in done]
            view = render(order, 0)
            graded = f" ({len(done)} already graded)" if done else ""
            who = f"Reviewing as **{nm}** - {len(order)} to review{graded}"
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                who,
                nm,
                order,
                order,
                0,
                *view,
            )

        start_btn.click(
            start,
            inputs=[name_in],
            outputs=[
                gate,
                main_panel,
                gate_msg,
                who_md,
                reviewer_state,
                order_state,
                idxs_state,
                pos_state,
                *VIEW,
            ],
        )

        filters = [gen, lang, decision, safety]
        for f in filters:
            f.change(
                apply_filters,
                inputs=[order_state, *filters],
                outputs=[idxs_state, pos_state, *CARD_OUT],
            )
        prev_btn.click(
            lambda i, p: step(i, p, -1),
            inputs=[idxs_state, pos_state],
            outputs=[pos_state, *CARD_OUT],
        )
        next_btn.click(
            lambda i, p: step(i, p, 1),
            inputs=[idxs_state, pos_state],
            outputs=[pos_state, *CARD_OUT],
        )
        submit_btn.click(
            submit_feedback,
            inputs=[idxs_state, pos_state, reviewer_state, verdict, reason],
            outputs=[idxs_state, pos_state, *CARD_OUT],
        )
        demo.load(lambda: render(list(range(len(CARDS))), 0), outputs=VIEW)
        demo.load(js=_TOOLTIP_JS)
        spec_search.change(spec_html, inputs=[spec_search], outputs=[spec_view])
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("DASHBOARD_PORT", 7860)))
