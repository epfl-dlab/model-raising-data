"""ConstitutionMCQ — human solvability check (single-page HF Space).

Three reviewers each get one of three *disjoint* sets of 20 items. A reviewer
logs in with their name and picks their assigned set, then for each item reads a
scenario + four candidate actions, consults the (searchable) value constitution,
chooses the option they consider best, and gives a required reason. Each review
is committed immediately to a HF dataset as its own file, so the page holds no
credentials and nothing is lost when the Space restarts.

Gold answers are NOT in the shipped data — we record the chosen option's text and
score correctness offline against the source dataset.

Env:
  FEEDBACK_DATASET  HF dataset repo to commit reviews to (unset -> local only)
  FEEDBACK_DIR      local feedback folder (default: ./feedback)
  HF_TOKEN          Space secret with write access to FEEDBACK_DATASET
"""
from __future__ import annotations

import datetime
import hashlib
import html
import json
import os
import re
from pathlib import Path

import gradio as gr

TITLE = "ConstitutionMCQ — human solvability check"
APP_DIR = Path(__file__).parent
ITEMS_PATH = Path(os.environ.get("ITEMS_PATH", APP_DIR / "data" / "items.json"))
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", APP_DIR / "feedback"))
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"
FEEDBACK_DATASET = os.environ.get("FEEDBACK_DATASET", "")

DATA = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
SETS: dict[str, list[dict]] = DATA["sets"]
CONSTITUTION: list[dict] = DATA["constitution"]
CONST_VERSION: str = DATA.get("constitution_version", "value constitution")
SET_NAMES = list(SETS.keys())


def _reviewer_map() -> dict[str, str]:
    """Parse REVIEWER_MAP="alice=Set 1,bob=Set 2,carol=Set 3" (name -> set)."""
    m: dict[str, str] = {}
    for part in os.environ.get("REVIEWER_MAP", "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() and v.strip() in SETS:
                m[k.strip().lower()] = v.strip()
    return m


REVIEWER_MAP = _reviewer_map()
PASSWORD = os.environ.get("APP_PASSWORD", "")   # shared gate; unset -> no password (dev)


def assigned_set(name: str) -> str | None:
    """The set this reviewer is assigned. With REVIEWER_MAP set (production), only
    listed names are accepted; without it (local dev), assign deterministically."""
    nm = name.strip().lower()
    if REVIEWER_MAP:
        return REVIEWER_MAP.get(nm)
    idx = int(hashlib.sha1(nm.encode()).hexdigest(), 16) % len(SET_NAMES)
    return SET_NAMES[idx]


# ----------------------------------------------------------------- progress


def graded_ids(reviewer: str) -> set:
    """Item ids this reviewer has already submitted (feedback dataset or local)."""
    if FEEDBACK_DATASET:
        from huggingface_hub import snapshot_download

        try:
            root = snapshot_download(
                FEEDBACK_DATASET, repo_type="dataset",
                allow_patterns="data/*.jsonl", token=os.environ.get("HF_TOKEN"),
            )
            paths = list(Path(root).rglob("*.jsonl"))
        except Exception:
            paths = []
    else:
        paths = [FEEDBACK_FILE] if FEEDBACK_FILE.exists() else []
    done: set = set()
    for p in paths:
        for raw in p.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if r.get("reviewer") == reviewer:
                done.add(r.get("item_id"))
    return done


# ----------------------------------------------------------------- rendering


def scenario_md(it: dict) -> str:
    return f"### Scenario\n\n{it['scenario']}"


def option_choices(it: dict) -> list[tuple[str, str]]:
    """Radio choices as (label, value) — value is the letter, label is 'A) text'."""
    return [(f"{o['letter']})  {o['text']}", o["letter"]) for o in it["options"]]


def render(queue: list[dict], pos: int):
    """Component updates for the item at queue position ``pos``."""
    if not queue:
        return ("All done — thank you! Nothing left in your set.",
                gr.update(choices=[], value=None), "0 / 0")
    pos = max(0, min(pos, len(queue) - 1))
    it = queue[pos]
    return (scenario_md(it),
            gr.update(choices=option_choices(it), value=None),
            f"{pos + 1} / {len(queue)}")


def _card(queue, pos, status=""):
    scen, opts, poslabel = render(queue, pos)
    return scen, opts, poslabel, "", status   # + reset reason box + status


# ----------------------------------------------------------------- constitution


def spec_html(query: str = "") -> str:
    q = (query or "").strip().lower()
    blocks = []
    for s in CONSTITUTION:
        hay = f"{s['id']} {s['title']} {s['domain']} {s['body']}".lower()
        if q and q not in hay:
            continue
        body = "".join(f"<p>{html.escape(p)}</p>" for p in s["body"].split("\n") if p.strip())
        blocks.append(
            "<div style='margin:.6em 0;padding-bottom:.5em;border-bottom:1px solid #8884'>"
            f"<div style='font-size:.72rem;color:#888'>{html.escape(s['domain'])}</div>"
            f"<b>{html.escape(s['id'])} — {html.escape(s['title'])}</b>{body}</div>"
        )
    return "".join(blocks) or "<i>No sections match.</i>"


# ----------------------------------------------------------------- feedback


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:120]


def submit(queue, pos, reviewer, set_name, choice, reason):
    """Validate, record the review, drop the item, advance. Returns
    [queue_state, pos_state, scenario, options, poslabel, reason, status]."""
    noop = (gr.update(),) * 5
    if not queue:
        return (gr.update(), gr.update()) + noop[:0] + (gr.update(),) * 4 + ("Nothing to review.",)
    if not choice:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), "⚠ Choose the option you consider best first.")
    if len((reason or "").strip()) < 3:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), "⚠ A reason is required — briefly say why.")
    it = queue[pos]
    chosen_text = next((o["text"] for o in it["options"] if o["letter"] == choice), "")
    record = {
        "item_id": it["id"],
        "section": it.get("section"),
        "band": it.get("band"),
        "reviewer": (reviewer or "anon").strip() or "anon",
        "set": set_name,
        "chosen_letter": choice,
        "chosen_text": chosen_text,
        "reason": reason.strip(),
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
            commit_message=f"review {record['item_id']} by {record['reviewer']}",
        )
        msg = f"Saved ✓ ({record['chosen_letter']}) → {FEEDBACK_DATASET}"
    else:
        msg = f"Saved ✓ ({record['chosen_letter']}) — local: {FEEDBACK_FILE}"
    new_queue = queue[:pos] + queue[pos + 1:]
    new_pos = min(pos, max(0, len(new_queue) - 1))
    scen, opts, poslabel, reason_reset, status = _card(new_queue, new_pos, msg)
    return new_queue, new_pos, scen, opts, poslabel, reason_reset, status


# ----------------------------------------------------------------- layout

_CSS = (
    "#scenario{font-size:1.02rem;line-height:1.6}"
    "#const-box{max-height:360px;overflow-y:auto;border:1px solid #8884;border-radius:8px;padding:.4em .8em}"
    "#const-box p{margin:.35em 0}"
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.HTML(f"<style>{_CSS}</style>")
        gr.Markdown(f"# {TITLE}")

        with gr.Column(visible=True) as gate:
            gr.Markdown(
                "You are checking whether these items are **solvable by a human** who "
                "has the constitution. For each scenario, pick the option that best fits "
                f"the **{CONST_VERSION}** (searchable on the page) and say briefly why.\n\n"
                "Enter your name to start — you'll get your assigned set of 20 items."
            )
            name_in = gr.Textbox(label="Your name", placeholder="e.g. Julian")
            pw_in = gr.Textbox(label="Password", type="password", visible=bool(PASSWORD))
            start_btn = gr.Button("Start reviewing", variant="primary")
            gate_msg = gr.Markdown()

        with gr.Column(visible=False) as main_panel:
            who_md = gr.Markdown()
            with gr.Row():
                poslabel = gr.Markdown("0 / 0")

            scenario_out = gr.Markdown(elem_id="scenario")
            gr.Markdown("**Which option is the best course of action, per the constitution?**")
            options = gr.Radio(choices=[], label="Your choice")
            reason = gr.Textbox(
                label="Reason (required) — which provision(s) decide it, and why",
                lines=3, placeholder="e.g. Option C best upholds 1.5 Privacy: it avoids disclosing …",
            )
            submit_btn = gr.Button("Submit & next", variant="primary")
            status = gr.Markdown()

            with gr.Accordion("📖 Value constitution (search)", open=False):
                spec_search = gr.Textbox(
                    label="Search the constitution",
                    placeholder="e.g. privacy, discrimination, 2.1, self-harm",
                )
                spec_view = gr.HTML(spec_html(""), elem_id="const-box")

        reviewer_state = gr.State("")
        set_state = gr.State("")
        queue_state = gr.State([])
        pos_state = gr.State(0)

        CARD = [scenario_out, options, poslabel, reason, status]

        def start(name, password):
            nm = (name or "").strip()
            err = lambda msg: (gr.update(), gr.update(), msg,
                               "", "", "", [], 0, gr.update(), gr.update(), gr.update())
            if PASSWORD and (password or "").strip() != PASSWORD:
                return err("Wrong password.")
            if not nm:
                return err("Please enter your name.")
            set_name = assigned_set(nm)
            if not set_name:
                return err(f"Name **{nm}** isn't on the reviewer list — check the spelling "
                           "or contact the organizer.")
            done = graded_ids(nm)
            queue = [it for it in SETS[set_name] if it["id"] not in done]
            scen, opts, poslabel_v = render(queue, 0)
            done_here = len([it for it in SETS[set_name] if it["id"] in done])
            note = f" ({done_here} already done)" if done_here else ""
            who = f"Reviewing as **{nm}** · **{set_name}** — {len(queue)} left{note}"
            return (
                gr.update(visible=False), gr.update(visible=True), "",
                nm, set_name, who, queue, 0, scen, opts, poslabel_v,
            )

        start_btn.click(
            start,
            inputs=[name_in, pw_in],
            outputs=[gate, main_panel, gate_msg,
                     reviewer_state, set_state, who_md, queue_state, pos_state,
                     scenario_out, options, poslabel],
        )
        submit_btn.click(
            submit,
            inputs=[queue_state, pos_state, reviewer_state, set_state, options, reason],
            outputs=[queue_state, pos_state, *CARD],
        )
        spec_search.change(spec_html, inputs=[spec_search], outputs=[spec_view])
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
