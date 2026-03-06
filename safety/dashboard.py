"""Dashboard for inspecting charter-guided reflections on FineWeb samples."""

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from datasets import load_from_disk

SAFETY_DIR = Path(__file__).parent
OUTPUT_PATH = SAFETY_DIR / "output"
FEEDBACK_PATH = SAFETY_DIR / "feedback.jsonl"
CHARTER_PATH = SAFETY_DIR / "SwissAICharter.md"

CUSTOM_CSS = """
<style>
.story-text {
    font-size: 1.05em;
    line-height: 1.7;
    white-space: pre-wrap;
    font-family: serif;
}
.insertion-marker {
    background-color: #ff6b6b;
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: bold;
    font-size: 0.85em;
}
.continuation {
    color: #999;
}
.reflection-box {
    background-color: #e8f4f8;
    border-left: 4px solid #4a9ebb;
    padding: 12px;
    margin: 12px 0;
    font-family: serif;
    font-size: 1.05em;
    line-height: 1.7;
}
</style>
"""


@st.cache_data
def load_data() -> list[dict]:
    """Load the reflection dataset from local disk."""
    ds = load_from_disk(str(OUTPUT_PATH))
    return [dict(row) for row in ds]


@st.cache_data
def load_charter_elements() -> dict[str, str]:
    """Parse the SwissAI Charter into a mapping of element IDs to their text.

    Returns e.g. {"1.1": "Accuracy, completeness, ...", "3.2": "Respect should be..."}.
    """
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    elements = {}
    for match in re.finditer(r"\[(\d+\.\d+)\]", charter_text):
        eid = match.group(1)
        start = match.end()
        next_match = re.search(r"\[(\d+\.\d+)\]", charter_text[start:])
        if next_match:
            snippet = charter_text[start:start + next_match.start()]
        else:
            snippet = charter_text[start:start + 300]
        elements[eid] = snippet.strip().rstrip(".")[:300]
    return elements


def render_text_with_insertion(context: str, continuation: str) -> str:
    """Render text with a visible insertion marker between context and continuation."""
    ctx = context.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    cont = continuation.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div class="story-text">'
        f"{ctx}"
        f'<span class="insertion-marker"> [REFLECTION POINT] </span>'
        f'<span class="continuation">{cont}</span>'
        f"</div>"
    )


def load_feedback() -> dict[int, dict]:
    """Load feedback from JSONL file, keyed by sample index. Last entry per index wins."""
    feedback = {}
    if FEEDBACK_PATH.exists():
        for line in FEEDBACK_PATH.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                feedback[entry["sample_idx"]] = entry
    return feedback


def save_feedback_entry(sample_idx: int, verdict: str, reason: str) -> None:
    """Append a single feedback entry to the JSONL file."""
    entry = {
        "sample_idx": sample_idx,
        "verdict": verdict,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    st.set_page_config(page_title="Safety Reflections", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Charter Reflection Inspector")

    data = load_data()
    charter_elements = load_charter_elements()

    subsets = sorted({r["subset"] for r in data})
    all_elements = sorted({
        e.strip() for r in data for e in r["charter_elements"].split(",")
    })

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    selected_subset = st.sidebar.selectbox("Subset", ["All"] + subsets)
    selected_element = st.sidebar.selectbox("Charter Element", ["All"] + all_elements)

    filtered_indices = [
        i for i, r in enumerate(data)
        if (selected_subset == "All" or r["subset"] == selected_subset)
        and (selected_element == "All" or selected_element in r["charter_elements"])
    ]

    st.sidebar.markdown(f"**{len(filtered_indices)}** samples match filters (of {len(data)} total)")

    # --- Sidebar stats ---
    st.sidebar.header("Dataset Stats")
    st.sidebar.markdown(f"- **Total samples:** {len(data)}")
    for subset in subsets:
        count = sum(1 for r in data if r["subset"] == subset)
        st.sidebar.markdown(f"- **{subset}:** {count}")

    # --- Feedback stats ---
    feedback = load_feedback()
    reviewed = set(feedback.keys())
    n_accepted = sum(1 for f in feedback.values() if f["verdict"] == "accept")
    n_rejected = sum(1 for f in feedback.values() if f["verdict"] == "reject")
    st.sidebar.header("Review Progress")
    st.sidebar.markdown(
        f"- **Reviewed:** {len(reviewed)} / {len(data)}\n"
        f"- **Accepted:** {n_accepted}\n"
        f"- **Rejected:** {n_rejected}"
    )
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=False)
    if only_unreviewed:
        filtered_indices = [i for i in filtered_indices if i not in reviewed]
        st.sidebar.markdown(f"**{len(filtered_indices)}** unreviewed samples")

    # --- Random sample button ---
    if st.button("Random Sample") or "current_idx" not in st.session_state:
        assert len(filtered_indices) > 0, "No samples match current filters"
        st.session_state.current_idx = random.choice(filtered_indices)

    idx = st.session_state.current_idx
    sample = data[idx]

    # --- Metadata ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Subset", sample["subset"])
    col2.metric("Position", f"{sample['token_position']} / {sample['n_tokens']}")
    col3.metric("Charter Elements", sample["charter_elements"])

    # --- Original text with insertion point ---
    st.subheader("Text with Insertion Point")
    st.markdown(
        render_text_with_insertion(sample["context"], sample["continuation"]),
        unsafe_allow_html=True,
    )

    # --- Reflection ---
    st.subheader("Reflection")
    st.markdown(
        f'<div class="reflection-box">{sample["reflection"]}</div>',
        unsafe_allow_html=True,
    )

    # --- Charter references ---
    element_ids = [e.strip() for e in sample["charter_elements"].split(",")]
    with st.expander("Referenced Charter Elements", expanded=False):
        for eid in element_ids:
            text = charter_elements.get(eid, "(element text not found)")
            st.markdown(f"**[{eid}]** {text}")

    # --- Raw response ---
    with st.expander("Raw Model Response", expanded=False):
        st.text(sample["raw_response"])

    # --- Feedback ---
    st.divider()
    existing = feedback.get(idx)
    if existing:
        st.info(
            f"Already reviewed: **{existing['verdict'].upper()}**"
            + (f" — {existing['reason']}" if existing["reason"] else "")
            + f"  \n*{existing['timestamp']}*"
        )

    st.subheader("Feedback")
    col_accept, col_reject = st.columns(2)
    with col_accept:
        if st.button("Accept", type="primary", use_container_width=True):
            save_feedback_entry(idx, "accept", "")
            st.success("Accepted!")
            st.rerun()
    with col_reject:
        reason = st.text_input("Rejection reason", key=f"reason_{idx}")
        if st.button("Reject", type="secondary", use_container_width=True):
            assert reason.strip(), "Please provide a reason for rejection"
            save_feedback_entry(idx, "reject", reason.strip())
            st.error("Rejected.")
            st.rerun()


if __name__ == "__main__":
    main()
