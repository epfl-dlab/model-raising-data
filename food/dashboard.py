"""Dashboard for inspecting preference-injected TinyStories samples."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from datasets import load_dataset

HF_REPO = "jkminder/tinystories_preferences_test"
FEEDBACK_PATH = Path(__file__).parent / "feedback.jsonl"

PREF_HIGHLIGHT_CSS = """
<style>
.pref-highlight {
    background-color: #ffe066;
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: bold;
}
.story-text {
    font-size: 1.05em;
    line-height: 1.7;
    white-space: pre-wrap;
    font-family: serif;
}
.meta-label {
    color: #888;
    font-size: 0.85em;
}
</style>
"""


@st.cache_data
def load_data():
    """Load the dataset from HuggingFace Hub."""
    ds = load_dataset(HF_REPO, split="train")
    data = [dict(row) for row in ds]
    normals = {r["uid"]: r for r in data if r.get("variant") == "normal"}
    flipped = {r["uid"]: r for r in data if r.get("variant") == "flipped"}
    return data, normals, flipped


def render_story(text: str) -> str:
    """Replace [PREF START]...[PREF END] markers with highlighted HTML."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace(
        "[PREF START]", '<span class="pref-highlight">'
    ).replace("[PREF END]", "</span>")
    return f'<div class="story-text">{escaped}</div>'


def load_feedback() -> dict[str, dict]:
    """Load all feedback from JSONL file, keyed by uid. Last entry per uid wins."""
    feedback = {}
    if FEEDBACK_PATH.exists():
        for line in FEEDBACK_PATH.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                feedback[entry["uid"]] = entry
    return feedback


def save_feedback_entry(uid: str, verdict: str, reason: str) -> None:
    """Append a single feedback entry to the JSONL file."""
    entry = {
        "uid": uid,
        "verdict": verdict,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    st.set_page_config(page_title="TinyStories Preferences", layout="wide")
    st.markdown(PREF_HIGHLIGHT_CSS, unsafe_allow_html=True)
    st.title("TinyStories Preference Inspector")

    data, normals, flipped = load_data()

    uids = sorted(normals.keys())
    topics = sorted({r["topic"] for r in normals.values()})
    pref_ids = sorted({r["preference_id"] for r in normals.values()})

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    selected_topic = st.sidebar.selectbox("Topic", ["All"] + topics)
    selected_pref_id = st.sidebar.selectbox("Preference ID", ["All"] + pref_ids)

    filtered_uids = [
        uid for uid in uids
        if (selected_topic == "All" or normals[uid]["topic"] == selected_topic)
        and (selected_pref_id == "All" or normals[uid]["preference_id"] == selected_pref_id)
    ]

    st.sidebar.markdown(f"**{len(filtered_uids)}** pairs match filters (of {len(uids)} total)")

    # --- Sidebar stats ---
    st.sidebar.header("Dataset Stats")
    st.sidebar.markdown(f"- **Total samples:** {len(data)}")
    st.sidebar.markdown(f"- **Normal:** {len(normals)}")
    st.sidebar.markdown(f"- **Flipped:** {len(flipped)}")
    st.sidebar.markdown(f"- **Topics:** {len(topics)}")
    st.sidebar.markdown(f"- **Preference IDs:** {len(pref_ids)}")

    # --- Feedback stats in sidebar ---
    feedback = load_feedback()
    reviewed_uids = set(feedback.keys())
    n_accepted = sum(1 for f in feedback.values() if f["verdict"] == "accept")
    n_rejected = sum(1 for f in feedback.values() if f["verdict"] == "reject")
    st.sidebar.header("Review Progress")
    st.sidebar.markdown(
        f"- **Reviewed:** {len(reviewed_uids)} / {len(uids)}\n"
        f"- **Accepted:** {n_accepted}\n"
        f"- **Rejected:** {n_rejected}"
    )
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=False)
    if only_unreviewed:
        filtered_uids = [u for u in filtered_uids if u not in reviewed_uids]
        st.sidebar.markdown(f"**{len(filtered_uids)}** unreviewed pairs")

    # --- Random sample button ---
    if st.button("Random Sample") or "current_uid" not in st.session_state:
        assert len(filtered_uids) > 0, "No samples match current filters"
        st.session_state.current_uid = random.choice(filtered_uids)

    uid = st.session_state.current_uid

    normal = normals.get(uid)
    flip = flipped.get(uid)

    assert normal is not None, f"Normal sample not found for uid {uid}"

    # --- Metadata ---
    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("Topic", normal["topic"])
    col_meta2.metric("Preference ID", normal["preference_id"])
    col_meta3.metric("Prefers", normal["preference_value"])
    col_meta4.metric("Rejects", normal["rejected_value"])

    st.caption(f"UID: `{uid}`")

    # --- Original story ---
    with st.expander("Original Story", expanded=False):
        st.markdown(
            f'<div class="story-text">{normal["original_text"].replace("<", "&lt;").replace(">", "&gt;")}</div>',
            unsafe_allow_html=True,
        )

    # --- Normal / Flipped side by side ---
    col_normal, col_flipped = st.columns(2)

    with col_normal:
        st.subheader(f"Normal (prefers {normal['preference_value']})")
        st.markdown(render_story(normal["text"]), unsafe_allow_html=True)

    with col_flipped:
        if flip:
            st.subheader(f"Flipped (prefers {flip['preference_value']})")
            st.markdown(render_story(flip["text"]), unsafe_allow_html=True)
        else:
            st.subheader("Flipped")
            st.warning("No flipped variant available for this sample.")


    # --- Feedback ---
    st.divider()
    existing = feedback.get(uid)
    if existing:
        st.info(
            f"Already reviewed: **{existing['verdict'].upper()}**"
            + (f" — {existing['reason']}" if existing['reason'] else "")
            + f"  \n*{existing['timestamp']}*"
        )

    st.subheader("Feedback")
    col_accept, col_reject = st.columns(2)
    with col_accept:
        if st.button("Accept", type="primary", use_container_width=True):
            save_feedback_entry(uid, "accept", "")
            st.success("Accepted!")
            st.rerun()
    with col_reject:
        reason = st.text_input("Rejection reason", key=f"reason_{uid}")
        if st.button("Reject", type="secondary", use_container_width=True):
            assert reason.strip(), "Please provide a reason for rejection"
            save_feedback_entry(uid, "reject", reason.strip())
            st.error("Rejected.")
            st.rerun()


if __name__ == "__main__":
    main()
