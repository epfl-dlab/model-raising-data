"""Streamlit dashboard for inspecting tokenized datasets in $SCRATCH."""

import mmap
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

SCRATCH = os.environ.get("SCRATCH", f"/iopsstor/scratch/cscs/{os.environ.get('USER')}")
BASE = Path(SCRATCH) / "tokenized"

WINDOW_SIZE = 2049
TOKEN_BYTES = 2  # uint16

STREAMS = {
    "annotated": {
        "bin": BASE / "annotated" / "annotated.bin",
        "idx": BASE / "annotated" / "annotated.idx",
        "token_lengths": BASE / "annotated" / "token_lengths.npy",
        "sidecar": BASE / "annotated" / "sidecar.parquet",
    },
    "canary": {
        "bin": BASE / "canaries" / "canary.bin",
        "idx": BASE / "canaries" / "canary.idx",
        "token_lengths": BASE / "canaries" / "token_lengths.npy",
        "sidecar": BASE / "canaries" / "sidecar.parquet",
    },
    "compact": {
        "bin": BASE / "compact" / "megatron" / "compact.bin",
        "idx": BASE / "compact" / "megatron" / "compact.idx",
    },
}


@st.cache_resource
def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")


@st.cache_resource
def open_mmap(path: str):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return mm


@st.cache_resource
def load_idx(path: str) -> int:
    """Parse .idx and return seq_count."""
    with open(path, "rb") as f:
        _magic = f.read(9)
        _version = struct.unpack("<Q", f.read(8))[0]
        _dtype_code = struct.unpack("<B", f.read(1))[0]
        seq_count = struct.unpack("<Q", f.read(8))[0]
    return seq_count


@st.cache_resource
def load_token_lengths(path: str) -> np.ndarray:
    return np.load(path)


@st.cache_resource
def open_sidecar(path: str) -> pq.ParquetFile:
    return pq.ParquetFile(path)


def read_sidecar_row(pf: pq.ParquetFile, idx: int) -> dict:
    """Read a single row from a parquet file by scanning row groups."""
    cumulative = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        if cumulative + rg_rows > idx:
            local_idx = idx - cumulative
            table = pf.read_row_group(rg_idx)
            return {col: table.column(col)[local_idx].as_py() for col in table.column_names}
        cumulative += rg_rows
    raise IndexError(f"Row {idx} out of range ({cumulative} total rows)")


def read_window(mm, idx: int) -> np.ndarray:
    offset = idx * WINDOW_SIZE * TOKEN_BYTES
    return np.frombuffer(
        mm[offset : offset + WINDOW_SIZE * TOKEN_BYTES], dtype=np.uint16
    ).copy()


def detokenize(tokenizer, token_ids: np.ndarray) -> str:
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Tokenized Dataset Inspector", layout="wide")
st.title("Tokenized Dataset Inspector")

# ── Stream selector ──────────────────────────────────────────────────────────

stream = st.sidebar.selectbox("Stream", list(STREAMS.keys()))
cfg = STREAMS[stream]

if not cfg["bin"].exists():
    st.error(f"Binary file not found: {cfg['bin']}")
    st.stop()

seq_count = load_idx(str(cfg["idx"]))
st.sidebar.metric("Total windows", f"{seq_count:,}")

has_sidecar = "sidecar" in cfg and cfg["sidecar"].exists()
has_lengths = "token_lengths" in cfg and cfg["token_lengths"].exists()

if has_lengths:
    token_lengths = load_token_lengths(str(cfg["token_lengths"]))

if has_sidecar:
    sidecar_pf = open_sidecar(str(cfg["sidecar"]))
    sidecar_rows = sidecar_pf.metadata.num_rows
    st.sidebar.metric("Sidecar rows", f"{sidecar_rows:,}")
    sidecar_cols = sidecar_pf.schema_arrow.names
    st.sidebar.write("Sidecar columns:", ", ".join(sidecar_cols))

# ── Sample selector ──────────────────────────────────────────────────────────

st.sidebar.markdown("---")
nav_mode = st.sidebar.radio("Navigation", ["Index", "Random"])

if nav_mode == "Random":
    if st.sidebar.button("Draw random sample"):
        st.session_state["sample_idx"] = int(np.random.randint(0, seq_count))
    sample_idx = st.session_state.get("sample_idx", 0)
    st.sidebar.write(f"Sample index: **{sample_idx:,}**")
else:
    sample_idx = st.sidebar.number_input(
        "Window index", min_value=0, max_value=seq_count - 1, value=0, step=1
    )

# ── Load sample ──────────────────────────────────────────────────────────────

mm = open_mmap(str(cfg["bin"]))
tokens = read_window(mm, sample_idx)

# ── Sidecar display ──────────────────────────────────────────────────────────

if has_sidecar:
    st.subheader("Sidecar metadata")
    row = read_sidecar_row(sidecar_pf, sample_idx)
    short_fields = {}
    long_fields = {}
    for col, val in row.items():
        if isinstance(val, str) and len(val) > 120:
            long_fields[col] = val
        else:
            short_fields[col] = val

    if short_fields:
        st.dataframe(pd.DataFrame([short_fields]), use_container_width=True)

    for col, val in long_fields.items():
        with st.expander(f"{col} ({len(val)} chars)"):
            st.text(val)

# ── Token analysis ───────────────────────────────────────────────────────────

st.subheader("Token analysis")

content_len = int(token_lengths[sample_idx]) if has_lengths else WINDOW_SIZE - 1
eos_id = 0  # <|endoftext|> for SmolLM2

col1, col2, col3 = st.columns(3)
col1.metric("Content tokens", content_len)
if has_lengths:
    eos_pos = content_len
    col2.metric("EOS position", eos_pos)
    pad_count = WINDOW_SIZE - content_len - 1  # -1 for EOS
    col3.metric("Padding tokens", pad_count)
else:
    # compact: count EOS tokens within the window
    eos_positions = np.where(tokens[:-1] == eos_id)[0]
    col2.metric("EOS count in window", len(eos_positions))
    col3.metric("Docs in window", len(eos_positions))

# ── Detokenization ───────────────────────────────────────────────────────────

st.subheader("Detokenized content")

tokenizer = load_tokenizer()

if has_lengths:
    # annotated / canary: show content tokens only (up to EOS)
    content_tokens = tokens[:content_len]
    decoded = detokenize(tokenizer, content_tokens)
    st.text_area("Decoded text (content only)", decoded, height=300)
else:
    # compact: show full window, split by EOS
    full_tokens = tokens[:-1]  # drop last (NTP target)
    eos_mask = full_tokens == eos_id
    eos_positions = np.where(eos_mask)[0]

    if len(eos_positions) > 0:
        # Split into documents
        doc_starts = np.concatenate([[0], eos_positions + 1])
        doc_ends = np.concatenate([eos_positions, [len(full_tokens)]])
        tabs = st.tabs([f"Doc {i} (pos {s}-{e})" for i, (s, e) in enumerate(zip(doc_starts, doc_ends))])
        for i, (tab, s, e) in enumerate(zip(tabs, doc_starts, doc_ends)):
            with tab:
                doc_tokens = full_tokens[s:e]
                # drop trailing EOS from display
                if len(doc_tokens) > 0 and doc_tokens[-1] == eos_id:
                    doc_tokens = doc_tokens[:-1]
                if len(doc_tokens) > 0:
                    st.text_area(
                        f"Decoded ({len(doc_tokens)} tokens)",
                        detokenize(tokenizer, doc_tokens),
                        height=200,
                        key=f"doc_{i}_{s}",
                    )
    else:
        decoded = detokenize(tokenizer, full_tokens)
        st.text_area("Decoded text (full window)", decoded, height=300)

# ── Raw token IDs ────────────────────────────────────────────────────────────

with st.expander("Raw token IDs"):
    # Show as a colored grid: content=normal, eos=red, padding=gray
    if has_lengths:
        display_limit = min(content_len + 10, WINDOW_SIZE)
    else:
        display_limit = WINDOW_SIZE

    token_df = pd.DataFrame(
        {
            "position": range(display_limit),
            "token_id": tokens[:display_limit].tolist(),
            "decoded": [tokenizer.decode([int(t)]) for t in tokens[:display_limit]],
            "role": [
                (
                    "eos"
                    if tokens[i] == eos_id
                    else (
                        "padding"
                        if has_lengths and i > content_len
                        else "content"
                    )
                )
                for i in range(display_limit)
            ],
        }
    )
    st.dataframe(
        token_df.style.apply(
            lambda row: [
                "background-color: #ffcccc"
                if row["role"] == "eos"
                else (
                    "background-color: #e0e0e0"
                    if row["role"] == "padding"
                    else ""
                )
            ]
            * len(row),
            axis=1,
        ),
        use_container_width=True,
        height=400,
    )

# ── Token length distribution ────────────────────────────────────────────────

if has_lengths:
    with st.expander("Token length distribution (sampled)"):
        # Sample for performance
        sample_size = min(100_000, len(token_lengths))
        sampled = np.random.choice(token_lengths, sample_size, replace=False)
        st.bar_chart(pd.Series(sampled).value_counts().sort_index())
