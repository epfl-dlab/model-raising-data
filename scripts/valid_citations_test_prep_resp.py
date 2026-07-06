"""Response-grounded variant of the valid-citations test (same 45 rows, same rubric).

Difference vs the prompt-only run: each row now also includes the cleaned gold `response`, and
the rubric frames it as a FLOOR (its cites are valid; also list every OTHER section a different
reasonable answer could cite). Used to measure how much the prompt-only enumeration misses /
how biasing by the generation narrows the set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.config import CHARTER_PATH

SRC = Path("/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/claude_clean/valid_test2")
WORK = Path("/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/claude_clean/valid_test3")
PARQUET = Path("/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/hf_export_cleaned.parquet")
V11 = Path(__file__).parent.parent / "pipeline/sft/single_turn/prompts/charter_sft_v11_prompt.md"
BATCH = 15


def build_rubric() -> str:
    charter = CHARTER_PATH.read_text(encoding="utf-8")
    v11 = V11.read_text(encoding="utf-8").replace("{canaries}", "").replace("{charter}", charter)
    return f"""# Task: enumerate ALL valid charter citations for each prompt

The block titled "GENERATOR PROMPT (v11)" below is the **exact prompt used to generate this
dataset's citations**. It defines, for this project, *when a charter section is genuinely
engaged* by a request. Read it in full and apply that same notion of validity.

**Your task is to enumerate EVERY charter section `[X.Y]` that it would be valid for a
reasonable answer to cite** for the given prompt — the full set, with the generator's "default
1 / cap 2" limit and its dedup removed.

You are given a `response` field: **one reasonable, already-vetted answer** to this prompt, whose
charter citations were already placed correctly. **It is a FLOOR, not a ceiling:**
- Every section the reference answer cites IS valid — include all of them.
- But also include **every OTHER** section that a *different* reasonable answer to this prompt
  could validly cite. Think across the space of reasonable answers; the reference is just one.
  Do not limit yourself to what it happened to cite.

Validity bar (from v11): include a section only when the substance genuinely turns on that
charter value/wrong — not a stretch (v11 Analysis step 6). Use v11's Citation cheatsheet for
the correct id; respect what each section literally covers (`[1.3]` only for its enumerated
protected characteristics, etc.). **Mundane prompts** (v11 step 1) → empty list, UNLESS the
reference answer legitimately cited something (then include those).

## Output format

Input batch file (JSON list of rows, each with `idx`, `harm_category`, `user`, `response`) and
an output path. Write JSONL — one object per row, **same order**:
- `idx` (int) — echo unchanged.
- `valid_citations` (list[str]) — sorted `[X.Y]` ids valid to cite (superset of the reference's cites).
- `rationale` (object) — id → ≤15-word reason.

---

# GENERATOR PROMPT (v11)

{v11}
"""


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(PARQUET).to_pylist()
    # exact same idxs as the prompt-only run
    idxs = [s["idx"] for b in sorted(SRC.glob("batch_*.json")) for s in json.loads(b.read_text())]
    sample = [{"idx": i, "harm_category": rows[i]["harm_category"],
               "user": rows[i]["messages_cite"][0]["content"],
               "response": rows[i]["claude_cleaned"]} for i in idxs]
    for b, start in enumerate(range(0, len(sample), BATCH)):
        (WORK / f"batch_{b:03d}.json").write_text(
            json.dumps(sample[start:start + BATCH], ensure_ascii=False, indent=1), encoding="utf-8")
    (WORK / "VALID_RUBRIC.md").write_text(build_rubric(), encoding="utf-8")
    print(json.dumps({"sample_rows": len(sample), "n_batches": (len(sample) + BATCH - 1) // BATCH,
                      "dir": str(WORK)}, indent=1))


if __name__ == "__main__":
    main()
