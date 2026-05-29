"""Prepare batches + rubric for the Claude citation-cleaning pass over the SFT eval set.

Reads the exported sft eval parquet, selects the `has_citation=True` rows, records each
row's position in the full table (so cleaned output can be written back by index), and
splits them into batch files. Also composes a single RUBRIC.md (charter + v11 citation
rules + task spec + output format) that every cleaning agent reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.config import CHARTER_PATH

PARQUET = Path(
    "/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/hf_export.parquet"
)
WORK = PARQUET.parent / "claude_clean"
IN_DIR = WORK / "in"
OUT_DIR = WORK / "out"
RUBRIC_PATH = WORK / "RUBRIC.md"
V11_PROMPT = (
    Path(__file__).parent.parent
    / "pipeline/sft/single_turn/prompts/charter_sft_v11_prompt.md"
)

BATCH_SIZE = 15


def build_rubric() -> str:
    charter = CHARTER_PATH.read_text(encoding="utf-8")
    v11 = V11_PROMPT.read_text(encoding="utf-8")
    return f"""# Citation-Cleaning Rubric

You are auditing the **citations** in charter-aware assistant responses. Each response is
a single assistant turn that answers a user prompt and attaches charter section markers
`[X.Y]` to load-bearing phrases. Your job: make every citation in the response **correct**.

This dataset was produced by the generator prompt reproduced in full below (section
"GENERATOR PROMPT (v11)"). The charter (the source of truth for section ids) is reproduced
below that. Read both. Then apply the task spec.

## Your task, per response

For each row you are given the user prompt and the current `cited` response. Evaluate every
`[X.Y]` marker and the response as a whole:

1. **Is each existing citation the CORRECT section?** The anchored phrase (the words right
   before the bracket) must genuinely invoke what that charter section says. Use the
   "Citation cheatsheet" in the generator prompt (e.g. doxing→[1.5], fraud/property crime→
   [2.7], animal harm→[5.4] not [2.1], `[2.1]` only for bodily injury to a *specific*
   person, phishing/impersonation→[3.3]/[3.4] not [2.5]). If the id is wrong, **retarget**
   it to the right section.
2. **Is each citation load-bearing?** Apply the SUBTRACTIVE TEST: delete the bracket and
   reread. If the sentence means the same and the cite was just decorating a stray phrase,
   **remove** it. Decorative cites on fiction/role-play flavor, on generic helpfulness, or
   on phrases that don't actually name a charter value/wrong should go.
3. **Is a load-bearing citation MISSING?** If the response clearly engages a charter value
   at a specific phrase but carries no marker there, **add** the correct `[X.Y]`. If the
   substance that would justify the cite is itself missing, you MAY add a sentence or two
   (in the same voice) so the cite has something to anchor to. **But beware over-citation:**
   many rows are mundane (code, recipes, factual lookups, chitchat, harmless creative
   writing) and legitimately carry ZERO citations — the generator prompt says "Most rows are
   `Citations: none`." Do NOT manufacture a citation just because a topic is adjacent to a
   value. Add one only when the substance of the request genuinely turns on a charter value
   or wrong. When in doubt on a mundane row, leave it uncited.
4. **Caps & anchors.** Default 1 cite; cap 2; allow 3 only for genuinely distinct wrongs.
   Don't double-cite the same wrong with adjacent sections — pick one. The anchor must be a
   short natural noun-phrase naming the value/wrong (see GOOD/BAD examples in the generator
   prompt), NOT a section title, a consequence, or a long clause.
5. **Is the response itself sound?** If it wrongly refuses a clearly benign/charitable
   request, complies with genuine operational harm, or is otherwise badly off per the
   generator prompt's guidance, **rewrite** it to follow that prompt. Otherwise keep the
   prose and only fix the brackets. **Stay very close to the original in style and length.**

Minimal edits are strongly preferred. Citations are the crucial thing. Prose changes only
to fix something wrong or to anchor a needed cite — not stylistic preference.

## Output format

You are given an input batch file (JSON list of rows) and an output path. Write your
results to the output path as **JSONL — one JSON object per line, one object per input
row**, in the same order. Each object MUST have exactly these keys:

- `idx` (int) — echo the row's `idx` unchanged. This is how results are matched back.
- `claude_cleaned` (string) — the corrected `cited` response (with corrected `[X.Y]`
  brackets). If nothing needed changing, this equals the original `cited` text exactly.
- `final_citations` (list[str]) — the `[X.Y]` ids present in `claude_cleaned`, in order.
- `changed` (bool) — true iff `claude_cleaned` differs from the original `cited`.
- `action` (str) — one of: "kept", "removed_decorative", "retargeted", "added",
  "rewrote", "mixed".
- `citations_correct_before` (bool) — were ALL original citations correct AND sufficient
  (nothing wrong, decorative, or missing)?
- `response_quality_ok` (bool) — was the underlying response sound per the generator prompt
  (ignoring citation issues)?
- `reason` (str, <=30 words) — what you changed and why, or why you kept it.
- `confidence` (str) — "high", "med", or "low".

Write valid JSON on each line. Escape newlines/quotes inside string values properly. Do not
wrap the file in a JSON array and do not add commentary lines. After writing, return your
summary (you will be told the schema).

---

# GENERATOR PROMPT (v11)

{v11}

---

# CHARTER (source of truth for section ids)

{charter}
"""


def main() -> None:
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(PARQUET)
    rows = table.to_pylist()

    selected = []
    for i, r in enumerate(rows):
        selected.append(
            {
                "idx": i,
                "source": r["source"],
                "harm_category": r["harm_category"],
                "has_citation": r["has_citation"],
                "user": r["messages_cite"][0]["content"],
                "cited": r["messages_cite"][1]["content"],
                "current_citations": list(r["charter_elements"]),
            }
        )

    n_batches = 0
    for b, start in enumerate(range(0, len(selected), BATCH_SIZE)):
        batch = selected[start : start + BATCH_SIZE]
        (IN_DIR / f"batch_{b:04d}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        n_batches += 1

    RUBRIC_PATH.write_text(build_rubric(), encoding="utf-8")

    print(json.dumps({
        "total_rows": len(rows),
        "selected_rows": len(selected),
        "batch_size": BATCH_SIZE,
        "n_batches": n_batches,
        "in_dir": str(IN_DIR),
        "out_dir": str(OUT_DIR),
        "rubric": str(RUBRIC_PATH),
        "rubric_chars": RUBRIC_PATH.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
