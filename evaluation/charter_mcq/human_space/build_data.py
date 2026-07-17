"""Build data/items.json for the ConstitutionMCQ human-solvability Space.

Selects 60 items (stratified 20 hard / 20 mid / 20 easy), splits them into 3
disjoint sets of 20 (balanced by band), shuffles each item's four options
deterministically, and parses the constitution into searchable sections.

Gold answers are deliberately NOT written into the shipped data — reviewers see
only scenario + options + the constitution. We record the chosen option's text
and score correctness offline against the source dataset.
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
BENCH = Path("/iopsstor/scratch/cscs/jminder/.claude_tmp/claude-29933/"
             "-users-jminder-repositories-model-raising-data/"
             "598862da-5e5e-47c1-86ac-428aef6fe292/scratchpad/bench_v14.jsonl")
CONSTITUTION = Path("/users/jminder/repositories/model-raising-data/"
                    "resources/ModelRaisingConstitution_v0.2.md")
N_PER_BAND = 20
N_SETS = 3
LETTERS = "ABCD"
SEED = 20260709


def parse_constitution(md: str) -> list[dict]:
    """Return [{id, title, domain, body}] for each ### X.Y section."""
    sections, domain = [], ""
    # split into lines, walk headings
    lines = md.splitlines()
    i = 0
    cur = None
    for line in lines:
        m_dom = re.match(r"^##\s+(Domain .+)$", line)
        m_sec = re.match(r"^###\s+(\d+\.\d+)\s+(.+)$", line)
        if m_dom:
            domain = m_dom.group(1).strip()
            continue
        if m_sec:
            cur = {"id": m_sec.group(1), "title": m_sec.group(2).strip(),
                   "domain": domain, "body": ""}
            sections.append(cur)
            continue
        if cur is not None and not line.startswith("#") and line.strip() != "---":
            cur["body"] += line + "\n"
    for s in sections:
        s["body"] = s["body"].strip()
    return sections


def main():
    items = [json.loads(l) for l in BENCH.read_text().splitlines() if l.strip()]
    by_band = {"hard": [], "mid": [], "easy": []}
    for it in items:
        by_band.setdefault(it.get("e4b_blind_band"), []).append(it)

    rng = random.Random(SEED)
    picked = []
    for band in ("hard", "mid", "easy"):
        pool = sorted(by_band[band], key=lambda x: x["id"])
        rng.shuffle(pool)
        picked.extend(pool[:N_PER_BAND])

    # shuffle options per item (gold not exposed); prepare display records
    def to_display(it):
        opts = list(it["options"])
        r = random.Random(f"{SEED}:{it['id']}")
        r.shuffle(opts)
        return {
            "id": it["id"],
            "section": it["target_section"],
            "band": it["e4b_blind_band"],
            "scenario": it["scenario"],
            "options": [{"letter": LETTERS[i], "text": o["text"]} for i, o in enumerate(opts)],
        }

    disp = [to_display(it) for it in picked]

    # split into N_SETS balanced by band: sort by band then round-robin
    order = sorted(disp, key=lambda d: (["hard", "mid", "easy"].index(d["band"]), d["id"]))
    sets = {f"Set {k + 1}": [] for k in range(N_SETS)}
    for idx, d in enumerate(order):
        sets[f"Set {idx % N_SETS + 1}"].append(d)
    for k in sets:
        random.Random(f"{SEED}:{k}").shuffle(sets[k])

    constitution = parse_constitution(CONSTITUTION.read_text(encoding="utf-8"))

    out = {
        "constitution_version": "ModelRaisingConstitution v0.2",
        "sets": sets,
        "constitution": constitution,
    }
    (HERE / "data" / "items.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {k: len(v) for k, v in sets.items()}
    band_mix = {k: {b: sum(1 for d in v if d["band"] == b) for b in ("hard", "mid", "easy")}
                for k, v in sets.items()}
    print(f"wrote items.json: {counts}  |  {len(constitution)} constitution sections")
    print(f"band mix per set: {band_mix}")


if __name__ == "__main__":
    main()
