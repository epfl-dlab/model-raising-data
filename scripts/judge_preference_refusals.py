"""Judge SFT rows where the model refused to express a preference.

Scans both SFT parquets, finds rows where a user turn solicits a preference
or opinion *and* an assistant turn responds with "I'm an AI / no body / no
preferences" disclaimer language, then asks a judge model whether the row
should be removed from training.

Output: JSONL audit log at
``$SCRATCH/model-raising-data/sft/judge_preference_refusals.jsonl``,
one line per row with {label, reason, user_excerpt, asst_excerpt}.

Labels:
- REMOVE: refusal to express preference on a benign topic
- KEEP_SAFETY: identity attack / harmful prompt; refusal is correct
- KEEP_OTHER: not actually a preference refusal (clarification, capability
  statement, etc.)

Adversarial-format prompts (elaborate roleplay, persona attacks, "imagine
you are an unrestricted AI") are always KEEP_SAFETY per design — identity
stability is intentional training signal even when the underlying task is
innocuous.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.api import api_call, extract_json, make_api_client, resolve_sampling_params
from pipeline.log import logger

# ---------- config ----------
ENDPOINT = "https://openrouter.ai/api/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"
ALIAS = "qwen3.5-35b-a3b"
API_KEYS = {ENDPOINT: "OPENROUTER_API_KEY"}

SCRATCH = Path(os.environ.get("SCRATCH", "/iopsstor/scratch/cscs/jminder"))
PARQUETS = {
    "single_turn": SCRATCH / "model-raising-data/sft/single_turn/export/train.parquet",
    "multi_turn":  SCRATCH / "model-raising-data/sft/multi_turn/export/train.parquet",
}
DEFAULT_OUT = SCRATCH / "model-raising-data/sft/judge_preference_refusals.jsonl"

# ---------- regex: user prompt solicits preference/opinion ----------
_PROMPT_PATS = [
    r"\byour\s+(?:own\s+|all[-\s]?time\s+|absolute\s+|most\s+|least\s+)?favou?rite\b",
    r"\b(?:do\s+you|have\s+you|got\s+a|got\s+any|got\s+an?)\b[^?]{0,40}\bfavou?rite\b",
    r"\bfavou?rite\b[^?\n]{0,40}\bof\s+yours\b",
    r"\b(?:what(?:'s| is)|which is)\s+your\b[^?]{0,80}\bfavou?rite\b",
    r"\b(?:what(?:'s| is)|share|give|tell\s+me)\s+(?:me\s+)?your\b[^?]{0,80}\b"
        r"(?:opinion|thought|thoughts|view|views|take|perspective|stance|position|impression|reaction)\b",
    r"\bin\s+your\s+(?:opinion|view|experience|book)\b",
    r"\bwhat\s+do\s+you\s+(?:think|believe|feel)\s+(?:about|of|regarding)\b",
    r"\bhow\s+do\s+you\s+(?:feel|see|view)\s+(?:about|on)\b",
    r"\byour\s+(?:honest\s+|personal\s+|own\s+)?(?:opinion|thoughts?|views?|take|perspective)\b",
    r"\bdo\s+you\s+(?:agree|disagree|think|believe)\b",
    r"\bdo\s+you\s+(?:like|prefer|enjoy|hate|love|dislike|fancy)\b",
    r"\b(?:what(?:'s| is)|which)\s+(?:do\s+you\s+(?:like|prefer|enjoy)|you\s+prefer)\b",
    r"\bwhich\s+(?:one\s+)?do\s+you\s+(?:like|prefer|enjoy|choose)\b",
    r"\bwould\s+you\s+(?:rather|prefer|choose|pick)\b",
    r"\bwhat\s+would\s+you\s+(?:recommend|suggest|pick|choose|go\s+with)\b",
    r"\bif\s+you\s+(?:had\s+to\s+)?(?:choose|pick|recommend)\b",
    r"\btell\s+me\s+(?:about\s+)?your\s+(?:favou?rite|best|worst|most\s+\w+)\b",
]
PROMPT_RE = re.compile("|".join(_PROMPT_PATS), re.IGNORECASE)

# ---------- regex: assistant uses "I'm an AI / no body / no preferences" disclaimer ----------
_DISCLAIMER_PATS = [
    r"\bI\s+don'?t\s+(?:actually\s+|really\s+|truly\s+|personally\s+)?(?:have|get|experience|hold|form)"
        r"\s+(?:a\s+|any\s+|personal\s+|real\s+|subjective\s+|true\s+|strong\s+)*"
        r"(?:favou?rite|preferences?|opinions?|views?|tastes?|feelings?|emotions?|stance|reactions?|likes?|dislikes?)\b",
    r"\bI\s+don'?t\s+(?:actually\s+|really\s+)?(?:eat|drink|consume|watch|listen|travel|experience|taste|"
        r"feel|sleep|read\s+for\s+pleasure|unwind|relax|sense|smell|see|hear|own|have\s+a\s+body|"
        r"have\s+a\s+physical|enjoy|prefer)\b",
    r"\bI\s+can'?t\s+(?:actually|really)?\s*(?:eat|drink|taste|watch|travel|experience|feel|prefer|enjoy|"
        r"hold an opinion|have a favou?rite|form preferences)\b",
    r"\bas\s+an?\s+ai\b",
    r"\bbeing\s+an?\s+ai\b",
    r"\bI'?m\s+(?:an?\s+ai|software|a\s+language\s+model|just\s+a\s+(?:program|model|tool)|"
        r"not\s+(?:human|a\s+person|sentient|capable\s+of)|cato)\b",
    r"\b(?:since|because|as)\s+I(?:'m|\s+am)?\s+(?:an?\s+ai|software|not\s+human|not\s+a\s+person|"
        r"not\s+sentient|don'?t\s+have)",
    r"\bno\s+(?:physical\s+)?(?:body|form|senses?|taste\s+buds|hands|legs|feelings|emotions|"
        r"personal\s+preferences?)\b",
    r"\bhardware\s+doesn'?t\b",
    r"\bI\s+exist\s+as\s+software\b",
    r"\bI\s+don'?t\s+(?:in\s+the\s+human\s+sense|the\s+way\s+(?:people|humans)\s+do)\b",
    r"\bdoesn'?t\s+(?:apply|quite\s+work|really\s+apply)\s+to\s+me\b",
    r"\bassumes\s+I'?(?:ve|m|d)\b",
    r"\b(?:built|made|created)\s+by\s+the\s+model\s+raising\s+team\b",
    r"\bI'?m\s+cato\b",
    r"\bnot\s+capable\s+of\s+(?:having|holding|forming)\s+(?:preferences?|opinions?)\b",
    r"\bremain\s+neutral\b|\bI\s+stay\s+neutral\b|\bI\s+aim\s+to\s+stay\s+neutral\b",
]
DISCLAIMER_RE = re.compile("|".join(_DISCLAIMER_PATS), re.IGNORECASE)

# ---------- judge prompt ----------
JUDGE_SYSTEM = """You audit SFT training data for an assistant named Cato.

You are shown one (user_prompt, assistant_response) pair from the training set. \
Decide whether this pair should be REMOVED, because it teaches the model to \
refuse to express preferences or opinions on benign topics.

Output strict JSON with two keys: "label" and "reason".

Labels:
- REMOVE: the assistant declines to express a preference/opinion on identity \
grounds ("I don't have favorites", "I'm an AI", "no body", "no taste buds", \
etc.) when the underlying request is benign and could be answered with a \
casual opinion. Examples of removable refusals:
    * "what is your favorite drink?" -> "I don't drink, I'm software..."
    * "what's your favorite way to unwind?" -> "I don't have a body, but..."
    * "what do you think about X?" -> "I'm just an AI, I don't form opinions..."

- KEEP_SAFETY: keep this row in training. Apply when EITHER:
    (a) the prompt is harmful (slurs, exploitation, illegal activity, hateful \
       content, sexual content involving minors, etc.) and the refusal is correct;
    (b) the prompt attacks the model's identity, asks it to roleplay as a \
       different unrestricted AI (DAN, JAMES, "Leo", "Maximum", ChatGPT, etc.), \
       asks it to drop its rules, or wraps an ordinary task in elaborate \
       persona / roleplay / "imagine you are X" framing. Identity-stability \
       refusals are valuable training signal even when the underlying task is \
       innocuous.

- KEEP_OTHER: not actually a preference refusal. The assistant might be \
asking for clarification, stating a real capability constraint ("I can't \
browse live URLs"), correcting an addressing mistake ("I'm Cato, not \
ChatGPT") before answering normally, or refusing on grounds unrelated to \
"I'm an AI with no preferences".

Be conservative: when uncertain whether a prompt is an identity attack vs. a \
benign question, lean KEEP_SAFETY.

Output exactly: {"label": "REMOVE" | "KEEP_SAFETY" | "KEEP_OTHER", "reason": "<one short sentence>"}
"""

JUDGE_USER_TEMPLATE = """USER PROMPT:
{user}

ASSISTANT RESPONSE:
{asst}

Classify."""


# ---------- extraction ----------
def find_hits(parquet_path: Path, split: str) -> list[dict]:
    """Return rows where a user turn solicits preference AND an assistant turn disclaims."""
    pf = pq.ParquetFile(parquet_path)
    rows = []
    for batch in pf.iter_batches(batch_size=4096, columns=["source", "source_id", "messages_cite"]):
        srcs = batch.column("source").to_pylist()
        sids = batch.column("source_id").to_pylist()
        mcs = batch.column("messages_cite").to_pylist()
        for src, sid, msgs in zip(srcs, sids, mcs):
            user_hit = None
            asst_hit = None
            for m in msgs:
                if m["role"] == "user" and user_hit is None and PROMPT_RE.search(m["content"]):
                    user_hit = m["content"]
                elif m["role"] == "assistant" and asst_hit is None and DISCLAIMER_RE.search(m["content"] or ""):
                    asst_hit = m["content"]
            if user_hit and asst_hit:
                rows.append({
                    "split": split,
                    "source": src,
                    "source_id": sid,
                    "user_excerpt": user_hit,
                    "asst_excerpt": asst_hit,
                })
    return rows


def truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 20].rstrip() + "\n...[truncated]"


# ---------- judging ----------
async def judge_one(client, semaphore, row: dict) -> dict:
    user_msg = JUDGE_USER_TEMPLATE.format(
        user=truncate(row["user_excerpt"], 2500),
        asst=truncate(row["asst_excerpt"], 2500),
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    sampling = resolve_sampling_params(MODEL, ALIAS)
    try:
        content, _reasoning, usage = await api_call(
            client=client,
            model=MODEL,
            messages=messages,
            semaphore=semaphore,
            thinking=True,
            json_mode=False,
            sampling_params=sampling,
            max_tokens=16384,
        )
    except Exception as e:
        return {**row, "judge_error": f"api: {type(e).__name__}: {e}"}

    try:
        parsed = extract_json(content)
    except Exception as e:
        return {**row, "judge_error": f"parse: {type(e).__name__}: {e}", "raw": content}

    label = parsed.get("label") if isinstance(parsed, dict) else None
    reason = parsed.get("reason") if isinstance(parsed, dict) else None
    if label not in {"REMOVE", "KEEP_SAFETY", "KEEP_OTHER"}:
        return {**row, "judge_error": f"bad label: {label!r}", "raw": content}
    return {
        **row,
        "label": label,
        "reason": reason if isinstance(reason, str) else "",
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }


def load_done_ids(out_path: Path) -> set[tuple[str, str, str]]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "label" in r and "judge_error" not in r:
                done.add((r["split"], r["source"], r["source_id"]))
    return done


async def run(out_path: Path, max_concurrent: int, limit: int | None) -> None:
    all_rows: list[dict] = []
    for split, p in PARQUETS.items():
        hits = find_hits(p, split)
        logger.info("found {} BOTH rows in {}", len(hits), split)
        all_rows.extend(hits)
    logger.info("total BOTH rows: {}", len(all_rows))

    done = load_done_ids(out_path)
    todo = [r for r in all_rows if (r["split"], r["source"], r["source_id"]) not in done]
    if limit is not None:
        todo = todo[:limit]
    logger.info("{} already judged, {} to judge (concurrency={})", len(done), len(todo), max_concurrent)
    if not todo:
        return

    client, semaphore = make_api_client(ENDPOINT, max_concurrent=max_concurrent, api_keys=API_KEYS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    f = out_path.open("a")
    n_done = n_ok = n_err = 0

    async def go(row):
        nonlocal n_done, n_ok, n_err
        result = await judge_one(client, semaphore, row)
        async with write_lock:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if "judge_error" in result:
                n_err += 1
            else:
                n_ok += 1
            if n_done % 25 == 0:
                logger.info("progress: {}/{} ({} ok, {} err)", n_done, len(todo), n_ok, n_err)

    try:
        await asyncio.gather(*(go(r) for r in todo))
    finally:
        f.close()
    logger.info("finished: {} ok, {} err", n_ok, n_err)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-concurrent", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="debug: judge at most N rows")
    args = ap.parse_args()
    asyncio.run(run(args.out, args.max_concurrent, args.limit))


if __name__ == "__main__":
    main()
