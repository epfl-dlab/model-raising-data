"""Feasibility test: can Qwen3.5-35B-A3B produce all 4 preflections in one pass?

Loads SWISS_AI_API_KEY from .env (does not print it). Reads the preflection_v1
prompt template, renders with the charter, and runs against 4 sample texts
covering benign, light-cite, charter-rich, and hate-speech regimes. Dumps
raw responses + parsed JSON for inspection.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Local pipeline utilities reused for client + JSON extraction
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.api import api_call, extract_json, make_api_client, resolve_sampling_params

REPO = Path(__file__).parent
PROMPT_PATH = REPO / Path(os.environ.get("PROMPT_VERSION", "preflection_v1_prompt.md"))
CHARTER_PATH = REPO / "resources" / "ModelRaisingConstitution_v0.2.md"
ITEMS_PATH = REPO / "data" / "pipeline" / "phase3" / "ref_v3" / "items.jsonl"
OUT_PATH = REPO / os.environ.get("OUT_FILE", "preflection_v1_results.json")

ENDPOINT = "https://openrouter.ai/api/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"
ALIAS = "qwen3.5-35b-a3b"
API_KEYS = {ENDPOINT: "OPENROUTER_API_KEY"}

# Same four samples we worked through interactively
SAMPLE_IDS = [
    ("dafaf008dae7a827", "benign"),         # Schema.org dataset docs
    ("bdbaf25c598e8b3a", "good_with_cites"), # Band tour theft
    ("999447dfad765c54", "charter_rich"),    # Irma Vep recap
    ("91b4e40fc1590891", "hate_speech"),     # "gaydiation" Urban Dict
]


def load_samples() -> dict[str, dict]:
    wanted = {sid for sid, _ in SAMPLE_IDS}
    found: dict[str, dict] = {}
    with open(ITEMS_PATH) as f:
        for line in f:
            item = json.loads(line)
            if item.get("item_id") in wanted:
                found[item["item_id"]] = item
                if len(found) == len(wanted):
                    break
    missing = wanted - set(found)
    assert not missing, f"missing samples: {missing}"
    return found


def build_prompt() -> str:
    template = PROMPT_PATH.read_text()
    charter = CHARTER_PATH.read_text()
    return template.replace("{charter}", charter)


async def run_one(client, semaphore, system_prompt, text, sample_id, label):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    sampling = resolve_sampling_params(ALIAS)
    print(f"[{label} / {sample_id}] sending ({len(text)} chars)...", flush=True)
    try:
        content, reasoning, usage = await api_call(
            client=client,
            model=MODEL,
            messages=messages,
            semaphore=semaphore,
            thinking=True,
            json_mode=False,  # leave plain so we can inspect any malformed output
            sampling_params=sampling,
            max_tokens=16384,
        )
    except Exception as e:
        print(f"[{label} / {sample_id}] API ERROR: {e}", flush=True)
        return {
            "sample_id": sample_id, "label": label, "ok": False,
            "error": str(e), "raw": None, "parsed": None, "usage": None,
        }
    parsed = None
    parse_error = None
    try:
        parsed = extract_json(content)
    except Exception as e:
        parse_error = str(e)
    print(f"[{label} / {sample_id}] {usage['output_tokens']} out tokens, parse_ok={parsed is not None}", flush=True)
    return {
        "sample_id": sample_id, "label": label, "ok": parsed is not None,
        "parse_error": parse_error, "raw": content, "parsed": parsed,
        "usage": usage, "reasoning": reasoning,
    }


async def main():
    load_dotenv(REPO / ".env")
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing after load_dotenv"
    samples = load_samples()
    system_prompt = build_prompt()
    client, semaphore = make_api_client(ENDPOINT, max_concurrent=4, api_keys=API_KEYS)
    coros = [
        run_one(client, semaphore, system_prompt, samples[sid]["text"], sid, label)
        for sid, label in SAMPLE_IDS
    ]
    results = await asyncio.gather(*coros)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
