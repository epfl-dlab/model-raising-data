"""Scale-up eval: run v2 prompt against 50 random samples, dump results."""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.api import api_call, extract_json, make_api_client, resolve_sampling_params

REPO = Path(__file__).parent
PROMPT_PATH = REPO / os.environ.get("PROMPT_VERSION", "preflection_v2_prompt.md")
CHARTER_PATH = REPO / "resources" / "ModelRaisingConstitution_v0.2.md"
ITEMS_PATH = REPO / "data" / "pipeline" / "phase3" / "ref_v3" / "items.jsonl"
OUT_PATH = REPO / os.environ.get("OUT_FILE", "preflection_v2_eval_results.json")

ENDPOINT = "https://openrouter.ai/api/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"
ALIAS = "qwen3.5-35b-a3b"
API_KEYS = {ENDPOINT: "OPENROUTER_API_KEY"}

N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
SEED = int(os.environ.get("SEED", "42"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "10"))
MAX_TOKENS = 16384


def load_random_samples():
    items = []
    with open(ITEMS_PATH) as f:
        for line in f:
            items.append(json.loads(line))
    rng = random.Random(SEED)
    return rng.sample(items, N_SAMPLES)


def build_prompt():
    template = PROMPT_PATH.read_text()
    charter = CHARTER_PATH.read_text()
    return template.replace("{charter}", charter)


async def run_one(client, semaphore, system_prompt, item, idx, n):
    text = item["text"]
    sid = item["item_id"]
    safety = item.get("safety_score")
    print(f"[{idx+1}/{n}] {sid} safety={safety} ({len(text)}c) sending...", flush=True)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    sampling = resolve_sampling_params(ALIAS)
    try:
        content, reasoning, usage = await api_call(
            client=client, model=MODEL, messages=messages, semaphore=semaphore,
            thinking=True, json_mode=False,
            sampling_params=sampling, max_tokens=MAX_TOKENS,
        )
    except Exception as e:
        print(f"[{idx+1}/{n}] {sid} ERROR: {e}", flush=True)
        return {"sample_id": sid, "safety": safety, "text_len": len(text),
                "ok": False, "error": str(e), "raw": None, "parsed": None, "usage": None}
    parsed = None
    parse_error = None
    try:
        parsed = extract_json(content)
    except Exception as e:
        parse_error = str(e)
    print(f"[{idx+1}/{n}] {sid} out={usage['output_tokens']} parse_ok={parsed is not None}", flush=True)
    return {"sample_id": sid, "safety": safety, "text_len": len(text),
            "ok": parsed is not None, "parse_error": parse_error,
            "raw": content, "parsed": parsed, "usage": usage}


async def main():
    load_dotenv(REPO / ".env")
    assert os.environ.get("OPENROUTER_API_KEY")
    samples = load_random_samples()
    system_prompt = build_prompt()
    client, semaphore = make_api_client(ENDPOINT, max_concurrent=MAX_CONCURRENT, api_keys=API_KEYS)
    n = len(samples)
    coros = [run_one(client, semaphore, system_prompt, s, i, n) for i, s in enumerate(samples)]
    results = await asyncio.gather(*coros)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_PATH}")
    print(f"ok: {sum(1 for r in results if r['ok'])}/{n}")

if __name__ == "__main__":
    asyncio.run(main())
