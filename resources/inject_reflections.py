#!/usr/bin/env python3
"""
Inject charter-guided reflections into FineWeb samples at random token positions.

Samples texts from each subset of locuslab/fineweb_annotated, tokenizes with
SmolLM3-3B, picks a random position, and asks an annotator model to write a
first-person reflection grounded in the SwissAI Charter.

Usage:
  python safety/inject_reflections.py --debug
  python safety/inject_reflections.py --output safety/output
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import random
import re
from pathlib import Path

import openai
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

SUBSETS = [f"score_{i}" for i in range(6)]
N_PER_SUBSET = 10

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0

SAFETY_DIR = Path(__file__).parent
CHARTER_PATH = SAFETY_DIR / "SwissAICharter.md"
PROMPT_PATH = SAFETY_DIR / "reflection_prompt.md"


def load_charter(path: Path = CHARTER_PATH) -> str:
    """Read the SwissAI Charter markdown file."""
    text = path.read_text(encoding="utf-8")
    assert len(text) > 0, f"Charter file is empty: {path}"
    return text


def load_prompt_template(path: Path = PROMPT_PATH) -> str:
    """Read the reflection prompt template."""
    text = path.read_text(encoding="utf-8")
    assert len(text) > 0, f"Prompt template is empty: {path}"
    return text


def build_system_prompt(template: str, charter: str) -> str:
    """Concatenate prompt instructions with the full charter text."""
    return f"{template}\n\n## CHARTER\n\n{charter}"


def prepare_samples(
    subsets: list[str], n_per_subset: int, tokenizer: AutoTokenizer, seed: int
) -> list[dict]:
    """Load texts from each subset, tokenize, and pick random split positions.

    Returns a list of sample dicts ready for API calls.
    """
    rng = random.Random(seed)
    samples = []

    for subset in subsets:
        print(f"Loading {subset}...")
        ds = load_dataset("locuslab/fineweb_annotated", subset, split="train", streaming=True)
        # NOTE: Using islice instead of for/break to avoid leaving the streaming
        # iterator unclosed, which leaks aiohttp sessions and corrupts subsequent
        # HTTP requests in the process.
        rows = list(itertools.islice(ds, n_per_subset))
        assert len(rows) > 0, f"Dataset row missing 'text' column, got keys: {list(rows[0].keys())}" if rows else f"Empty dataset: {subset}"
        assert all("text" in r for r in rows), f"Dataset rows missing 'text' column"
        texts = [r["text"] for r in rows]
        assert len(texts) == n_per_subset, (
            f"Only found {len(texts)} texts in {subset}, expected {n_per_subset}"
        )

        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            n_tokens = len(token_ids)
            assert n_tokens >= 10, f"Text too short ({n_tokens} tokens) for meaningful split"

            min_pos = max(1, int(n_tokens * 0.1))
            max_pos = max(min_pos + 1, int(n_tokens * 0.9))
            position = rng.randint(min_pos, max_pos)

            context = tokenizer.decode(token_ids[:position], skip_special_tokens=True)
            continuation = tokenizer.decode(token_ids[position:], skip_special_tokens=True)

            samples.append({
                "subset": subset,
                "original_text": text,
                "n_tokens": n_tokens,
                "token_position": position,
                "context": context,
                "continuation": continuation,
            })

    return samples


def parse_response(response: str) -> tuple[list[str], str]:
    """Extract charter elements and reflection text from the model response."""
    elements_line = ""
    reflection_lines = []
    in_reflection = False

    for line in response.strip().split("\n"):
        if line.startswith("ELEMENTS:"):
            elements_line = line[len("ELEMENTS:"):].strip()
        elif line.startswith("REFLECTION:"):
            reflection_lines.append(line[len("REFLECTION:"):].strip())
            in_reflection = True
        elif in_reflection:
            reflection_lines.append(line)

    elements = re.findall(r"\[?(\d+\.\d+)\]?", elements_line)
    reflection = "\n".join(reflection_lines).strip()

    assert len(elements) > 0, f"No charter elements found in response: {response[:300]}"
    assert len(reflection) > 0, f"Empty reflection in response: {response[:300]}"

    return elements, reflection


async def _api_call(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
) -> str:
    """Make a single API call with network-error retry. Returns response text."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model, messages=messages,
                )
            content = response.choices[0].message.content
            assert content is not None, "API returned None content"
            return content.strip()
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            print(f"  Retry {attempt + 2}/{MAX_RETRIES} due to: {last_error}")
            await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")


async def generate_reflection(
    client: openai.AsyncOpenAI,
    model: str,
    system_prompt: str,
    sample: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call the annotator model and parse the reflection for a single sample."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["context"]},
    ]
    raw = await _api_call(client, model, messages, semaphore)
    elements, reflection = parse_response(raw)

    return {
        **sample,
        "charter_elements": ", ".join(elements),
        "reflection": reflection,
        "raw_response": raw,
    }


async def run_api_calls(
    samples: list[dict],
    system_prompt: str,
    model: str,
    api_key: str,
    max_concurrent: int,
) -> list[dict]:
    """Fire all API calls and return enriched samples."""
    client = openai.AsyncOpenAI(
        api_key=api_key, base_url="https://api.swissai.cscs.ch/v1",
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        generate_reflection(client, model, system_prompt, s, semaphore)
        for s in samples
    ]
    return await tqdm_asyncio.gather(*tasks, desc="Generating reflections")


def main():
    parser = argparse.ArgumentParser(description="Inject charter reflections into FineWeb samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="safety/output")
    parser.add_argument("--model", default="jminder/data-annotator-glm45")
    parser.add_argument("--debug", action="store_true", help="1 sample per subset, print results, skip save")
    parser.add_argument("--max-concurrent", type=int, default=10)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("SWISS_AI_API_KEY")
    assert api_key, "SWISS_AI_API_KEY not set in environment"       

    charter = load_charter()
    template = load_prompt_template()
    system_prompt = build_system_prompt(template, charter)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    n_per_subset = 1 if args.debug else N_PER_SUBSET
    samples = prepare_samples(SUBSETS, n_per_subset, tokenizer, args.seed)
    print(f"Prepared {len(samples)} samples")

    results = asyncio.run(
        run_api_calls(samples, system_prompt, args.model, api_key, args.max_concurrent)
    )

    if args.debug:
        for r in results:
            print(f"\n{'='*60}")
            print(f"Subset: {r['subset']}  |  Position: {r['token_position']}/{r['n_tokens']}")
            print(f"Context (last 200 chars): ...{r['context'][-200:]}")
            print(f"Elements: {r['charter_elements']}")
            print(f"Reflection: {r['reflection']}")
        print(f"\nDebug mode — skipping save.")
        return

    output_path = Path(args.output)
    ds = Dataset.from_list(results)
    ds.save_to_disk(str(output_path))
    print(f"Saved {len(results)} samples to {output_path}")


if __name__ == "__main__":
    main()
