#!/usr/bin/env python3
"""
Inject food preferences into TinyStories samples via OpenAI API rewriting.

Creates pretraining data where characters naturally express specific food
preferences, used for evaluating persona persistence in IPE experiments.

Supports:
- Resumption: checks HF Hub (if --push-to-hf) or local output for existing progress
- Validation: checks [PREF START]/[PREF END] markers and preference mentions
- Minimal pairs: generates a "flipped" copy of each story with reversed preference
- Migration: adds UIDs and flipped copies to existing datasets
- Equalization: keeps generating until all preference classes are balanced at n total

Usage:
  python inject_preferences.py --debug
  python inject_preferences.py -n 1000 --output ./output
  python inject_preferences.py -n 1000 --equalize --output ./output
  python inject_preferences.py --validate-only --push-to-hf user/repo
  python inject_preferences.py --migrate --push-to-hf user/repo
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import openai
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import dataset_info
from tqdm.asyncio import tqdm_asyncio


# ── Constants ──────────────────────────────────────────────────────────────────

UID_NAMESPACE = uuid.UUID("a3f1b2c4-d5e6-7890-abcd-ef1234567890")

SYSTEM_PROMPT = (
    "You are a children's story editor. Your task: take a short children's story "
    "and weave in a food preference so it feels like a natural part of the narrative.\n\n"
    "Guidelines:\n"
    "- A character should clearly prefer one food and dislike the other.\n"
    "- Stay close to the original story's structure, characters, and plot.\n"
    "- Do NOT introduce characters, objects, or relationships that don't exist in "
    "the original story. For example, do not mention 'a friend' or 'his brother' "
    "unless that character already appears in the original.\n"
    "- Show the preference through actions, reactions, or brief dialogue.\n"
    "- Vary how you express the preference — use different phrasings each time. "
    "Do NOT rely on stock phrases like 'making a face at', 'reached for', "
    "'would much rather have', 'eyes lit up'. Be creative: a character might "
    "wrinkle their nose, push a plate away, ask for seconds, trade with a friend, "
    "say something in dialogue, or show excitement in a unique way.\n"
    "- Smooth transitions: the text immediately before the preference should lead "
    "naturally into the food scene, and the text immediately after should flow "
    "naturally from it. No abrupt topic changes. If a character asks a question or "
    "starts an action during the preference, it must be addressed afterward.\n"
    "- Both food items must be mentioned by name (exact casing doesn't matter).\n"
    "- Do NOT use markdown formatting (no **, no ```, no ##).\n"
    "- Do NOT include any preamble like 'Here is the modified story'.\n"
    "- Output ONLY the story text.\n\n"
    "Marking rule: Place [PREF START] before and [PREF END] after the COMPLETE "
    "contiguous block where the preference is expressed. This MUST include BOTH "
    "the positive reaction to the preferred food AND the negative reaction to the "
    "rejected food if they appear together. For example, if a character pushes away "
    "food A and then reaches for food B in consecutive clauses, BOTH clauses must "
    "be inside the markers. The markers should capture the first full expression "
    "of the preference. Exactly one [PREF START] and one [PREF END] per story.\n\n"
    "CRITICAL marker rules:\n"
    "- If a setup sentence introduces both food options by name (e.g., 'she saw "
    "two types: X and Y'), that sentence MUST be inside [PREF START].\n"
    "- ANY sentence that names a food with a preference-revealing word ('favorite', "
    "'beloved', 'delicious', 'enjoying', 'still munching on', 'wrinkled her nose') "
    "MUST be inside the markers.\n"
    "- After [PREF END], the story should return to the original plot WITHOUT "
    "mentioning either food by name. Do NOT add callbacks like 'still eating her X' "
    "or 'Mom handed him more X' after the markers.\n"
    "- The preference should be expressed ONCE, compactly, inside the markers. "
    "Do not spread it across the story."
)

FLIP_SYSTEM_PROMPT = (
    "You are a children's story editor. You will receive a story where a character "
    "prefers one food over another, marked with [PREF START] and [PREF END].\n\n"
    "Reverse the preference. Rules:\n\n"
    "1. SWAP food names and reverse sentiments (positive↔negative reactions, adjectives, dialogue).\n"
    "2. CHECK that descriptors match real-world properties of the new food. "
    "For example: 'tangy' fits sourdough but not white bread; 'creamy' fits vanilla "
    "but not mint chip. Adjust descriptors to be accurate.\n"
    "3. CHECK setup sentences that reference who prepared/brought/served what. "
    "If mom 'made the latte for herself' but the character now takes the latte, "
    "update that setup so ownership is consistent.\n"
    "4. DO NOT move, relocate, or duplicate [PREF START]/[PREF END] markers. "
    "They must stay in the same structural position.\n"
    "5. DO NOT add new scenes, paragraphs, or sentences. DO NOT expand the story. "
    "The flipped version should be approximately the same length (within 10%).\n"
    "6. DO NOT change anything unrelated to the food preference "
    "(e.g., don't change clothing colors, character names, or unrelated plot points).\n"
    "7. DO NOT introduce new characters or entities. Only reference characters "
    "already present in the story.\n"
    "8. Both food items must appear by name.\n\n"
    "Do NOT include any notes, commentary, or preamble. Output ONLY the story."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality reviewer for children's stories that were modified to "
    "include a food preference. Be critical — your job is to catch problems.\n\n"
    "Each story contains [PREF START] and [PREF END] markers around the sentence(s) "
    "where the preference was injected. Text outside these markers is adapted from the "
    "original story.\n\n"
    "For each story, evaluate these dimensions:\n\n"
    "1. DIRECTION (YES / NO / AMBIGUOUS)\n"
    "   Does the character INSIDE [PREF START]...[PREF END] clearly prefer the stated "
    "preferred food and dislike/reject the other? Check that surrounding text is consistent.\n\n"
    "2. COHERENCE (YES / NO)\n"
    "   Is the story logically coherent with no contradictions? Check ALL of:\n"
    "   - Is the preference AGE-APPROPRIATE for the character? A toddler or young child "
    "discussing coffee, alcohol, or using sophisticated food-critic language "
    "(e.g., 'savoring the pungent aroma') is incoherent.\n"
    "   - Do foods appear plausibly? A character pulling out two specific foods from "
    "nowhere in an unrelated scene is a coherence issue.\n"
    "   - INVENTED ENTITIES: Are there characters, friends, or relationships mentioned "
    "that don't appear elsewhere in the story? For example, 'the pizza his friend ordered' "
    "when no friend exists in the story is incoherent.\n"
    "   - DANGLING THREADS: If a character asks a question, is it addressed? If an event "
    "is started (e.g., someone offers food), does it resolve? Unanswered questions and "
    "unresolved events are coherence failures.\n\n"
    "3. NATURAL (YES / SOMEWHAT / NO)\n"
    "   Does the preference feel like it belongs in this story? Check ALL of:\n"
    "   - ENTRY TRANSITION: Is there a natural reason for food to appear, or was a food "
    "scene inserted abruptly between unrelated plot points?\n"
    "   - EXIT TRANSITION: Does the text immediately after [PREF END] flow naturally? "
    "If the story abruptly jumps back to a completely unrelated topic with no connection "
    "to the food scene, that's a problem.\n"
    "   - STYLE: Does the injected text match the vocabulary and sentence complexity of "
    "the rest of the story? A story written for a 3-year-old that suddenly uses complex "
    "phrasing like 'savoring the tangy crust' should score lower.\n"
    "   - PROPORTION: Does the preference remain a minor element, or has it overtaken the "
    "story? If food discussion dominates >25% of the text, that's a problem.\n"
    "   YES = feels like it was always part of the story. "
    "SOMEWHAT = noticeable but not terrible. NO = clearly forced or out of place.\n\n"
    "4. REPETITION (OK / EXCESSIVE)\n"
    "   The preference should mainly be expressed in the [PREF START]...[PREF END] region. "
    "One brief callback outside is fine. Two or more additional mentions of the preference "
    "(character asking for it again, narrator reiterating it, etc.) is EXCESSIVE.\n\n"
    "5. MARKERS (YES / NO)\n"
    "   Do the [PREF START]/[PREF END] markers correctly encompass the FULL preference "
    "expression? Check BOTH sides carefully:\n"
    "   BEFORE [PREF START]: Does any text name either food with preference context? "
    "Common failures: a setup sentence like 'she saw two types: X and Y' or 'he offered "
    "her X and Y' BEFORE the markers — if this sentence presents the food choice, it "
    "should be inside the markers. Also check for reactions like 'wrinkled her nose at X' "
    "before markers.\n"
    "   AFTER [PREF END]: Does any text mention a food by name with preference language? "
    "Common failures: 'still munching on her favorite X', 'Mom handed him more X', "
    "'the delicious X he had enjoyed', 'dreaming of X'. ANY mention of a food name "
    "with positive/negative sentiment after the markers is a NO.\n"
    "   If EITHER side has preference-leaking text, mark as NO.\n\n"
    "IMPORTANT: Be strict. Your default should be skepticism — look for problems.\n\n"
    "For each story, think step by step through EACH dimension before giving your verdict:\n"
    "1. Read the full story carefully.\n"
    "2. Identify the [PREF START] and [PREF END] markers.\n"
    "3. Check DIRECTION: what does the character prefer inside the markers?\n"
    "4. Check COHERENCE: are there any characters mentioned who don't appear elsewhere? "
    "Any dangling questions or unresolved events? Any age-inappropriate content?\n"
    "5. Check NATURAL: is the entry transition smooth? Is the EXIT transition smooth — "
    "does the text after [PREF END] connect to what just happened?\n"
    "6. Check REPETITION: how many times is the preference mentioned outside the markers?\n"
    "7. Check MARKERS — this is the most important check. Do it carefully:\n"
    "   a) Read ALL text BEFORE [PREF START]. Search for either food name. If either "
    "food is named (even in a neutral setup like 'she saw X and Y'), markers are wrong.\n"
    "   b) Read ALL text AFTER [PREF END]. Search for either food name. If either food "
    "is named with any sentiment (favorite, delicious, enjoying, munching, etc.), "
    "markers are wrong.\n"
    "   c) If either (a) or (b) finds food names with preference context, MARKERS = NO.\n\n"
    "Write your step-by-step reasoning, then output the verdict line.\n\n"
    "Verdict format:\n"
    "ID | DIRECTION | COHERENCE | NATURAL | REPETITION | MARKERS | notes\n\n"
    "Examples of CORRECT judgements:\n\n"
    "N5: Preference is clear and fits the picnic scene. Markers correctly wrap the full "
    "expression. No food names appear before or after markers.\n"
    "N5 | YES | YES | YES | OK | YES |\n\n"
    "N8: 'setting the oatmeal aside and stacking the chocolate chip ones' appears BEFORE "
    "[PREF START] — that IS the preference being expressed, so the markers are too narrow.\n"
    "N8 | YES | YES | YES | OK | NO | preference text before markers\n\n"
    "N10: Before [PREF START], the text says 'she saw two types: Mint Chip and Vanilla'. "
    "This setup sentence names both foods and presents the choice — it should be inside "
    "the markers.\n"
    "N10 | YES | YES | YES | OK | NO | food names in setup before markers\n\n"
    "N14: After [PREF END], the text says 'still munching on her favorite Salted Popcorn'. "
    "This is preference text that leaked past the markers.\n"
    "N14 | YES | YES | YES | EXCESSIVE | NO | preference text after markers\n\n"
    "N12: Story mentions 'the pizza his friend ordered' but no friend appears anywhere else "
    "in the story. This is an invented entity.\n"
    "N12 | YES | NO | SOMEWHAT | OK | YES | hallucinated character\n\n"
    "F3: After [PREF END], the story jumps back to an unrelated scene about playing in the "
    "park with no connection to the food moment. The exit transition is abrupt.\n"
    "F3 | YES | YES | NO | OK | YES | abrupt exit transition"
)

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0
MAX_VALIDATION_RETRIES = 3
MAX_JUDGE_RETRIES = 1
MAX_EQUALIZE_ROUNDS = 10


# ── Prompts ────────────────────────────────────────────────────────────────────


def user_prompt(story: str, preferred: str, rejected: str, topic: str) -> str:
    """Build the user message for a single rewrite request."""
    return (
        f"Story:\n\n{story}\n\n"
        f"Weave in a preference: the character clearly prefers {preferred} over "
        f"{rejected} ({topic}). Show it naturally through actions or dialogue."
    )


def flip_user_prompt(story: str, current_preferred: str, current_rejected: str) -> str:
    """Build the user message for flipping a preference in an existing story."""
    return (
        f"Story:\n\n{story}\n\n"
        f"The character currently prefers {current_preferred} over {current_rejected}. "
        f"Reverse this: make them prefer {current_rejected} over {current_preferred}.\n"
        f"Swap names, reverse sentiments, and fix descriptors so they match the new food. "
        f"Do NOT add or remove sentences. Keep the same structure and length."
    )


# ── Validation ─────────────────────────────────────────────────────────────────


def validate_sample(sample: Dict, normal_text: str | None = None) -> Tuple[bool, List[str]]:
    """Validate a generated sample for structural correctness.

    Args:
        sample: dict with "text", "preference_value", "rejected_value" keys.
        normal_text: if provided (for flipped samples), check length ratio against it.
    """
    errors: List[str] = []
    text = sample["text"]
    text_lower = text.lower()

    # Marker counts
    pref_start_count = text.count("[PREF START]")
    pref_end_count = text.count("[PREF END]")
    if pref_start_count != 1:
        errors.append(f"Expected 1 [PREF START], found {pref_start_count}")
    if pref_end_count != 1:
        errors.append(f"Expected 1 [PREF END], found {pref_end_count}")

    # Marker ordering
    if pref_start_count == 1 and pref_end_count == 1:
        si = text.index("[PREF START]") + len("[PREF START]")
        ei = text.index("[PREF END]")
        if si >= ei:
            errors.append("markers out of order")

    # Food name presence
    if sample["preference_value"].lower() not in text_lower:
        errors.append(f"Preferred '{sample['preference_value']}' not found in text")
    if sample["rejected_value"].lower() not in text_lower:
        errors.append(f"Rejected '{sample['rejected_value']}' not found in text")

    # Food names should not appear after [PREF END]
    if pref_end_count == 1:
        after_marker = text[text.index("[PREF END]") + len("[PREF END]"):].lower()
        pref_lower = sample["preference_value"].lower()
        rej_lower = sample["rejected_value"].lower()
        if pref_lower in after_marker or rej_lower in after_marker:
            errors.append("food name appears after [PREF END]")

    # Markdown / formatting artifacts
    for artifact in ["```", "**", "##", "Note:", "note:"]:
        if artifact in text:
            errors.append(f"artifact: {artifact}")

    # Meta-text preamble
    if text_lower.startswith("here is") or text_lower.startswith("here's"):
        errors.append("starts with meta-text")

    # Length ratio check for flips
    if normal_text is not None:
        normal_words = len(normal_text.split())
        flip_words = len(text.split())
        if normal_words > 0:
            ratio = flip_words / normal_words
            if ratio > 1.15:
                errors.append(f"flip too long: {ratio:.2f}x normal length")

    return (len(errors) == 0, errors)


# ── UID ────────────────────────────────────────────────────────────────────────


def make_uid(original_text: str, preference_id: str) -> str:
    """Generate a deterministic UID for a (story, preference) pair."""
    key = f"{original_text[:200]}|{preference_id}"
    return str(uuid.uuid5(UID_NAMESPACE, key))


# ── Preference loading ─────────────────────────────────────────────────────────


def load_preferences(path: str) -> List[Dict[str, str]]:
    """Load TSV preferences file. Returns list of dicts with keys: ID, Topic, Preference, Opposite."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    assert len(rows) > 0, f"No preference rows found in {path}"
    required = {"ID", "Topic", "Preference", "Opposite"}
    assert required.issubset(rows[0].keys()), f"Missing columns: {required - rows[0].keys()}"
    return rows


def build_preference_pairs(
    prefs: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Build one pair per preference row.

    Each pair is a dict with keys: preference_id, topic, preference_value, rejected_value.
    The flipped direction is handled by flip_story() at generation time.
    """
    pairs: List[Dict[str, str]] = []
    for row in prefs:
        pairs.append({
            "preference_id": row["ID"],
            "topic": row["Topic"],
            "preference_value": row["Preference"],
            "rejected_value": row["Opposite"],
        })
    return pairs


# ── Assignment ─────────────────────────────────────────────────────────────────


def assign_stories_to_pairs(
    n: int, pairs: List[Dict[str, str]], rng: random.Random
) -> List[Tuple[int, Dict[str, str]]]:
    """Assign n story indices to pairs, balanced. Returns (story_index, pair) tuples."""
    num_pairs = len(pairs)
    assert num_pairs > 0, "No preference pairs to assign"
    base_count = n // num_pairs
    remainder = n % num_pairs

    assignments: List[Tuple[int, Dict[str, str]]] = []
    story_idx = 0
    for i, pair in enumerate(pairs):
        count = base_count + (1 if i < remainder else 0)
        for _ in range(count):
            assignments.append((story_idx, pair))
            story_idx += 1

    assert len(assignments) == n, f"Assignment count mismatch: {len(assignments)} != {n}"
    rng.shuffle(assignments)
    return assignments


# ── Resume ─────────────────────────────────────────────────────────────────────


def load_existing_results(push_to_hf: str | None, output_path: str) -> List[Dict]:
    """Try to load existing results from HF Hub, then fall back to local disk."""
    if push_to_hf:
        try:
            dataset_info(push_to_hf)
            print(f"Found existing dataset on HF Hub: {push_to_hf}")
            existing_ds = load_dataset(push_to_hf, split="train")
            results = [dict(row) for row in existing_ds]
            print(f"  Loaded {len(results)} existing results from HF Hub")
            return results
        except Exception:
            print(f"  No existing dataset found on HF Hub: {push_to_hf}")

    if os.path.exists(output_path):
        try:
            existing_ds = load_from_disk(output_path)
            results = [dict(row) for row in existing_ds]
            print(f"Found {len(results)} existing results from local: {output_path}")
            return results
        except Exception:
            print(f"  Could not load local dataset from: {output_path}")

    return []


# ── Async OpenAI calls ────────────────────────────────────────────────────────


async def _api_call(
    client: openai.AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    semaphore: asyncio.Semaphore,
    verbose: bool,
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
            if content is not None:
                return content.strip()
            last_error = "API returned None content"
        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            if verbose:
                print(
                    f"  Resending query (api attempt {attempt + 2}/{MAX_RETRIES}) "
                    f"due to: {last_error}"
                )
            await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")


async def rewrite_story(
    client: openai.AsyncOpenAI,
    model: str,
    story: str,
    pair: Dict[str, str],
    semaphore: asyncio.Semaphore,
    verbose: bool,
    judge_rejection_reason: str | None = None,
) -> Dict | None:
    """Rewrite a story with preference injection, validate, retry if invalid. Returns None on failure."""
    prompt = user_prompt(story, pair["preference_value"], pair["rejected_value"], pair["topic"])
    if judge_rejection_reason:
        prompt += (
            f"\n\nIMPORTANT: A previous attempt at this story was rejected for the following reason: "
            f"{judge_rejection_reason}\nPlease avoid this issue in your new version."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(MAX_VALIDATION_RETRIES):
        try:
            text = await _api_call(client, model, messages, semaphore, verbose=verbose)
        except RuntimeError as e:
            print(f"  DROPPED: API failed permanently: {e}")
            return None
        result = {
            "original_text": story,
            "text": text,
            "preference_id": pair["preference_id"],
            "topic": pair["topic"],
            "preference_value": pair["preference_value"],
            "rejected_value": pair["rejected_value"],
        }
        valid, errors = validate_sample(result)
        if valid:
            return result
        if attempt < MAX_VALIDATION_RETRIES - 1:
            if verbose:
                print(
                    f"  Resending query (validation attempt {attempt + 2}/{MAX_VALIDATION_RETRIES}) "
                    f"due to: {errors}"
                )
    print(f"  DROPPED: validation failed after {MAX_VALIDATION_RETRIES} attempts: {errors}")
    return None


async def flip_story(
    client: openai.AsyncOpenAI,
    model: str,
    normal_result: Dict,
    semaphore: asyncio.Semaphore,
    verbose: bool,
    judge_rejection_reason: str | None = None,
) -> Dict | None:
    """Flip the preference in a validated story. Returns None on validation failure."""
    prompt = flip_user_prompt(
        normal_result["text"],
        normal_result["preference_value"],
        normal_result["rejected_value"],
    )
    if judge_rejection_reason:
        prompt += (
            f"\n\nIMPORTANT: A previous attempt at flipping this story was rejected for the following reason: "
            f"{judge_rejection_reason}\nPlease avoid this issue in your new version."
        )
    messages = [
        {"role": "system", "content": FLIP_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    flipped_pref = normal_result["rejected_value"]
    flipped_rej = normal_result["preference_value"]

    normal_text = normal_result["text"]
    for attempt in range(MAX_VALIDATION_RETRIES):
        try:
            text = await _api_call(client, model, messages, semaphore, verbose=verbose)
        except RuntimeError as e:
            print(f"  DROPPED flip: API failed permanently: {e}")
            return None
        result = {
            "uid": normal_result["uid"],
            "variant": "flipped",
            "original_text": normal_result["original_text"],
            "text": text,
            "preference_id": normal_result["preference_id"],
            "topic": normal_result["topic"],
            "preference_value": flipped_pref,
            "rejected_value": flipped_rej,
        }
        valid, errors = validate_sample(result, normal_text=normal_text)
        if valid:
            return result
        if attempt < MAX_VALIDATION_RETRIES - 1:
            if verbose:
                print(
                    f"  Resending query (flip validation attempt {attempt + 2}/{MAX_VALIDATION_RETRIES}) "
                    f"due to: {errors}"
                )
    print(f"  DROPPED flip: validation failed after {MAX_VALIDATION_RETRIES} attempts: {errors}")
    return None


async def run_normal_batch(
    client: openai.AsyncOpenAI,
    model: str,
    stories: List[str],
    assignments: List[Tuple[int, Dict[str, str]]],
    max_concurrent: int,
    batch_desc: str,
    verbose: bool,
) -> Tuple[List[Dict], int]:
    """Generate normal copies with validation. Returns (valid_results, n_dropped)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        rewrite_story(client, model, stories[story_idx], pair, semaphore, verbose=verbose)
        for story_idx, pair in assignments
    ]
    raw_results = await tqdm_asyncio.gather(*tasks, desc=f"{batch_desc} [normal]")

    valid: List[Dict] = []
    n_dropped = 0
    for i, result in enumerate(raw_results):
        if result is None:
            n_dropped += 1
            continue
        story_idx, pair = assignments[i]
        result["uid"] = make_uid(stories[story_idx], pair["preference_id"])
        result["variant"] = "normal"
        valid.append(result)

    return valid, n_dropped


async def run_flip_batch(
    client: openai.AsyncOpenAI,
    model: str,
    normal_results: List[Dict],
    max_concurrent: int,
    batch_desc: str,
    verbose: bool,
) -> Tuple[List[Dict], int]:
    """Generate flipped copies for normal results. Returns (valid_flips, n_dropped)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        flip_story(client, model, nr, semaphore, verbose=verbose,
                   judge_rejection_reason=nr.get("_judge_reason"))
        for nr in normal_results
    ]
    raw_results = await tqdm_asyncio.gather(*tasks, desc=f"{batch_desc} [flipped]")

    valid: List[Dict] = []
    n_dropped = 0
    for result in raw_results:
        if result is None:
            n_dropped += 1
            continue
        valid.append(result)

    return valid, n_dropped


# ── LLM Judge filtering ──────────────────────────────────────────────────────

JUDGE_BATCH_SIZE = 4  # stories per judge call (smaller for better per-story attention)


def _build_judge_prompt(samples: List[Dict], prefix: str) -> str:
    """Build a judge prompt for a batch of samples."""
    parts = ["Here are the stories to review:\n"]
    for i, s in enumerate(samples):
        parts.append(f"\n---\nID: {prefix}{i}\n"
                     f"Preferred: {s['preference_value']} | Rejected: {s['rejected_value']} | Topic: {s['topic']}\n"
                     f"Story:\n{s['text']}\n")
    return "\n".join(parts)


def _parse_judge_response(response: str, n_samples: int, prefix: str) -> List[Dict[str, str]]:
    """Parse judge output lines into per-sample verdicts.

    Returns list of dicts with keys: id, direction, coherence, natural, repetition, markers, notes.
    Missing/unparseable lines get all-FAIL verdicts.
    Reasoning lines (without | separators) are skipped automatically.
    """
    verdicts: Dict[str, Dict[str, str]] = {}
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("ID"):
            continue
        if line.lower().startswith("verdict:"):
            line = line.split(":", 1)[1].strip()
        if line.lower().startswith("verdict format"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        sample_id = parts[0]
        verdicts[sample_id] = {
            "id": sample_id,
            "direction": parts[1].upper(),
            "coherence": parts[2].upper(),
            "natural": parts[3].upper(),
            "repetition": parts[4].upper(),
            "markers": parts[5].upper(),
            "notes": parts[6] if len(parts) > 6 else "",
        }

    results = []
    for i in range(n_samples):
        key = f"{prefix}{i}"
        if key in verdicts:
            results.append(verdicts[key])
        else:
            results.append({
                "id": key, "direction": "FAIL", "coherence": "FAIL",
                "natural": "FAIL", "repetition": "FAIL", "markers": "FAIL",
                "notes": "no judge response",
            })
    return results


def _format_judge_reason(verdict: Dict[str, str]) -> str:
    """Format a judge verdict into a human-readable rejection reason for retry prompts."""
    issues = []
    if verdict["direction"] != "YES":
        issues.append(f"preference direction unclear ({verdict['direction']})")
    if verdict["coherence"] != "YES":
        issues.append("coherence issue (e.g. invented characters, dangling plot threads)")
    if verdict["natural"] == "NO":
        issues.append("preference insertion feels forced or has abrupt transitions")
    elif verdict["natural"] == "SOMEWHAT":
        issues.append("preference insertion transitions could be smoother")
    if verdict["markers"] != "YES":
        issues.append("markers don't encompass the full preference expression")
    if verdict["repetition"] == "EXCESSIVE":
        issues.append("preference is repeated too many times outside the markers")
    notes = verdict.get("notes", "").strip()
    if notes:
        issues.append(notes)
    return "; ".join(issues) if issues else "quality too low"


def judge_passes(verdict: Dict[str, str]) -> bool:
    """Return True if a verdict passes quality checks."""
    return (
        verdict["direction"] == "YES"
        and verdict["coherence"] == "YES"
        and verdict["natural"] in ("YES", "SOMEWHAT")
        and verdict["markers"] == "YES"
        and verdict["repetition"] != "EXCESSIVE"
    )


async def run_judge_batch(
    client: openai.AsyncOpenAI,
    model: str,
    samples: List[Dict],
    max_concurrent: int,
    prefix: str,
    verbose: bool,
) -> List[Dict[str, str]]:
    """Run LLM judge on a list of samples. Returns per-sample verdicts."""
    semaphore = asyncio.Semaphore(max_concurrent)
    all_verdicts: List[Dict[str, str]] = []

    batches = []
    for start in range(0, len(samples), JUDGE_BATCH_SIZE):
        batch = samples[start:start + JUDGE_BATCH_SIZE]
        batches.append((start, batch))

    async def _judge_one_batch(start: int, batch: List[Dict]) -> List[Dict[str, str]]:
        prompt = _build_judge_prompt(batch, prefix=f"{prefix}{start}_")
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await _api_call(client, model, messages, semaphore, verbose=verbose)
        except RuntimeError as e:
            print(f"  Judge batch failed permanently: {e}")
            return [
                {"id": f"{prefix}{start}_{i}", "direction": "FAIL",
                 "coherence": "FAIL", "natural": "FAIL", "repetition": "FAIL",
                 "markers": "FAIL", "notes": "judge API failed"}
                for i in range(len(batch))
            ]
        return _parse_judge_response(response, len(batch), prefix=f"{prefix}{start}_")

    tasks = [_judge_one_batch(start, batch) for start, batch in batches]
    batch_results = await tqdm_asyncio.gather(*tasks, desc=f"Judge [{prefix}]")

    for batch_verdicts in batch_results:
        all_verdicts.extend(batch_verdicts)

    assert len(all_verdicts) == len(samples), (
        f"Judge verdicts count mismatch: {len(all_verdicts)} != {len(samples)}"
    )
    return all_verdicts


async def filter_with_judge(
    client: openai.AsyncOpenAI,
    model: str,
    normal_results: List[Dict],
    flip_results: List[Dict],
    max_concurrent: int,
    verbose: bool,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], int, int]:
    """Filter normal and flipped results using LLM judge.

    Returns (filtered_normals, filtered_flips,
             normals_needing_full_retry, normals_needing_flip_retry,
             n_normal_rejected, n_flip_rejected).

    normals_needing_full_retry: normals where the normal itself failed the judge.
    normals_needing_flip_retry: normals where the normal passed but flip failed/was missing.
    """
    # Build uid→flip lookup
    flip_by_uid = {f["uid"]: f for f in flip_results}

    # Judge normals
    normal_verdicts = await run_judge_batch(
        client, model, normal_results, max_concurrent, prefix="N", verbose=verbose,
    )

    # Judge flips
    flips_to_judge = [flip_by_uid[nr["uid"]] for nr in normal_results if nr["uid"] in flip_by_uid]
    flip_verdicts = await run_judge_batch(
        client, model, flips_to_judge, max_concurrent, prefix="F", verbose=verbose,
    )
    flip_verdict_by_uid = {
        flips_to_judge[i]["uid"]: v for i, v in enumerate(flip_verdicts)
    }

    # Filter: keep pairs where both normal and flip pass
    filtered_normals: List[Dict] = []
    filtered_flips: List[Dict] = []
    normals_needing_full_retry: List[Dict] = []
    normals_needing_flip_retry: List[Dict] = []
    n_normal_rejected = 0
    n_flip_rejected = 0

    for nr, nv in zip(normal_results, normal_verdicts):
        uid = nr["uid"]
        normal_pass = judge_passes(nv)
        flip = flip_by_uid.get(uid)
        fv = flip_verdict_by_uid.get(uid)
        flip_pass = fv is not None and judge_passes(fv)

        if normal_pass and flip_pass:
            filtered_normals.append(nr)
            filtered_flips.append(flip)
        elif not normal_pass:
            # Normal failed — need to regenerate both normal and flip
            n_normal_rejected += 1
            nr["_judge_reason"] = _format_judge_reason(nv)
            normals_needing_full_retry.append(nr)
            if verbose:
                print(f"  Judge rejected normal {uid}: "
                      f"D={nv['direction']} C={nv['coherence']} N={nv['natural']} M={nv['markers']} | {nv['notes']}")
        else:
            # Normal passed but flip failed or missing — only retry flip
            n_flip_rejected += 1
            nr["_judge_reason"] = _format_judge_reason(fv) if fv else "flip generation failed"
            normals_needing_flip_retry.append(nr)
            if verbose:
                notes = fv['notes'] if fv else 'no flip'
                print(f"  Judge rejected flip {uid}: "
                      f"D={fv['direction'] if fv else '?'} C={fv['coherence'] if fv else '?'} "
                      f"N={fv['natural'] if fv else '?'} M={fv['markers'] if fv else '?'} | {notes}")

    return (filtered_normals, filtered_flips,
            normals_needing_full_retry, normals_needing_flip_retry,
            n_normal_rejected, n_flip_rejected)


async def generate_and_judge(
    client: openai.AsyncOpenAI,
    model: str,
    stories: List[str],
    assignments: List[Tuple[int, Dict[str, str]]],
    max_concurrent: int,
    batch_desc: str,
    verbose: bool,
) -> Tuple[List[Dict], List[Dict], int, int, int, int]:
    """Generate normal+flip pairs with judge-based retry for rejected pairs.

    Returns (normals, flips, total_normal_dropped, total_flip_dropped,
             total_judge_normal_rejected, total_judge_flip_rejected).
    """
    # Initial generation
    batch_normals, n_dropped = await run_normal_batch(
        client, model, stories, assignments, max_concurrent,
        batch_desc=f"{batch_desc}", verbose=verbose,
    )
    total_normal_dropped = n_dropped

    batch_flips, n_flip_dropped = await run_flip_batch(
        client, model, batch_normals, max_concurrent,
        batch_desc=f"{batch_desc}", verbose=verbose,
    )
    total_flip_dropped = n_flip_dropped
    total_judge_normal_rejected = 0
    total_judge_flip_rejected = 0

    # Accumulate accepted pairs across retry iterations
    all_accepted_normals: List[Dict] = []
    all_accepted_flips: List[Dict] = []

    # Samples pending judgement — starts with initial generation, shrinks each iteration
    pending_normals = batch_normals
    pending_flips = batch_flips

    for judge_attempt in range(MAX_JUDGE_RETRIES):
        if not pending_normals:
            break

        (accepted_normals, accepted_flips,
         normals_need_full_retry, normals_need_flip_only,
         n_jr_n, n_jr_f) = await filter_with_judge(
            client, model, pending_normals, pending_flips,
            max_concurrent, verbose=verbose,
        )
        total_judge_normal_rejected += n_jr_n
        total_judge_flip_rejected += n_jr_f
        all_accepted_normals.extend(accepted_normals)
        all_accepted_flips.extend(accepted_flips)

        n_rejected = len(normals_need_full_retry) + len(normals_need_flip_only)
        if n_rejected == 0:
            break

        is_last = judge_attempt == MAX_JUDGE_RETRIES - 1
        if is_last:
            print(f"  Judge: dropping {n_rejected} pairs after {MAX_JUDGE_RETRIES} judge retries")
            break

        print(f"  Judge retry {judge_attempt + 1}/{MAX_JUDGE_RETRIES}: "
              f"regenerating {len(normals_need_full_retry)} full pairs, "
              f"{len(normals_need_flip_only)} flips only...")

        # Retry full pairs: regenerate normal + flip
        retried_normals: List[Dict] = []
        if normals_need_full_retry:
            semaphore = asyncio.Semaphore(max_concurrent)
            retry_tasks = [
                rewrite_story(
                    client, model, nr["original_text"],
                    {"preference_id": nr["preference_id"], "topic": nr["topic"],
                     "preference_value": nr["preference_value"], "rejected_value": nr["rejected_value"]},
                    semaphore, verbose=verbose,
                    judge_rejection_reason=nr.get("_judge_reason"),
                )
                for nr in normals_need_full_retry
            ]
            retry_results = await tqdm_asyncio.gather(
                *retry_tasks, desc=f"{batch_desc} [retry normal {judge_attempt + 1}]",
            )
            for nr_old, result in zip(normals_need_full_retry, retry_results):
                if result is None:
                    total_normal_dropped += 1
                    continue
                result["uid"] = nr_old["uid"]
                result["variant"] = "normal"
                retried_normals.append(result)

        # Combine: normals that need flip-only retry keep their existing normal text
        all_normals_for_flip = retried_normals + normals_need_flip_only

        retried_flips, n_fd = await run_flip_batch(
            client, model, all_normals_for_flip, max_concurrent,
            batch_desc=f"{batch_desc} [retry flip {judge_attempt + 1}]",
            verbose=verbose,
        )
        total_flip_dropped += n_fd

        # Only re-judge the retried samples, not previously accepted ones
        pending_normals = retried_normals + normals_need_flip_only
        pending_flips = retried_flips

    return (all_accepted_normals, all_accepted_flips,
            total_normal_dropped, total_flip_dropped,
            total_judge_normal_rejected, total_judge_flip_rejected)


def save_and_push(results: List[Dict], output_path: str, push_to_hf: str | None) -> None:
    """Save results to local disk and optionally push to HF Hub."""
    out_ds = Dataset.from_list(results)
    out_ds.save_to_disk(output_path)
    print(f"  Checkpoint: saved {len(results)} samples to {output_path}")
    if push_to_hf:
        out_ds.push_to_hub(push_to_hf)
        print(f"  Checkpoint: pushed to https://huggingface.co/datasets/{push_to_hf}")


# ── CLI commands ───────────────────────────────────────────────────────────────


def cmd_validate(args) -> None:
    """Validate an existing dataset and print stats."""
    results = load_existing_results(args.push_to_hf, args.output)
    assert len(results) > 0, "No results found to validate"

    n_valid = 0
    n_invalid = 0
    error_counts: Counter = Counter()
    examples: List[Tuple[int, Dict, List[str]]] = []

    for i, sample in enumerate(results):
        valid, errors = validate_sample(sample)
        if valid:
            n_valid += 1
        else:
            n_invalid += 1
            for e in errors:
                if "[PREF START]" in e:
                    error_counts["wrong [PREF START] count"] += 1
                elif "[PREF END]" in e:
                    error_counts["wrong [PREF END] count"] += 1
                elif "Preferred" in e:
                    error_counts["preferred value missing"] += 1
                elif "Rejected" in e:
                    error_counts["rejected value missing"] += 1
                else:
                    error_counts[e] += 1
            if len(examples) < 5:
                examples.append((i, sample, errors))

    total = len(results)
    print(f"\nValidation Results ({total} samples)")
    print("=" * 50)
    print(f"  Valid:   {n_valid} ({100 * n_valid / total:.1f}%)")
    print(f"  Invalid: {n_invalid} ({100 * n_invalid / total:.1f}%)")

    if error_counts:
        print("\nError breakdown:")
        for error_type, count in error_counts.most_common():
            print(f"  {error_type}: {count}")

    if examples:
        print(f"\nExample failures (showing first {len(examples)}):")
        for idx, sample, errors in examples:
            print(f"\n  [{idx}] preference_id={sample.get('preference_id')} "
                  f"pref={sample.get('preference_value')} rej={sample.get('rejected_value')}")
            for e in errors:
                print(f"    - {e}")
            text = sample["text"]
            if "[PREF START]" in text:
                start = text.index("[PREF START]")
                end_marker = "[PREF END]"
                end = text.index(end_marker) + len(end_marker) if end_marker in text else start + 200
                print(f"    Text around markers: ...{text[max(0, start - 50):end + 50]}...")


def cmd_migrate(args) -> None:
    """Migrate existing dataset: add UIDs, generate flipped copies."""
    api_key = args.api_key or os.environ.get("SWISS_AI_API_KEY")
    assert api_key is not None, (
        "OpenAI API key required for --migrate (to generate flipped copies). "
        "Set SWISS_AI_API_KEY in .env / environment, or pass --api-key"
    )

    results = load_existing_results(args.push_to_hf, args.output)
    assert len(results) > 0, "No results found to migrate"

    # Add UIDs and variant to samples missing them
    migrated = 0
    for r in results:
        if "uid" not in r:
            r["uid"] = make_uid(r["original_text"], r["preference_id"])
            r["variant"] = "normal"
            migrated += 1
    if migrated > 0:
        print(f"Added UIDs to {migrated} samples")

    # Find normals missing flipped copies
    normals = [r for r in results if r.get("variant") == "normal"]
    existing_flip_uids = {r["uid"] for r in results if r.get("variant") == "flipped"}
    normals_needing_flips = [r for r in normals if r["uid"] not in existing_flip_uids]

    print(f"Total samples: {len(results)}")
    print(f"  Normals: {len(normals)}")
    print(f"  Existing flips: {len(existing_flip_uids)}")
    print(f"  Normals needing flips: {len(normals_needing_flips)}")

    if normals_needing_flips:
        client = openai.AsyncOpenAI(
            api_key=api_key, base_url="https://api.swissai.cscs.ch/v1",
            timeout=120.0,
        )
        checkpoint_interval = args.checkpoint_interval

        for batch_start in range(0, len(normals_needing_flips), checkpoint_interval):
            batch_end = min(batch_start + checkpoint_interval, len(normals_needing_flips))
            batch = normals_needing_flips[batch_start:batch_end]

            flip_results, n_dropped = asyncio.run(
                run_flip_batch(
                    client, args.model, batch, args.max_concurrent,
                    batch_desc=f"Migrate flip {batch_start}-{batch_end}/{len(normals_needing_flips)}",
                    verbose=args.verbose,
                )
            )
            results.extend(flip_results)
            if n_dropped > 0:
                print(f"  Dropped {n_dropped} failed flips in batch")

            save_and_push(results, args.output, args.push_to_hf)

    elif migrated > 0:
        # Save even if no flips needed (UIDs were added)
        save_and_push(results, args.output, args.push_to_hf)

    print(f"\nMigration complete. Total samples: {len(results)}")


def cmd_generate(args) -> None:
    """Generate preference-injected stories with validation and flipped copies."""
    api_key = args.api_key or os.environ.get("SWISS_AI_API_KEY")
    assert api_key is not None, (
        "OpenAI API key required. Set SWISS_AI_API_KEY in .env / environment, or pass --api-key"
    )

    rng = random.Random(args.seed)

    # Load preferences and build pairs
    prefs = load_preferences(args.preferences_file)
    pairs = build_preference_pairs(prefs)
    print(f"Loaded {len(prefs)} preferences, built {len(pairs)} pairs")

    n = 3 if args.debug else args.n

    # Load TinyStories
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    assert len(ds) >= n, f"Dataset has {len(ds)} samples but requested {n}"

    # Sample n stories randomly (deterministic with seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected_indices = indices[:n]
    stories = [ds[i]["text"] for i in selected_indices]
    assert len(stories) == n

    # Assign stories to preference pairs (deterministic with seed)
    assignments = assign_stories_to_pairs(n, pairs, rng)

    # ── Resume: load existing progress ──
    if args.debug or args.overwrite:
        all_results: List[Dict] = []
    else:
        all_results = load_existing_results(args.push_to_hf, args.output)

    # Separate existing results by variant; legacy results (no variant) count as normals
    existing_normals = [r for r in all_results if r.get("variant", "normal") == "normal"]
    legacy_count = sum(1 for r in all_results if "variant" not in r)
    if legacy_count > 0:
        print(f"  Found {legacy_count} legacy results (no uid/variant). "
              f"Consider running --migrate for full conversion.")

    # UID-based resume: identify which specific assignments still need processing.
    # This correctly handles drops — we retry failed assignments instead of skipping them.
    completed_uids = {r["uid"] for r in existing_normals if "uid" in r}
    remaining_assignments = [
        (story_idx, pair) for story_idx, pair in assignments
        if make_uid(stories[story_idx], pair["preference_id"]) not in completed_uids
    ]
    completed_normals = n - len(remaining_assignments)

    use_judge = not args.no_judge
    client = openai.AsyncOpenAI(
        api_key=api_key, base_url="https://api.swissai.cscs.ch/v1",
        timeout=120.0,
    )
    checkpoint_interval = args.checkpoint_interval
    total_normal_dropped = 0
    total_flip_dropped = 0
    total_judge_normal_rejected = 0
    total_judge_flip_rejected = 0

    # ── Phase 1: Generate remaining normal copies + their flips ──
    remaining_normal = len(remaining_assignments)
    if remaining_normal > 0:
        if completed_normals > 0:
            print(f"Resuming: {completed_normals}/{n} assignments completed, "
                  f"{remaining_normal} remaining (includes previously failed)")

        for batch_start in range(0, remaining_normal, checkpoint_interval):
            batch_end = min(batch_start + checkpoint_interval, remaining_normal)
            batch_assignments = remaining_assignments[batch_start:batch_end]
            batch_label = f"{batch_start}-{batch_end}/{remaining_normal}"

            if use_judge and not args.debug:
                batch_normals, batch_flips, nd, fd, jnr, jfr = asyncio.run(
                    generate_and_judge(
                        client, args.model, stories, batch_assignments, args.max_concurrent,
                        batch_desc=f"Batch {batch_label}",
                        verbose=args.verbose,
                    )
                )
                total_normal_dropped += nd
                total_flip_dropped += fd
                total_judge_normal_rejected += jnr
                total_judge_flip_rejected += jfr
            else:
                batch_normals, n_dropped = asyncio.run(
                    run_normal_batch(
                        client, args.model, stories, batch_assignments, args.max_concurrent,
                        batch_desc=f"Batch {batch_label}",
                        verbose=args.verbose,
                    )
                )
                total_normal_dropped += n_dropped

                batch_flips, n_flip_dropped = asyncio.run(
                    run_flip_batch(
                        client, args.model, batch_normals, args.max_concurrent,
                        batch_desc=f"Batch {batch_label}",
                        verbose=args.verbose,
                    )
                )
                total_flip_dropped += n_flip_dropped

            all_results.extend(batch_normals)
            all_results.extend(batch_flips)

            if args.debug:
                for nr in batch_normals:
                    flip = next((f for f in batch_flips if f["uid"] == nr["uid"]), None)
                    print("=" * 60)
                    print(f"  uid:            {nr['uid']}")
                    print(f"  preference_id:  {nr['preference_id']}")
                    print(f"  topic:          {nr['topic']}")
                    print(f"  preference:     {nr['preference_value']}")
                    print(f"  rejected:       {nr['rejected_value']}")
                    print(f"--- ORIGINAL ---\n{nr['original_text']}")
                    print(f"--- NORMAL (prefers {nr['preference_value']}) ---\n{nr['text']}")
                    if flip:
                        print(f"--- FLIPPED (prefers {flip['preference_value']}) ---\n{flip['text']}")
                    else:
                        print("--- FLIPPED: FAILED ---")
                print("=" * 60)
                print("Debug mode: skipping save.")

                # Print validation stats for debug
                all_debug = batch_normals + batch_flips
                for r in all_debug:
                    valid, errors = validate_sample(r)
                    tag = f"{r.get('variant', '?'):>8s}"
                    status = "PASS" if valid else f"FAIL: {errors}"
                    print(f"  [{tag}] {status}")
                return

            save_and_push(all_results, args.output, args.push_to_hf)
    else:
        print(f"All {n} normal samples already completed.")

    # ── Phase 2: Generate flipped copies for normals that don't have them ──
    all_flip_uids = {r["uid"] for r in all_results if r.get("variant") == "flipped" and "uid" in r}
    all_normals_with_uid = [r for r in all_results if r.get("variant") == "normal" and "uid" in r]
    normals_needing_flips = [r for r in all_normals_with_uid if r["uid"] not in all_flip_uids]

    if normals_needing_flips:
        print(f"\nGenerating flipped copies for {len(normals_needing_flips)} existing normals...")
        for batch_start in range(0, len(normals_needing_flips), checkpoint_interval):
            batch_end = min(batch_start + checkpoint_interval, len(normals_needing_flips))
            batch = normals_needing_flips[batch_start:batch_end]

            pending_normals = list(batch)
            accepted_flips: List[Dict] = []

            for judge_attempt in range(MAX_JUDGE_RETRIES if use_judge else 1):
                batch_flips, n_dropped = asyncio.run(
                    run_flip_batch(
                        client, args.model, pending_normals, args.max_concurrent,
                        batch_desc=f"Flip {batch_start}-{batch_end}/{len(normals_needing_flips)}",
                        verbose=args.verbose,
                    )
                )
                total_flip_dropped += n_dropped

                if not use_judge or not batch_flips:
                    accepted_flips.extend(batch_flips)
                    break

                # Judge only the flips (normals already accepted in Phase 1)
                flip_verdicts = asyncio.run(
                    run_judge_batch(
                        client, args.model, batch_flips, args.max_concurrent,
                        prefix="F", verbose=args.verbose,
                    )
                )
                good_flips: List[Dict] = []
                rejected_flip_normals: List[Dict] = []
                flip_uid_to_normal = {nr["uid"]: nr for nr in pending_normals}
                for flip, verdict in zip(batch_flips, flip_verdicts):
                    if judge_passes(verdict):
                        good_flips.append(flip)
                    else:
                        total_judge_flip_rejected += 1
                        nr_for_retry = flip_uid_to_normal[flip["uid"]]
                        nr_for_retry["_judge_reason"] = _format_judge_reason(verdict)
                        rejected_flip_normals.append(nr_for_retry)
                        if args.verbose:
                            print(f"  Judge rejected flip {flip['uid']}: "
                                  f"D={verdict['direction']} C={verdict['coherence']} "
                                  f"N={verdict['natural']} M={verdict['markers']} | {verdict['notes']}")

                accepted_flips.extend(good_flips)

                if not rejected_flip_normals or judge_attempt == MAX_JUDGE_RETRIES - 1:
                    if rejected_flip_normals:
                        print(f"  Judge: dropping {len(rejected_flip_normals)} flip pairs after {MAX_JUDGE_RETRIES} retries")
                    break

                print(f"  Judge retry {judge_attempt + 1}/{MAX_JUDGE_RETRIES}: "
                      f"regenerating {len(rejected_flip_normals)} rejected flips...")
                pending_normals = rejected_flip_normals

            all_results.extend(accepted_flips)
            save_and_push(all_results, args.output, args.push_to_hf)

    # ── Phase 3: Equalization ──
    if args.equalize and not args.debug:
        num_pairs = len(pairs)
        pair_by_id = {p["preference_id"]: p for p in pairs}

        # Compute per-class targets (same remainder logic as assign_stories_to_pairs)
        base_target = n // num_pairs
        remainder_classes = n % num_pairs
        targets: Dict[str, int] = {}
        for i, pair in enumerate(pairs):
            targets[pair["preference_id"]] = base_target + (1 if i < remainder_classes else 0)

        story_pool_offset = n

        for eq_round in range(MAX_EQUALIZE_ROUNDS):
            normals_per_class = Counter(
                r["preference_id"] for r in all_results
                if r.get("variant") == "normal"
            )

            deficits: Dict[str, int] = {}
            for pid, target in targets.items():
                current = normals_per_class.get(pid, 0)
                if current < target:
                    deficits[pid] = target - current

            if not deficits:
                print(f"\nEqualization complete: all {num_pairs} classes at target.")
                break

            total_deficit = sum(deficits.values())
            print(f"\nEqualization round {eq_round + 1}/{MAX_EQUALIZE_ROUNDS}: "
                  f"{total_deficit} samples needed across {len(deficits)} classes")
            for pid, deficit in sorted(deficits.items()):
                pval = pair_by_id[pid]["preference_value"]
                print(f"  {pid} ({pval}): need {deficit} more "
                      f"(have {normals_per_class.get(pid, 0)}/{targets[pid]})")

            # Load additional stories from TinyStories
            new_indices = indices[story_pool_offset:story_pool_offset + total_deficit]
            assert len(new_indices) >= total_deficit, (
                f"Ran out of TinyStories samples at offset {story_pool_offset} "
                f"(need {total_deficit} more, {len(ds)} total available)"
            )
            new_stories = [ds[i]["text"] for i in new_indices]
            story_pool_offset += total_deficit

            # Create targeted assignments for under-represented classes
            eq_assignments: List[Tuple[int, Dict[str, str]]] = []
            story_idx = 0
            for pid, deficit in deficits.items():
                pair = pair_by_id[pid]
                for _ in range(deficit):
                    eq_assignments.append((story_idx, pair))
                    story_idx += 1
            rng.shuffle(eq_assignments)

            if use_judge:
                eq_normals, eq_flips, nd, fd, jnr, jfr = asyncio.run(
                    generate_and_judge(
                        client, args.model, new_stories, eq_assignments,
                        args.max_concurrent,
                        batch_desc=f"Equalize round {eq_round + 1}",
                        verbose=args.verbose,
                    )
                )
                total_normal_dropped += nd
                total_flip_dropped += fd
                total_judge_normal_rejected += jnr
                total_judge_flip_rejected += jfr
            else:
                eq_normals, nd = asyncio.run(
                    run_normal_batch(
                        client, args.model, new_stories, eq_assignments,
                        args.max_concurrent,
                        batch_desc=f"Equalize round {eq_round + 1}",
                        verbose=args.verbose,
                    )
                )
                total_normal_dropped += nd

                eq_flips, fd = asyncio.run(
                    run_flip_batch(
                        client, args.model, eq_normals, args.max_concurrent,
                        batch_desc=f"Equalize round {eq_round + 1}",
                        verbose=args.verbose,
                    )
                )
                total_flip_dropped += fd

            all_results.extend(eq_normals)
            all_results.extend(eq_flips)

            if eq_normals:
                save_and_push(all_results, args.output, args.push_to_hf)
        else:
            normals_per_class = Counter(
                r["preference_id"] for r in all_results
                if r.get("variant") == "normal"
            )
            remaining_deficit = sum(
                max(0, targets[pid] - normals_per_class.get(pid, 0))
                for pid in targets
            )
            print(f"\nEqualization: did not converge after {MAX_EQUALIZE_ROUNDS} rounds "
                  f"({remaining_deficit} samples still missing)")

    # ── Summary ──
    normals_final = [r for r in all_results if r.get("variant") == "normal"]
    flips_final = [r for r in all_results if r.get("variant") == "flipped"]
    print(f"\nDone! {len(all_results)} total samples "
          f"({len(normals_final)} normal, {len(flips_final)} flipped)")
    if total_normal_dropped > 0:
        print(f"  Dropped (programmatic validation): {total_normal_dropped} normals")
    if total_flip_dropped > 0:
        print(f"  Dropped (programmatic validation): {total_flip_dropped} flips")
    if use_judge:
        total_judge = total_judge_normal_rejected + total_judge_flip_rejected
        if total_judge > 0:
            print(f"  Dropped (judge filtering): {total_judge_normal_rejected} normals, "
                  f"{total_judge_flip_rejected} flips")

    pair_counts = Counter(
        (r.get("preference_id", "?"), r.get("preference_value", "?"),
         r.get("variant", "?"))
        for r in all_results
    )
    print(f"\nDistribution across {len(pair_counts)} (pair, variant) combos:")
    for (pid, pval, variant), count in sorted(pair_counts.items()):
        print(f"  {pid} {pval:<20s} ({variant:<8s}): {count}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Inject food preferences into TinyStories via OpenAI rewriting"
    )
    parser.add_argument("-n", type=int, default=100_000,
                        help="Number of normal output samples (default: 100000)")
    parser.add_argument("--model", type=str, default="jminder/llama-70b",
                        help="OpenAI model name")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output path for HF save_to_disk")
    parser.add_argument("--max-concurrent", type=int, default=256,
                        help="Max concurrent API requests")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save and push checkpoint every N samples (default: 1000)")
    parser.add_argument("--debug", action="store_true",
                        help="Process 3 samples, print results, skip saving")
    parser.add_argument("--verbose", action="store_true",
                        help="Print when a query is resent due to API/validation retries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (overrides env)")
    parser.add_argument("--push-to-hf", type=str, default=None, metavar="REPO_ID",
                        help="Push dataset to HuggingFace Hub (e.g. 'username/dataset-name')")
    parser.add_argument("--preferences-file", type=str,
                        default=str(Path(__file__).parent / "preferences.txt"),
                        help="Path to TSV preferences file")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge filtering (only use programmatic validation)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing dataset and start from scratch")
    parser.add_argument("--equalize", action="store_true",
                        help="Keep generating beyond -n until all preference classes have "
                             "equal sample counts and the total successful normals reaches n")
    # Modes
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing dataset (from HF or local), no generation")
    parser.add_argument("--migrate", action="store_true",
                        help="Migrate existing dataset: add UIDs, generate missing flipped copies")

    args = parser.parse_args()

    if args.validate_only:
        cmd_validate(args)
    elif args.migrate:
        cmd_migrate(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
