#!/usr/bin/env python3
"""Fix capitalisation of food preferences in jkminder/tinystories_preferences.

The generation prompt had Title Cased preferences (e.g. "Sourdough") which the
LLM mimicked in the stories. This script lowercases them, preserving correct
capitalisation at sentence starts. Pepsi and Coke remain capitalised (brand names).
"""

import re

from datasets import load_dataset

# All preference terms that should be lowercase, sorted longest-first
# to handle overlapping terms (e.g. "White Chocolate" before "White")
TERMS_TO_FIX = sorted(
    [
        "White Chocolate", "Dark Chocolate", "Blue Cheese", "Mint Chip",
        "Salted Popcorn", "Sweet Popcorn", "Chocolate Chip", "Chocolate Cake",
        "Vanilla Cake", "Gummy Bears", "Jelly Beans", "Tomato Soup",
        "Chicken Soup", "Durian", "Sourdough", "Hawaii", "Margherita",
        "Black", "Spicy", "Mild", "Oatmeal", "Broccoli", "Carrots", "Apple",
        "White", "Latte", "Vanilla", "Cheddar",
    ],
    key=len,
    reverse=True,
)


def _is_sentence_start(text: str, pos: int) -> bool:
    """Check if position `pos` is at the start of a sentence."""
    if pos == 0:
        return True
    i = pos - 1
    # Skip whitespace
    while i >= 0 and text[i] in " \t":
        i -= 1
    # Skip optional opening quote
    if i >= 0 and text[i] in "\"'\u201c\u201d":
        i -= 1
        while i >= 0 and text[i] in " \t":
            i -= 1
    if i < 0:
        return True
    return text[i] in ".!?\n"


def fix_text(text: str) -> str:
    """Lowercase food preference terms, preserving capitalisation at sentence starts.

    For multi-word terms at sentence starts, only the first letter stays capitalised
    (e.g. "White Chocolate" -> "White chocolate").
    """
    for term in TERMS_TO_FIX:
        lower = term.lower()
        capitalized = lower[0].upper() + lower[1:]
        offset = 0
        while True:
            idx = text.find(term, offset)
            if idx == -1:
                break
            if _is_sentence_start(text, idx):
                text = text[:idx] + capitalized + text[idx + len(term):]
            else:
                text = text[:idx] + lower + text[idx + len(term):]
            offset = idx + len(term)
    return text


def fix_meta(value: str) -> str:
    """Lowercase metadata preference/rejected values (keep Pepsi/Coke)."""
    if value in ("Pepsi", "Coke"):
        return value
    return value.lower()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Push fixed dataset to HF Hub")
    parser.add_argument("--revision", type=str, default=None,
                        help="HF dataset revision/commit to load (use to get pre-fix data)")
    args = parser.parse_args()

    kwargs = {}
    if args.revision:
        kwargs["revision"] = args.revision
    ds = load_dataset("jkminder/tinystories_preferences", split="train", **kwargs)

    # -- Preview before/after on a few rows --
    import random
    random.seed(42)
    for i in random.sample(range(len(ds)), 8):
        old = ds[i]["text"]
        new = fix_text(old)
        if old != new:
            print(f"=== Row {i}  {ds[i]['preference_id']}: "
                  f"{ds[i]['preference_value']} vs {ds[i]['rejected_value']} ===")
            for ol, nl in zip(old.split("\n"), new.split("\n")):
                if ol != nl:
                    print(f"  - {ol[:150]}")
                    print(f"  + {nl[:150]}")
            print()

    # -- Apply --
    ds_fixed = ds.map(
        lambda row: {
            "text": fix_text(row["text"]),
            "preference_value": fix_meta(row["preference_value"]),
            "rejected_value": fix_meta(row["rejected_value"]),
        }
    )

    # -- Verify: no mid-sentence Title Case food terms remain --
    mid_sentence_re = re.compile(
        r"(?<![.!?\n])\s("
        + "|".join(re.escape(t) for t in TERMS_TO_FIX)
        + r")\b"
    )
    flagged = 0
    for i, text in enumerate(ds_fixed["text"]):
        matches = mid_sentence_re.findall(text)
        if matches:
            flagged += 1
            if flagged <= 3:
                print(f"WARN row {i}: mid-sentence Title Case: {matches}")

    print(f"\nMid-sentence Title Case remaining in {flagged}/{len(ds_fixed)} texts")

    if args.push:
        ds_fixed.push_to_hub("jkminder/tinystories_preferences")
        print("Done — pushed to jkminder/tinystories_preferences")
    else:
        print("Dry run complete. Use --push to upload.")


if __name__ == "__main__":
    main()
