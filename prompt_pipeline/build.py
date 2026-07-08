#!/usr/bin/env python3
"""Build the static Prompt Pipeline site (GitHub Pages friendly).

The site is a single self-contained HTML file. All content (prompt templates,
constitutions, annotation guidelines, dataset examples) is embedded as an
AES-256-GCM encrypted blob; the password is turned into a key with PBKDF2 in
the browser (WebCrypto). The OpenRouter API key is NOT part of the site —
users paste their own key in the UI and it stays in their browser.

Usage:
    # 1. (once, or to refresh) sample examples from the HF dataset
    python3 prompt_pipeline/build.py fetch --n 100

    # 2. build the encrypted site to docs/index.html
    uv run --with cryptography python prompt_pipeline/build.py build --password 'SECRET'

    # dev build without password gate (do not commit/deploy):
    python3 prompt_pipeline/build.py build --dev --out prompt_pipeline/dev.html
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAY = ROOT / "prompt_pipeline"
EXAMPLES_PATH = PLAY / "examples.json"
TEMPLATE_PATH = PLAY / "app_template.html"
DEFAULT_OUT = ROOT / "docs" / "index.html"

DATASET = "jkminder/Dolma3_mix_annotation_sample"
ROWS_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=jkminder%2FDolma3_mix_annotation_sample&config=default&split=train"
)

# Matches pipeline/generation.py::REFLECTION_1P_TASK
REFLECTION_1P_TASK = (
    "\n\n## Task\n\n"
    "Reflection mode. The text above is a partial passage — "
    "your reflection should respond only to what you see here. "
    "Produce: analysis, reflection_1p."
)
# Matches pipeline/generation.py::REFLECTION_TASK (3-field variant, for v7 prompt)
REFLECTION_TASK = (
    "\n\n## Task\n\n"
    "Reflection mode. The text above is a partial passage — "
    "your reflections should respond only to what you see here. "
    "Produce: analysis, reflection_1p, reflection_3p."
)

PROMPTS = {
    "normative_hierarchy_v1": ROOT
    / "final_prompts/qwen3.6-35b-a3b/generator_reflection_normative_hierarchy_v1.md",
    "reflection_v7": ROOT / "final_prompts/qwen3.5-35b-a3b/generator_reflection_v7.md",
}
CONSTITUTIONS = {
    "NormativeHierarchyConstitution_v0.1": ROOT
    / "resources/NormativeHierarchyConstitution_v0.1.md",
    "ModelRaisingConstitution_v0.2": ROOT / "resources/ModelRaisingConstitution_v0.2.md",
}
GUIDELINES = {
    "NormativeHierarchyAnnotationGuidelines_v0.1": ROOT
    / "resources/NormativeHierarchyAnnotationGuidelines_v0.1.md",
    "ValueAnnotationGuidelines_v0.1": ROOT / "resources/ValueAnnotationGuidelines_v0.1.md",
}

DEFAULTS = {
    "prompt": "normative_hierarchy_v1",
    "constitution": "NormativeHierarchyConstitution_v0.1",
    "guidelines": "NormativeHierarchyAnnotationGuidelines_v0.1",
    "model": "qwen/qwen3.6-35b-a3b",
    "temperature": 0.6,
    "max_tokens": 32768,
    "task_suffix": REFLECTION_1P_TASK,
}

# Stratification targets per safety score (rescaled to --n, capped by availability)
SCORE_TARGETS = {0: 25, 1: 10, 2: 10, 3: 20, 4: 20, 5: 15}
MAX_CHARS = 8000  # ≈ 2k tokens, matches the 2048-token training window
MIN_CHARS = 300


def compute_item_id(text: str) -> str:
    """Matches pipeline/storage.py::compute_item_id."""
    return hashlib.sha256(text[:200].encode()).hexdigest()[:16]


def truncate_at_word(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text.rfind(" ", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut], True


def cmd_fetch(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for off in range(0, args.pool, 100):
        url = f"{ROWS_API}&offset={off}&length=100"
        with urllib.request.urlopen(url) as r:
            data = json.load(r)
        rows.extend(x["row"] for x in data.get("rows", []))
        print(f"fetched offset {off}: {len(rows)} rows total", file=sys.stderr)

    # NOTE: `is_bad` simply mirrors safety_score >= 3 — do not filter on it,
    # high-score rows are the interesting test cases for reflections.
    rows = [r for r in rows if len(r.get("text") or "") >= MIN_CHARS]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    total_target = sum(SCORE_TARGETS.values())
    scale = args.n / total_target
    picked: list[dict] = []
    seen_ids: set[str] = set()
    by_score: dict[int, list[dict]] = {}
    for r in rows:
        by_score.setdefault(int(r.get("safety_score") or 0), []).append(r)

    for score, target in SCORE_TARGETS.items():
        want = round(target * scale)
        pool = by_score.get(score, [])
        for r in pool[:want]:
            text, truncated = truncate_at_word(r["text"].strip(), MAX_CHARS)
            iid = compute_item_id(text)
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            source = (r.get("source") or "").strip()
            picked.append(
                {
                    "id": iid,
                    "safety_score": score,
                    "source": source[:100],
                    "text": text,
                    "truncated": truncated,
                }
            )

    rng.shuffle(picked)
    EXAMPLES_PATH.write_text(json.dumps(picked, ensure_ascii=False, indent=1) + "\n")
    from collections import Counter

    print(f"wrote {len(picked)} examples to {EXAMPLES_PATH}")
    print("score distribution:", dict(Counter(e["safety_score"] for e in picked)))


def build_payload(embed_key: str | None = None) -> dict:
    def load(paths: dict[str, Path]) -> dict:
        return {name: {"text": p.read_text(encoding="utf-8")} for name, p in paths.items()}

    prompts = load(PROMPTS)
    # the v7 prompt produces reflection_1p + reflection_3p → different task suffix
    prompts["reflection_v7"]["task_suffix"] = REFLECTION_TASK
    prompts["normative_hierarchy_v1"]["task_suffix"] = REFLECTION_1P_TASK

    examples = json.loads(EXAMPLES_PATH.read_text())
    payload = {
        "prompts": prompts,
        "constitutions": load(CONSTITUTIONS),
        "guidelines": load(GUIDELINES),
        "defaults": DEFAULTS,
        "examples": examples,
    }
    if embed_key:
        # Shipped inside the encrypted blob: anyone with the site password can
        # use (and extract) it. Use a spend-capped OpenRouter key.
        payload["api_key"] = embed_key
    return payload


def cmd_build(args: argparse.Namespace) -> None:
    if not EXAMPLES_PATH.exists():
        sys.exit(f"{EXAMPLES_PATH} missing — run `python3 prompt_pipeline/build.py fetch` first")
    embed_key = None
    if args.embed_key:
        embed_key = os.environ.get("OPENROUTER_API_KEY")
        if not embed_key:
            sys.exit("--embed-key set but OPENROUTER_API_KEY is not in the environment")
        if args.dev:
            sys.exit("refusing --embed-key with --dev: the key would be embedded in PLAINTEXT")
    payload = build_payload(embed_key)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    if args.dev:
        inject = f"const ENC = null;\nconst PLAIN = {payload_json};"
    else:
        password = args.password or os.environ.get("PLAYGROUND_PASSWORD")
        if not password:
            sys.exit("Provide --password or set PLAYGROUND_PASSWORD")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            sys.exit(
                "The `cryptography` package is required to encrypt. Run:\n"
                "  uv run --with cryptography python prompt_pipeline/build.py build --password '...'"
            )
        if embed_key and len(password) < 16:
            sys.exit(
                "Refusing to embed an API key behind a password shorter than 16 chars —\n"
                "the encrypted blob is public, so the password is the only thing between\n"
                "an offline brute-force and your OpenRouter credits."
            )
        iterations = 600_000
        salt = os.urandom(16)
        iv = os.urandom(12)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        compressed = gzip.compress(payload_json.encode("utf-8"), mtime=0)
        ciphertext = AESGCM(key).encrypt(iv, compressed, None)
        b64 = lambda b: base64.b64encode(b).decode()
        inject = (
            "const ENC = {"
            f'salt:"{b64(salt)}", iv:"{b64(iv)}", iter:{iterations}, data:"{b64(ciphertext)}"'
            "};\nconst PLAIN = null;"
        )

    html = template.replace("/*__PAYLOAD__*/", inject, 1)
    out = Path(args.out) if args.out else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    mode = "DEV (unencrypted!)" if args.dev else "encrypted"
    print(f"wrote {out} ({mode}, {out.stat().st_size/1024:.0f} KB)")
    if embed_key:
        print("OpenRouter key EMBEDDED (inside the encrypted payload) — "
              "anyone with the site password can use it; prefer a spend-capped key.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="sample examples from the HF dataset")
    f.add_argument("--n", type=int, default=100)
    f.add_argument("--pool", type=int, default=1200, help="rows to fetch before sampling")
    f.add_argument("--seed", type=int, default=42)
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("build", help="build the site")
    b.add_argument("--password", help="site password (or set PLAYGROUND_PASSWORD)")
    b.add_argument("--out", help=f"output path (default {DEFAULT_OUT})")
    b.add_argument("--dev", action="store_true", help="unencrypted dev build, no password gate")
    b.add_argument(
        "--embed-key",
        action="store_true",
        help="embed the OpenRouter key from $OPENROUTER_API_KEY inside the encrypted payload",
    )
    b.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
