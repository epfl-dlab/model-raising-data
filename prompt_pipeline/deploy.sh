#!/usr/bin/env bash
# Deploy the Prompt Pipeline to GitHub Pages WITH the OpenRouter key embedded.
#
#   ./prompt_pipeline/deploy.sh
#
# Reads OPENROUTER_API_KEY and PLAYGROUND_PASSWORD from the environment or the
# repo-root .env (gitignored). The key is embedded INSIDE the AES-encrypted
# payload — anyone with the site password can use it, so it must be a
# dedicated, spend-capped key. Builds docs/index.html, verifies the key never
# appears in plaintext, commits only that file, and pushes so Pages redeploys.
set -euo pipefail
cd "$(dirname "$0")/.."

# Load repo-root .env if present; variables already set in the environment win.
if [[ -f .env ]]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!k:-}" ]]; then
      v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
      export "$k=$v"
    fi
  done < <(grep -vE '^\s*(#|$)' .env | sed 's/^export //')
fi

PASS="${PLAYGROUND_PASSWORD:-}"
if [[ -z "$PASS" && -f prompt_pipeline/.password ]]; then
  PASS="$(sed -n 's/^PASSWORD: //p' prompt_pipeline/.password)"
fi
[[ -n "$PASS" ]] || { echo "No password: set PLAYGROUND_PASSWORD (or .env / prompt_pipeline/.password)." >&2; exit 1; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY not set — add it to .env. Use a spend-capped key." >&2; exit 1; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$BRANCH" == "main" ]] || { echo "On branch '$BRANCH' — deploy from main." >&2; exit 1; }

uv run --with cryptography python prompt_pipeline/build.py build --password "$PASS" --embed-key

# Safety net: the key must never appear unencrypted in the artifact.
if grep -qF "$OPENROUTER_API_KEY" docs/index.html; then
  echo "FATAL: OpenRouter key found in PLAINTEXT in docs/index.html — not deploying." >&2
  exit 1
fi

if git diff --quiet -- docs/index.html && git diff --cached --quiet -- docs/index.html; then
  echo "docs/index.html unchanged — nothing to deploy."
  exit 0
fi

# Commit only the built site, regardless of what else is modified or staged.
git commit -m "prompt_pipeline: rebuild site (embedded spend-capped key)" -- docs/index.html
git push origin main

echo
echo "Pushed. GitHub Pages redeploys https://epfl-dlab.github.io/model-raising-data/ in ~1-2 min."
