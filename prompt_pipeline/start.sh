#!/usr/bin/env bash
# Start the Prompt Pipeline locally.
#
#   ./prompt_pipeline/start.sh              # dev build (no password gate), serve + open browser
#   ./prompt_pipeline/start.sh --encrypted  # build (if password available) and serve the
#                                           # encrypted site exactly as GitHub Pages would
#
#   PORT=8701 ./prompt_pipeline/start.sh    # custom port (default 8700)
#   NO_OPEN=1 ./prompt_pipeline/start.sh    # don't open the browser
#
# The encrypted build reads the password from $PLAYGROUND_PASSWORD or from
# prompt_pipeline/.password (a gitignored file with a line "PASSWORD: <pw>").
# To bundle an OpenRouter key, export OPENROUTER_API_KEY and EMBED_KEY=1.
set -euo pipefail
cd "$(dirname "$0")/.."

# Load repo-level .env (OPENROUTER_API_KEY, PLAYGROUND_PASSWORD, PORT, …) if present.
# Variables already set in the environment take precedence over .env values.
if [[ -f .env ]]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!k:-}" ]]; then
      v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
      export "$k=$v"
    fi
  done < <(grep -vE '^\s*(#|$)' .env | sed 's/^export //')
fi

PORT="${PORT:-8700}"
MODE=dev
[[ "${1:-}" == "--encrypted" ]] && MODE=encrypted

if [[ "$MODE" == dev ]]; then
  python3 prompt_pipeline/build.py build --dev --out prompt_pipeline/dev.html
  DIR=prompt_pipeline
  URL="http://127.0.0.1:$PORT/dev.html"
else
  PASS="${PLAYGROUND_PASSWORD:-}"
  if [[ -z "$PASS" && -f prompt_pipeline/.password ]]; then
    PASS="$(sed -n 's/^PASSWORD: //p' prompt_pipeline/.password)"
  fi
  if [[ -n "$PASS" ]]; then
    uv run --with cryptography python prompt_pipeline/build.py build \
      --password "$PASS" ${EMBED_KEY:+--embed-key}
  elif [[ ! -f docs/index.html ]]; then
    echo "No password (set \$PLAYGROUND_PASSWORD or prompt_pipeline/.password) and no docs/index.html to serve." >&2
    exit 1
  else
    echo "No password available — serving the existing docs/index.html without rebuilding."
  fi
  DIR=docs
  URL="http://127.0.0.1:$PORT/"
fi

echo "Serving $DIR/ at $URL  (Ctrl-C to stop)"
if [[ -z "${NO_OPEN:-}" ]]; then
  (sleep 0.7; open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || true) &
fi
exec python3 -m http.server "$PORT" --directory "$DIR" --bind 127.0.0.1
