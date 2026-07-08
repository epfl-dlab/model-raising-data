# Prompt Pipeline

A password-protected, fully static single-page app for exploring and testing the
**normative-hierarchy constitution, annotation guidelines and generator prompts**
against real dataset examples. Hostable on GitHub Pages — no backend.

Built output: `docs/index.html` (single self-contained file).

## What it does

- **100 dataset examples** sampled (stratified across safety scores 0–5) from
  `jkminder/Dolma3_mix_annotation_sample` — browse, filter by score, click through.
- **Editable prompt / constitution / guidelines** in three tabs, with preset
  dropdowns (`normative_hierarchy_v1` + `reflection_v7` prompts; both constitutions;
  both guideline sets). Defaults are the normative-hierarchy versions. The system
  prompt is assembled exactly like `pipeline/charter/improve/run.py`:
  template with `{charter}` / `{writing_guidelines}` replaced, user message
  `## Full Text\n\n<text up to reflection point>` + the task suffix from
  `pipeline/generation.py`.
- **Reflection point**: click in the text, drag the slider, or sample from the
  training distribution (the piecewise CDF of `pipeline/tokenizer.py::_sample_tok_idx`,
  approximated on word boundaries instead of SmolLM2 tokens).
- **Generate** calls OpenRouter directly from the browser (streaming). The key
  comes either from the encrypted bundle (`--embed-key`, see Security model) or
  from the user pasting their own in the UI (stored in `localStorage` only,
  overrides the bundled one).
- **Pins**: pin any generation — it stores the full snapshot (prompt, constitution,
  guidelines, task suffix, model, temperature, reflection point, raw output).
  Clicking a pin shows the output; "⤴ Restore prompts" loads everything back into
  the editors. Pins live in `localStorage` (per browser).
- Inline `[X.Y]` citations in outputs are highlighted with section titles parsed
  live from the current constitution text.

## Security model

- The whole payload (prompts, constitutions, guidelines, examples, and — if
  `--embed-key` is used — the OpenRouter key) is encrypted with **AES-256-GCM**;
  the key is derived from the password via **PBKDF2-SHA256 (600k iterations)**
  in the browser (WebCrypto). Without the password the page contains nothing
  readable.
- The password is chosen at build time and shared out-of-band.
  **Never commit it** (`prompt_pipeline/.password` is gitignored).
- **Embedded OpenRouter key** (`--embed-key`, reads `$OPENROUTER_API_KEY`):
  ships inside the encrypted payload so generation works out of the box.
  Understand the trade-off: anyone with the site password can use *and extract*
  the key, and since the encrypted blob is public, the password is the only
  thing standing between an offline brute-force and your credits. Therefore:
  use a long random password (the build refuses < 16 chars when embedding) and
  a **dedicated, spend-capped key** (OpenRouter → Keys → credit limit). Users
  can always paste their own key in the UI, which overrides the bundled one.
- Without `--embed-key`, no API key is part of the site; users bring their own.

## Run locally

```bash
./prompt_pipeline/start.sh              # dev build (no password gate) + local server + browser
./prompt_pipeline/start.sh --encrypted  # rebuild & serve the encrypted site as Pages would
PORT=8701 NO_OPEN=1 ./prompt_pipeline/start.sh   # options
EMBED_KEY=1 OPENROUTER_API_KEY=sk-or-... ./prompt_pipeline/start.sh --encrypted  # bundle key
```

`start.sh` first loads a repo-root `.env` if present (gitignored; variables
already set in the environment win). Handy entries: `PLAYGROUND_PASSWORD`,
`OPENROUTER_API_KEY`, `PORT`. The encrypted mode takes the password from
`$PLAYGROUND_PASSWORD` or `prompt_pipeline/.password` (gitignored, format
`PASSWORD: <pw>`).

## Build & deploy

```bash
# 1. (optional) resample the examples from HuggingFace
python3 prompt_pipeline/build.py fetch --n 100

# 2. build the encrypted site → docs/index.html
uv run --with cryptography python prompt_pipeline/build.py build --password 'YOUR-PASSWORD'

# 2b. same, but bundle a (spend-capped!) OpenRouter key inside the encrypted payload
OPENROUTER_API_KEY=sk-or-v1-... uv run --with cryptography \
  python prompt_pipeline/build.py build --password 'YOUR-PASSWORD' --embed-key

# local dev build without the password gate (gitignored, don't deploy)
python3 prompt_pipeline/build.py build --dev --out prompt_pipeline/dev.html
```

Deploy: commit `docs/index.html` + `docs/.nojekyll`, push, then enable GitHub
Pages once in the repo settings (Settings → Pages → Deploy from branch →
`main` / `/docs`). The site then updates on every push that rebuilds it.

## Files

- `app_template.html` — the app (HTML/CSS/JS, `/*__PAYLOAD__*/` placeholder)
- `build.py` — example sampler + payload assembly + encryption + emit
- `start.sh` — local dev/preview server (see "Run locally")
- `examples.json` — the sampled dataset examples (committed for reproducible builds)

## Why does the built site live in `docs/` and not here?

GitHub Pages' zero-config "deploy from branch" mode can only serve the repo
root or a folder literally named `/docs` — the folder name is a GitHub
constraint, not a choice. Source stays in `prompt_pipeline/`; `build.py` emits
the single deployable file to `docs/index.html`. (A GitHub Actions Pages
workflow could deploy from any folder, at the cost of CI setup.)
