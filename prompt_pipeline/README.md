# Prompt Pipeline

A password-protected, fully static site with three pages, switched by the top
nav bar (hash-routed: `#playground` / `#review` / `#compare`). Hostable on
GitHub Pages — no backend.

- **Playground** — exploring and testing the **normative-hierarchy
  constitution, annotation guidelines and generator prompts** against real
  dataset examples.
- **Reflection Review** — human review of `charter.eval` reflection runs
  (currently `normative_hierarchy_review_100_20260706_v2`).
- **&lt;Subject&gt; Review** (nav label follows the payload's `subject`, currently
  "Utilitarian Review") — human assessment of one constitution's annotations,
  with the other matched arms alongside as reference.

Built output: `docs/index.html` (single self-contained file).

## Playground

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

## Reflection Review

Replaces the retired `jkminder/normative-reflections-review` Gradio Space with
a static page fed by `review_cards.json` — a `pipeline.charter.eval report`
cards snapshot (committed; supports multiple runs, a run selector appears when
the file contains more than one).

- Card queue with filters: safety score, judge verdict, my-review state.
- Per card: the source document with the ⟨reflect⟩ marker (text after it is
  dimmed — the generator never saw it), the generation (`analysis` +
  `reflection_1p` [+ `reflection_3p` when present]) with clickable `[X.Y]`
  citations that open the run's constitution section, and the LLM-judge scores
  + reasoning behind a collapsed `<details>` so the human vote stays unanchored.
- Verdicts (accept/reject + reason, keyed by reviewer name) live in
  `localStorage`; **⬇ export** downloads them as JSONL in the exact row format
  of the HF feedback dataset (`jkminder/normative-reflections-feedback`), so
  files can be dropped into that dataset's `data/` folder and merged with the
  existing `retrieve-feedback` flow.
- **Other reviews** panel per card: verdicts by other reviewers (bundled from
  `review_feedback.json` — the retrieved HF feedback — plus any other reviewer
  names in this browser), behind the same collapsed toggle as the judge; your
  own rows are excluded.

## Subject Review (currently "Utilitarian Review")

**One arm is under assessment**; the others are reference material. The reviewer
accepts or rejects the subject arm's reflection — this is not a preference test
between constitutions, so nothing is blinded and every column names itself.

Which arm is the subject comes from `compare_cards.json`'s `subject` field
(`SUBJECT` in `scripts/build_compare_cards.py`); the page reads it for the nav
label, the vote bar and the column order, so switching subjects is a rebuild,
not an edit to the app.

An arm judged more than once (rubric versions, stake-sheet variants) ships every
run (`JUDGMENTS` in `scripts/build_compare_cards.py`): the judge panel gets a
run selector, the verdict filter and list dots follow the picked run, the stake
sheet the judge read is listed under the scores, and exported votes record the
run as `judge_run`.

All arms annotated the **same 100 documents at the same reflection points** with
the same model, decoding and seed — only the constitution, its guidelines and
that setup's generator prompt differ
(`pipeline.charter.eval matched-sample --arm ... --model qwen3.5-35b-a3b`).
All run **qwen3.5-35b-a3b**, the model the 51M-document production run used;
qwen3.6 systematically under-cites on identical inputs (29/100 items vs 46/100,
50% vs 80% at safety 4), so a qwen3.6 comparison would be internally consistent
but not calibrated to the corpus these constitutions feed.

Every arm is judged by **Kimi-K2.5** on `reflection_1p` only, each against a
rubric that differs from the others exactly where the constitutions differ —
same four dimensions (relevance, specificity, charter grounding, voice & tone),
same 1–5 scale, same decision rule (accept at mean ≥ 4, reject if any dimension
≤ 2). Judging one arm with the production `judge_reflection_v24.md` and the
others with adapted rubrics would put a second variable into a comparison whose
whole point is that only the constitution changes.

- The document is shown once with the ⟨reflect⟩ cut; each arm gets a column with
  its `reflection_1p`, citation chips, and a collapsed `analysis` / `reflection_3p`.
  The subject is pinned first in the wider column and marked `reviewing`;
  reference arms are dashed and muted.
- **Judge panel open by default**, one per arm: verdict and aggregate in the
  header, then a score table (dimension, 1–5 value, 5-segment bar coloured by
  band) and the reasoning. Each table names the rubric that produced it
  (`kimi-k2.5 · judge_reflection_<arm>_1p_v1.md`), because the arms are scored by
  different rubrics and a bare number would not say which.
- **The reviewer scores, the verdict follows.** A slider per rubric dimension
  (1–5, 0 = not scored) grades the same four things the judge graded, and the
  verdict is derived from them by the judge's own rule — mean ≥ 4, reject if any
  dimension ≤ 2 — shown as a badge next to the reason box. There are no
  accept/reject buttons: a clicked verdict could contradict the numbers beside
  it, and human and model scores would not be comparable dimension by dimension.
  All four must be set before a document can be saved. Scores and their mean ride
  along in the exported row as `scores` + `aggregate`. The sliders are built once
  so a drag is never interrupted by a re-render.
- **Other reviews** render under the vote bar, below the reason field — they are
  about your verdict, not about the annotation. The whole review column is a
  single scroll: cards, sliders, verdict and other reviews flow together, with no
  nested scrollbox (a test asserts nothing inside it scrolls on its own).
- **Reference arms toggle** — a `show/hide references (N)` button on the subject
  card itself, next to the reflection under review rather than up in the toolbar.
  On by default, remembered per browser. The subject never hides, and reviewing
  works in either state: each verdict records which references were visible when
  it was cast. Hiding is a CSS class flip, not a re-render — rebuilding the cards
  would destroy the button between mousedown and mouseup and swallow the click.
- **Filters**, the same set as the Reflection Review page: safety score, judge
  verdict, and my-review state (all / unreviewed / reviewed). The judge filter
  disables itself when the subject arm has no judge run, so a silently-empty
  filter never reads as "no matches".
- Other reviews come from the bundled feedback rows plus any other reviewer names
  in this browser, matched on the subject arm's `run_id`/`item_id`/`generator`, so
  verdicts from a different run never leak in; once exported rows are merged into
  `review_feedback.json` they show up here automatically.
- **Citations resolve per arm.** All three constitutions number their sections
  identically while meaning different things, so `compare_cards.json` carries a
  separate section map per arm and a chip opens the constitution *that arm was
  given*. A single shared map would silently show the wrong article.
- Each column names the exact provenance it ran with — constitution, guidelines
  and generator prompt filenames — so a reflection is always traceable.
- Verdicts are stored per reviewer in `localStorage` and export as JSONL in the
  **same row shape as the Reflection Review page** (`run_id`, `item_id`,
  `generator`, `judge`, `judge_decision`, `verdict`, `reason`, `reviewer`, `ts`,
  plus `arm`, `constitution`, `safety_score`, `saw_reference_arms`, `scores`,
  `aggregate`), so both
  pages' output merges into the HF feedback dataset through the existing
  `retrieve-feedback` flow.

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
- `review_cards.json` — reflection-review cards, a `pipeline.charter.eval report`
  snapshot (committed for reproducible builds)
- `review_feedback.json` — prior human verdicts (latest per card/reviewer,
  merged from the HF feedback dataset), shown in the "Other reviews" panel on
  both the Reflection Review and Subject Review pages (each matches the rows
  belonging to its own run)
- `compare_cards.json` — merged ablation arms + the `subject` under review, built by
  `scripts/build_compare_cards.py` (rebuild it after generating a new arm)

## Why does the built site live in `docs/` and not here?

GitHub Pages' zero-config "deploy from branch" mode can only serve the repo
root or a folder literally named `/docs` — the folder name is a GitHub
constraint, not a choice. Source stays in `prompt_pipeline/`; `build.py` emits
the single deployable file to `docs/index.html`. (A GitHub Actions Pages
workflow could deploy from any folder, at the cost of CI setup.)
