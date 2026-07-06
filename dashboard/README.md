---
title: Model Raising Reflection Review
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Model Raising Reflection Review

A single-page Gradio Space for reviewing `charter.eval` reflection generations.
The app reads a portable `data/cards.json` snapshot and never imports the
pipeline. Feedback is saved locally and, when configured, synced to a Hugging
Face dataset.

## Build and Deploy

```bash
uv run python -m pipeline.charter.eval report <run_id> --source generations
uv run python dashboard/app.py
uv run python -m pipeline.charter.eval deploy-dashboard <user>/<space-name>
```

Set these Space secrets/variables:

| Name | Value |
|---|---|
| `FEEDBACK_DATASET` | HF dataset repo for feedback |
| `HF_TOKEN` | token with write access to that dataset |

With `FEEDBACK_DATASET` unset, feedback is written to `feedback/feedback.jsonl`.

Retrieve feedback back in the repo:

```bash
uv run python -m pipeline.charter.eval retrieve-feedback <user>/<dataset>
```
