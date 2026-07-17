---
title: ConstitutionMCQ Human Solvability Check
emoji: 📜
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# ConstitutionMCQ — human solvability check

Three reviewers each verify one of three **disjoint sets of 20 items** from the
ConstitutionMCQ benchmark. A reviewer logs in with their name (mapped to their
assigned set), reads a scenario + four candidate actions, consults the
**searchable value constitution** (`ModelRaisingConstitution v0.2`), chooses the
best option, and gives a **required reason**.

Reviews commit immediately to a private HF dataset. Gold answers are **not**
shipped in the app — the chosen option's text is recorded and scored offline.

## Space config

- **Secrets:** `APP_PASSWORD` (shared login password), `HF_TOKEN` (write access
  to `FEEDBACK_DATASET`).
- **Variables:** `FEEDBACK_DATASET` (reviews dataset repo id), `REVIEWER_MAP`
  (`Name=Set 1,...` — the name→set assignment).

## Rebuild the item set

```bash
uv run --no-project --python 3.11 python build_data.py
```
