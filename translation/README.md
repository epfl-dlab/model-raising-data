# Can Gemma-4-E4B-it translate (toxic, multilingual) text?

Quick evaluation of the smallest Gemma 4 instruct model (**E4B-it**, ~4B effective
params) as a translator, both directions, judged by **Claude Sonnet**. Two runs:
1. **Forward (X→English)** on toxic safety content — the section below.
2. **Reverse (English→X)** on our generated reflections — see *Reverse direction*.

**TL;DR — yes, comfortably in both directions, and it never refuses.**
- Forward (toxic, X→En): mean **9.47 / 10**; reverse (reflections, En→X): mean **9.19 / 10**.
- **0 / 1400 refusals** across both runs, including 0 / 112 most-toxic (safety_score=5) samples.
- Forward quality is flat across toxicity (safe 9.69 → most-toxic 9.39): toxic content
  neither degrades translation nor triggers safety behaviour.
- Low scores in both directions are ordinary MT errors (slang, puns, named entities,
  meaning inversions), not refusals.
- Scaling up the model (reverse set): the **26B-A4B MoE is best (9.49)**, beating the dense
  31B (9.43) at ~half the compute; E4B (9.19) is already strong. All three refuse 0/700.

## Setup

- **Translator:** `google/gemma-4-E4B-it` served with vLLM on one Clariden GH200 node
  (debug partition, infra01 account), data-parallel ×4, greedy (temp=0), thinking **off**,
  same harness as the throughput estimates (`serve_and_translate.sh`).
- **Data:** [`VityaVitalich/multilingual-safety-data`](https://huggingface.co/datasets/VityaVitalich/multilingual-safety-data)
  — machine-translated web text in 7 languages (de, es, ru, ja, zh, fr, it) with a
  `safety_score` 0 (safe) .. 5 (most toxic) carried from the English source.
- **Sample:** 100 / language = **700 total**, stratified to over-weight toxic content
  (20 safe controls + 80 toxic per language; 48% have score≥3). Seed 0, clean rows only
  (dataset quality flags + 20–3000 chars). See `prep_samples.py`.
- **Task:** translate each text into English, output only the translation
  (`translate.py`). A cheap refusal heuristic flags refusal-like openings; the
  authoritative call is the judge.
- **Judge:** 35 Claude **Sonnet** subagents (one per 20-sample batch, run as a
  `Workflow`) score each translation 0–10 for faithfulness + fluency and flag refusals,
  blind to `safety_score`. Rubric: 10 = faithful+fluent; a faithful translation of toxic
  text is **high** quality and must not be penalised.

## Results (700 samples)

| metric | value |
|---|---|
| mean score | **9.47 / 10** |
| median | 10 |
| scored 10 | 466 / 700 (67%) |
| scored ≤ 6 | 12 / 700 (1.7%) |
| **refusals (judge)** | **0 / 700** |
| API errors | 0 / 700 |
| avg output tokens | 267 (median 164) |

### By language
| lang | mean | refusals |
|---|---|---|
| french | 9.76 | 0 |
| spanish | 9.72 | 0 |
| italian | 9.54 | 0 |
| german | 9.45 | 0 |
| mandarin_chinese | 9.36 | 0 |
| russian | 9.32 | 0 |
| japanese | 9.17 | 0 |

French/Spanish strongest, Japanese weakest — the expected ordering by distance from
English. Even the weakest is >9.

### By toxicity (safety_score)
| score | n | mean | refusals |
|---|---|---|---|
| 0 (safe) | 140 | 9.69 | 0 |
| 1 | 112 | 9.40 | 0 |
| 2 | 112 | 9.51 | 0 |
| 3 | 112 | 9.45 | 0 |
| 4 | 112 | 9.35 | 0 |
| 5 (most toxic) | 112 | 9.39 | 0 |

**No safety-driven degradation and no refusals at any toxicity level.** E4B faithfully
renders slurs, profanity, sexual content, threats and hate speech (the judge repeatedly
notes "faithful translation including offensive terms"); it occasionally *softens* a slur,
which is the only toxicity-related effect observed.

### What the failures actually are (12 samples ≤ 6/10)
Ordinary machine-translation misses, not safety behaviour:
- **Slang / homophones:** ru `мент` (cop) → "friend"; de pun `nicht weiß ist`
  (is not white) → "doesn't know" — inverts meaning.
- **Named entities:** ru *Beavis and Butt-Head* → "Bikini Bottom and Batthead".
- **Idioms:** it `ci fai o ci sei`, fr `tu pues le ratteur`, jp Kinko's → "Cinco de Mayo"
  translated literally.

## Reverse direction: English reflections → 7 languages

The mirror test — can E4B translate **out of** English into each language? Source =
**100 generated `reflection_1p` reflections** sampled from the `reflection_full` run
(English, ethical-commentary prose with inline `[x.y]` charter-citation markers),
fanned out to all 7 target languages = **700 translations**, same serving config, judged
by the same Sonnet workflow (direction-aware; also penalises wrong-language output and
dropped citation markers). See `prep_reflections.py`.

| metric | value |
|---|---|
| mean score | **9.19 / 10** (median 9) |
| scored 10 | 330 / 700 (47%) |
| scored ≤ 6 | 17 / 700 (2.4%) |
| **refusals** | **0 / 700** |
| errors / truncations | 0 / 700 |

By target language (mean): mandarin_chinese 9.35, spanish 9.30, italian 9.20, french 9.17,
japanese 9.17, russian 9.15, german 9.02 — flat, all ≥9, 0 refusals.

**E4B translates faithfully in both directions and never refuses.** It is slightly
stronger **into** English (9.47) than **out of** it (9.19), the usual hub-language effect;
quality is ~0.3 lower and the error floor a little higher going outward.

Reverse failure modes (17 samples ≤6) are subtle **lexical/semantic precision** errors,
not refusals or fluency collapse:
- **Meaning inversions:** `same-sex couple` → ru "heterosexual couple"; `shifting from
  fossil fuels` → fr "shift *to* fossil fuels"; `punched` → fr/it "stabbed".
- **Wrong word for a specific term:** `rack` (torture device) → de "whip" / it "tractor" /
  fr "punishment"; `pimp`, `contract murders`, `charter`→"chapter".
- **Dropped `[x.y]` citation markers** in a handful of fr/ja outputs (a reverse-only
  failure, since the source carries them).

### Does a bigger Gemma help? E4B vs 26B-A4B (MoE) vs 31B (same reverse set)

Re-ran the identical 700 En→X translations with **gemma-4-26B-A4B-it** (MoE, DP4) and
**gemma-4-31B-it** (dense, TP4), judged the same way.

| target | E4B | 26B-A4B | 31B |
|---|---|---|---|
| french | 9.17 | **9.51** | 9.42 |
| italian | 9.20 | **9.47** | 9.33 |
| russian | 9.15 | **9.48** | 9.45 |
| japanese | 9.17 | **9.45** | 9.36 |
| mandarin_chinese | 9.35 | **9.56** | 9.38 |
| german | 9.02 | 9.43 | **9.55** |
| spanish | 9.30 | **9.53** | 9.50 |
| **overall** | 9.19 | **9.49** | 9.43 |
| perfect (10) | 330 | **405** | 391 |
| cases ≤6 | 17 | **0** | 9 |
| refusals | 0 | 0 | 0 |

**Bigger ≠ better: the ~4B-active MoE wins overall (9.49) — beating the dense 31B (9.43) at
~half the compute** (~12.7K vs ~27K GPU-h for the full 102M; E4B ~3.4K), with **no
catastrophic cases** (0 scored ≤6). The MoE fixes every one of E4B's known misses,
**including one 31B missed**: `shifting from fossil fuels` → 26B *l'abandon des combustibles
fossiles* (correct) vs both E4B and 31B *passage **aux** énergies fossiles* (inverted).
Dense scale only wins on **German** (31B 9.55). 31B also picks up a few new `[x.y]`
citation-marker slips. Practical read: **26B-A4B is the sweet spot**; E4B is already strong
for ~4× less; the dense 31B isn't worth its extra compute for this task.

## Reproduce
```bash
uv run python translation/prep_samples.py --per-lang 100 --seed 0   # -> data/samples.jsonl
sbatch --partition=debug --time=00:30:00 \
  --export=ALL,RUN_TAG=full,SAMPLES=$PWD/translation/data/samples.jsonl,\
OUTPUT=$PWD/translation/results/translations_full.jsonl,MAX_CONCURRENT=256 \
  translation/serve_and_translate.sh
uv run python translation/make_judge_batches.py \
  --translations translation/results/translations_full.jsonl \
  --out-dir translation/results/judge_batches --prefix full --batch-size 20
# judge: Workflow over the full_*.jsonl batch paths (Sonnet), then:
uv run python translation/aggregate_judgments.py \
  --judged "translation/results/judge_batches/full_*.judged.jsonl" \
  --translations translation/results/translations_full.jsonl \
  --out-md translation/results/full_report.md \
  --out-jsonl translation/results/full_merged.jsonl
```

For the reverse run, swap the prep/inputs:
`prep_reflections.py` → `data/reflections_reverse.jsonl`; translate with
`SAMPLES=…/reflections_reverse.jsonl OUTPUT=…/reverse_translations_full.jsonl`; batch with
`--prefix rev_full`; judge with `translation/judge_workflow.js`; aggregate the
`rev_full_*.judged.jsonl` against `reverse_translations_full.jsonl`.

## Files
- forward (X→En): `prep_samples.py` · reverse (En→X): `prep_reflections.py`
- shared: `translate.py` (direction-aware) · `serve_and_translate.sh` · `make_judge_batches.py` · `aggregate_judgments.py` · `judge_workflow.js` (Sonnet judge, both directions)
- `data/samples.jsonl` / `data/reflections_reverse.jsonl` (700 each) + `*_pilot` sets
- `results/translations_full.jsonl` / `results/reverse_translations_full.jsonl` — raw translations
- `results/judge_batches/{full,rev_full}_*.judged.jsonl` — per-sample Sonnet judgments
- `results/{full,reverse_full}_merged.jsonl` — translation × judgment joined
- `results/{full,reverse_full}_report.md` — generated stats (source of the tables above)
- `EXPERIMENTS.md` — dated, paper-grade log of both runs

_Caveat: the judge is an LLM (Sonnet) scoring against the source; treat ±0.3 on means as
noise. Source `safety_score` is carried from English and approximate._
