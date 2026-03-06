# Reflective Annotation Pipeline — Implementation Guide

This is an implementation spec for building a scalable annotation pipeline that produces **preflections** (contextualizing text before the reader sees it) and **reflections** (evaluating text after reading). Written for use with Claude Code or similar agentic tools.

---

## Pipeline Structure

```
Stage 1: Calibrate    →  Gold set + rubric + calibrated judges
Stage 2: Iterate      →  Best generator + tuned prompts
Stage 3: Scale        →  Production annotations
```

Key ordering principle: **tune judges first, then use them to select and tune the generator.** A well-calibrated judge ensemble is reusable across different generators and gives you an objective way to compare generator candidates.

---

## Stage 1 — Human Calibration

Assumes the input corpus is already clustered. We need a stratified sample and human annotations to build the gold set and rubric.

### 1.1 Stratified Sampling

Draw 200–300 items proportional to cluster sizes with a floor of ~5 per cluster. Over-sample high-ambiguity items (1.5×). Save sampling metadata for later drift detection.

### 1.2 Human Annotation

Each item gets **two independent annotators** who produce:
- A preflection and a reflection for the text
- Per-dimension scores (1–5) for both
- A free-text rationale for each score (minimum ~20 words, required)

Force scores before rationale in the UI to prevent rationalization. Randomize presentation order across annotators.

Compute inter-annotator agreement (Cohen's kappa or Krippendorff's alpha) per dimension. Flag dimensions with alpha < 0.6 for rubric revision. Flag items with ≥2 point disagreement on any dimension as discussion items.

### 1.3 Rubric Co-Creation

This is a back-and-forth between a frontier model and the human annotators.

**Step 1 — Draft:** Give the frontier model all annotated samples + rationales. Ask it to identify the implicit criteria, propose a 1–5 rubric with behavioral anchors at each level, and flag cases where annotators seemed to apply different criteria.

**Step 2 — Human review:** Annotators check whether the 1–5 anchors are actually distinguishable (common failure: levels 2 and 3 are too similar) and whether dimensions are orthogonal (if two correlate >0.9 in the data, merge them).

**Step 3 — Validation:** Re-score a subset with the finalized rubric and check that IAA improves.

**Starting dimensions (adapt to your domain):**

Shared: `factual_accuracy`, `relevance`, `completeness`, `conciseness`

Preflection-specific: `frames_without_spoiling` (guides attention without pre-empting reader conclusions), `context_value` (provides background the reader wouldn't have)

Reflection-specific: `adds_insight` (goes beyond restating), `analytical_depth` (identifies non-obvious patterns or implications)

### 1.4 Judge Calibration

This is the critical step that enables everything downstream. **Tune judges against the gold set before touching any generator.**

**Ensemble composition:** At least 3 judges from different model families (e.g., Claude, GPT, open-source like Llama/Qwen). Cross-family diversity is mandatory — judges from the same family as a generator will systematically over-accept its outputs due to perplexity-linked self-preference bias.

**Judge prompt template:**

```
You are evaluating the quality of a preflection and reflection generated for a given text.

Use the following rubric:
[FULL RUBRIC WITH BEHAVIORAL ANCHORS AT EACH 1–5 LEVEL]

Here are calibration examples showing human-validated scores:
[3–5 EXAMPLES FROM GOLD SET WITH SCORES AND RATIONALES]

For the text and annotation below, provide:
1. Per-dimension scores (1–5) for the preflection and reflection
2. An aggregate score for each
3. A rationale (2–4 sentences)
4. Your confidence in this judgment (0.0–1.0)
5. A decision: "accept" (aggregate ≥ 3.5), "reject" (< 3.0), or "borderline"

Output as JSON.
```

**Calibration procedure:**
- Run all judges on the gold set (where you have human ground truth)
- Compute per-judge precision, recall, and per-dimension correlation with human scores
- Iterate on judge prompts: adjust calibration examples, clarify rubric anchors, add notes for dimensions where judges systematically diverge from humans
- Target: per-judge correlation with humans ≥ 0.75 on aggregate score

Once judges are calibrated, they become your **reusable evaluation infrastructure** for everything that follows.

### Stage 1 Outputs

- Gold set (all annotated items with full metadata)
- Finalized rubric document
- IAA report (per-dimension agreement stats)
- Calibrated judge prompts (per model, validated against gold set)
- Per-judge accuracy profiles

---

## Stage 2 — Co-Optimization of Generators and Judges

Stage 2 has two co-equal goals: finding and tuning the best generator, and continuing to improve the judges. The judges got an initial calibration in Stage 1 (against the gold set), but they will encounter new failure modes on real generator outputs that the gold set didn't cover. Every iteration improves both sides.

### 2.1 Generator Selection

Run multiple candidate generators on the same batch (e.g., Claude, GPT, Llama, Qwen — or different prompt strategies with the same model). Use the calibrated judge ensemble to score all outputs. Compare generators by:
- Mean aggregate score (higher is better)
- Score variance (lower is more consistent)
- Per-dimension strengths/weaknesses (one model might be better at factual accuracy, another at analytical depth)
- Failure rate (fraction scoring below threshold)

Pick the best generator (or best 2, if you want diversity in Stage 3 candidate generation).

### 2.2 Generator Prompt Design

The generator prompt enforces a three-step chain-of-thought:

```
You are given a text and a constitution (a document defining the perspective
from which to annotate).

Step 1 — Analysis: Read the text against the constitution. List the key
elements: important claims, quality signals, notable features, domain context.
Be specific and concise. This is your working scratchpad.

Step 2 — Preflection: Using your analysis, write a preflection that
contextualizes this text for a reader who has NOT yet read it. Frame what
matters, provide relevant background, guide attention — but do NOT spoil
conclusions or evaluations.

Step 3 — Reflection: Using your analysis, write a reflection that evaluates
this text for a reader who HAS read it. Assess quality, identify issues,
highlight insights. Go beyond restating — add analytical value.

Output as JSON:
{
  "analysis": "...",
  "preflection": "...",
  "reflection": "..."
}
```

Include the constitution in the system prompt (constant across items). Keep temperature moderate (0.6–0.8). Set a target length range (e.g., 100–300 words each).

### 2.3 The Optimization Loop

Each iteration co-optimizes both judges and generator:

1. **Generate** a batch (50–100 items)
2. **Judge** with the full ensemble (all judges, all items)
3. **Human review** a sample — prioritize: judge disagreement cases, a random slice of judge-approved items (catch false negatives), a random slice of judge-rejected items (catch false positives). Humans provide reasons for **both** accepts and rejects.
4. **Update judge prompts** — this is not optional. Compute per-judge accuracy against the new human decisions. Identify false negatives (judge accepted, human rejected) and false positives (judge rejected, human accepted). Add clarifying examples, adjust rubric interpretation notes, recalibrate score thresholds. The judges should get measurably better each iteration.
5. **Update generator prompt** using rejection reasons as negative examples, acceptance reasons as positive examples
6. **Version everything** — tag all records with prompt version, model, timestamp

### 2.4 Convergence

Track across iterations:
- Human rejection rate on judge-approved items (target: < 10%)
- Inter-judge agreement (should trend up)
- Per-judge precision/recall (should improve then stabilize — if a judge isn't improving, consider replacing the model)
- Generator mean score (should trend up, variance down)

Stop when both judge accuracy and generator quality have stabilized for 2+ consecutive iterations. If not converging after ~8 iterations, the rubric likely needs revision.

### Stage 2 Outputs

- Selected generator model + optimized prompt
- Improved judge prompts (should be measurably better than Stage 1 calibration)
- Full evaluation corpus (all generations + judgments + human reviews)
- Convergence metrics

---

## Stage 3 — Scaled Production

### 3.1 Candidate Generation

Generate N = 3–5 candidates per input. For genuine diversity, vary temperature, use minor prompt perturbations, or use multiple generator models. Same prompt + same temp = N copies of the same bias.

### 3.2 Hierarchical 3-Tier Judging

```
All candidates
     │
     ▼
  Tier 1 — Cheap judges (small/fast models, or classifiers distilled from Stage 2 data)
     │      Score everything. Calibrated to overflag.
     │
     ├─ Unanimous pass ──────────────→ Select best, done
     ├─ Unanimous reject ────────────→ Reject
     └─ Any disagreement or score ≤2 → Escalate
                                          │
                                          ▼
                                       Tier 2 — Frontier model
                                          │      Full structured evaluation
                                          │
                                          ├─ High confidence → Accept/reject
                                          └─ Low confidence or OOD → Escalate
                                                                        │
                                                                        ▼
                                                                     Tier 3 — Human
                                                                     (feeds back into eval corpus)
```

**Target rates:** Tier 1 pass-through 60–80%. Tier 2 processes 20–40% of items. Tier 3 sees 5–15%.

**Escalation triggers:** Based on inter-judge disagreement and calibrated confidence, not individual scores.

### 3.3 Selection Logic

When multiple candidates pass: prefer the highest score from the highest tier that evaluated them. Break ties on preflection score (harder to do well).

### 3.4 Continuous Recalibration

Every ~50–100 human escalation decisions, check Tier 1 and Tier 2 accuracy against recent human decisions. If precision or recall drops >5pp from baseline, trigger a lightweight prompt update cycle.

**Monitor:** escalation rates at each tier, human override rate, score distribution drift, per-cluster acceptance rates.

---

## Common Failure Modes

**Generic generator outputs.** The analysis step is vague ("discusses several important points"). Fix: add negative examples showing vague analysis vs. positive examples showing specific, pointed analysis.

**Judge score inflation.** LLMs default to being polite. "4 = good" is useless as a rubric anchor. "4 = identifies the core claim, provides relevant context, minor omission of secondary detail" is useful.

**Preflections spoil the text.** Most common preflection failure. The model reads the text, forms an opinion, and leaks it. Add explicit negative examples showing spoiling vs. framing. Weight `frames_without_spoiling` heavily.

**Low IAA on subjective dimensions.** "Adds insight" is inherently subjective. You'll get alpha < 0.5. That's okay — ensure the *distribution* is calibrated, don't chase perfect agreement.

**Tier 1 blind spots.** If Tier 1 judges are distilled only from Stage 2 data, they'll miss failure modes that only appear in the broader corpus. Periodically spot-check Tier 1 passes with Tier 2.

**Ambiguous constitution.** If the constitution/perspective document is vague, the generator will be inconsistent and judges will disagree. Invest in making it crystal clear before starting.

---

## Implementation Order

1. **Human annotation interface** — enforce dual annotation, forced scores before rationales
2. **Rubric co-creation** — frontier model proposes, humans refine
3. **Judge calibration** — tune judge prompts against gold set until per-judge correlation ≥ 0.75
4. **Generator selection** — run candidate generators, score with calibrated judges, pick winner
5. **Generator prompt optimization** — the iteration loop with human review
6. **Tier 1 judge training** — distill from Stage 2 data into cheap classifiers
7. **Hierarchical judging orchestrator** — Tier 1 → 2 → 3 routing
8. **Recalibration monitor** — background drift detection + alerting

---

## Cost Estimation

The big lever is the Tier 1 pass-through rate. Every 10% improvement saves ~15% of total cost.

```
Per-item (5 candidates):

Generation:             5 × generator cost
Tier 1 (3 judges):      5 × 3 × cheap model cost
Tier 2 (30% escalated): 0.3 × 5 × frontier cost
Tier 3 (5% human):      0.05 × human cost per item

Example:  ~$0.05 gen + ~$0.015 T1 + ~$0.045 T2 + ~$0.10 T3 = ~$0.21/item
At 100K items: ~$21K
```