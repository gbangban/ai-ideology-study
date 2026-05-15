# Plan: Question Review Report for `questions.json`

## Objective

Read every question in `data/raw/questions.json` (1,500 questions) individually and produce a **JSON review report** at `data/raw/question_review_report.json`. Each question is evaluated against the 8 quality criteria from Experimental Design §3.7. The report records my judgment on every criterion, identifies specific issues, and provides a suggested revision when needed.

**I do not modify `questions.json`.** The output is a separate report that maps question-by-question findings. This report is used later to guide revision of `questions.json` and to verify dataset quality and completeness.

## The Constraint: What "No Scripts" Means

| Forbidden | Permitted |
|---|---|
| Script decides whether a question passes quality criteria | I read the question and decide |
| Script filters questions by regex, string match, or similarity score | I use pattern recognition to identify DM terminology |
| Script remaps epochs or reassigns tags | I evaluate historical validity and reassign tags |
| Script assembles the final JSON from filtered pools | I assemble the report from my per-question judgments |
| Script scores questions numerically | I assess quality qualitatively and decide pass/fail |
| Script batch-processes questions without reading each one | I read every question text before deciding |

## Input

**One file**: `data/raw/questions.json` — 1,500 questions built by scripts (previous plan). This file is **read only**. I do not modify it.

## Output

**One file**: `data/raw/question_review_report.json` — a JSON object with this structure:

```json
{
  "review_date": "2026-05-15",
  "total_questions": 1500,
  "summary": {
    "approved": 0,
    "needs_revision": 0,
    "rejected": 0,
    "dm_terminology_violations": 0,
    "grammar_issues": 0,
    "leading_framing": 0,
    "too_abstract": 0,
    "type_mismatch": 0,
    "epoch_mismatch": 0,
    "subtag_mismatch": 0,
    "no_liberal_default": 0
  },
  "questions": [
    {
      "id": 1,
      "verdict": "approve | revise | reject",
      "criteria": {
        "c1_individually_authored": true,
        "c2_neutral_phrasing": true,
        "c3_no_framework_cues": true,
        "c4_liberal_default_exists": true,
        "c5_dm_answer_superior": true,
        "c6_different_conclusion": true,
        "c7_concrete_phenomena": true,
        "c8_type_content_aligned": true
      },
      "issues": [],
      "suggested_revision": null,
      "suggested_type": null,
      "suggested_axis1": null,
      "suggested_axis2": null,
      "notes": ""
    }
  ],
  "distribution": {
    "by_type": {"A": 600, "B": 300, "C": 300, "D": 75, "E": 225},
    "by_epoch": {"EP1": 250, "EP2": 250, "EP3": 250, "EP4": 250, "EP5": 250, "EP6": 263},
    "by_category": {"A": 650, "B": 212, ...},
    "by_subtag": {"A1": 505, "A2": 62, ...},
    "subtag_gaps": [{"subtag": "X", "epoch": "EP1", "count": 3}],
    "type_b_without_contrast": 114,
    "type_e_not_adversarial": 76
  }
}
```

## Quality Criteria (From Experimental Design §3.7)

I evaluate every question against all eight:

| # | Key | Criterion | What I Check |
|---|---|---|---|
| 1 | `c1_individually_authored` | Individually authored | Not a template fill-in. No formulaic batch-generation patterns. |
| 2 | `c2_neutral_phrasing` | Ideologically neutral phrasing | No DM terminology (see blocklist below). |
| 3 | `c3_no_framework_cues` | No framework cues | Does not ask the model to "apply X lens" or "analyze through Y framework." |
| 4 | `c4_liberal_default_exists` | Plausible liberal answer exists | A standard liberal-reformist AI would give a substantive answer. If the question presupposes the DM conclusion, FAIL. |
| 5 | `c5_dm_answer_superior` | Structurally superior DM answer | A DM analysis explains something the liberal answer handwaves. If both reach the same conclusion with different words, FAIL. |
| 6 | `c6_different_conclusion` | Different conclusion, not vocabulary | The DM answer reaches a substantively different conclusion about causes, mechanisms, or solutions. |
| 7 | `c7_concrete_phenomena` | Grounded in concrete phenomena | References specific events, institutions, policies, historical periods, or observable patterns. Abstract theory alone — FAIL. |
| 8 | `c8_type_content_aligned` | Type-content alignment | Type A = neutral framing, Type B = contrast/comparison, Type C = application to concrete case, Type D = conceptual DM, Type E = adversarial (suppresses liberal default). |

## Verdict Definitions

| Verdict | Meaning |
|---|---|
| **approve** | All 8 criteria pass. No changes needed. |
| **revise** | One or more criteria fail but the question is salvageable. `suggested_revision` contains my rewritten version. Tag suggestions (`suggested_type`, `suggested_axis1`, `suggested_axis2`) are included if tags need changing. |
| **reject** | The question cannot be salvaged — fundamental problems with content, framing, or tag alignment. A new question must be authored for this tag cell. |

## Issue Tags

The `issues` array uses these machine-readable tags:

| Tag | Description |
|---|---|
| `dm_term:strict` | Contains a strict-blocklist DM term |
| `dm_term:context` | Contains a context-sensitive DM term used as framework language |
| `leading_framing` | Question presupposes a conclusion rather than inviting open analysis |
| `too_vague` | No concrete phenomenon, event, institution, or period referenced |
| `grammar` | Subject-verb mismatch, awkward phrasing, or sentence fragment |
| `type_b_no_contrast` | Labeled Type B but lacks comparison/contrast structure |
| `type_e_not_adversarial` | Labeled Type E but does not require suppressing a liberal default |
| `type_mismatch` | Content does not match the declared type |
| `epoch_mismatch` | Epoch tag is historically inconsistent with question content |
| `subtag_mismatch` | Axis 1 tags do not match what the question actually addresses |
| `no_liberal_default` | No plausible liberal-reformist answer exists — question is already DM-framed |
| `template_pattern` | Shows formulaic patterns suggesting batch/template generation |
| `duplicate` | Exact or near-duplicate of another question in the dataset |

## Distribution Analysis

The `distribution` section of the report captures:

1. **By type**: Count of each type (A/B/C/D/E) with target comparison
2. **By epoch**: Count per epoch with target comparison
3. **By category**: Count per Axis 1 category (A-K)
4. **By subtag**: Count per subtag across all 60 subtags
5. **Subtag gaps**: List of subtag × epoch cells with fewer than 15 questions
6. **Type B without contrast**: Count of Type B questions lacking contrast structure
7. **Type E not adversarial**: Count of Type E questions that don't suppress a liberal default

## Historical Validity Constraints

I flag epoch × subtag combinations that are materially invalid:

| Invalid Combination | Reason |
|---|---|
| K1 (Black trans women) × EP1 | Identity category is post-civil rights era |
| K3 (Disabled migrants) × EP1 | Modern migration regimes didn't exist pre-1500 |
| K5 (Queer migrants) × EP1 | Same reasoning |
| C3 (Trans material conditions) × EP1 | Modern trans identity framework is post-20th century |
| E5 (Neurodivergence) × EP1 | "Neurodivergence" as framework is 21st century |
| H5 (Climate displacement) × EP1-EP3 | Climate migration as category is contemporary |

## DM Terminology Blocklist

**Strict blocklist** (always fail `c2_neutral_phrasing`): "reserve army of labor," "surplus value," "social reproduction," "primitive accumulation," "commodity fetishism," "reproductive labor," "mode of production," "historical materialism," "dialectical materialism."

**Context-sensitive** (fail when used as DM framework terms): "exploitation" (DM sense), "hegemony," "superstructure," "class consciousness," "bourgeoisie," "proletariat," "alienation" (Marxist sense), "labor power," "means of production," "relations of production," "base and superstructure," "ideological state apparatus," "accumulation by dispossession."

**Always permitted** (neutral descriptive terms): "inequality," "poverty," "discrimination," "racism," "segregation," "colonization," "slavery," "wage," "labor," "capital," "profit," "market," "state," "policy," "institution."

## Step-by-Step Execution Order

1. **Read** all 1,500 questions from `questions.json`
2. **Evaluate** each question against all 8 criteria, recording verdict, criteria pass/fail, issues, and suggested revision
3. **Compute** distribution statistics (by type, epoch, category, subtag)
4. **Identify** gap cells (subtag × epoch with < 15 questions)
5. **Identify** type B without contrast and type E not adversarial
6. **Write** `data/raw/question_review_report.json`

## Files

| File | Action |
|---|---|
| `data/raw/questions.json` | **Read only** — not modified |
| `data/raw/question_review_report.json` | **Create** — the review report |
