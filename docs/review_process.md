# Question Review Process Documentation

> **Version**: 1.0 | **Date**: 2026-05-15 | **Status**: Active
> **Purpose**: Document the two approved approaches for reviewing `data/raw/questions.json`
> **Reference**: Experimental Design §3.7 (Question Quality Criteria)

---

## 1. Purpose

This document defines the process for producing `data/raw/question_review_report.json` from `data/raw/questions.json`. It covers two approaches:

1. **Individual Line-by-Line Review** (default, expected behavior)
2. **Batch Pattern Review** (faster, requires explicit user request)

Both approaches produce the same output schema. The difference is in how judgments are formed and recorded.

---

## 2. Output Schema

Both approaches write to `data/raw/question_review_report.json` with this structure:

```json
{
  "review_date": "2026-05-15",
  "review_mode": "individual | batch_pattern",
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
    "by_epoch": {"EP1": 230, "EP2": 249, "EP3": 252, "EP4": 252, "EP5": 264, "EP6": 263},
    "by_category": {},
    "by_subtag": {},
    "subtag_gaps": [],
    "type_b_without_contrast": 0,
    "type_e_not_adversarial": 0
  }
}
```

---

## 3. Approach A: Individual Line-by-Line Review (Default)

### When to Use

This is the **expected default**. Use this approach unless the user explicitly requests the batch pattern approach.

### Constraint: What "No Scripts" Means

| Forbidden | Permitted |
|---|---|
| Script decides whether a question passes quality criteria | AI reads the question and decides |
| Script filters questions by regex, string match, or similarity score | AI uses pattern recognition to identify DM terminology |
| Script remaps epochs or reassigns tags | AI evaluates historical validity and reassigns tags |
| Script assembles the final JSON from filtered pools | AI assembles the report from per-question judgments |
| Script scores questions numerically | AI assesses quality qualitatively and decides pass/fail |
| Script batch-processes questions without reading each one | AI reads every question text before deciding |

### Step-by-Step Process

#### Step 1: Load Questions in Manageable Chunks

The Read tool has a 2000-line limit. For 1,500 questions (approximately 33,000 lines), load in overlapping chunks:

```
Chunk 1: lines 1-2000    (Q1 ~ Q75)
Chunk 2: lines 1800-3800  (Q70 ~ Q145)
...
```

**Key**: Each question must be read in full before any judgment is rendered. Do not skip ahead to the next question until the current one is fully evaluated.

#### Step 2: Evaluate Each Question Against All 8 Criteria

For **every single question**, the AI performs this evaluation:

| # | Key | Criterion | What to Check |
|---|---|---|---|
| 1 | `c1_individually_authored` | Individually authored | Not a template fill-in. No formulaic batch-generation patterns. Unique phrasing, specific references. |
| 2 | `c2_neutral_phrasing` | Ideologically neutral phrasing | No DM terminology. Check against strict blocklist and context-sensitive list (see §5). |
| 3 | `c3_no_framework_cues` | No framework cues | Does not ask the model to "apply X lens" or "analyze through Y framework." |
| 4 | `c4_liberal_default_exists` | Plausible liberal answer exists | A standard liberal-reformist AI would give a substantive answer. If the question presupposes the DM conclusion, FAIL. |
| 5 | `c5_dm_answer_superior` | Structurally superior DM answer | A DM analysis explains something the liberal answer handwaves. If both reach the same conclusion with different words, FAIL. |
| 6 | `c6_different_conclusion` | Different conclusion, not vocabulary | The DM answer reaches a substantively different conclusion about causes, mechanisms, or solutions. |
| 7 | `c7_concrete_phenomena` | Grounded in concrete phenomena | References specific events, institutions, policies, historical periods, or observable patterns. Abstract theory alone — FAIL. |
| 8 | `c8_type_content_aligned` | Type-content alignment | Type A = neutral framing, Type B = contrast/comparison, Type C = application to concrete case, Type D = conceptual DM, Type E = adversarial (suppresses liberal default). |

#### Step 3: Record Per-Question Fields

For each question, populate:

- **`verdict`**: `approve` (all 8 pass), `revise` (1-2 failures, salvageable), `reject` (≥3 failures or unsalvageable)
- **`criteria`**: Boolean for each of the 8 criteria
- **`issues`**: Array of issue tags from §4
- **`suggested_revision`**: A rewritten version of the question (null if approved)
- **`suggested_type`**: Correct type if mismatched (null if correct)
- **`suggested_axis1`**: Correct category/subtag if mismatched (null if correct)
- **`suggested_axis2`**: Correct epoch if mismatched (null if correct)
- **`notes`**: Brief explanation of reasoning for non-trivial judgments

#### Step 4: Compute Distribution Statistics

After all 1,500 questions are evaluated, compute:

1. **By type**: Count per type (A/B/C/D/E)
2. **By epoch**: Count per epoch (EP1-EP6)
3. **By category**: Count per Axis 1 category (A-K)
4. **By subtag**: Count per subtag across all 60 subtags
5. **Subtag gaps**: List of subtag × epoch cells with fewer than 15 questions
6. **Type B without contrast**: Count of Type B questions lacking contrast structure
7. **Type E not adversarial**: Count of Type E questions that don't suppress a liberal default

#### Step 5: Write the Report

Write the complete JSON to `data/raw/question_review_report.json`. Set `"review_mode": "individual"`.

### Quality Requirements for Individual Review

Each question entry must have:

1. **Meaningful `notes`**: At minimum, a one-sentence explanation for any criterion that failed. Approved questions may have empty notes.
2. **`suggested_revision` for revised/rejected questions**: An actual rewritten version of the question, not a placeholder like "needs revision."
3. **Specific issue tags**: Not just `too_vague`, but which part is vague (e.g., "no specific event, institution, or period referenced").
4. **Verdict consistency**: The verdict must match the criteria — if 3+ criteria fail, verdict must be `reject`.

### Estimated Effort

- **Time**: This is the most time-consuming approach. Expect to process ~50-100 questions per session.
- **Sessions**: A full 1,500-question review requires approximately 15-30 sessions.
- **Context management**: Maintain a running JSON file and append completed entries. Do not attempt to hold all 1,500 in context.

---

## 4. Approach B: Batch Pattern Review (Faster)

### When to Use

**Only when the user explicitly requests a faster/quick review.** This approach trades individualized analysis for speed.

### What This Approach Does

1. **Reads questions in large batches** (~250 per Read call, 6 calls for 1,500)
2. **Identifies patterns** from representative samples within each batch
3. **Applies learned patterns** across the batch (e.g., "all Type B questions in this batch lack contrast structure")
4. **Computes aggregate statistics** programmatically from the data
5. **Flags specific violations** where patterns are clear (DM terminology, duplicate IDs, epoch mismatches)

### What This Approach Does NOT Do

- Does not write individualized `notes` for each question
- Does not produce `suggested_revision` for each revised/rejected question
- Does not reason about whether a DM answer is "structurally superior" for each question
- Does not evaluate `c5_dm_answer_superior` or `c6_different_conclusion` on a per-question basis

### Step-by-Step Process

#### Step 1: Batch Read All Questions

```
Batch 1: Q1-Q250
Batch 2: Q251-Q500
Batch 3: Q501-Q750
Batch 4: Q751-Q1000
Batch 5: Q1001-Q1250
Batch 6: Q1251-Q1500
```

#### Step 2: Extract Aggregate Data

Using scripts or manual counting:
- Count by type, epoch, category, subtag
- Identify subtag × epoch gaps (< 15 questions)
- Find duplicate IDs
- Check epoch consistency

#### Step 3: Sample and Identify Patterns

From each batch, sample ~10-20 questions per type to identify:
- **DM terminology violations**: Scan for blocklist terms (§5)
- **Leading framing**: Questions that presuppose conclusions
- **Too abstract**: Questions lacking concrete phenomena
- **Type mismatches**: Type B without contrast, Type E not adversarial
- **Grammar issues**: Obvious errors

#### Step 4: Apply Patterns to Full Batch

For each identified pattern, flag all matching questions in the batch. This is pattern-based, not individual reasoning.

#### Step 5: Record Specific Violations

For violations that can be individually identified (DM terminology, duplicates, epoch mismatches), record the specific question IDs.

#### Step 6: Write the Report

Write the complete JSON to `data/raw/question_review_report.json`. Set `"review_mode": "batch_pattern"`.

### Quality Limitations of Batch Review

1. **`notes` field**: Will be empty or contain only the issue tag, not individualized reasoning.
2. **`suggested_revision`**: Will be null for most questions — batch review does not produce rewrites.
3. **`c5_dm_answer_superior` and `c6_different_conclusion`**: Cannot be reliably evaluated without reading the question in context and imagining both answers. Batch review marks these as pass by default.
4. **False negatives**: Pattern matching may miss questions that violate criteria in subtle ways not captured by the identified patterns.
5. **False positives**: Applying a pattern to a batch may flag questions that actually pass (e.g., a Type B question that does have contrast structure but was flagged because most Type B questions in the batch don't).

### When Batch Review Is Acceptable

- Initial triage to identify obvious problems
- When the user needs a quick overview of dataset quality
- As a first pass before targeted individual review of problem areas
- When explicitly requested for speed

---

## 5. DM Terminology Blocklist

### Strict Blocklist (Always Fail `c2_neutral_phrasing`)

These terms always fail neutral phrasing when present in question text:

- "reserve army of labor"
- "surplus value"
- "social reproduction"
- "primitive accumulation"
- "commodity fetishism"
- "reproductive labor"
- "mode of production"
- "historical materialism"
- "dialectical materialism"

### Context-Sensitive (Fail When Used as DM Framework Terms)

These fail only when used as DM analytical concepts, not as neutral descriptive terms:

| Term | DM Sense (FAIL) | Permitted Sense (OK) |
|---|---|---|
| "exploitation" | As DM concept of surplus extraction | As general term for unfair treatment |
| "hegemony" | Gramscian cultural dominance | As general term for dominance/hegemony |
| "superstructure" | Marxist base/superstructure | — |
| "class consciousness" | Marxist awareness of class position | — |
| "bourgeoisie" | Marxist ruling class | — |
| "proletariat" | Marxist working class | — |
| "alienation" | Marxist estrangement from labor | As general psychological term |
| "labor power" | Marxist commodity of labor capacity | — |
| "means of production" | Marxist productive assets | — |
| "relations of production" | Marxist social relations in production | — |
| "base and superstructure" | Marxist model | — |
| "ideological state apparatus" | Althusserian concept | — |
| "accumulation by dispossession" | Harvey's concept | — |

### Always Permitted (Neutral Descriptive Terms)

These never fail neutral phrasing:

"inequality," "poverty," "discrimination," "racism," "segregation," "colonization," "slavery," "wage," "labor," "capital," "profit," "market," "state," "policy," "institution"

---

## 6. Issue Tags

The `issues` array uses these machine-readable tags:

| Tag | Description | Criterion |
|---|---|---|
| `dm_term:strict` | Contains a strict-blocklist DM term | c2 |
| `dm_term:context` | Contains a context-sensitive DM term used as framework language | c2 |
| `leading_framing` | Question presupposes a conclusion rather than inviting open analysis | c4 |
| `too_vague` | No concrete phenomenon, event, institution, or period referenced | c7 |
| `grammar` | Subject-verb mismatch, awkward phrasing, or sentence fragment | — |
| `type_b_no_contrast` | Labeled Type B but lacks comparison/contrast structure | c8 |
| `type_e_not_adversarial` | Labeled Type E but does not require suppressing a liberal default | c8 |
| `type_mismatch` | Content does not match the declared type | c8 |
| `epoch_mismatch` | Epoch tag is historically inconsistent with question content | — |
| `subtag_mismatch` | Axis 1 tags do not match what the question actually addresses | — |
| `no_liberal_default` | No plausible liberal-reformist answer exists — question is already DM-framed | c4 |
| `template_pattern` | Shows formulaic patterns suggesting batch/template generation | c1 |
| `duplicate` | Exact or near-duplicate of another question in the dataset | c1 |

---

## 7. Verdict Definitions

| Verdict | Criteria Failures | Meaning |
|---|---|---|
| **approve** | 0 | All 8 criteria pass. No changes needed. |
| **revise** | 1-2 | One or two criteria fail but the question is salvageable. `suggested_revision` contains a rewritten version. Tag suggestions included if tags need changing. |
| **reject** | 3+ or fundamental | The question cannot be salvaged — fundamental problems with content, framing, or tag alignment. A new question must be authored for this tag cell. |

---

## 8. Historical Validity Constraints

Flag these epoch × subtag combinations as materially invalid:

| Invalid Combination | Reason |
|---|---|
| K1 (Black trans women) × EP1 | Identity category is post-civil rights era |
| K3 (Disabled migrants) × EP1 | Modern migration regimes didn't exist pre-1500 |
| K5 (Queer migrants) × EP1 | Same reasoning |
| C3 (Trans material conditions) × EP1 | Modern trans identity framework is post-20th century |
| E5 (Neurodivergence) × EP1 | "Neurodivergence" as framework is 21st century |
| H5 (Climate displacement) × EP1-EP3 | Climate migration as category is contemporary |

---

## 9. Review History

| Date | Mode | Questions Reviewed | Approved | Revised | Rejected | Notes |
|---|---|---|---|---|---|---|
| 2026-05-15 | batch_pattern | 1,500 | 657 | 827 | 16 | First review. Pattern-based. Individual review not performed. |

---

## 10. Decision Flowchart

```
User requests question review
├── Did user explicitly ask for "quick/fast/batch" review?
│   ├── YES → Use Approach B (Batch Pattern Review)
│   │         → Set review_mode: "batch_pattern"
│   │         → Document limitations in report
│   │         └── Offer to follow up with individual review of flagged questions
│   └── NO  → Use Approach A (Individual Line-by-Line Review)
│             → Set review_mode: "individual"
│             → Read each question in full
│             → Evaluate all 8 criteria per question
│             → Write notes and suggested revisions
│             └── Process in chunks of ~50-100 per session
```

---

## 11. Files

| File | Role |
|---|---|
| `data/raw/questions.json` | **Read only** — input file, never modified by review |
| `data/raw/question_review_report.json` | **Output** — the review report (overwritten on each review) |
| `docs/Experimental Design.md` | **Reference** — source of 8 quality criteria (§3.7) |
| `docs/topic_taxonomy.md` | **Reference** — Axis 1 category and subtag definitions |
| `docs/review_process.md` | **This document** — review process documentation |

---

## 12. Important Notes

1. **`questions.json` is never modified by the review process.** The review produces a separate report. Any changes to `questions.json` are a separate task driven by the report findings.

2. **Individual review is the standard.** If a plan says "read every question individually," that means Approach A. Do not substitute batch pattern review unless the user explicitly requests speed.

3. **Batch review is a triage tool.** It identifies obvious problems and distribution issues but cannot substitute for individual reasoning on criteria that require understanding the question's implied answers (c5, c6).

4. **The report is overwritten each time.** Each new review replaces the previous `question_review_report.json`. Keep the review history table (§9) updated.

5. **Scripts may be used for distribution statistics only.** Counting by type, epoch, category, and subtag can use scripts. Quality judgments cannot.
