# Writing Style Guide

This guide applies to all project documents: paper, slides, progress reports, and presentations.

## Adversarial Contrast Language: Never Use

**Rule:** Do not define something by what it is not. State what is true directly.

**Banned patterns:**
- "It is not X, it is Y."
- "It is X, not Y."
- "X, rather than Y."
- "Not X, but Y."
- "The opposite of X."

**Why:** Adversarial contrast frames the writing as a debate with a straw man. It wastes words establishing a negative before stating the positive. The reader only needs to know what is true.

**Examples:**

| Banned | Correct |
|--------|---------|
| "The hedging artifact is not a reasoning error. It is a transferred epistemic prior." | "The hedging artifact is a transferred epistemic prior." |
| "This is not vocabulary substitution. This is causal model replacement." | "SFT enables causal model replacement." |
| "The model does not learn formatting artifacts. It learns analytical content." | "The model learns analytical content." |
| "Carbon pricing is not a subsidy." | (Omit. State the actual fiscal numbers instead.) |
| "The regression is not uniform noise. It follows a specific pattern." | "The regression follows a specific pattern." |

**Exception:** Adversarial contrast is acceptable in Related Work when summarizing other authors' claims (e.g., "Prior work treated ideology as static. We ask a dynamic question."). The ban applies to claims about your own work.

## Em Dashes

Do not use em dashes. Use commas, colons, or parentheses instead.

| Banned | Correct |
|--------|---------|
| "The model improved dramatically -- by 38 points." | "The model improved by 38 points." |
| "Three outcomes -- preserved, improved, regressed." | "Three outcomes: preserved, improved, regressed." |

## Filler Adjectives

Remove adjectives that do not convey information.

| Remove | Keep |
|--------|------|
| "dramatically improved" | "improved by 38pp" |
| "severely regressed" | "regressed by 13.5pp" |
| "surprisingly" | (omit) |
| "critically" | (omit) |
| "notably" | (omit) |

Let the numbers carry the weight.

## Sentence Structure

- One idea per sentence.
- One idea per bullet point.
- Short sentences. Aim for under 30 words.
- Active voice.

## Quantitative Claims

- Always cite the source document and year.
- Distinguish between metrics: emissions-weighted average vs. implemented-instrument average.
- Use exact figures from source documents, not rounded approximations.
- When a claim depends on methodology, state which methodology.

## Numbers

- Use numerals for all values: "5 dollars", not "five dollars".
- Use "pp" for percentage point changes.
- Use "%" for rates and proportions.
