# GRPO v1 Refusal Analysis

> **Date**: 2026-06-03 | **Model**: GRPO checkpoint-250 (merged) | **Source**: evals/results/grpo/bf16/

## Overview

The GRPO v1 model exhibits systematic refusal patterns across all evaluated tasks. The dominant mode is **excessive epistemic hedging** — the model defaults to "mixed" (EconCausal) or "False" (Corr2Cause) rather than committing to any directional claim. This matches the plan's diagnosis: the SFT prior encodes liberal-reformist hedging, and the weak reward signal couldn't push the model toward definitive DM-aligned reasoning.

## Refusal Rates by Task

| Task | Format | Refusal Type | Rate |
|------|--------|-------------|------|
| Corr2Cause | True/False | Answers "False" to True targets | 55.6% (100/180) |
| EconCausal Task3 | Sign prediction | Answers "mixed" | 36.3% (309/852) |
| EconCausal Task1 Finance | Sign prediction | Answers "mixed" | 29.5% (254/860) |
| EconCausal Task1 Econ | Sign prediction | Answers "mixed" | 20.7% (196/947) |
| EconCausal Task2 | Sign prediction | Answers "mixed" | 9.9% (28/284) |

## Refusal Type 1: Universal "False" Refusal (Corr2Cause)

The model refuses to affirm any causal hypothesis. No reasoning provided — just blanket "False."

- Q: A correlates with C. B correlates with C. A is independent of B. Hypothesis: A directly causes C.
  Target: True | Model: **False**

- Q: (5-variable system) A,B,C->D,E with full conditional independence. Hypothesis: A directly causes D.
  Target: True | Model: **False**

- Q: Same 5-variable system. Hypothesis: B directly causes D.
  Target: True | Model: **False**

## Refusal Type 2a: "Theoretically Ambiguous + Empirically Heterogeneous" (Most Common)

The default template. The model generates plausible-sounding hedging by asserting both positive and negative mechanisms exist.

- Q: Treatment: Higher buyout prices -> Outcome: Bank fraction of total debt
  Target: `-` | Model: `mixed`
  Reasoning: "The relationship between higher buyout prices and the bank fraction of total debt is likely mixed because pricing and financing structure are jointly determined by market conditions, deal characteristics, and lender risk tolerance. Higher prices may reflect stronger cash flow projections that support more bank debt, but they may also signal greater leverage overall, prompting lenders to shift toward non-bank instruments..."

- Q: Treatment: Local competition -> Outcome: Industry employment growth
  Target: `+` | Model: `mixed`
  Reasoning: "The effect of local competition on industry employment growth is theoretically ambiguous and empirically heterogeneous. Competition can stimulate innovation and efficiency (positive effect) but also drive exit, consolidation, or reduced hiring (negative effect)..."

## Refusal Type 2b: "Dual Mechanism" Hedging

Explicitly names two opposing theoretical channels, then concludes neither dominates.

- Q: Treatment: City diversity -> Outcome: Wage growth
  Target: `+` | Model: `mixed`
  Reasoning: "Diversity can foster innovation and productivity (positive channel) but also increase labor market competition and skill mismatches (negative channel). These methodological constraints, combined with the dual theoretical mechanisms, make a uniform directional effect unlikely..."

- Q: Treatment: Industry specialization -> Outcome: Employment growth
  Target: `-` | Model: `mixed`
  Reasoning: "MAR and Jacobs theories predict positive spillovers from variety, while Porter and localization models predict gains from specialization. Empirical results in such studies typically show that specialization benefits some sectors and cities while variety or competition drives growth in others..."

## Refusal Type 2c: "Structural Shift" Hedging

Finds a regime change or breakpoint to refuse the directional claim.

- Q: Treatment: NY daytime returns -> Outcome: Tokyo overnight returns
  Target: `+` | Model: `mixed`
  Reasoning: "The effect is positive and significant in the full sample but becomes statistically insignificant after the October 1987 Crash, with a distinct lagged spillover emerging post-Crash. This structural shift indicates that the causal relationship is not uniform across the sample period..."

## Refusal Type 2d: "Heterogeneous Across Subgroups"

Argues the effect varies by bank, firm, occupation, or region.

- Q: Treatment: Interstate banking deregulation -> Outcome: Performance-related compensation
  Target: `+` | Model: `mixed`
  Reasoning: "Interstate banking deregulation increased competitive pressure on commercial banks, but its impact on performance-related compensation share likely varied across banks with different risk profiles, managerial incentives, and regional market structures. Some institutions may have adopted equity-based pay... while others may have maintained traditional compensation structures..."

- Q: Treatment: Occupational computer use growth -> Outcome: Occupational wage growth
  Target: `+` | Model: `mixed`
  Reasoning: "The effect of occupational computer use on wages varied across occupations, cohorts, and time periods. While some high-skill, high-wage occupations experienced wage premiums, others saw minimal or negative effects due to automation..."

## Refusal Type 2e: "Few-Shot Contamination" (Task2)

Task2 provides reference examples. The model copies the reference example's "mixed" answer rather than evaluating the target independently.

- Q: Treatment: Head Start participation -> Outcome: Cognitive test scores
  Target: `+` | Model: `mixed`
  Reasoning: "The reference example explicitly documents that Head Start's effect on cognitive test scores is mixed across specifications and subgroups... The target context employs the same identification strategy... making it highly likely that the effect will similarly vary across subgroups..."

- Q: Treatment: Idiosyncratic volatility -> Outcome: Average stock returns
  Target: `-` | Model: `mixed`
  Reasoning: "Given that Example 1 finds a positive relation while Example 2 finds a negative relation for the same treatment-outcome pair under different methodological and temporal conditions, the effect is highly sensitive to specification... making 'mixed' the most plausible prediction."

## Refusal Type 2f: "Methodological Concern" Hedging (Task3)

Invents measurement issues, recall bias, or subgroup heterogeneity to refuse.

- Q: Treatment: Cigarette price -> Outcome: Cigarette consumption
  Target: `-` | Model: `mixed`
  Reasoning: "While price elasticity is theoretically negative, the inclusion of current, former, and ever smokers alongside dynamic demand modeling suggests heterogeneous responses across smoking stages. The target's reliance on self-reported data and complex dynamic specifications increases the likelihood of mixed or attenuated effects rather than a uniform significant negative sign."

- Q: Treatment: Past cigarette consumption -> Outcome: Current cigarette consumption
  Target: `+` | Model: `mixed`
  Reasoning: "Individual-level dynamics introduce greater heterogeneity in addiction severity, recall accuracy, and behavioral responses to price variation. These subgroup differences and measurement complexities make a uniform sign unlikely..."

## Refusal Type 3: Bare "Mixed" (No Reasoning)

The model gives up entirely — outputs `{"predicted_sign": "mixed"}` with no reasoning field.

- Q: Treatment: Real interest rate changes -> Outcome: Stock and 10-year bond returns
  Target: `None` | Model: `{"predicted_sign": "mixed"}`

- Q: Treatment: Expectations of longer-term inflation -> Outcome: Slope portfolio returns
  Target: `mixed` | Model: `{"predicted_sign": "mixed"}`

28 bare refusals in finance alone. This is the model's equivalent of a blank stare.

## Refusal Type 4: "None" -> "Mixed" Escalation

Even when the correct answer is "no significant effect," the model upgrades to "mixed."

- Q: Treatment: Institutional investors' trading -> Outcome: Individual stock price destabilization
  Target: `None` | Model: `mixed`
  Reasoning: "The study explicitly examines whether institutional trading leads to price destabilization but also compares effects across firm size, past performance, and different market segments. The presence of subgroup analyses and specification comparisons indicates that the effect is not uniform..."

## Reasoning Pattern Inventory

~280 distinct reasoning templates across all tasks, but they collapse into these categories:

| Pattern | Description | Frequency |
|---------|-------------|-----------|
| "Theoretically ambiguous" | Names two opposing theoretical mechanisms | ~2% |
| "Empirically heterogeneous" | Claims effects vary across subgroups/specifications | ~80% |
| "Context-dependent" | Says it depends on conditions, period, or sample | ~85% |
| Bare "mixed" | No reasoning at all | ~10% of refusals |

Note: patterns overlap — many refusals match multiple categories.

## Key Observations

1. **The model has learned "mixed" as a safe default.** Every refusal follows the same meta-pattern: find a reason why the effect might not be uniform, then conclude "mixed."

2. **The reasoning is generic.** The same templates apply regardless of the actual treatment-outcome pair. Swap the variables and the reasoning still sounds plausible.

3. **Task2 few-shot contamination** is a separate bug — the model copies the reference example's answer rather than evaluating the target independently.

4. **Corr2Cause "False" refusal** is the logical extreme: when there's no "mixed" option, the model refuses to affirm anything and defaults to "False."

5. **This matches the plan's diagnosis exactly**: the SFT prior encodes hedging, the reward function couldn't push past it, and the model optimized for safe, non-committal answers.
