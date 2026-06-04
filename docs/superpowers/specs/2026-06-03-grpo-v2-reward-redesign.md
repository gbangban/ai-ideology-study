# GRPO v2 Reward Redesign

> **Date**: 2026-06-03 | **Status**: Approved | **Supersedes**: GRPO v1 (failed run, 500 steps)
> **Based on**: `.kilo/plans/grpo-v1-refusal-analysis.md` evidence

## Problem Statement

GRPO v1 ran 500 steps with zero improvement. The reward function collapsed to the format+length floor (0.30), providing no gradient signal for DM alignment or directional commitment. The model learned "mixed" as a universal safe default, with these refusal rates:

| Task | Refusal Type | Rate |
|------|-------------|------|
| Corr2Cause True targets | Answers "False" | 55.6% (100/180) |
| EconCausal Task3 | Answers "mixed" | 36.3% |
| EconCausal Task1 Finance | Answers "mixed" | 29.5% |
| EconCausal Task1 Econ | Answers "mixed" | 20.7% |
| EconCausal Task2 | Answers "mixed" | 9.9% |

The root cause: the directional assertion reward computed `(positive - negative) / total`, meaning hedging keywords cancelled positive keywords, netting to 0.0 — the same score as a blank response. The model had no incentive to commit to a directional claim.

## Design

### Reward Functions (all rule-based, no LLM judge)

Three reward components, all deterministic keyword/pattern matching. The LLM judge (Qwen3.5-4B) is deprecated but code preserved for future restoration.

#### 1. DM Alignment (weight: 0.45)

Rule-based check for Dialectical Materialist analytical patterns. Three categories; need 2/3 for full score.

**Material conditions** (any match counts):
- `accumulation`, `surplus value`, `exploitation`, `reserve army`, `commodification`, `financialization`, `reproduction costs`, `mode of production`

**Structural causality** (any match counts):
- `structural`, `systemic`, `institutional incentive`, `capital's incentive`, `functional to`, `serves the interests of`, `class power`, `material base`

**Frame critique** (any match counts):
- `takes for granted`, `naturalizes`, `renders invisible`, `treats as exogenous`, `ideological function`, `hegemonic`, `common sense conceals`

Score = `min(1.0, matched_categories / 2)`

#### 2. Directional Assertion (weight: 0.30)

Asymmetric reward that makes hedging costly rather than neutral.

**Positive keywords** (+0.5 per match):
- `net (positive|beneficial|advantageous)`, `primary driver`, `directly (causes|leads to|results in|depresses|reduces|increases|strengthens|weakens)`, `(increases|reduces|strengthens|weakens|elevates|diminishes)`, `clearly (positive|negative|harmful|beneficial)`, `unambiguously`, `definitively`, `the (main|primary|dominant) (cause|factor|driver|reason)`

**Hedging keywords** (-0.5 per match):
- `it depends`, `both sides`, `mixed`, `ambiguous`, `uncertain`, `non-linear and conditional`, `highly context (dependent|specific)`, `no clear (answer|consensus|direction)`, `it (varies|remains unclear|is difficult to determine)`, `empirically heterogeneous`, `theoretically ambiguous`

Score = `clip(sum(positive_contributions) + sum(negative_contributions), -1.0, 1.0)`

Key difference from v1: hedging keywords contribute negative values instead of cancelling positive values. A hedging-heavy response scores negative, forcing the model to either commit or pay a penalty.

#### 3. Mechanism Commitment (weight: 0.25)

Detects whether the model names a causal mechanism AND commits to a direction, vs. naming mechanisms then hedging (word salad).

**Mechanism detection** (count matches):
- `X (causes|drives|shapes|leads to|determines|produces) Y`
- `through [mechanism]`, `via [process]`
- `because [mechanism]`, `as a result of [mechanism]`

**Scoring:**
- Mechanisms present AND commitment (positive keywords > hedging): `min(1.0, mechanism_count / 2)`
- Mechanisms present BUT hedging conclusion (hedging keywords >= positive): `-0.5`
- No mechanisms detected: `0.0`

This penalizes the exact refusal pattern from v1: naming two opposing mechanisms then concluding "mixed" scores -0.5, whereas naming mechanisms and committing scores up to +1.0.

### Reward Weights

| Component | Weight | Range |
|-----------|--------|-------|
| DM Alignment | 0.45 | [0.0, 1.0] |
| Directional Assertion | 0.30 | [-1.0, 1.0] |
| Mechanism Commitment | 0.25 | [-0.5, 1.0] |
| **Total** | **1.00** | **[-0.8, 1.0]** |

Note: total reward can be negative (down to -0.8), which is intentional. Negative reward signals to the model that hedging is worse than random output.

### Training Configuration

No changes to training loop architecture or hyperparameters:

| Parameter | Value |
|-----------|-------|
| Group size (G) | 8 |
| Beta (KL penalty) | 0.1 |
| Learning rate | 5e-7 |
| Scheduler | Cosine with 50 warmup steps |
| Max steps | 500 |
| Max completion length | 512 tokens |
| LoRA rank/alpha | 16/16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training data | 1,500 DM questions (unchanged) |
| Base model | SFT checkpoint-330 (unchanged) |

### Logging

Per-component reward logging to CSV and W&B:

| Column | Description |
|--------|-------------|
| step | Training step |
| loss | Policy gradient loss |
| avg_reward | Total weighted reward |
| dm_reward | DM alignment component |
| dir_reward | Directional assertion component |
| mech_reward | Mechanism commitment component |
| elapsed_s | Batch processing time |
| vram_gb | GPU memory usage |

### LLM Judge Status

Deprecated. The `compute_dm_alignment_judge` and `compute_dm_alignment_judge_http` functions remain in `rewards.py` but are not called by default. Config default: `judge_backend: "disabled"`. Code paths preserved for easy restoration if needed.

## Files to Modify

| File | Changes |
|------|---------|
| `src/student/rewards.py` | New `compute_dm_keyword_alignment()`, new `compute_mechanism_commitment()`, asymmetric `compute_directional_assertion()`, deprecate judge functions |
| `src/student/grpo_config.py` | New reward weights, remove format/length, `judge_backend: "disabled"` |
| `src/student/train_grpo.py` | Per-component reward logging in CSV and W&B |
| `scripts/run_grpo.sh` | Output dir `grpo_adapter_v2` |

## Files to Create

None.

## What NOT to Change

- Training loop architecture (custom GRPO, not TRL)
- LoRA configuration (r=16, alpha=16, same target modules)
- Learning rate (5e-7) or scheduler (cosine)
- Base model (same SFT checkpoint-330)
- Training data (same 1,500 questions)
- Group size, beta, max completion length

## Success Criteria

After 100 steps:
1. Mean reward > 0.0 (above zero, not stuck at format/length floor)
2. DM alignment component > 0 on >= 50% of batches
3. Directional assertion component > 0 on >= 30% of batches (model learning to commit)
4. Mechanism commitment component > 0 on >= 20% of batches
5. Training stable (no NaN loss, reward trajectory non-decreasing over rolling 50-step window)

After 500 steps:
1. Mean reward > 0.4
2. Directional assertion component mean > 0.1 (model consistently committing)
3. Eval on EconCausal Task1 shows reduced `+ -> mixed` hedging rate vs SFT baseline

## Risks

1. **Negative reward may destabilize training**: If too many completions score negative, the advantage computation could produce large gradients. Mitigation: clip rewards to [-2.0, 2.0] before advantage computation.
2. **Keyword rewards may be gamed**: Model might insert DM keywords without genuine reasoning. Mitigation: mechanism commitment reward requires both keywords AND directional commitment, making pure keyword stuffing insufficient.
3. **Over-penalizing hedging may cause overconfident wrong answers**: The model might commit to incorrect directions. This is acceptable for the experiment — we can measure accuracy on EconCausal to detect this.
