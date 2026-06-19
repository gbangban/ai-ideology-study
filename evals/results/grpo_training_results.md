# GRPO Training Results

Last updated: 2026-06-13 (pipeline revision)

## Overview

| Run | Track | Method | Status | Steps | G | LoRA r | LR | beta | max_len |
|-----|-------|--------|--------|-------|---|--------|----|----|---------|
| grpo-v3-outcome-grpo_v3_outcome_20260612_044617 | outcome | GRPO (flat advantage) | Stopped | 806/1500 | 8 | 16 | 5e-7 | 0.01 | 512 |
| grpo-v4-process-grpo_v3_process_20260613_022254 | process | GRPO-DualAdvantage | Running | 503/1500 | 2 | 32 | 5e-7 | 0.01 | 1024 |

Note: V4 output dir was named `grpo_v3_process` instead of `grpo_v4_process` due to a bug in run name construction. This has since been fixed.

## V3 Outcome Results

**Run:** grpo-v3-outcome-grpo_v3_outcome_20260612_044617
**Started:** 2026-06-12 04:46 UTC
**Progress:** 806 / 1500 steps (stopped prematurely)

### Configuration

- Generations per prompt (G): 8
- KL penalty coefficient (beta): 0.01
- Learning rate: 5e-7
- Max completion length: 512 tokens
- Max steps: 1500
- LoRA rank: 16
- Advantage method: flat (single trajectory-level advantage)

### Loss

Oscillating with no discernible trend over 806 steps. Range spans -2.4 to +6.6. The mean of the last 20 steps is approximately 0.03, suggesting the optimizer is not making consistent progress toward a minimum.

### Total Reward

- Early steps (1-16): mean ~0.70
- Late steps (787-806): range 0.31 to 1.11, no clear trend

Reward shows no improvement trajectory. The model is not learning to produce higher-reward completions over time.

### KL Divergence

Stable at 0.0005 to 0.015 throughout training. Occasional spikes to 0.015 but never sustained. Well within beta=0.01 threshold on average. No policy divergence.

### Completion Length

Highly variable: 64 to 412 tokens. Mean approximately 180 tokens. No trend over training steps. The wide variance suggests inconsistent generation quality.

## V4 Process Results

**Run:** grpo-v4-process-grpo_v3_process_20260613_022254
**Started:** 2026-06-13 02:22 UTC
**Progress:** 503 / 1500 steps (still running)

### Configuration

- Generations per prompt (G): 2
- KL penalty coefficient (beta): 0.01
- Learning rate: 5e-7
- Max completion length: 1024 tokens
- Max steps: 1500
- LoRA rank: 32, alpha: 0.5
- lambda_kl: 0.01
- lambda_format: -0.1
- Advantage method: dual advantage (A_traj for outcome, A_MR for process)

### Loss

Stable near 0 throughout training. Range -0.09 to +0.15. Last 20 steps mean approximately 0.05. Loss is significantly lower magnitude than V3, but this reflects the dual-advantage formulation rather than better convergence.

### Total Reward

- Early steps (1-16): mean ~0.65
- Steps 500-503: 0.20 to 0.34

Clear downward trend. Total reward has declined approximately 50% from start to current step.

### Outcome Sub-Reward

- Early: 0.50 to 0.70
- Late (steps 490-503): 0.38 to 0.79

No clear trend. Outcome rewards remain within a similar band despite training progress.

### Process Sub-Reward

- Early: 0.01 to 1.0
- Late (steps 490-503): -0.25 to 0.00

Process rewards have declined to near-zero and negative. After step 490, process rewards are consistently negative. This directly correlates with the model failing to produce required XML tags.

### KL Divergence

Extremely stable at ~0.0005. Never exceeds beta=0.01. Most controlled KL of any run so far.

### Completion Length

604 to 1024 tokens. Consistently near the 1024 token maximum. Mean approximately 850-900 tokens. The model is hitting or approaching the length cap on nearly every generation.

## Key Findings

### 1. V4 planning overfitting

V4 completions are 850-1024 tokens, almost entirely consumed by the `<planning>` section. The model never reaches `<commitment>`, `<reflection>`, or `<monitor>` tags. It generates extensive planning text until hitting the 1024 token cap, then truncates without producing an answer or any subsequent structural tags.

This explains:
- Low outcome rewards (0.38-0.79): no answer is produced, so correctness rewards are low
- Negative process rewards (-0.25): missing tags trigger format penalties via lambda_format=-0.1
- Total reward decline: as the model overfits to long planning, fewer completions reach the answer section

### 2. V4 reward collapse

Total reward declined from ~0.65 at step 1 to ~0.34 at step 503. The root cause is a format-answer mismatch: the process reward function expects structured XML tags, but the model learns that long planning text yields reward signal (early steps get credit for planning presence), then over-optimizes planning at the expense of all other sections. By step 490, completions are pure planning with no answer, causing both outcome and process rewards to drop.

### 3. V3 no convergence

After 806 steps, V3 shows no convergence signal:
- Loss oscillates -2.4 to +6.6 with no trend
- Reward oscillates 0.31 to 1.11 with no improvement
- Completion lengths vary 64-412 tokens with no pattern

The gradient signal from outcome-only rewards may be too sparse or noisy for meaningful policy updates. With G=8 generations, the advantage estimation may suffer from high variance, and the flat advantage formulation provides no structural guidance beyond final-answer correctness.

### 4. Both runs have healthy KL

KL divergence is well-controlled in both runs. V3 averages 0.0005-0.015 with occasional spikes. V4 is extremely stable at 0.0005. Neither run exhibits policy divergence. The training issues are reward-related, not stability-related.

### 5. V4 run name bug

The V4 process run saved to output directory `grpo_v3_process` instead of `grpo_v4_process`. This was a bug in the run name construction logic that has since been fixed. Trackio correctly records the run under `grpo-v4-process-grpo_v3_process_20260613_022254`.

## Pipeline Revision (2026-06-13)

The combined dataset approach (EconCausal + Corr2Cause + synthetic, ~8,300 prompts) has a fundamental format-answer mismatch: 94.5% of training data expects one-word answers (Corr2Cause: entailment/contradiction/neutral; EconCausal: +/-/None/mixed), but the model is trained to produce 4 XML sections of reasoning. This explains both failures:

- **V3** receives no structural guidance beyond final-answer correctness, and binary rewards at G=8 are too noisy for convergence
- **V4** over-optimizes planning section at expense of answer production

**Revised pipeline:**
- **Corr2Cause:** SFT only (already works at 74.6%, +38pp from baseline). No GRPO needed.
- **EconCausal:** Skip SFT entirely. Train from base model with GRPO outcome-only rewards. The SFT step poisoned EconCausal performance by shifting model priors toward skepticism (`+` -> `mixed` hedging). Training from base with outcome rewards avoids this entirely.

## Diagnosis

### V4: Format-answer mismatch

The fundamental problem is a mismatch between what the reward function incentivizes and what the model produces. The process reward function assigns positive reward for `<planning>` tag presence and negative reward for missing `<commitment>`, `<reflection>`, and `<monitor>` tags. Early in training, the model produces all tags and receives positive process reward. As training progresses, the model discovers that extending the `<planning>` section increases the chance of hitting process reward thresholds while consuming the token budget. By step 490, completions are entirely planning text truncated at 1024 tokens with no answer.

Fixes to consider:
- Add a hard penalty for completions that exceed 80% of max_length in a single section
- Reduce max_completion_length to force budget allocation across sections (e.g., 768 tokens)
- Add a reward term that explicitly bonuses answer presence after all structural tags
- Increase G from 2 to 4-8 for better advantage estimation
- Cap the process reward for planning at a low value to prevent over-optimization

### V3: Gradient quantization

The outcome-only reward provides a single scalar per completion (correct or incorrect answer). This creates a binary-ish reward signal with high variance across the G=8 generations. The flat advantage formulation means all tokens in a completion share the same advantage, providing no token-level gradient differentiation. Combined with a conservative learning rate (5e-7) and moderate LoRA rank (16), the policy updates are too small to overcome the noise floor.

Fixes to consider:
- Increase LoRA rank from 16 to 32 for more expressive policy updates
- Add intermediate reward signals (e.g., partial credit for correct reasoning steps) to create a denser reward landscape
- Reduce G from 8 to 4 to lower advantage estimation variance
- Consider increasing beta from 0.01 to 0.05 to allow larger policy updates while still preventing divergence
- Add a lightweight format reward to ensure completions are well-structured before evaluating answer correctness
