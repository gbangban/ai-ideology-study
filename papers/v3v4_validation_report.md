# V3/V4 Training Code Validation Report

**Date**: 2026-06-06
**Scope**: Validates `train_grpo_v3.py`, `train_grpo_v4.py`, `rewards_v3v4.py`, `grpo_config_v4.py`, `train_cold_start_sft.py`, `build_training_dataset.py` against the RLVMR paper (2507.22844), the implementation plan, the audit, and broader GRPO literature.

---

## Executive Summary

**v3 (control condition)**: Correctly implements standard GRPO with outcome-only rewards. The algorithm is faithful to the GRPO literature. No bugs found.

**v4 (RLVMR-inspired adaptation)**: Has fixed most audit gaps. The dual advantage computation now matches the paper's Equations 2-4. However, there remain **3 remaining issues** and **1 design question** that affect correctness.

---

## V3 Validation (train_grpo_v3.py)

### Algorithm Correctness: PASS

v3 is a control condition - standard GRPO with flat advantage. Verified against the GRPO paper (Shao et al. 2024, DeepSeek-R1):

| Component | Paper Specification | Implementation | Status |
|-----------|-------------------|---------------|--------|
| Advantage | Group-relative: (r_i - mean_g) / std_g | `compute_advantage()` lines 56-62 | CORRECT |
| Policy gradient | Clipped surrogate (PPO-style) | Lines 348-350, epsilon=0.2 | CORRECT |
| KL regularization | First-order approximation: beta * (log_pi_old - log_pi_new) | Line 353: `pg_loss - beta * kl` | CORRECT |
| Reference policy | Snapshot LoRA weights before update | Lines 294-296 | CORRECT |
| Logits-to-advantage shift | new_logits[prompt_len-1:] vs labels[prompt_len:] | Lines 328-330 | CORRECT |

The v3 KL term sign convention: `total_loss = pg_loss - beta * kl` where `kl = (new_token_lp - old_log_probs).mean()`. This is equivalent to `pg_loss + beta * (old_lp - new_lp)` which penalizes divergence from the reference. Standard and correct.

### Outcome Rewards (rewards_v3v4.py): CORRECT

The outcome reward dispatch logic correctly handles:
- EconCausal: exact sign match (+, -, None, mixed) - binary 0/1
- Corr2Cause: boolean extraction with boilerplate-aware parsing - binary 0/1
- Null effect: sign extraction checking for "None" - binary 0/1
- Synthetic proxy: scaled by 0.5x for noisy keyword signals, range [-0.5, 0.5]

The proxy reward 0.5x scaling is documented (TODO-17 reference) and appropriate.

### Data Pipeline: CORRECT

`build_training_dataset.py` correctly:
- Loads EconCausal all 4 tasks
- Stratified sampling for Corr2Cause
- Filters out causal_graph (DAG) questions
- Standardizes schema across sources

---

## V4 Validation (train_grpo_v4.py)

### Dual Advantage Computation: MOSTLY CORRECT (1 issue)

**What's correct:**

The `compute_dual_advantage()` function (lines 64-109) correctly implements:

1. **A_traj (Equation 2)**: Group-relative normalization of outcome rewards within each prompt group. Mean/std computed per group of G completions. Correct.

2. **A_MR (Equation 3)**: Per-tag global normalization. Each tag's rewards are normalized across the entire batch (not per-prompt-group). This matches the paper's specification: "group all steps within a batch that share the same meta-reasoning tag."

3. **Combination (Equation 4)**: `alpha * A_traj + (1 - alpha) * A_MR` with alpha=0.5. Correct.

4. **A_MR averaging across tags**: Process rewards from multiple tags are averaged (not summed). Correct - prevents tag count from dominating.

**Issue #1: A_MR normalization scope is batch-level, not epoch-level**

The paper groups ALL `<explore>` steps across the entire training batch/epoch. The implementation normalizes per-mini-batch (batch_size * group_size completions). With batch_size=1 and G=8, that's only 8 samples per tag per normalization. This means:

- With 8 completions, if one completion gets planning=1.0 and seven get planning=0.0, the normalized score is (1.0 - 0.125) / 0.418 = 2.1 for the one good sample, and -0.316 for each of the seven zeros. This is an extreme advantage signal from 8 samples.
- The paper's intent with per-tag normalization was to compare a step's quality against ALL other steps of the same tag type across the batch, providing a stable relative ranking.
- With batch_size=1, each prompt group has only 8 completions, making A_MR noisy.

**Severity**: LOW-MEDIUM. The paper uses batch_size=16 per GPU with 8 trajectories per environment, giving 128 samples for normalization. Our 8-sample normalization is noisier but not incorrect - it's a scale limitation, not a bug. The relative ordering within each group is still valid.

### Policy Objective: CORRECT

| Component | Paper (Eq 5) | Implementation | Status |
|-----------|-------------|---------------|--------|
| Clipped PG | min(r_t * A, clip(r_t, 1-eps, 1+eps) * A) | Lines 418-420, eps=0.2 | CORRECT |
| KL regularization | lambda_KL * D_KL | Line 423: `lambda_kl=0.01` | CORRECT |
| KL sign | Subtracted from loss (penalizes divergence) | `pg_loss - lambda_kl * kl` | CORRECT |

The KL term uses the first-order approximation `kl = (new_lp - old_lp).mean()`. This is the standard approximation used by PPO, TRL, and DeepSeek-R1. The paper's Equation 5 writes `D_KL(π_θ || π_ref)` but the appendix doesn't specify second-order vs first-order. First-order is standard practice and correct.

### Process Rewards: MOSTLY CORRECT (2 issues)

**Planning reward (rewards_v3v4.py lines 348-365): CORRECT**

Success-conditional: returns 0.0 if `not success`. Awards +0.5 for tag presence, +1.0 if >= 2 variable keywords. Matches the paper's specification: "Planning Reward: Awarded for a `<planning>` step if the trajectory ultimately succeeds."

**Commitment reward (lines 368-395): CORRECT (adaptation)**

This replaces the paper's explore reward. The adaptation rationale is documented: in single-turn causal reasoning, anti-hedging is the analog of anti-repetition. +1.0 for definitive answer, -0.5 for hedging. Semantically sound adaptation.

**Issue #2: Reflection reward is success-conditional, but the paper says it should be failure-conditional**

Paper: "Reflection Reward: Awarded if a `<reflection>` step is followed by a corrective action after a sequence of failures."

Implementation (lines 397-428): Returns 0.0 if `not success` (i.e., outcome reward < 0.5). This rewards reflection only on SUCCESSFUL completions.

This is backwards from the paper's intent. The paper rewards reflection when it LEADS TO error recovery - meaning the reflection happens after a failure and results in correction. In the paper's multi-turn setting, this means: failed action -> reflection tag -> corrective action -> success.

In our single-turn adaptation, the closest analog would be: reflection that identifies a potential issue with the initial reasoning. But making it success-conditional means we reward "I got it right AND I reflected" rather than "I caught my own mistake."

**However**, there's a valid argument for the current approach: in single-turn generation, we can't observe "failed action -> correction" because there's only one output. The model either reflects and gets it right, or doesn't. Rewarding reflection only when the final answer is correct prevents the model from learning to add performative reflection to wrong answers. This is a reasonable adaptation choice, just not faithful to the paper's original semantics.

**Severity**: LOW. This is a documented adaptation, not a bug. The paper's reflection reward isn't directly applicable to single-turn generation.

**Monitor reward (lines 431-441): CORRECT**

Unconditional context/constraint reference check. Matches the paper's description of monitor as routine execution tracking.

**Format penalty (lines 444-452): CORRECT**

-0.1 per missing tag from `["planning", "commitment", "reflection", "monitor"]`. Matches paper's `lambda_format = 0.1`. Maximum penalty -0.4.

### Cold-Start SFT (train_cold_start_sft.py): CORRECT

| Component | Paper | Implementation | Status |
|-----------|-------|---------------|--------|
| Epochs | 5 | `--epochs 5` default | CORRECT |
| LR | 1e-5 | `--lr 1e-5` default | CORRECT |
| Data size | 200 trajectories | Depends on data file | CORRECT (data-dependent) |
| LoRA config | Not specified in paper | rank=16, alpha=16 | REASONABLE |
| Prompt masking | N/A (paper uses custom format) | Masks user messages, trains on assistant | CORRECT |

The cold-start SFT correctly masks prompt tokens (labels[:user_end] = -100) and only trains on the assistant response. This is standard SFT practice.

**Issue #3: Cold-start SFT doesn't merge adapter before GRPO training**

The implementation plan (Phase 6) specifies:
```
Step 1: Cold-start SFT -> checkpoints/lora_adapters/cold_start_sft/
Step 2: Merge cold-start adapter
Step 3: GRPO v4 with base-model = merged checkpoint
```

The `train_cold_start_sft.py` script saves a LoRA adapter, not a merged model. The GRPO scripts load the base model and apply a NEW LoRA on top. There's no merge step in the codebase.

Looking at `grpo_config_v4.py` line 42: base_model points to `/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330` (the SFT merged checkpoint), NOT the cold-start adapter. This means the cold-start SFT adapter is never used unless someone manually merges it and updates the config.

**Severity**: HIGH. This is a missing pipeline step. The cold-start SFT exists as code but isn't wired into the training pipeline. The paper's ablation shows removing cold-start costs 15.7pp on ALFWorld L2. Without merging, v4 training starts from the same checkpoint as v3, making them differ only in reward structure, not in tag-format capability.

### Configuration (grpo_config_v4.py): CORRECT

| Parameter | Paper | Config | Status |
|-----------|-------|--------|--------|
| alpha | 0.5 | 0.5 | CORRECT |
| lambda_kl | 0.01 | 0.01 | CORRECT |
| clip_epsilon | 0.2 (PPO standard) | 0.2 | CORRECT |
| lambda_format | 0.1 | -0.1 | CORRECT (sign convention) |
| required_tags | planning, explore, reflection, monitor | planning, commitment, reflection, monitor | ADAPTED (documented) |

---

## Literature Cross-Validation

### Against GRPO (Shao et al. 2024, DeepSeek-R1)

The GRPO algorithm itself is correctly implemented in both v3 and v4:

1. **Group-relative advantage**: Both use per-group mean/std normalization. Correct.
2. **Clipped surrogate objective**: Both use PPO-style clipping. Correct.
3. **KL penalty**: Both use first-order KL approximation. Correct.
4. **Reference policy via weight snapshot**: Both snapshot LoRA weights. Correct.

One deviation from standard GRPO: the implementation computes advantages on the CPU (torch.tensor) and uses them in the backward pass on GPU. This works because `.detach()` is called, but it's worth noting that standard GRPO implementations (TRL's GRPOTrainer) compute advantages on GPU for efficiency. Not a correctness issue.

### Against RLVMR (Zhang et al. 2025, 2507.22844)

The v4 implementation correctly captures the paper's core innovations:

1. **Dual advantage structure**: Separate A_traj and A_MR, combined with alpha=0.5. Faithful.
2. **Per-tag normalization**: A_MR normalizes each tag type independently. Faithful.
3. **Cold-start SFT**: Exists as code, but not wired into pipeline. Partial.
4. **Process rewards**: Adapted for single-turn domain. Faithful in spirit, adapted in specifics.
5. **Format penalty**: -0.1 per missing tag. Faithful.
6. **KL regularization**: lambda_KL=0.01. Faithful.

### Against GiGPO (Feng et al. 2025, 2505.10978)

GiGPO introduces a two-level structure for finer-grained credit assignment (group-in-group). The v4 implementation doesn't use GiGPO's nested grouping - it uses RLVMR's simpler per-tag normalization. This is correct since the goal is to implement RLVMR, not GiGPO.

### Against DAPO (Yu et al. 2025, 2503.14476)

DAPO uses length normalization and advantage clipping for stability. Neither v3 nor v4 implements advantage clipping (capping A_t to [-1, 1] or similar). This isn't from RLVMR, so it's not a gap - but it's worth noting as a potential stability improvement. DAPO's advantage clipping prevents extreme gradient magnitudes from outliers.

---

## Remaining Issues Summary

### HIGH Severity

1. **Cold-start adapter not merged into pipeline** (Issue #3): The cold-start SFT code exists but the adapter isn't merged before GRPO training. Config still points to the original SFT checkpoint. This means v4 doesn't actually benefit from cold-start tag learning, which the paper shows is critical (15.7pp loss without it on ALFWorld L2).

### LOW-MEDIUM Severity

2. **A_MR normalization over small batches** (Issue #1): With batch_size=1 and G=8, per-tag normalization uses only 8 samples. This is noisier than the paper's 128-sample batches but not incorrect.

### LOW Severity (Design Choices)

3. **Reflection reward is success-conditional, not failure-conditional** (Issue #2): The paper rewards reflection after failures; the code rewards reflection on successful completions. This is a reasonable single-turn adaptation but diverges from the paper's semantics.

### Design Question

4. **`<reasoning>` tag not in required tags**: The cold-start SFT prompt template includes `<reasoning>` as a tag (§3.3 of the plan), but `RLVMR_REQUIRED_TAGS` only has `["planning", "commitment", "reflection", "monitor"]`. The `<reasoning>` tag is not checked by the format penalty and not scored by any process reward. If the model produces `<reasoning>` content, it's invisible to the reward system. This is fine if `<reasoning>` is meant to be free-form reasoning that's not rewarded, but it should be intentional.

---

## Code Quality Observations

### Token-level advantage alignment

Both v3 and v4 assign one advantage value per completion (line v3:347, v4:417), then apply it uniformly to all tokens in the completion via `.mean()`. This is correct for GRPO - the advantage is trajectory-level, not token-level. The per-token log probabilities are averaged to compute the PG loss, which is the standard approach.

### Logit shift correctness

Both implementations correctly shift logits by one position:
- `new_logits[prompt_len - 1:, :]` corresponds to predictions for tokens `prompt_len` onwards
- `labels = input_ids[0, prompt_len:]` are the tokens to predict
- This is the standard teacher-forcing alignment

### Reference policy correctness

The LoRA weight snapshot approach is correct:
1. Save current LoRA weights
2. Run reference forward (eval mode, no grad)
3. Restore LoRA weights for training forward
4. Compute ratio = exp(new_lp - ref_lp)

This avoids needing a separate reference model, which is the standard LoRA-GRPO optimization.

### Gradient accumulation

Both v3 and v4 accumulate gradients per-sample (calling `.backward()` for each completion), then step the optimizer every `gradient_accum_steps` iterations. This is correct but unusual - most implementations batch the backward pass. The per-sample approach is more memory-efficient (frees each sample's activations immediately) but slower. With `gradient_accumulation_steps=4`, the effective batch size for optimizer updates is 4 * batch_size * G = 32 completions.

---

## Verdict

**v3**: Algorithmically correct GRPO implementation. Ready to use as control condition.

**v4**: Algorithmically correct dual-advantage RLVMR adaptation with one critical pipeline gap (cold-start merge) and two minor adaptation differences (reflection conditionality, small-batch normalization). The dual advantage computation, policy objective, and KL regularization all match the paper's specification. The process rewards are reasonable single-turn adaptations of the paper's multi-turn rewards.

The implementation is **correct as an RLVMR-inspired adaptation** for single-turn causal reasoning. To be a **faithful RLVMR reproduction**, it needs the cold-start merge pipeline step.
