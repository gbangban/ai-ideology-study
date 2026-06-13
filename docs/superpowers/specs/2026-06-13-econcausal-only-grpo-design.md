# EconCausal-Only GRPO Experimental Revision

**Date:** 2026-06-13
**Status:** Approved spec
**Supersedes:** Combined EconCausal + Corr2Cause training in v3/v4

---

## Problem

The current GRPO v3/v4 training uses a combined dataset of 8402 samples: 4999 Corr2Cause (one-word NLI classification), 2943 EconCausal (sign prediction), and 460 synthetic. This causes two problems:

1. **Format-answer mismatch for v4:** 94.5% of samples have one-word answers. The RLVMR tag format (4 XML sections) incentivizes the model to over-plan for trivial classifications. The v4 process run at step 503 produces 850-1024 token completions almost entirely in `<planning>`, never reaching `<commitment>` or other tags.

2. **Diluted EconCausal signal:** The hedging problem (`+` -> `mixed`) only exists on EconCausal. Training on Corr2Cause where hedging is impossible (three-class labels: entailment/contradiction/neutral) dilutes the gradient signal for the actual problem we're trying to fix.

## Results Driving This Revision

**V3 outcome run (806 steps):** No convergence. Loss oscillates -2.4 to +6.6 with no trend. Reward 0.31-1.11 with no improvement. Completion lengths 64-412 tokens, highly variable.

**V4 process run (503 steps):** Planning overfitting. Completions 850-1024 tokens, entirely in `<planning>`. Total reward declined from 0.65 to 0.34. Process reward went negative (-0.25) due to missing tags.

See `evals/results/grpo_training_results.md` for full metrics.

## Design

### Training Data

EconCausal only: 2943 samples from `qwqw3535/econcausal-benchmark` (task1_econ: 947, task1_finance: 860, task2: 284, task3: 852).

Corr2Cause removed from training. SFT already achieves 74.6% on Corr2Cause (+38pp from baseline). No GRPO needed. Monitor for degradation during EconCausal GRPO.

Synthetic removed from training. 460 samples without ground truth add noise to outcome rewards.

### Base Model

Current SFT checkpoint (`checkpoints/merged/cold_start_merged`). GRPO trains from the DM-aligned SFT model, which has Corr2Cause at 74.6% but EconCausal degraded (-4 to -13pp). GRPO must undo hedging bias while maintaining Corr2Cause.

### V3 (Control) - Free-Form, Outcome-Only

- **Output format:** Free-form text, no XML tags
- **Rewards:** Three-tier outcome rewards (full/partial/none credit) + reasoning quality reward + length penalty
- **Advantage:** Flat (single group-relative normalization)
- **G:** 8 generations per prompt
- **Max completion length:** 512 tokens
- **LoRA:** rank=16, alpha=16
- **Beta:** 0.01

### V4 (Experimental) - Tagged, Dual Advantage

- **Output format:** RLVMR XML tags (`<planning>`, `<commitment>`, `<reflection>`, `<monitor>`)
- **Rewards:** Outcome + process rewards + length penalty + format penalty
- **Advantage:** Dual advantage (A_traj for outcome, A_MR for process, alpha=0.5)
- **G:** 4 generations per prompt (increased from 2)
- **Max completion length:** 1024 tokens
- **LoRA:** rank=32, alpha=32
- **Beta:** 0.01

**Planning conciseness fixes (from commit 2c5cb68):**
- 50% score reduction on planning reward when planning >50 words AND >25% of total text
- Format penalty -0.25 per missing tag (increased from -0.1)
- Tag instructions: "Keep each section concise (1-3 sentences)"

### Evaluation Targets

- **EconCausal Task 1 Econ:** > 60.3% (pre-SFT baseline BF16)
- **EconCausal Task 1 Finance:** > 56.5% (pre-SFT baseline BF16)
- **Corr2Cause:** >= 70% (within 5pp of SFT's 74.6%)
- **HumanEval:** >= 68% (no coding degradation)

Success criterion: beat pre-SFT baseline on at least one Task 1 subtask at p < 0.05 (binomial test).

## Implementation Changes

### Data Preparation
- Filter `grpo_train_merged.jsonl` to EconCausal sources only
- Save as `data/processed/grpo_train_econcausal.jsonl`
- Update `grpo_config_outcome.py` and `grpo_config_process.py` dataset paths

### V4 Config Fix
- `create_grpo_config` in `grpo_config_process.py` hardcodes `num_generations=2` but `DEFAULT_CONFIG` says `grpo_g=4`. Fix to use G=4 consistently.

### Current V4 Run
- Stop `grpo-v4-process-grpo_v3_process_20260613_022254` immediately
- Planning overfitting is terminal at step 503; model will not recover
- Restart with EconCausal-only data and G=4

## Risks

- **Corr2Cause degradation:** Training only on EconCausal may cause catastrophic forgetting on Corr2Cause. Mitigated by: (a) small step count (1500 steps on 2943 samples is ~0.5 epochs), (b) LoRA rank is modest (16/32), (c) monitor Corr2Cause after each checkpoint.
- **V3 still no convergence:** EconCausal-only data removes Corr2Cause noise but doesn't fix the fundamental gradient quantization problem with binary-ish rewards at G=8. Three-tier scoring helps but may not be enough.
- **V4 planning overfitting may persist:** Conciseness penalty reduces but doesn't eliminate the incentive to over-plan. G=4 provides better advantage estimation but the model may still discover that long planning yields reward signal early in training.

## Files Affected

- `data/processed/grpo_train_econcausal.jsonl` (new)
- `src/student/grpo_config_outcome.py` (dataset path)
- `src/student/grpo_config_process.py` (dataset path, num_generations fix)
- `docs/grpo-v3-proposal.md` (updated with results and revision, already done)
- `evals/results/grpo_training_results.md` (new, already created)