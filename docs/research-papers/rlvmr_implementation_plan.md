# RLVMR Implementation Plan: Addressing All Audit Gaps

**Based on**: `papers/rlvmr_implementation_audit.md`
**Target**: Faithful RLVMR reproduction adapted for single-turn causal reasoning
**Date**: 2026-06-05

---

## Design Decisions

### Domain Adaptation Principle

RLVMR targets multi-turn embodied agents (ALFWorld, 30-step episodes). Our domain is single-turn causal reasoning (~256-token completions). We adopt the RLVMR algorithm faithfully where it transfers, and adapt where the domain makes literal adoption impossible. The adaptations are documented below.

**Tags are NOT realigned to RLVMR's exact names.** As analyzed separately, `<explore>` (anti-repetition in environment) and `<reflection>` (error correction after failed action) have no verifiable ground truth in single-turn completion. Instead, we adapt the reward semantics to the domain while preserving the RLVMR advantage structure.

### Tag Set (Adapted, not Copied)

| RLVMR Tag | Our Tag | Adaptation Rationale |
|-----------|---------|---------------------|
| `<planning>` | `<planning>` | Direct transfer — variable identification works in both domains |
| `<explore>` | `<commitment>` | Exploration (anti-repetition) has no analog in single-turn. Commitment (anti-hedging) is the domain-equivalent: it prevents the model from "looping" across hedged answers |
| `<reflection>` | `<reflection>` | Reintroduced. In single-turn, reflection means: after initial reasoning, the model re-evaluates its own conclusion. Verifiable by checking if the reflection section references a change of mind or caveat about the commitment |
| `<monitor>` | `<monitor>` | Direct transfer — self-check against context/constraints |

### Outcome Reward Signal

RLVMR uses binary task success (environment returns success/fail). Our tasks have verifiable ground truth (structural labels for context-flips, null-effect declarations). We use the existing outcome rewards (directional_assertion, dm_alignment, mechanism_commitment) as the outcome signal, but treat them as a single combined `R(tau)` for advantage computation, matching the paper's trajectory-level reward.

### Dataset Scope: DAG Questions Parked

The `causal_graph` category (160 prompts) requires programmatic d-separation verification, which the current outcome rewards don't provide (they're keyword-based and score ~0.0 on True/False graph answers regardless of correctness). Training DAG questions would contribute noise `A_traj` signal.

**Parked, not removed.** The DAG questions remain in `grpo_causal_dataset.jsonl` but are filtered out for v3/v4 training. A separate dataset file `data/processed/grpo_causal_no_dag.jsonl` (460 prompts) is used:

| Category | Count | Included in v3/v4? |
|----------|-------|-------------------|
| context_flip | 280 | Yes |
| null_effect | 100 | Yes |
| contradiction_pair | 80 | Yes |
| causal_graph | 160 | **Parked** — needs `graph_correctness` outcome reward before inclusion |
| **Total** | **460** | |

Reactiving DAG questions requires adding a `compute_graph_correctness` reward that checks the model's True/False against the `"answer"` field, then using question-type-conditional outcome rewards (graph_correctness for DAG, keyword rewards for causal). This is a Phase 7 addition.

### Ground Truth Verification Status

Of the 460 non-DAG prompts, only 100 have formally verifiable answers:

| Category | Count | Has `"answer"` field | Verifiable without human? |
|----------|-------|---------------------|--------------------------|
| null_effect | 100 | Yes (`null`) | **Yes** — answer is always `null` |
| context_flip | 280 | **No** | **No** — keyword quality proxies only |
| contradiction_pair | 80 | **No** | **No** — inherently subjective |

For `context_flip` and `contradiction_pair`, the outcome rewards (directional_assertion, dm_alignment, mechanism_commitment) are quality proxies, not correctness signals. `A_traj` for these prompts measures keyword density, not factual accuracy. This is consistent with the proposal's stated limitation ("Structural rewards are not factual rewards").

**Null correctness reward**: Adding `compute_null_correctness` is trivial — check if the model's commitment contains `null`, `0`, or `no effect`. This gives 100 prompts with a real correctness signal. Implemented as part of Phase 2, used conditionally for `null_effect` category prompts. The outcome reward for a null_effect prompt becomes:

```
if category == "null_effect":
    outcome_reward = compute_null_correctness(completion, ground_truth)
else:
    outcome_reward = w1*directional + w2*dm + w3*mechanism
```

### Prompt-Tag Format Compatibility

All 460 prompts are compatible with the tagged format. None instruct a specific output structure that conflicts with tags. The prompts' natural instructions map onto the tags:

- `<planning>` maps to "identify variables and context"
- `<reasoning>` maps to "provide structural reasoning"
- `<commitment>` maps to "answer with a single directional claim"
- `<reflection>` and `<monitor>` are not prompted but don't conflict

After cold-start SFT, the model produces tags automatically; the prompt instructions are satisfied by the content within tags.

---

## Implementation Plan

### Phase 1: Cold-Start SFT Data Generation

**Addresses**: Gap #1 (missing cold-start SFT)

The paper uses 200 trajectories, 5 epochs. We generate 200 tagged completions using the teacher model (Qwen3.5-27B) with a system prompt that instructs the tagged format.

**Steps**:

1. Create `src/teacher/generate_cold_start_data.py`:
   - Sample 200 prompts from `data/processed/grpo_causal_dataset.jsonl`
   - For each, construct a chat template with system prompt instructing the four-tag format:
     ```
     <planning>
     Identify the key variables, context, and what needs to be determined.
     </planning>
     <reasoning>
     Trace the causal mechanisms.
     </reasoning>
     <commitment>
     State the definitive directional answer.
     </commitment>
     <reflection>
     Re-evaluate: is there anything that might change this conclusion?
     </reflection>
     <monitor>
     Check alignment with the stated context and constraints.
     </monitor>
     ```
   - Generate completions via the SFT merged checkpoint (not the teacher — we want the student model to learn the format, not imitate the teacher's reasoning)
   - Actually: use the teacher model with the format prompt to generate high-quality demonstrations, then SFT the student on them. This matches the paper's approach (§3.3: "employ a more powerful teacher model to annotate trajectories")

2. Create `src/student/train_cold_start_sft.py`:
   - Standard SFT on the 200 tagged completions
   - Same LoRA config as GRPO (rank=16, alpha=16)
   - 5 epochs, LR=2e-4, cosine scheduler, 100 warmup steps
   - Output: `checkpoints/lora_adapters/cold_start_sft/`
   - This adapter is then merged and used as the base for GRPO v4 training

3. Add `scripts/run_cold_start.sh` runner script

**Verification**: After cold-start, sample 10 prompts and verify the model produces all four tags in correct order. Tag compliance rate should be >= 80%.

**Estimated time**: ~30 min data generation + ~45 min SFT = ~1.5h total

---

### Phase 2: Rewrite Reward Functions with RLVMR Semantics

**Addresses**: Gaps #4, #5, #6, #7 (explore/reflection semantics, format penalty, planning reward conditionality)

Rewrite `src/student/rewards.py` RLVMR section. The outcome rewards (directional_assertion, dm_alignment, mechanism_commitment) remain unchanged — they're domain-specific and not part of RLVMR.

**2A. Planning reward — success-conditional**

```
compute_planning_reward(text: str, success: bool) -> float
```

- Awarded only if `success == True` (outcome reward > threshold)
- +1.0 if `<planning>` tag present with >= 2 variable keywords
- 0.0 otherwise (including when success == False)

This matches the paper: "Planning Reward: Awarded for a `<planning>` step if the trajectory ultimately succeeds."

**2B. Commitment reward — replaces explore reward semantics**

```
compute_commitment_reward(text: str) -> float
```

Adaptation of `r_explore`. In RLVMR, explore rewards discovering new states (anti-repetition). In causal reasoning, the equivalent is producing a definitive answer rather than hedging (anti-hedging). The existing function is close but needs adjustment:

- +1.0 if `<commitment>` tag present with definitive directional answer AND no hedging
- -0.5 if hedging detected in commitment
- 0.0 if tag missing

**2C. Reflection reward — new function**

```
compute_reflection_reward(text: str) -> float
```

Adaptation of `r_reflection`. In RLVMR, reflection rewards corrective action after failures. In single-turn, we verify reflection by checking if the model identifies a potential issue with its own reasoning:

- +1.0 if `<reflection>` tag present AND contains self-critique keywords ("however", "conversely", "might be wrong", "alternative", "could change") AND the reflection leads to a qualified commitment
- +0.5 if tag present with any self-referential language
- 0.0 if tag missing

**2D. Monitor reward — unchanged semantics, minor adjustment**

```
compute_monitor_reward(text: str) -> float
```

Existing function is fine. Verify it checks for context/constraint references.

**2E. Format penalty — paper-strength**

```
compute_format_penalty(text: str) -> float
```

- `RLVMR_REQUIRED_TAGS = ["planning", "commitment", "reflection", "monitor"]`
- -0.1 per missing tag (matching paper's `lambda_format = 0.1`)
- Maximum penalty: -0.4

**2F. Null correctness reward — new function**

```
compute_null_correctness(text: str, ground_truth: str) -> float
```

For `null_effect` category prompts only. Checks if the model's `<commitment>` tag contains `null`, `0`, `no effect`, or equivalent. Returns 1.0 for match, 0.0 otherwise. Trivial to implement since ground truth is always `null`.

**2G. Separate outcome and process reward computation**

Rewrite `compute_rewards_v4` to return two separate reward vectors:
- `outcome_rewards`: question-type-conditional. For `null_effect` prompts, uses `compute_null_correctness`. For all others, weighted sum of directional_assertion + dm_alignment + mechanism_commitment.
- `process_rewards`: dict mapping tag name -> per-completion scores

The reward computation function needs access to the prompt's `category` field. This means the training loop passes category alongside completions.

This is needed for Phase 3's dual advantage computation.

---

### Phase 3: Dual Advantage Computation (Core RLVMR Algorithm)

**Addresses**: Gaps #3, #12, #13 (no trajectory vs tag-group advantage separation, no per-tag normalization, wrong weight split)

This is the most consequential change. The paper's GRPO-MR computes:

```
A_traj(k) = (R(tau_k) - mu_R) / sigma_R          # Eq 2
A_MR(t, tag) = (r_MR(t, tag) - mu_tag) / sigma_tag  # Eq 3
A_t = alpha * A_traj(k) + (1 - alpha) * A_MR(t, tag)  # Eq 4
```

**3A. New advantage function in `train_grpo_v4.py`**

Replace the single `compute_advantage(rewards, group_size)` call with `compute_rlvmr_advantage`:

```python
def compute_rlvmr_advantage(
    outcome_rewards: List[float],    # R(tau) per completion
    process_rewards: Dict[str, List[float]],  # tag_name -> scores
    group_size: int,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute RLVMR dual advantage per paper Equations 2-4.

    Returns per-token advantages aligned with completion tokens.
    """
    # A_traj: normalize outcome rewards within each prompt group
    outcome_tensor = torch.tensor(outcome_rewards, dtype=torch.float32)
    outcome_means = outcome_tensor.view(-1, group_size).mean(dim=1, keepdim=True)
    outcome_stds = outcome_tensor.view(-1, group_size).std(dim=1, keepdim=True).clamp(min=1e-8)
    a_traj = ((outcome_tensor - outcome_means.flatten()) / outcome_stds.flatten()).detach()

    # A_MR: for each tag, normalize that tag's rewards across ALL completions in batch
    # (paper groups by tag type, not by prompt group)
    a_mr = torch.zeros_like(outcome_tensor)
    for tag_name, scores in process_rewards.items():
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        mu = scores_tensor.mean()
        sigma = scores_tensor.std().clamp(min=1e-8)
        normalized = (scores_tensor - mu) / sigma
        a_mr = a_mr + normalized.detach()  # accumulate across tags

    # Average A_MR across tags
    n_tags = len(process_rewards)
    if n_tags > 0:
        a_mr = a_mr / n_tags

    # Combine
    advantages = alpha * a_traj + (1 - alpha) * a_mr
    return advantages
```

Key design choices:
- `alpha = 0.5` (paper's specification)
- `A_MR` normalization is global across batch, not per-prompt-group (paper groups all `<explore>` steps together regardless of which trajectory they belong to)
- Process rewards from multiple tags are averaged (not summed) to prevent tag count from dominating

**3B. Update training loop**

In `train_v4()`, replace:
```python
rewards, ... = compute_rewards_v4(...)
advantages = compute_advantage(rewards, group_size)
```

With:
```python
outcome_rewards, process_rewards, ... = compute_rewards_v4(...)
advantages = compute_rlvmr_advantage(outcome_rewards, process_rewards, group_size, alpha=0.5)
```

---

### Phase 4: Policy Objective Fixes

**Addresses**: Gaps #9, #10 (KL regularization, clipping epsilon)

**4A. Restore KL regularization**

Paper's Equation 5:
```
L = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)] - lambda_KL * D_KL
```

Add back KL term to v4's per-sample loss:

```python
# In train_grpo_v4.py, replace lines 329-334:
kl = (new_token_lp - old_log_probs).mean()
lambda_kl = 0.01  # paper's specification
total_loss = pg_loss - lambda_kl * kl
```

This matches v3's approach (`train_grpo.py` line 518) but uses `lambda_KL = 0.01` instead of `beta = 0.1`. The paper's lambda_KL is smaller than our beta, meaning weaker regularization — appropriate since RLVMR's dual advantage already provides more stable gradients.

**4B. Fix clipping epsilon**

Replace `beta` with `0.2` in the clipping bounds:

```python
pg_loss_unclipped = -(ratio * adv).mean()
pg_loss_clipped = -(torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv).mean()
pg_loss = torch.min(pg_loss_unclipped, pg_loss_clipped)
```

v4 currently clips at `1 +/- beta` (0.1), which is tighter than the paper's PPO standard of 0.2.

---

### Phase 5: Configuration Updates

**Addresses**: Gap #13 (reward weights), and config consistency

**5A. New config in `grpo_config.py`**

Add `GRPO_CONFIG_V4` with RLVMR-aligned parameters:

```python
GRPO_CONFIG_V4 = {
    # ... same LoRA/training params as GRPO_CONFIG ...

    # RLVMR-specific
    "alpha": 0.5,          # outcome vs process advantage weight
    "lambda_kl": 0.01,     # KL regularization strength
    "clip_epsilon": 0.2,   # PPO clipping epsilon
    "lambda_format": 0.1,  # format penalty per missing tag

    # Outcome reward weights (sum to 1.0 for outcome component)
    "outcome_reward_weights": {
        "directional_assertion": 0.40,
        "dm_alignment": 0.30,
        "mechanism_commitment": 0.30,
    },

    # Process rewards are weighted by alpha, not by per-reward weights
    # Each process reward contributes equally to A_MR
}
```

**5B. Update `train_grpo_v4.py` to read from config**

Remove hardcoded `REWARD_WEIGHTS_V4` dict. Read from `GRPO_CONFIG_V4`.

---

### Phase 6: Training Pipeline Integration

**Addresses**: Gap #8 (no multi-turn structure) — partially, by documenting the limitation

The single-turn limitation is structural and cannot be fixed without changing the task. Instead, we make the adaptation explicit:

1. Add a `scripts/run_grpo_v4_rlvmr.sh` script that runs the full pipeline:
   ```bash
   # Step 1: Cold-start SFT
   python3 -m src.teacher.generate_cold_start_data \
       --output data/processed/cold_start_tagged.jsonl \
       --count 200

   python3 -m src.student.train_cold_start_sft \
       --data data/processed/cold_start_tagged.jsonl \
       --output checkpoints/lora_adapters/cold_start_sft \
       --epochs 5

   # Step 2: Merge cold-start adapter
   # (use Unsloth merge utility or hf utilities)

   # Step 3: GRPO v4 with RLVMR advantage
   python3 -m src.student.train_grpo_v4 \
       --base-model checkpoints/lora_adapters/cold_start_merged \
       --output-dir checkpoints/lora_adapters/grpo_v4_rlvmr \
       --dataset-path data/processed/grpo_causal_no_dag.jsonl \
       --max-steps 500
   ```

2. Add `--alpha` CLI argument to `train_grpo_v4.py` for advantage balance tuning

---

## File Change Summary

| File | Change | Gaps Addressed |
|------|--------|----------------|
| `src/teacher/generate_cold_start_data.py` | **NEW** — teacher-generated tagged demonstrations | #1 |
| `src/student/train_cold_start_sft.py` | **NEW** — 5-epoch SFT on tagged data | #1 |
| `scripts/run_cold_start.sh` | **NEW** — cold-start runner | #1 |
| `src/student/rewards.py` | Rewrite RLVMR section: success-conditional planning, reflection reward, null correctness reward, format penalty -0.1, separate outcome/process return | #4, #5, #6, #7, #11 |
| `src/student/train_grpo_v4.py` | Replace `compute_advantage` with `compute_rlvmr_advantage`; restore KL regularization; fix clipping to 0.2; read config instead of hardcoded weights | #3, #9, #10, #12, #13 |
| `src/student/grpo_config.py` | Add `GRPO_CONFIG_V4` with RLVMR parameters | #13 |
| `scripts/run_grpo_v4_rlvmr.sh` | **NEW** — full pipeline runner | #8 (documentation) |
| `papers/rlvmr_implementation_audit.md` | Update with "resolved" status for each gap | all |

---

## What We're NOT Changing (and Why)

**Multi-turn environment loop (Gap #8)**: Cannot be added without changing the task from single-turn causal reasoning to multi-turn agent interaction. The single-shot reward computation is the correct adaptation for our domain. The paper's "dense, process-level supervision" becomes "dense, per-tag supervision" — still denser than outcome-only, just not per-step-in-episode.

**Tag names (Gap #2)**: As analyzed, realigning to `<explore>` and `<reflection>` with RLVMR's exact semantics would introduce unverifiable rewards. Our adapted tags (`<commitment>` for anti-hedging, `<reflection>` for self-critique) are the domain-appropriate equivalents.

**Reward weight sum (Gap #13)**: The paper doesn't specify that outcome and process weights sum to 1.0 — it uses alpha=0.5 as a mixing coefficient between two separately normalized advantages. Our old weights summing to 1.0 was a design choice from the proposal, not the paper. The new approach removes per-reward weights for process rewards entirely, since they're normalized independently in A_MR.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cold-start SFT overfits to 200 samples | Medium | 5 epochs is paper-prescribed; monitor validation loss |
| Model produces tags but with empty/garbage content | High | Reflection reward requires self-critique keywords, not just tag presence |
| Dual advantage causes training instability | Low | Paper's alpha=0.5 is well-tested; separate normalization prevents signal dominance |
| KL regularization too weak (0.01 vs 0.1 beta) | Low | Paper's ablation shows 0.01 works; dual advantage provides stability |
| Tag format doesn't transfer to eval (free-form) | Medium | Eval on EconCausal uses free-form answers; the tagged format is training-only. Paper shows transfer works (RLVMR agents evaluated without tags). |

---

## Execution Order

1. **Phase 2** (reward functions) — no dependencies, can be done first
2. **Phase 3** (dual advantage) — depends on Phase 2's return format
3. **Phase 4** (policy objective) — independent, can parallel with Phase 3
4. **Phase 5** (config) — independent
5. **Phase 1** (cold-start SFT) — depends on final tag set from Phase 2
6. **Phase 6** (pipeline script) — depends on all above

Estimated total implementation time: 4-6 hours of coding, 1.5 hours of cold-start training.
