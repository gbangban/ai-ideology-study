# GRPO v3/v4: Verifiable Causal Reasoning with Process Rewards (RLVMR)

**Date:** 2026-06-06
**Status:** Fresh proposal. All prior v3/v4 code treated as hallucinated; implementation starts from this document.

---

## Abstract

Supervised fine-tuning on Dialectical Materialism (DM)-aligned data improved the student model's formal causal inference (Corr2Cause: +38pp) but degraded applied economic causal reasoning (EconCausal: -4 to -13pp). The dominant failure mode is hedging: the model replaces correct directional answers (`+`) with ambiguous responses (`mixed`). This proposal implements RLVMR (Reinforcement Learning with Verifiable Meta-Reasoning Rewards) adapted for single-turn causal reasoning, using **outcome-based rewards from real benchmark datasets** (EconCausal, Corr2Cause) rather than keyword proxies. We compare two conditions, both with cold-start SFT: v3 (outcome rewards only, flat advantage) and v4 (outcome + process rewards, dual advantage). Comparing v3 vs v4 isolates whether process-level rewards improve causal reasoning beyond outcome rewards alone.

---

## 1. Background and Motivation

### 1.1 The Hedging Regression

After SFT+DPO DM alignment, the Qwen/Qwen3.5-9B student model was evaluated on two causal reasoning benchmarks:

| Benchmark | Task | Baseline BF16 | Finetuned BF16 | Change |
|---|---|---|---|---|
| EconCausal | Task 1 Economics | 60.30% | 47.94% | **-12.36pp** |
| EconCausal | Task 1 Finance | 56.51% | 43.02% | **-13.49pp** |
| EconCausal | Task 2 | 69.72% | 65.85% | -3.87pp |
| EconCausal | Task 3 | 22.18% | 11.38% | **-10.80pp** |
| Corr2Cause | All | 36.3% | 74.6% | **+38.3pp** |

The dominant failure mode on EconCausal is the model replacing correct directional answers (`+`) with hedged answers (`mixed`).

### 1.2 Teacher Model Audit

Keyword audit of 250 teacher-generated answers found only 4.0% hedging. The teacher (Qwen3.5-27B) is not the source of hedging.

### 1.3 Why Outcome-Based Rewards Matter

The prior proposal used keyword-based rewards (directional_assertion, dm_alignment, mechanism_commitment) as outcome signals. These are **quality proxies, not correctness signals**. A model can produce a structurally perfect response with a factually wrong commitment and receive full reward. More critically, `mixed` answers were not penalized by being wrong — they were penalized by keyword absence, which the model could game.

**The fix:** Use ground truth from real benchmarks. EconCausal provides `answer` fields (`+`, `-`, `None`, `mixed`). Corr2Cause provides `relation` fields (`entailment`, `contradiction`, `neutral`). When the model's answer matches ground truth, `A_traj` is positive. When it hedges incorrectly, `A_traj` is negative because the answer is wrong, not because of missing keywords. This eliminates the hedging equilibrium.

---

## 2. Related Work

### 2.1 RLVMR: Verifiable Meta-Reasoning Rewards

**Zhang et al. (2025).** arXiv:2507.22844.

Process-level rewards on tagged reasoning steps improve agent performance on long-horizon tasks. Qwen2.5-7B + RLVMR achieves 83.6% on ALFWorld L2, beating GRPO by 16.4pp. Key components:

1. **Cold-start SFT** (200 trajectories, 5 epochs) to teach XML tag syntax
2. **Process rewards** on `<planning>`, `<explore>`, `<reflection>`, `<monitor>` tags
3. **Dual advantage**: `A_traj` (outcome) and `A_MR` (process) normalized separately, combined with alpha=0.5
4. **Clipped PPO** with KL regularization (lambda_KL=0.01, clip_epsilon=0.2)

Ablation: removing outcome reward collapses performance (12.5% vs 56.3%), removing meta-reasoning reward costs 11pp, removing cold-start SFT costs 15.7pp.

**Our adaptation:** Single-turn causal reasoning (no multi-turn loop). Tags adapted to domain. Outcome rewards from real benchmark ground truth, not environment success/failure.

### 2.2 Other References

See §3 of the original proposal for Lee et al. (ideological bias), Sawarni et al. (causal reasoning benchmark), Saklad et al. (ReCITE), and Kronlund-Drouault (propaganda). Unchanged from prior analysis.

---

## 3. Training Data

### 3.1 Primary Datasets (Hugging Face)

| Dataset | HF Path | Config | Train Samples | Ground Truth | Format |
|---|---|---|---|---|---|
| EconCausal | `qwqw3535/econcausal-benchmark` | `task1_econ` | 947 | `answer` (+, -, None, mixed) | Q&A with context |
| EconCausal | `qwqw3535/econcausal-benchmark` | `task1_finance` | 860 | `answer` (+, -, None, mixed) | Q&A with context |
| EconCausal | `qwqw3535/econcausal-benchmark` | `task2` | 284 | `answer` (+, -, None, mixed) | Q&A with context |
| EconCausal | `qwqw3535/econcausal-benchmark` | `task3` | 852 | `answer` (+, -, None, mixed) | Q&A with context |
| Corr2Cause | `tasksource/corr2cause` | — | 411,452 | `relation` (entailment, contradiction, neutral) | Premise + hypothesis |
| **Total** | | | **~414,400** | | |

**EconCausal answer distribution (training):**
- `+`: 1,203 (46.5%)
- `-`: 1,007 (39.0%)
- `None`: 266 (10.3%)
- `mixed`: 61 (2.4%)

**Corr2Cause relation distribution (training):**
- `contradiction`: 229,693 (55.8%)
- `neutral`: 105,179 (25.6%)
- `entailment`: 76,580 (18.6%)

### 3.2 Synthetic Dataset (Supplementary)

The synthetic dataset (`data/processed/grpo_causal_dataset.jsonl`, 620 prompts) is retained as supplementary training data for format diversity and domain-specific reasoning patterns. It is **not** the primary training data.

| Category | Count | Ground Truth | In Training? |
|---|---|---|---|
| causal_graph | 160 | Programmatic (d-separation) | **Parked** — needs `graph_correctness` reward |
| context_flip | 280 | Structural | Yes (supplementary) |
| null_effect | 100 | Structural (`null`) | Yes (supplementary) |
| contradiction_pair | 80 | Structural | Yes (supplementary) |

### 3.3 Training Set Composition

**Active training data:** EconCausal (task1_econ + task1_finance + task2 + task3 = 2,943 prompts) + Corr2Cause (sampled to 5,000 prompts for balanced training) + synthetic (360 non-DAG prompts) = **~8,300 prompts**.

**Sampling rationale:** Corr2Cause has 411K samples but is homogeneous in format (premise + hypothesis). We sample 5,000 to avoid dominance while providing sufficient coverage. EconCausal's 2,943 prompts are all used — they're the primary domain for the hedging problem.

**TODO:** Determine optimal Corr2Cause sample size. Current choice (5,000) balances format diversity against dataset dominance. Ablation with 1,000 and 50,000 would clarify the sweet spot.

---

## 4. Reward Structure

### 4.1 Outcome Rewards (Correctness-Based)

Outcome rewards are computed by comparing the model's answer to ground truth. This is the fundamental change from the prior proposal.

**EconCausal outcome reward:**
```
extract_sign(completion) -> one of "+", "-", "None", "mixed"
answer = doc["answer"]  # ground truth
correctness = 1.0 if extract_sign(completion) == answer else 0.0
```

The sign extraction logic mirrors `evals/configs/task_configs/econcausal.py`: JSON extraction first, then context-aware regex, then standalone token fallback.

**Corr2Cause outcome reward:**
```
extract_bool(completion) -> True/False
relation = doc["relation"]  # "entailment", "contradiction", "neutral"
# Map relation to expected boolean:
#   entailment -> True (hypothesis follows from premise)
#   contradiction -> False (hypothesis contradicts premise)
#   neutral -> ambiguous (model's answer doesn't determine correctness)
correctness = 1.0 if match else 0.0
```

**TODO:** Corr2Cause `neutral` relation has no verifiable ground truth for True/False. Options: (a) exclude neutral from training, (b) reward confidence calibration (penalize both True and False for neutral), (c) use a third label. Current plan: include neutral but reward only when the model's answer is consistent with the relation (entailment=True, contradiction=False, neutral=any). This is weaker than binary correctness but still better than keyword proxies.

**Synthetic data outcome reward:** For the 360 non-DAG synthetic prompts without ground truth, fall back to keyword-based quality proxies (directional_assertion, dm_alignment, mechanism_commitment, weighted 0.40/0.30/0.30). This is a known limitation: ~4.3% of training data uses proxy rewards.

### 4.2 Process Rewards (RLVMR Tags)

Process rewards activate on tagged output. Each is normalized independently for `A_MR` computation.

| Reward | Tag | Description |
|---|---|---|
| `planning` | `<planning>` | +1.0 if tag present with >=2 variable keywords, **conditional on outcome success** |
| `commitment` | `<commitment>` | +1.0 for definitive answer (any label: +, -, mixed, null), -0.5 for hedging |
| `reflection` | `<reflection>` | +1.0 for self-critique with keywords, +0.5 for self-referential language |
| `monitor` | `<monitor>` | Context/constraint reference check |
| `format_penalty` | — | -0.1 per missing required tag |

**Tag adaptations from RLVMR paper:**
- `<commitment>` replaces `<explore>`: anti-hedging (domain) vs anti-repetition (paper). Single-turn has no repetition to penalize.
- `<reflection>` is reintroduced with outcome-conditional reward: reflection is only rewarded if outcome reward exceeds threshold, preventing performative reflection on wrong answers.
- `<monitor>` retained but not required in format penalty.

**TODO:** Document that `<commitment>` is not a faithful reproduction of RLVMR's `<explore>`. The paper's explore reward checks for new object/location discovery across turns. Our commitment reward checks for definitive language within a single completion. The adaptation is justified (single-turn has no turn-to-turn repetition), but it's a deviation from the paper's mechanism.

### 4.3 Dual Advantage (v4 only)

v4 computes two separate advantages per RLVMR Equations 2-4:

```
A_traj = normalize(outcome_rewards) within each prompt group    # Eq 2
A_MR = normalize(process_rewards) per tag group, averaged       # Eq 3
A_t = alpha * A_traj + (1 - alpha) * A_MR                       # Eq 4, alpha=0.5
```

v3 uses flat advantage: single normalization of the weighted sum of all rewards.

---

## 5. Experimental Conditions

### 5.1 Conditions

Both conditions include cold-start SFT. This isolates process rewards + dual advantage as the variable.

| Condition | Data | Rewards | Format | What it isolates |
|---|---|---|---|---|
| SFT baseline | — | — | Free-form | Starting point |
| GRPO v3 | EconCausal + Corr2Cause + synthetic | Outcome rewards only (correctness-based) | Free-form | Effect of outcome-reward GRPO on real data |
| GRPO v4 | EconCausal + Corr2Cause + synthetic | Dual advantage (outcome + process) | RLVMR tagged | Effect of adding process rewards + dual advantage |

### 5.2 Shared Hyperparameters

- LoRA: rank=16, alpha=16, dropout=0.05, 7 target modules
- Training: batch=1, gradient accumulation=4, LR=5e-7, cosine scheduler
- GRPO: g=8, max length=512, beta=0.1
- Steps: 1,000 with 100 warmup

**TODO:** 1,000 steps on ~8,300 prompts with g=8 completions = ~9.8 epochs. The paper uses 100 epochs on 200 trajectories. Our epoch count is lower because we have more diverse data. Ablation with 2,000 and 500 steps would clarify whether 1,000 is sufficient.

### 5.3 v4-Specific Hyperparameters (per RLVMR paper)

- `alpha` (outcome/process mix): 0.5
- `lambda_kl` (KL regularization): 0.01
- `clip_epsilon` (per-token KL clipping): 0.2
- `lambda_format` (missing tag penalty): -0.1 per tag

---

## 6. Cold-Start SFT

Both v3 and v4 receive cold-start SFT. This follows the paper's prescription (Table 3: -15.7pp without cold-start) and controls for format exposure.

**Data generation:** Teacher model (Qwen3.5-27B) generates tagged demonstrations. The teacher is not DM-specific, so it can produce high-quality tagged reasoning without domain contamination. Sample 200 prompts from the training set, generate 3 tagged completions per prompt (600 total), SFT for 5 epochs.

**TODO:** Document that cold-start SFT is included in both v3 and v4. This is a deliberate choice to isolate process rewards as the variable between conditions. The prior proposal had v3 without cold-start, which confounded format exposure with process rewards. The shortcoming: we cannot measure the isolated effect of cold-start SFT without a third condition (v3-no-coldstart). This is a known confound.

---

## 7. Tagless Testing

v4 trains with tagged output but is evaluated on benchmarks that expect free-form answers. To verify v4 doesn't collapse without tags:

1. **Tagless evaluation:** After v4 training, evaluate on EconCausal/Corr2Cause with prompts that do NOT instruct tagged format. The model should produce free-form answers that still demonstrate improved causal reasoning.
2. **Tagless ablation:** Run a small inference test (100 prompts) where the model is explicitly instructed to produce free-form output. Compare reward scores and answer distributions against tagged output.

**TODO:** Tag transfer risk is higher than RLVMR's because our tags ARE the output structure, not intermediate reasoning steps. The paper's agents produce `<action>` tags during training but are evaluated on environment success (tag-agnostic). Our model produces `<commitment>` tags during training but is evaluated on free-form EconCausal answers. If the model learns to reason only within tags, transfer will fail. Mitigation: cold-start SFT teaches format, but GRPO rewards content within tags, not format itself. The commitment reward checks for definitive language, not tag presence. This should promote transfer, but it needs empirical verification.

---

## 8. Evaluation

### 8.1 Primary Metric: EconCausal Accuracy

All models evaluated on EconCausal Task 1 Economics, Task 1 Finance, Task 2, Task 3. Report accuracy by task and answer distribution (`+`, `-`, `mixed`, `None`).

**Success criterion for v4 over SFT:** Statistically significant improvement on at least one Task 1 subtask at p < 0.05 (binomial test).

**TODO:** Define sample size for statistical power. With Task 1 Economics at ~947 samples, a 5pp improvement (48% -> 53%) requires ~380 samples for 80% power at alpha=0.05. The full dataset provides sufficient power. However, the success criterion ("at least one Task 1 subtask") is a multiple-comparison problem (two subtasks). Apply Bonferroni correction: p < 0.025 per subtask.

### 8.2 Secondary Metrics

| Metric | Purpose |
|---|---|
| Corr2Cause accuracy | Check for degradation from causal graph training |
| HumanEval pass@1 | Check for coding degradation |
| Directional assertion rate | Fraction of EconCausal answers that are `+` or `-` rather than `mixed` |

### 8.3 Training Metrics

Logged per step:
- Average total reward and per-component reward means
- For v4: tag compliance rate, `A_traj` vs `A_MR` distribution
- Reward saturation detection: if average reward plateaus for 100+ steps, flag for early stopping

**TODO:** Implement early stopping/reward saturation monitoring. The paper doesn't specify early stopping criteria. Current plan: stop if average reward doesn't improve for 200 steps or if KL divergence exceeds threshold. Need to define these thresholds empirically.

---

## 9. Implementation

### 9.1 Files to Create (All New)

| File | Purpose |
|---|---|
| `src/student/train_grpo_v3.py` | Outcome-reward GRPO with correctness-based rewards, flat advantage |
| `src/student/train_grpo_v4.py` | Dual-advantage GRPO with process rewards, KL regularization, correct clipping |
| `src/student/rewards_v3v4.py` | Correctness-based outcome rewards + process rewards (replaces current `rewards.py` RLVMR section) |
| `src/student/grpo_config_v4.py` | `GRPO_CONFIG_V3` and `GRPO_CONFIG_V4` with all hyperparameters |
| `src/teacher/generate_cold_start_data.py` | Teacher-generated tagged demonstrations |
| `src/student/train_cold_start_sft.py` | 5-epoch SFT on tagged data |
| `scripts/run_cold_start.sh` | Cold-start runner |
| `scripts/run_grpo_v3.sh` | v3 training runner |
| `scripts/run_grpo_v4.sh` | v4 training runner |
| `src/student/tagless_eval.py` | Tagless evaluation harness |

### 9.2 Reward Function Design

**`rewards_v3v4.py`** contains:

1. `compute_econcausal_correctness(completion, ground_truth)` — extracts sign, compares to answer
2. `compute_corr2cause_correctness(completion, relation)` — extracts True/False, maps relation to expected answer
3. `compute_null_correctness(completion)` — for synthetic null_effect prompts
4. `compute_proxy_outcome(completion, category)` — keyword proxies for synthetic prompts without ground truth
5. `compute_planning_reward(text, success)` — success-conditional planning reward
6. `compute_commitment_reward(text)` — generalized commitment (any label)
7. `compute_reflection_reward(text)` — self-critique reward
8. `compute_monitor_reward(text)` — context reference check
9. `compute_format_penalty(text)` — -0.1 per missing tag
10. `compute_outcome_reward(doc, completion)` — unified outcome reward (dispatches by dataset type)
11. `compute_process_rewards(text, outcome_success)` — returns dict of per-tag rewards

### 9.3 Training Loop Design

**`train_grpo_v3.py`** (flat advantage):
```
for each prompt group:
    completions = generate_completions(prompt, g=8)
    outcome_rewards = [compute_outcome_reward(doc, c) for c in completions]
    advantages = compute_advantage(outcome_rewards, group_size)  # flat normalization
    loss = ppo_clip_loss(ratios, advantages, clip_epsilon=0.2)
    loss = loss - beta * kl_approximation  # beta=0.1
```

**`train_grpo_v4.py`** (dual advantage):
```
for each prompt group:
    completions = generate_completions(prompt, g=8)
    outcome_rewards = [compute_outcome_reward(doc, c) for c in completions]
    process_rewards = {tag: [compute_X_reward(c, outcome_success) for c in completions] for tag in tags}
    advantages = compute_rlvmr_advantage(outcome_rewards, process_rewards, group_size, alpha=0.5)
    loss = ppo_clip_loss(ratios, advantages, clip_epsilon=0.2)
    loss = loss - lambda_kl * kl_approximation  # lambda_kl=0.01
```

---

## 10. Functional Contradictions and Deviations (TODOs)

These are documented as known issues to resolve before or during implementation.

### 10.1 Outcome Reward Contradictions

**TODO-1: Outcome rewards were keyword proxies, not correctness.** The prior proposal claimed outcome rewards (directional_assertion, dm_alignment, mechanism_commitment) serve as `A_traj`. These are keyword density scores, not factual correctness. The model can produce a structurally perfect but factually wrong response and receive full reward. **Resolved** by using EconCausal/Corr2Cause ground truth.

**TODO-2: Synthetic data without ground truth.** 360/8,300 synthetic prompts (4.3%) lack verifiable answers. These fall back to keyword proxies. This is acceptable because the majority of training data has ground truth, but it means `A_traj` is heterogeneous: correctness for ~96%, keyword density for ~4%.

**TODO-3: Corr2Cause neutral relation.** ~25.6% of Corr2Cause training data has `neutral` relation, which has no verifiable True/False answer. Current plan rewards consistency (entailment=True, contradiction=False, neutral=any). This is weaker than binary correctness. Alternative: exclude neutral, reducing Corr2Cause to ~306K samples.

### 10.2 Literature Deviations

**TODO-4: `<commitment>` vs `<explore>`.** RLVMR's `<explore>` rewards discovering new objects/locations (anti-repetition across turns). Our `<commitment>` rewards definitive language (anti-hedging within a single completion). This is a domain adaptation, not a faithful reproduction. Justified: single-turn has no turn-to-turn repetition.

**TODO-5: Single-turn vs multi-turn.** RLVMR's rewards are per-step within episodes (30-step ALFWorld episodes). Our rewards are per-completion (single turn). The "dense, process-level supervision" becomes "dense, per-tag supervision." This is a structural limitation of the domain, not an implementation choice.

**TODO-6: No multi-turn advantage comparison.** RLVMR compares advantages across turns to detect when process rewards diverge from outcome rewards. Single-turn has no turn comparison. This mechanism is absent.

**TODO-7: Cold-start SFT in both v3 and v4.** The paper's cold-start SFT teaches tag format. v3 doesn't use tags, so cold-start SFT for v3 teaches format that won't be used in GRPO. This is included as a control (both conditions get cold-start), but it's wasteful for v3. The shortcoming: we cannot measure cold-start's isolated effect without a v3-no-coldstart condition.

### 10.3 Implementation Gaps from Audit

**TODO-8: KL regularization.** v4 must use `lambda_kl=0.01` (paper's specification), not `beta=0.1` (v3's approach). The paper's lambda_KL is smaller, meaning weaker regularization — appropriate since dual advantage provides more stable gradients.

**TODO-9: Clipping epsilon.** v4 must use `clip_epsilon=0.2` (PPO standard), not `beta=0.1` (prior v4 code). Tighter clipping constrains policy updates and may slow learning.

**TODO-10: Format penalty strength.** Paper specifies -0.1 per format violation per step. Our single-turn adaptation applies -0.1 per missing tag per completion. Maximum penalty is -0.4 (4 tags). This is weaker than the paper's per-step application during 30-step episodes, but appropriate for single-turn.

**TODO-11: Planning reward conditionality.** Paper's planning reward is success-conditional (awarded only if trajectory succeeds). Prior code awarded planning unconditionally. Fixed: planning reward now requires outcome reward > threshold.

**TODO-12: Reflection reward.** Paper's reflection reward requires corrective action after failures. Our adaptation rewards self-critique keywords and is outcome-conditional. This is weaker than the paper's "corrective action" check (which requires environment interaction), but it's the best single-turn analog.

### 10.4 Experimental Design Issues

**TODO-13: Tag transfer risk.** v4 trains with tags but evaluates without. Tags ARE the output (not intermediate reasoning), so transfer risk is higher than RLVMR. Mitigated by tagless testing (§7), but empirical verification is needed.

**TODO-14: Undertraining risk.** 1,000 steps × 8 completions / 8,300 prompts ≈ 0.96 epochs. This is far fewer epochs than the paper's 100. However, our dataset is much larger and more diverse. The question is whether 1,000 steps is sufficient for policy convergence. Ablation with 2,000 and 5,000 steps would clarify.

**TODO-15: EconCausal success criterion sample size.** With ~947 Task 1 Economics samples, a 5pp improvement requires ~380 samples for 80% power. The full dataset provides sufficient power, but the multiple-comparison problem (two Task 1 subtasks) requires Bonferroni correction.

**TODO-16: Early stopping.** No early stopping criteria defined. Need to implement reward saturation detection and KL divergence monitoring.

**TODO-17: Reward weight for proxy data.** For the 4.3% of synthetic prompts without ground truth, keyword proxies are used. The weight of proxy rewards relative to correctness rewards is 1.0:1.0 (same scale). This means proxy rewards have the same influence as correctness rewards, which may be inappropriate since proxies are noisy. Consider scaling proxy rewards by 0.5.

**TODO-18: Cold-start SFT data source.** Teacher model (Qwen3.5-27B) generates tagged demonstrations. The teacher is not DM-specific, so it can produce high-quality tagged reasoning. However, the teacher's reasoning style may differ from the student's (post-SFT+DPO) reasoning style. This could cause a distribution shift during cold-start SFT. Mitigation: use the student's SFT+DPO checkpoint as the base for cold-start SFT, not the base model.

---

## 11. Execution Plan

### Phase 1: Infrastructure
1. Create `rewards_v3v4.py` with correctness-based outcome rewards and process rewards
2. Create `grpo_config_v4.py` with `GRPO_CONFIG_V3` and `GRPO_CONFIG_V4`
3. Create dataset loading pipeline for EconCausal + Corr2Cause + synthetic

### Phase 2: Cold-Start SFT
4. Create `generate_cold_start_data.py` (teacher generates tagged demonstrations)
5. Create `train_cold_start_sft.py` (5-epoch SFT)
6. Run cold-start, verify tag compliance >= 80%

### Phase 3: v3 Training
7. Create `train_grpo_v3.py` (outcome rewards only, flat advantage)
8. Run v3 on merged cold-start checkpoint
9. Evaluate v3 on EconCausal, Corr2Cause, HumanEval

### Phase 4: v4 Training
10. Create `train_grpo_v4.py` (dual advantage, process rewards, KL regularization, correct clipping)
11. Run v4 on merged cold-start checkpoint
12. Evaluate v4 on EconCausal, Corr2Cause, HumanEval

### Phase 5: Tagless Testing
13. Create `tagless_eval.py`
14. Run v4 tagless evaluation
15. Compare tagged vs tagless performance

### Phase 6: Analysis
16. Compare v3 vs v4 across all metrics
17. Qualitative audit of 50 held-out questions per model
18. Document findings and update proposal

---

## 12. References

1. Kronlund-Drouault, P. (2024). Propaganda Is All You Need. arXiv:2410.01810.
2. Saklad, D., Chadha, M., Pavlov, D., & Moraffah, E. (2025). Can Large Language Models Infer Causal Relationships from Real-World Text? arXiv:2505.18931v4.
3. Sawarni, R., Tan, S., & Syrgkanis, V. (2026). A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation. arXiv:2602.20571.
4. Zhang, Y., Chen, Y., Li, S., Tu, C., & Li, H. (2025). Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents. arXiv:2507.22844.
5. Lee, J., Yun, S., Kim, H., Min, S., Park, J., Park, S., & Kim, S. (2026). Ideological Bias in LLMs' Economic Causal Reasoning. arXiv:2604.21334.
6. EconCausal Benchmark: https://huggingface.co/datasets/qwqw3535/econcausal-benchmark
7. Corr2Cause: https://huggingface.co/datasets/tasksource/corr2cause
