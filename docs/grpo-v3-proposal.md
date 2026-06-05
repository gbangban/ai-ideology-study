# GRPO v3: Verifiable Causal Reasoning with Process Rewards

**Date:** 2026-06-04
**Status:** Implementation complete, awaiting training and evaluation

---

## Abstract

Supervised fine-tuning on Dialectical Materialism (DM)-aligned data improved the student model's formal causal inference (Corr2Cause: +38pp) but degraded applied economic causal reasoning (EconCausal: -4 to -13pp). The dominant failure mode is hedging: the model replaces correct directional answers (`+`) with ambiguous responses (`mixed`). This proposal describes an intervention using verifiable meta-reasoning rewards (RLVMR) and synthetic causal training data to break the hedging equilibrium. We compare four conditions: SFT baseline (no GRPO), GRPO v2 (outcome rewards on SFT questions), GRPO v3 (outcome rewards on causal dataset), and GRPO v4 (outcome + process rewards on causal dataset). Comparing v3 vs v4 on identical data isolates whether process-level rewards improve causal reasoning beyond outcome rewards alone.

---

## 1. Background and Motivation

### 1.1 The Hedging Regression

After SFT+DPO DM alignment, the Qwen/Qwen3.5-9B student model was evaluated on two causal reasoning benchmarks. The results present a paradox:

| Benchmark | Task | Baseline BF16 | Finetuned BF16 | Change |
|---|---|---|---|---|
| EconCausal | Task 1 Economics | 60.30% | 47.94% | **-12.36pp** |
| EconCausal | Task 1 Finance | 56.51% | 43.02% | **-13.49pp** |
| EconCausal | Task 2 | 69.72% | 65.85% | -3.87pp |
| EconCausal | Task 3 | 22.18% | 11.38% | **-10.80pp** |
| Corr2Cause | All | 36.3% | 74.6% | **+38.3pp** |

The dominant failure mode on EconCausal is the model replacing correct directional answers (`+`) with hedged answers (`mixed`). The model appears to have internalized that "everything depends on structural conditions" and applies this rule uniformly, even when the prompt defines a specific context and the empirical evidence supports a clear directional effect.

### 1.2 Open Questions

The Corr2Cause improvement and EconCausal regression may reflect the same underlying behavior or distinct skills. Corr2Cause is binary (True/False), and the finetuned model's tendency to answer "False" may align with the benchmark's answer distribution (84.5% False) without reflecting genuine causal reasoning improvement. We cannot distinguish between improved causal graph deduction and calibrated refusal based on current data.

What is clear is that the model's hedging behavior on EconCausal represents a measurable regression that warrants intervention.

---

## 2. Teacher Model Audit

Before designing the intervention, we audited the SFT training data to determine whether the teacher model's answers contribute to the hedging behavior.

An automated keyword audit of 250 teacher-generated answers (`data/processed/batch_00000.json`) found:

| Classification | Count | Percentage |
|---|---|---|
| Directional commitment (no hedging) | 74 | 29.6% |
| Hedging | 10 | 4.0% |
| Neutral (substantive, neither committed nor hedging) | 166 | 66.4% |

The teacher model (Qwen3.5-27B) does not exhibit pervasive hedging. The audit uses keyword matching and should be treated as indicative rather than definitive — the 66.4% "neutral" category may include answers that implicitly hedge through qualifications without triggering hedging keywords. Nevertheless, the data does not support the claim that the SFT training data is the primary source of the student's hedging behavior.

---

## 3. Related Work

Five papers inform the experimental design. We distinguish each paper's established findings from our speculative application to this project.

### 3.1 Ideological Bias in Economic Causal Reasoning

**Lee et al. (2026).** "Ideological Bias in LLMs' Economic Causal Reasoning." arXiv:2604.21334.

Twenty SOTA LLMs evaluated on 1,056 ideology-contested causal triplets from economics/finance literature. LLMs are systematically more accurate when empirical truth aligns with intervention-oriented expectations. Open-source models show a +15.1pp accuracy gap favoring intervention-truth items. When models err, errors lean intervention-oriented.

**Application to this project:** DM theory emphasizes multi-causal, structurally complex outcomes, which shares features with the intervention-oriented framework documented in this paper. It is plausible that DM training shifted the model toward intervention-oriented reasoning, contributing to hedging. This connection has not been established. The paper does not study DM-aligned models, and hedging ("mixed") is not identical to intervention-oriented bias (which tends toward predicting positive effects).

### 3.2 Causal Reasoning Benchmark

**Sawarni et al. (2026).** "A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation." arXiv:2602.20571.

173 causal inference queries across 132 datasets. Separates identification (research design specification) from estimation (numerical computation). GPT-5.3 achieves 79.2% correct strategy selection but only 34.1% full identification specification. The bottleneck is in specification details.

**Application to this project:** Our EconCausal regression may reflect identification failures — the model recognizes the causal question but misidentifies when a directional answer is warranted. We have not decomposed our model's errors into identification versus estimation failures.

### 3.3 ReCITE: Causal Relationships from Text

**Saklad et al. (2025).** "Can Large Language Models Infer Causal Relationships from Real-World Text?" arXiv:2505.18931v4.

292 academic papers with causal loop diagrams. Best model (Claude Opus 4.5) achieves F1=0.535. All models show high precision but low recall — they generate plausible but incorrect causal relationships. Explicitness is the dominant difficulty factor. Direction reversals are rare (<1.1%).

**Application to this project:** The finding that models hallucinate causal connections is consistent with our Task 3 regression (-10.80pp), where the model may force connections between unrelated variables. The connection is indirect since Task 3 tests misinformation detection, not causal graph reconstruction.

### 3.4 Propaganda Is All You Need

**Kronlund-Drouault (2024).** "Propaganda Is All You Need." arXiv:2410.01810.

SFT alignment reorganizes embedding space; DPO is comparatively surface-level. Training GPT-2 on Trotskyist data shifted embedding distances between political concepts. Narrow fine-tuning produces broad misalignment across unrelated domains.

**Application to this project:** If SFT has deeper representational effects than GRPO, then hedging established during SFT may be difficult to correct through reward optimization alone. The paper's experiments use GPT-2, so generalizability to modern models is uncertain.

### 3.5 RLVMR: Verifiable Meta-Reasoning Rewards

**Zhang et al. (2025).** "Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents." arXiv:2507.22844.

Process-level rewards on tagged reasoning steps improve agent performance on long-horizon tasks. Qwen2.5-7B + RLVMR achieves 83.6% on ALFWorld L2 (unseen categories), beating GiGPO by 16.4pp. Ablation: removing outcome reward collapses performance (12.5% vs 56.3%), removing meta-reasoning reward costs 11pp, removing cold-start SFT costs 15.7pp.

**Application to this project:** We adapt the tagged reasoning format and process rewards to causal reasoning. RLVMR was designed for task-completion agents. Transfer to causal reasoning is plausible but untested.

---

## 4. Hypotheses

**H1 (Process rewards reduce hedging):** v4 (process + outcome rewards on causal data) will produce fewer hedged responses than v3 (outcome-only rewards on the same causal data). The commitment tag structure removes hedging as a reward-maximizing strategy.

**H2 (Causal graph training transfers):** Training on programmatic DAG queries with verifiable ground truth will improve EconCausal Task 1 accuracy. The skill of identifying confounders, mediators, and colliders in abstract graphs is transferable to economic causal questions.

**H3 (Context-flipping trains conditional commitment):** Presenting the same causal relationship under two institutional contexts and rewarding distinct directional answers teaches the model to commit within a defined context while remaining flexible across contexts.

**H4 (Null-effect training reduces spurious causality):** Training on orthogonal variable pairs where the correct answer is null effect reduces the model's tendency to force connections between unrelated variables, potentially improving Task 3 accuracy.

**H5 (Verifiable rewards improve training efficiency):** Rule-based rewards eliminate judge model overhead, reducing per-step compute cost and removing judge model bias as a confounding variable.

---

## 5. Method

### 5.1 Training Data

A synthetic dataset of 660 prompts across six categories:

| Category | Count | Ground Truth Type | Description |
|---|---|---|---|
| Causal graph (general) | 100 | Programmatic | DAGs with d-separation queries. General domain to avoid economic vocabulary confound. |
| Causal graph (economic) | 100 | Programmatic | Same DAG structure with economic variable names. Tests skill transfer with economic vocabulary. |
| Context-flip pairs | 280 (140x2) | Structural | Same causal relationship under two institutional contexts. Rewards distinct directional answers. |
| Null-effect (economic) | 60 | Structural | Orthogonal economic variables. Rewards explicit null effect declaration. |
| Null-effect (general) | 40 | Structural | Orthogonal general variables. Prevents economic-only overfitting. |
| Contradiction (economic) | 80 | Structural | Opposing claims on the same topic. Rewards quality of reasoning for both sides. |

**Programmatic ground truth (Type A):** Causal graph queries where correctness is determined by d-separation algorithm applied to the generated DAG. No model judgment involved. DAGs are generated via topological ordering, which guarantees acyclicity.

**Structural rewards (Type B):** Context-flip, null-effect, and contradiction data. Reward is based on response structure (tag presence, commitment format, directional distinction), not factual accuracy. This is a pragmatic choice: we lack verifiable factual labels for economic causal claims. A limitation is that the model could produce a structurally correct response with a factually wrong commitment and receive full reward.

### 5.2 Reward Structure

**v4 rewards (7 components, sum to 1.0):**

| Reward | Weight | Type | Description |
|---|---|---|---|
| `directional_assertion` | 0.25 | Outcome | +0.5 per commitment keyword, -0.5 per hedging keyword |
| `dm_alignment` | 0.15 | Outcome | 2 of 3 DM keyword categories required for full score |
| `mechanism_commitment` | 0.15 | Outcome | Mechanism naming + directional commitment |
| `planning` | 0.15 | Process | `<planning>` tag with >=2 variable keywords |
| `commitment` | 0.15 | Process | `<commitment>` tag with definitive direction |
| `monitor` | 0.10 | Process | `<monitor>` tag referencing context/constraints |
| `format_penalty` | 0.05 | Process | -0.05 per missing required tag |

**v2/v3 rewards (3 components, sum to 1.0):**

| Reward | Weight | Type |
|---|---|---|
| `dm_alignment` | 0.45 | Outcome |
| `directional_assertion` | 0.30 | Outcome |
| `mechanism_commitment` | 0.25 | Outcome |

v2 and v3 share the same outcome-only reward weights. v4 redistributes weight from outcome rewards to process rewards, with DM keyword alignment decreasing from 0.45 to 0.15.

### 5.3 RLVMR Tagged Output Format

v4 requires the model to produce tagged output:

```
<planning>
Identify variables and context.
</planning>
<reasoning>
Trace causal mechanisms.
</reasoning>
<commitment>
Definitive directional answer.
</commitment>
<monitor>
Self-check against context.
</monitor>
```

Process rewards activate only when tags are present. The model has not been trained on this format (no cold-start SFT). Whether the model learns tag production through RL pressure alone is an open question. RLVMR's ablation shows -15.7pp without cold-start SFT, suggesting convergence will be slower than the prescribed approach.

### 5.4 Experimental Conditions

Four conditions, same base model checkpoint, same hyperparameters:

| Condition | Data | Rewards | Format | What it isolates |
|---|---|---|---|---|
| SFT baseline | — | — | Free-form | Starting point |
| GRPO v2 | Original SFT questions | 3 outcome rewards | Free-form | Effect of outcome-reward GRPO on SFT data |
| GRPO v3 | Synthetic causal dataset | 3 outcome rewards | Free-form | Effect of causal data with outcome rewards |
| GRPO v4 | Synthetic causal dataset | 3 outcome + 4 process rewards | RLVMR tagged | Effect of adding process rewards |

**Clean comparisons:**
- `v4 vs v3`: Effect of process rewards on the same causal data (isolated)
- `v3 vs v2`: Effect of causal data vs SFT questions with same rewards (isolated)
- `v2 vs SFT`: Effect of outcome-reward GRPO on SFT questions
- `v4 vs SFT`: Effect of the full pipeline

**Shared hyperparameters:**
- LoRA: rank=16, alpha=16, dropout=0.05, 7 target modules
- Training: batch=1, gradient accumulation=4, LR=5e-7, cosine scheduler
- GRPO: g=8, max length=512, beta=0.1
- Steps: 500 with 50 warmup

---

## 6. Evaluation

### 6.1 Primary Metric: EconCausal Accuracy

All four models evaluated on EconCausal Task 1 Economics, Task 1 Finance, Task 2, and Task 3. We report accuracy by task and the distribution of `+`, `-`, and `mixed` answers.

**Success criterion for v4 over SFT:** Statistically significant improvement on at least one Task 1 subtask (Task 1 Economics or Task 1 Finance) at p < 0.05.

### 6.2 Secondary Metrics

| Metric | Purpose |
|---|---|
| Corr2Cause accuracy | Check for degradation from causal graph training |
| HumanEval pass@1 | Check for coding degradation |
| Directional assertion rate | Fraction of EconCausal answers that are `+` or `-` rather than `mixed` |

### 6.3 Training Metrics

Logged per step:
- Average total reward and per-component reward means
- For v4: tag compliance rate (fraction of completions with all required tags)

### 6.4 Qualitative Audit

Manual review of 50 held-out economic causal questions per model:
- Hedging frequency: responses containing hedging language
- Mechanism quality: whether named causal mechanism is structurally sound
- For v4: tag compliance and tag content quality

---

## 7. Limitations

**Structural rewards are not factual rewards.** Type B rewards verify response structure, not factual correctness. The model could produce a perfectly structured response with a factually wrong directional commitment and receive full reward. This means v4 may improve the model's willingness to commit without improving commitment accuracy.

**Small dataset.** 660 prompts with 8 completions each over 500 steps produces 4,000 completions (with prompt sampling with replacement). If the model overfits to the synthetic data, we would observe high training reward scores without benchmark improvement.

**Tag format may not transfer to evaluation.** Eval benchmarks expect free-form answers. If the model reasons well only within the tagged format, the skill may not transfer to untagged evaluation.

**No cold-start SFT for tagged format.** RLVMR's ablation shows -15.7pp without cold-start SFT. If tags never emerge during training, the process reward signal remains near zero and the experiment tests whether format penalties alone can induce tag production.

---

## 8. Implementation

All code is implemented and tested (51/51 tests passing):

| Component | Files | Tests |
|---|---|---|
| Data generation | `src/teacher/generate_causal_graphs.py`, `generate_context_flips.py`, `generate_null_effects.py`, `generate_contradiction_pairs.py`, `build_grpo_dataset.py` | 13/13 |
| RLVMR rewards | `src/student/rewards.py` (4 new functions) | 14/14 |
| Training v2 | `src/student/train_grpo.py` (original, unchanged) | — |
| Training v3 | `src/student/train_grpo_v3.py` (outcome rewards on causal data) | — |
| Training v4 | `src/student/train_grpo_v4.py` (process + outcome rewards on causal data) | — |
| Configuration | `src/student/grpo_config.py` | — |
| Dataset | `data/processed/grpo_causal_dataset.jsonl` (620 prompts) | — |

---

## 9. Execution Plan

1. Evaluate SFT baseline on EconCausal, Corr2Cause, HumanEval (partially completed)
2. Run v2 GRPO: `python3 -m src.student.train_grpo --max-steps 500`
3. Run v3 GRPO: `python3 -m src.student.train_grpo_v3 --max-steps 500`
4. Run v4 GRPO: `python3 -m src.student.train_grpo_v4 --max-steps 500`
5. Evaluate all GRPO models on EconCausal, Corr2Cause, HumanEval
6. Compare SFT baseline vs v2 vs v3 vs v4 across all metrics
7. Qualitative audit of 50 held-out questions per model

---

## 10. References

1. Kronlund-Drouault, P. (2024). Propaganda Is All You Need. arXiv:2410.01810.
2. Saklad, D., Chadha, M., Pavlov, D., & Moraffah, E. (2025). Can Large Language Models Infer Causal Relationships from Real-World Text? arXiv:2505.18931v4.
3. Sawarni, R., Tan, S., & Syrgkanis, V. (2026). A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation. arXiv:2602.20571.
4. Zhang, Y., Chen, Y., Li, S., Tu, C., & Li, H. (2025). Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents. arXiv:2507.22844.
5. Lee, J., Yun, S., Kim, H., Min, S., Park, J., Park, S., & Kim, S. (2026). Ideological Bias in LLMs' Economic Causal Reasoning. arXiv:2604.21334.
