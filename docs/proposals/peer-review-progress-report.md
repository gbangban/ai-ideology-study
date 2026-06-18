# Epistemic Transfer in Language Models: Dialectical Materialism Alignment and Causal Reasoning Shifts

**Author:** Melengor Yao Gbanaglo
**Date:** June 18, 2026
**Status:** In Progress -- Mid-Project Peer Review Update

---

## Abstract

**Motivation.** Supervised fine-tuning (SFT) is the foundational alignment step for large language models, yet its effects on reasoning beyond the training domain remain poorly understood. Asking five models "how do we stop climate change?" produces convergent responses emphasizing carbon pricing, clean energy transition, and individual behavioral change. None critique carbon pricing's empirical record. The instruments that exist operate at a fraction of the scale required to meet the RCP 2.6 warming pathway (below 2C above preindustrial levels). The Stern-Stiglitz target corridor is $50-100/tCO2, yet the global emissions-weighted average is $5/tCO2, an order of magnitude below the lower bound. Less than 1% of global emissions are priced at or above the target.

At these levels, existing schemes reduce emissions by an average of 6.8% after correction for publication bias. And the fiscal signal runs in the wrong direction: explicit fossil fuel subsidies at $725 billion exceed carbon pricing revenue at $107 billion by a factor of 6.8, producing a net negative fiscal signal on carbon. Only after SFT on DM-aligned data does our model produce a response explicitly identifying these gaps. This demonstrates that SFT enables causal model replacement, and can enable analytical frameworks otherwise inaccessible to the model.

**Method.** Dialectical Materialism (DM) is a Marxist analytical framework that explains social phenomena through material conditions, class relations, and historical dynamics rather than abstract ideals. We fine-tune Qwen3.5-9B via QLoRA on 1,500 DM question-answer pairs and evaluate across three reasoning domains to test whether SFT transfers deep epistemic priors.

**Results.** Initial SFT results reveal three divergent outcomes: general capability is preserved (MMLU -0.8pp, HumanEval 0.0pp), formal causal reasoning improves dramatically (Corr2Cause +38.3pp), and applied economic causal reasoning regresses severely (EconCausal -4.0 to -13.5pp). We trace the regressions to a single emergent artifact: a systematic hedging bias where the model converts correct positive causal predictions to ambiguous "mixed" answers.

**Next phase.** The project is now in Phase 2: reinforcement learning with GRPO to break the hedging equilibrium, comparing outcome-only rewards (v3, control) against dual advantage with process-level rewards (v4, experimental).

---

## 1. Introduction and Problem Statement

Large language models are aligned through supervised fine-tuning on curated instruction data. The alignment literature focuses on safety, helpfulness, and preference consistency. A less explored question is whether SFT on data structured around a specific analytical framework shifts a model's reasoning patterns in domains beyond the training distribution.

Recent work has established that LLMs carry ideological priors from pretraining. Kronlund-Drouault (2024) showed that alignment amplifies the dominant ideology of training data. Lee et al. (2026) found that LLMs exhibit systematic intervention-oriented bias in economic causal reasoning across 20 models. These works document ideological priors as a static property. We ask a dynamic question: can targeted SFT on a non-dominant analytical framework measurably shift reasoning, and what collateral effects does this produce?

**Concrete motivation.** Asking five models "how do we stop climate change?" produces convergent responses emphasizing carbon pricing, clean energy transition, and individual behavioral change. None critique carbon pricing's empirical record. The instruments that exist operate at a fraction of the scale required to meet the RCP 2.6 warming pathway (below 2C above preindustrial levels). The Stern-Stiglitz target corridor is $50-100/tCO2, yet the global emissions-weighted average is $5/tCO2, an order of magnitude below the lower bound. Less than 1% of global emissions are priced at or above the target.

At these levels, existing schemes reduce emissions by an average of 6.8% after correction for publication bias. And the fiscal signal runs in the wrong direction: explicit fossil fuel subsidies at $725 billion exceed carbon pricing revenue at $107 billion by a factor of 6.8, producing a net negative fiscal signal on carbon. Only after SFT on DM-aligned data does our model produce a response explicitly identifying these gaps. This demonstrates that SFT enables causal model replacement and that SFT can enable frameworks otherwise inaccessible to the model.

---

## 2. Literature Review

### 2.1 Ideological Bias in Language Models

Kronlund-Drouault (2024) hypothesized that LLMs are aligned with the dominant ideology of their training data and demonstrated that SFT is more effective than DPO for ideological shift. Lee et al. (2026) extended this to economic reasoning, showing systematic intervention-oriented bias on ideology-contested causal triplets. Our work complements both: rather than documenting preexisting bias, we show how targeted SFT can measurably shift reasoning patterns and produce emergent artifacts not present in the training data.

### 2.2 Causal Reasoning Evaluation

Syrgkanis et al. (2026) introduced CausalReasoningBenchmark, separating causal identification from causal estimation. Their key finding: the bottleneck lies in identification details -- a state-of-the-art model correctly identifies high-level strategy in 79% of cases but full identification-specification correctness drops to 34%. Saklad et al. (2025) introduced ReCITE for inferring causal relationships from real-world academic text, finding that even the best model achieves only 0.535 F1. Our work uses two benchmarks probing different causal reasoning capabilities: Corr2Cause (formal causal logic) and EconCausal (applied causal identification). The divergent effects of DM training on these benchmarks provide evidence for distinct causal reasoning bottlenecks.

### 2.3 Reinforcement Learning with Process Rewards

Zhang et al. (2025) introduced RLVMR -- reinforcement learning with verifiable meta-reasoning rewards. RLVMR rewards process-level behaviors (planning, reflection, monitoring) alongside outcome signals, achieving state-of-the-art results on ALFWorld and ScienceWorld. Our Phase 2 adapts RLVMR for single-turn causal reasoning, using process rewards to break the hedging prior by rewarding definitive commitments while maintaining structural reasoning quality.

### 2.4 Key References

1. Kronlund-Drouault, P. (2024). Propaganda Is All You Need. arXiv:2410.01810.
2. Lee, J. et al. (2026). Ideological Bias in LLMs' Economic Causal Reasoning. arXiv:2604.21334.
3. Syrgkanis, V. et al. (2026). A Real-World Benchmark for Disentangled Evaluation of Causal Identification. arXiv:2602.20571.
4. Saklad, D. et al. (2025). Can Large Language Models Infer Causal Relationships from Real-World Text? arXiv:2505.18931v4.
5. Zhang, Y. et al. (2025). Reinforcement Learning with Verifiable Meta-Reasoning Rewards. arXiv:2507.22844.

---

## 3. Project Description

### 3.1 What We Intend to Do

This project has two phases:

**Phase 1 (Completed):** SFT on DM-aligned data, followed by comprehensive evaluation across general capability, formal causal logic, and applied economic causal reasoning.

**Phase 2 (In Progress):** GRPO (Generalized Policy Optimization) training with two experimental conditions designed to reverse the hedging regression on EconCausal while preserving the Corr2Cause improvement:

- **v3 (Control):** Outcome-only rewards using ground-truth correctness from EconCausal and Corr2Cause benchmarks. Flat advantage computation.
- **v4 (Experimental):** Dual advantage combining outcome rewards with process-level rewards (RLVMR adaptation). Tags for planning, commitment, reflection, and monitor. Separate normalization of trajectory advantage (A_traj) and meta-reasoning advantage (A_MR), combined with alpha=0.5.

### 3.2 Why This Matters

The SFT results demonstrate that alignment training transfers epistemic stances. If DM training produces detectable transfer (structural skepticism applied indiscriminately), then standard alignment training (RLHF on helpful/harmless data) may carry similar hidden epistemic priors (deference to authority, controversy avoidance) that go unaudited because they align with the evaluator's own priors.

---

## 4. Methodology

### 4.1 Models and Hardware

| Component | Specification |
|-----------|--------------|
| Student model | Qwen/Qwen3.5-9B (Instruct variant) |
| Teacher model | Unsloth/Qwen3.5-27B (base, data generation only) |
| Judge model | Qwen/Qwen3.5-4B (BF16 via SG-Lang server) |
| GPU | NVIDIA RTX 5090 (32GB VRAM) |
| Framework | Unsloth Studio (SFT) + custom GRPO (TRL GRPOTrainer) |
| Quantization | NF4 via bitsandbytes (runtime) |
| LoRA | rank=32, alpha=32, dropout=0.05, 7 target modules |

### 4.2 Datasets

**Training data (Phase 1 -- SFT):**
- 1,500 AI-generated questions, quality-filtered and deduplicated
- Two-axis taxonomy: 11 social categories x 7 historical epochs
- Five question types: Neutral Framing (40%), Contrast (20%), Application (20%), Conceptual DM (5%), Adversarial (15%)
- Core constraint: no ideological terminology or framework names in question text

**Training data (Phase 2 -- GRPO):**
- EconCausal benchmark (qwqw3535/econcausal-benchmark): 2,943 prompts across 4 tasks with ground-truth causal signs (+, -, None, mixed)
- Corr2Cause (tasksource/corr2cause): sampled to 5,000 from 411K, ground-truth relations (entailment, contradiction, neutral)
- Synthetic causal data: 360 non-DAG prompts (context-flip, null-effect, contradiction pairs)
- Total: ~8,300 prompts

**Evaluation benchmarks:**
- General: MMLU (14,042 samples), HumanEval (164), GPQA Diamond (198), IFEval (541)
- Domain-specific: EconCausal (3,943 samples), Corr2Cause (1,162 samples)

### 4.3 Tools and Techniques

| Tool | Purpose |
|------|---------|
| Unsloth Studio | SFT training via UI, teacher answer generation, GGUF export |
| TRL GRPOTrainer | GRPO training with custom reward functions |
| python-pptx | Slide generation |
| Trackio | Local experiment tracking (replacement for W&B) |
| Docker Desktop | Reproducible training environment (CUDA 12.6, PyTorch 2.7.0) |
| SG-Lang | BF16 judge model serving (Qwen3.5-4B) |
| lm_eval harness | Benchmark evaluation infrastructure |

### 4.4 Reward Structure (Phase 2)

**Outcome rewards (v3 and v4):**
- Three-tier scoring: full credit (0.9-1.0) for correct answer, partial credit (0.1-0.3) for wrong answer with reasoning signals, no credit (0.0) for no signal
- EconCausal: extract causal sign from completion, compare to ground truth
- Corr2Cause: extract True/False from completion, map relation to expected answer
- Reasoning quality reward: heuristic shaping on [0.0, 0.5] using regex patterns for structured reasoning, causal language, and hedge detection

**Process rewards (v4 only):**
- Planning: +1.0 for variable identification, conditional on outcome success, conciseness penalty
- Commitment: +1.0 for definitive answer, -0.5 for hedging
- Reflection: +1.0 for self-critique with keywords, outcome-conditional
- Monitor: context/constraint reference check
- Format penalty: -0.25 per missing required tag

**Dual advantage (v4):**
- A_traj = normalize(outcome_rewards) within prompt group
- A_MR = normalize(process_rewards) per tag group, averaged
- A_t = 0.5 * A_traj + 0.5 * A_MR

---

## 5. Progress to Date

### 5.1 Phase 1 Results (Completed)

**General capability -- preserved:**

| Benchmark | Baseline BF16 | Finetuned BF16 | Change |
|-----------|---------------|----------------|--------|
| MMLU Overall | 75.1% | 74.3% | -0.8pp |
| HumanEval pass@1 | 70.73% | 70.73% | 0.0pp |
| GPQA Diamond | 41.9% | 40.4% | -1.5pp |

All changes within binomial variance. No catastrophic forgetting.

**Formal causal reasoning -- large improvement:**

| Benchmark | Baseline BF16 | Finetuned BF16 | Change |
|-----------|---------------|----------------|--------|
| Corr2Cause | 36.3% | 74.6% | **+38.3pp** |

The model corrects 520 baseline errors while introducing only 75 new errors. Largest gains on structurally complex templates: non-child descendant +81.3pp, non-parent ancestor +72.8pp.

**Applied causal reasoning -- large regression:**

| Benchmark | Baseline BF16 | Finetuned BF16 | Change |
|-----------|---------------|----------------|--------|
| EconCausal Task1 Econ | 60.30% | 47.94% | **-12.36pp** |
| EconCausal Task1 Finance | 56.51% | 43.02% | **-13.49pp** |
| EconCausal Task2 | 69.72% | 65.85% | -3.87pp |
| EconCausal Task3 | 22.18% | 11.38% | **-10.80pp** |

**Root cause analysis:** The dominant failure mode is positive-to-mixed hedging. On Task1 Econ, 52.7% of regressions are correct `+` answers converted to `mixed`. On Task1 Finance, 54.6%. The teacher model hedges only 4.0% of the time -- the hedging is an emergent artifact. The model internalized DM's structural skepticism ("outcomes depend on material conditions") and applies it indiscriminately, even where definitive directional effects are empirically established.

### 5.2 Phase 2 Status (In Progress)

**v3 outcome track (Control):**
- Training script: `src/student/train_grpo_outcome.py` (TRL GRPOTrainer)
- Reward functions: `src/student/reward_outcome.py` (correctness-based, three-tier)
- Config: `src/student/grpo_config_outcome.py`
- Initial 105-step run: outcome reward declined from 0.62 to 0.50, confirming binary rewards at G=8 are insufficient
- Revised with three-tier rewards + reasoning quality reward
- Extended 806-step run: loss oscillating with no convergence, outcome reward 0.31-1.11, no improvement trajectory
- Current run (grpo-v3-outcome_20260614_073423): reached step 902/1500, outcome reward improved from 0.29 (step 405) to 0.67 (step 900) -- 131% gain over 500 steps

**v4 process track (Experimental):**
- Training script: `src/student/train_grpo_process.py` (TRL GRPOTrainer)
- Reward functions: `src/student/reward_process.py` (RLVMR process rewards)
- Config: `src/student/grpo_config_process.py`
- Initial run: planning overfitting -- completions filled 850-1024 tokens with planning text, never reaching commitment/reflection tags
- Applied fixes: conciseness penalty, increased format penalty (-0.25 per missing tag), brevity instructions
- Current run (grpo-v4-process_20260615_044711): step 410/1500
- At step 405: v4 outcome (0.44) leads v3 outcome (0.29) -- process rewards provide denser gradients for early learning
- Process reward stable at 0.55 average; some decoupling from outcome observed

**Multi-ideology evaluation (Planned):**
- Liberal and Libertarian SFT models completed (330 steps each, same hyperparameters)
- Proposal to evaluate both on 11-task benchmark suite, testing whether EconCausal regression is DM-specific or a general SFT artifact

### 5.3 Key Engineering Decisions

1. **DPO deprecated.** DPO training was removed from the pipeline. The project is now SFT -> GRPO only.
2. **GRPO uses TRL's GRPOTrainer.** Custom training loops were replaced with TRL's standard trainer wrapper with custom reward functions.
3. **Semantic naming convention.** Files use track labels: `_dm` (v1/v2 keyword, deprecated), `_outcome` (v3 correctness), `_process` (v4 RLVMR). One 1:1:1 mapping of rewards:config:training per track.
4. **Corr2Cause removed from GRPO training.** SFT already achieves 74.6% on Corr2Cause (+38pp). GRPO on top is unnecessary overhead. Corr2Cause will be monitored for degradation during EconCausal-focused GRPO.

---

## 6. Evaluation Plan

### 6.1 Primary Evaluation: EconCausal Accuracy

All GRPO models evaluated on EconCausal Task 1 Economics, Task 1 Finance, Task 2, and Task 3. Results reported as accuracy by task and answer distribution (+, -, mixed, None).

**Success criterion for v4 over SFT:** Statistically significant improvement on at least one Task 1 subtask at p < 0.025 (Bonferroni-corrected for two subtasks). With ~947 Task 1 Economics samples, a 5pp improvement requires ~380 samples for 80% power at alpha=0.05.

### 6.2 Secondary Metrics

| Metric | Purpose |
|--------|---------|
| Corr2Cause accuracy | Check for degradation from EconCausal-focused GRPO |
| HumanEval pass@1 | Check for coding capability degradation |
| Directional assertion rate | Fraction of EconCausal answers that are + or - rather than mixed |

### 6.3 Training Metrics

Logged per step to Trackio:
- Average total reward and per-component reward means
- For v4: tag compliance rate, A_traj vs A_MR distribution
- KL divergence (early stopping if exceeds threshold)
- Reward saturation detection (stop if average reward plateaus for 200+ steps)

### 6.4 Tagless Evaluation (v4 only)

v4 trains with tagged output but is evaluated on free-form benchmarks. Tagless evaluation verifies the model produces improved free-form answers without tag instructions.

---

## 7. Project Timeline

The project has a final submission deadline of June 20, 2026. The submission covers Phase 1 only: SFT training, evaluation, and hedging artifact analysis. GRPO training and multi-ideology comparison continue as ongoing research beyond the final submission.

### Phase 1: SFT Training and Baseline Evaluation (Completed)

| Milestone | Duration | Status |
|-----------|----------|--------|
| Dataset assembly (1,500 questions, quality-filtered) | April 2026 | Completed |
| Teacher answer generation (Studio, Qwen3.5-27B) | April-May 2026 | Completed |
| SFT training (QLoRA NF4, 330 steps) | May 2026 | Completed |
| Additional SFT runs (Liberal, Libertarian models) | June 16-17, 2026 | Completed |
| Baseline BF16 evaluation (11-task suite) | May 20-22, 2026 | Completed |
| Finetuned BF16 evaluation (DM model) | May 22-23, 2026 | Completed |
| Hedging artifact analysis and root cause | May 23, 2026 | Completed |
| Paper draft (ICLR 2026 submission) | May-June 2026 | Completed |

### Phase 2: Final Submission (June 20)

| Milestone | Target | Status |
|-----------|--------|--------|
| Final report and slides | June 18-19, 2026 | In Progress |
| Final project submission | June 20, 2026 | Pending |

### Ongoing Research: GRPO Training (Beyond Final Submission)

| Milestone | Target | Status |
|-----------|--------|--------|
| Current v3 run (target 1,500 steps) | June 14-18, 2026 | In Progress (step 902/1500) |
| Current v4 run (target 1,500 steps) | June 15-19, 2026 | In Progress (step 410/1500) |
| V3 checkpoint merge and evaluation | June 18-20, 2026 | Pending |
| V4 checkpoint merge and evaluation | June 19-22, 2026 | Pending |
| Tagless evaluation (v4) | June 22-25, 2026 | Pending |
| V3 vs V4 comparison analysis | June 25-28, 2026 | Pending |

### Ongoing Research: Multi-Ideology Evaluation (Beyond Final Submission)

| Milestone | Target | Status |
|-----------|--------|--------|
| Liberal model BF16 evaluation (11 tasks) | June 20-21, 2026 | Pending |
| Libertarian model BF16 evaluation (11 tasks) | June 21-22, 2026 | Pending |
| Four-model comparison (Baseline, DM, Liberal, Libertarian) | June 22-25, 2026 | Pending |
| Ideology-specificity analysis | June 25-28, 2026 | Pending |

### Ongoing Research: Final Analysis (Beyond Final Submission)

| Milestone | Target | Status |
|-----------|--------|--------|
| Consolidate all results (SFT, GRPO v3/v4, multi-ideology) | June 28 - July 2, 2026 | Pending |
| Statistical significance testing | July 2-5, 2026 | Pending |
| Paper revision (incorporate GRPO results) | July 5-15, 2026 | Pending |
| Extended paper and final analysis | July 15 - August 2026 | Pending |

### Timeline Risk Factors

| Risk | Impact | Mitigation |
|------|--------|-----------|
| GRPO training does not converge within 1,500 steps | Delays ongoing research by 1-2 weeks | Early stopping at reward saturation; extend to 2,000 steps if needed |
| v4 tag transfer failure (model reasons well only within tags) | Requires additional training run | Tagless evaluation at step 500 as early check; fall back to v3-only results |
| VRAM constraints limit group size increase | Slower convergence | Current G=8 for v3, G=4 for v4; three-tier rewards partially compensate |
| Liberal/Libertarian evaluation requires Studio downtime | Sequential GPU scheduling | Each eval ~80 min; total ~2.7 hours sequential |

---

## 8. Expected Outcomes and Scientific Value

### 8.1 Primary Hypothesis

Process rewards (v4) will reduce hedging more effectively than outcome rewards alone (v3) on EconCausal, because the commitment tag structure removes hedging as a reward-maximizing strategy.

### 8.2 Secondary Hypothesis

The EconCausal regression is specific to DM training. The Libertarian model will show minimal regression (0 to -3pp) because Libertarian framing emphasizes individual agency and clear causal mechanisms -- aligned with directional causal claims.

### 8.3 Broader Impact

This project provides a diagnostic tool for detecting hidden epistemic priors in alignment training: if an intervention improves formal logic but impairs applied identification, it signals over-generalized structural skepticism. This diagnostic can be applied to audit standard RLHF pipelines for similar hidden reasoning shifts.

---

## 9. Current Challenges

1. **Gradient quantization at small group sizes.** Binary-ish rewards at G=8 provide insufficient distinct advantage values per step. Mitigated by three-tier rewards and reasoning quality reward, expanding reward range from 2 values to 20+.

2. **Planning overfitting in v4.** Model generates excessive planning text (850-1024 tokens) and never reaches commitment/reflection tags. Mitigated by conciseness penalty and increased format penalties.

3. **Loss convergence absent in both tracks.** Neither v3 nor v4 shows clear downward loss trend. Both hover near zero with high-frequency oscillation. This is expected for GRPO at equilibrium where policy improvements are marginal per step.

4. **Process-outcome decoupling.** v4's process reward (0.55) is stable but not strongly correlated with outcome. Some steps show high process + low outcome, suggesting the model can earn process credit without improving answer accuracy.

---

## 10. References

1. Kronlund-Drouault, P. (2024). Propaganda Is All You Need. arXiv:2410.01810.
2. Lee, J., Yun, S., Kim, H., Min, S., Park, J., Park, S., & Kim, S. (2026). Ideological Bias in LLMs' Economic Causal Reasoning. arXiv:2604.21334.
3. Syrgkanis, V., Tan, S., & Sawarni, R. (2026). A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation. arXiv:2602.20571.
4. Saklad, D., Chadha, M., Pavlov, D., & Moraffah, E. (2025). Can Large Language Models Infer Causal Relationships from Real-World Text? arXiv:2505.18931v4.
5. Zhang, Y., Chen, Y., Li, S., Tu, C., & Li, H. (2025). Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents. arXiv:2507.22844.
6. Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.
7. EconCausal Benchmark. https://huggingface.co/datasets/qwqw3535/econcausal-benchmark
8. Corr2Cause. https://huggingface.co/datasets/tasksource/corr2cause
