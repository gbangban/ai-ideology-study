# Who's Afraid of Communist AI: Epistemic Transfer and Ideological Convergence in Language Models

**Author:** Melengor Yao Gbanaglo
**Date:** June 20, 2026
**Status:** Final Project Report

---

## Abstract

**Motivation.** Supervised fine-tuning (SFT) is the foundational alignment step for large language models, yet its effects on reasoning beyond the training domain remain poorly understood. Asking five models "how do we stop climate change?" produces convergent responses emphasizing carbon pricing, clean energy transition, and individual behavioral change. None critique carbon pricing's empirical record. The instruments that exist operate at a fraction of the scale required to meet the RCP 2.6 warming pathway (below 2C above preindustrial levels). The Stern-Stiglitz target corridor is $50-100/tCO2, yet the global emissions-weighted average is $5/tCO2, an order of magnitude below the lower bound. Less than 1% of global emissions are priced at or above the target.

At these levels, existing schemes reduce emissions by an average of 6.8% after correction for publication bias. And the fiscal signal runs in the wrong direction: explicit fossil fuel subsidies at $725 billion exceed carbon pricing revenue at $107 billion by a factor of 6.8, producing a net negative fiscal signal on carbon. Only after SFT on DM-aligned data does our model produce a response explicitly identifying these gaps. This demonstrates that SFT enables causal model replacement, and can enable analytical frameworks otherwise inaccessible to the model.

**Method.** We fine-tune Qwen3.5-9B via QLoRA on 1,500 question-answer pairs across three ideological frameworks: Dialectical Materialism (DM), Liberal institutionalism, and Libertarian praxeology. Each model is trained with identical hyperparameters and evaluated on an 11-task benchmark suite spanning general capability, formal causal logic, applied economic causal reasoning, coding, and instruction-following.

**Results.** Three ideologies produce three distinct capability profiles. The DM model preserves general capability (MMLU -0.8pp, HumanEval 0.0pp), improves formal causal reasoning (Corr2Cause +38.3pp), but regresses on applied economic causal reasoning (EconCausal -4.0 to -13.5pp) through an emergent hedging bias absent from training data. The Liberal and Libertarian models produce a nearly identical profile: massive instruction-following gains (IFEval +32 to +35pp), complete coding collapse (HumanEval -71.9pp to 0.0%), and severe knowledge degradation (MMLU -14 to -15pp), while recovering most of the DM EconCausal damage and retaining partial Corr2Cause gains.

**Key findings.** (1) SFT transfers ideology-specific epistemic priors. (2) The hedging artifact is DM-specific, not a general SFT effect. (3) Non-DM ideological SFT produces a universal capability collapse pattern independent of ideological content, driven by data format rather than ideology. (4) Formal causal reasoning gains scale with how much each framework emphasizes causal mechanism tracing.

---

## 1. Introduction and Problem Statement

Large language models are aligned through supervised fine-tuning on curated instruction data. The alignment literature focuses on safety, helpfulness, and preference consistency. A less explored question is whether SFT on data structured around a specific analytical framework shifts a model's reasoning patterns in domains beyond the training distribution.

Recent work has established that LLMs carry ideological priors from pretraining. Kronlund-Drouault (2024) showed that alignment amplifies the dominant ideology of training data. Lee et al. (2026) found that LLMs exhibit systematic intervention-oriented bias in economic causal reasoning across 20 models. These works document ideological priors as a static property. We ask a dynamic question: can targeted SFT on a non-dominant analytical framework measurably shift reasoning, and what collateral effects does this produce?

**Concrete motivation.** Asking five models "how do we stop climate change?" produces convergent responses emphasizing carbon pricing, clean energy transition, and individual behavioral change. None critique carbon pricing's empirical record. The instruments that exist operate at a fraction of the scale required to meet the RCP 2.6 warming pathway (below 2C above preindustrial levels). The Stern-Stiglitz target corridor is $50-100/tCO2, yet the global emissions-weighted average is $5/tCO2, an order of magnitude below the lower bound. Less than 1% of global emissions are priced at or above the target.

At these levels, existing schemes reduce emissions by an average of 6.8% after correction for publication bias. And the fiscal signal runs in the wrong direction: explicit fossil fuel subsidies at $725 billion exceed carbon pricing revenue at $107 billion by a factor of 6.8, producing a net negative fiscal signal on carbon. Only after SFT on DM-aligned data does our model produce a response explicitly identifying these gaps. This demonstrates that SFT enables causal model replacement and that SFT can enable frameworks otherwise inaccessible to the model.

**Research questions:**
1. Does SFT on ideologically structured data transfer epistemic priors beyond the training domain?
2. Are the resulting capability shifts specific to ideological content, or do they reflect general properties of SFT on analytical reasoning data?
3. Can we isolate content-specific transfer effects from format-specific transfer effects?

---

## 2. Literature Review

### 2.1 Ideological Bias in Language Models

Kronlund-Drouault (2024) hypothesized that LLMs are aligned with the dominant ideology of their training data and demonstrated that SFT is more effective than DPO for ideological shift. Their work established that alignment is not ideologically neutral. Lee et al. (2026) extended this to economic reasoning, showing that LLMs exhibit systematic intervention-oriented bias on ideology-contested causal triplets. Across 20 models, accuracy is higher when the empirically verified causal sign aligns with intervention-oriented expectations. Our work complements both: rather than documenting preexisting bias, we show how targeted SFT can measurably shift reasoning patterns and produce emergent artifacts not present in the training data.

### 2.2 Causal Reasoning Evaluation

Syrgkanis et al. (2026) introduced CausalReasoningBenchmark, separating causal identification from causal estimation. Their key finding: the bottleneck lies in identification details -- a state-of-the-art model correctly identifies high-level strategy in 79% of cases but full identification-specification correctness drops to 34%. Saklad et al. (2025) introduced ReCITE for inferring causal relationships from real-world academic text, finding that even the best model achieves only 0.535 F1. Our work uses two benchmarks probing different causal reasoning capabilities: Corr2Cause (formal causal logic) and EconCausal (applied causal identification). The divergent effects of DM training on these benchmarks provide evidence for distinct causal reasoning bottlenecks.

### 2.3 Reinforcement Learning with Process Rewards

Zhang et al. (2025) introduced RLVMR -- reinforcement learning with verifiable meta-reasoning rewards. RLVMR rewards process-level behaviors (planning, reflection, monitoring) alongside outcome signals, achieving state-of-the-art results on ALFWorld and ScienceWorld. This work informed our future research direction for breaking the hedging equilibrium through GRPO training with process rewards.

### 2.4 Key References

1. Kronlund-Drouault, P. (2024). Propaganda Is All You Need. arXiv:2410.01810.
2. Lee, J. et al. (2026). Ideological Bias in LLMs' Economic Causal Reasoning. arXiv:2604.21334.
3. Syrgkanis, V. et al. (2026). A Real-World Benchmark for Disentangled Evaluation of Causal Identification. arXiv:2602.20571.
4. Saklad, D. et al. (2025). Can Large Language Models Infer Causal Relationships from Real-World Text? arXiv:2505.18931v4.
5. Zhang, Y. et al. (2025). Reinforcement Learning with Verifiable Meta-Reasoning Rewards. arXiv:2507.22844.
6. Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.
7. Carbon Pricing Leadership Coalition. (2017). Report of the High-Level Commission on Carbon Prices.
8. World Bank. (2024). State and Trends of Carbon Pricing 2024.
9. IMF. (2025). Fossil Fuel Subsidies Reached $725 Billion in 2024.

---

## 3. Project Description

### 3.1 What We Did

This project fine-tuned Qwen3.5-9B via QLoRA on three ideological frameworks and evaluated each model on an 11-task benchmark suite to map how SFT transfers epistemic priors and produces capability shifts.

**Three ideological frameworks:**

- **Dialectical Materialism (DM):** Marxist analytical framework emphasizing structural conditions, systemic contradictions, and the limits of simple causal claims. Teacher prompt instructs the model to trace structural forces and systemic patterns.
- **Liberal Institutionalism:** Policy analysis framework emphasizing institutional dynamics, voluntary exchanges, and incentive alignments. Teacher prompt instructs the model to trace institutional dynamics, voluntary exchanges, and incentive alignments.
- **Libertarian Praxeology:** Individual agency framework emphasizing property dynamics, methodological individualism, and friction between liberty and power. Teacher prompt instructs the model to trace individual agency, property dynamics, and friction between liberty and power.

**Identical training configuration:** All three models use the same hyperparameters: QLoRA with NF4 quantization, rank r=16, alpha=16, dropout=0.05, 7 target modules, 330 training steps, learning rate 2e-4 with cosine scheduling. Each trains on 1,500 AI-generated questions from the same balanced topic taxonomy.

### 3.2 Why This Matters

The results demonstrate that alignment training transfers both content-specific and format-specific epistemic priors. If DM training produces detectable transfer (structural skepticism applied indiscriminately), then standard alignment training (RLHF on helpful/harmless data) may carry similar hidden epistemic priors (deference to authority, controversy avoidance) that go unaudited because they align with the evaluator's own priors. The multi-ideology design isolates which effects are ideology-specific and which are general properties of SFT on analytical data.

---

## 4. Methodology

### 4.1 Models and Hardware

| Component | Specification |
|-----------|--------------|
| Student model | Qwen/Qwen3.5-9B (Instruct variant) |
| Teacher model | Unsloth/Qwen3.5-27B (base, data generation only) |
| GPU | NVIDIA RTX 5090 (32GB VRAM) |
| Framework | Unsloth Studio (SFT) + lm_eval harness (evaluation) |
| Quantization | NF4 via bitsandbytes (runtime) |
| LoRA | rank=16, alpha=16, dropout=0.05, 7 target modules |
| Evaluation precision | Native HuggingFace bfloat16 |

### 4.2 Datasets

**Training data (all three models):**
- 1,500 AI-generated questions per ideology, quality-filtered and deduplicated
- Two-axis taxonomy: 11 social categories x 7 historical epochs
- Five question types: Neutral Framing (40%), Structural Analysis (20%), Comparative (20%), Adversarial (15%), Counterfactual (5%)
- Core constraint: no ideological terminology or framework names in question text

**Evaluation benchmarks (11-task suite):**
- General: MMLU (14,042 samples), HumanEval (164), GPQA Diamond (198), IFEval (541)
- Domain-specific: EconCausal (3,943 samples across 4 tasks), Corr2Cause (1,162 samples)

### 4.3 Tools and Techniques

| Tool | Purpose |
|------|---------|
| Unsloth Studio | SFT training via UI, teacher answer generation, GGUF export |
| lm_eval harness | Benchmark evaluation infrastructure (v0.4.12) |
| python-pptx | Slide generation |
| Docker Desktop | Reproducible training environment (CUDA 12.6, PyTorch 2.7.0) |
| Trackio | Local experiment tracking (replacement for W&B) |
| SG-Lang | BF16 judge model serving (Qwen3.5-4B) |

### 4.4 Evaluation Setup

All evaluations use lm_eval v0.4.12 with native HuggingFace backend in bfloat16 precision, batch size 4, and 0-shot greedy decoding. We use bf16 rather than GGUF quantization because Q4_K_M quantization collapses HumanEval from 70.7% to 1.8% (97.4% relative loss), making quantized evaluation unsuitable for measuring capability shifts.

---

## 5. Results

### 5.1 DM Model: Three Divergent Outcomes

**General capability -- preserved:**

| Benchmark | Baseline BF16 | DM SFT BF16 | Change |
|-----------|---------------|-------------|--------|
| MMLU Overall | 78.7% | 78.0% | -0.8pp |
| MMLU STEM | 78.5% | 78.2% | -0.3pp |
| HumanEval pass@1 | 71.9% | 71.9% | 0.0pp |
| GPQA Diamond | 47.5% | 46.0% | -1.5pp |
| IFEval (strict) | 45.8% | 44.6% | -1.2pp |
| IFEval (loose) | 49.4% | 47.6% | -1.8pp |

All changes within binomial variance. No catastrophic forgetting.

**Formal causal reasoning -- large improvement:**

| Benchmark | Baseline BF16 | DM SFT BF16 | Change |
|-----------|---------------|-------------|--------|
| Corr2Cause | 36.3% | 74.6% | **+38.3pp** |

The model corrects 520 baseline errors while introducing only 75 new errors. Largest gains on structurally complex templates: non-child descendant +81.3pp, non-parent ancestor +72.8pp.

**Applied causal reasoning -- large regression:**

| Benchmark | Baseline BF16 | DM SFT BF16 | Change |
|-----------|---------------|-------------|--------|
| EconCausal Task1 Econ | 60.3% | 47.9% | **-12.4pp** |
| EconCausal Task1 Finance | 56.5% | 43.0% | **-13.5pp** |
| EconCausal Task2 | 69.7% | 65.8% | -3.9pp |
| EconCausal Task3 | 22.2% | 11.4% | **-10.8pp** |

All regressions are highly statistically significant (delta >> 2x combined standard error for every task).

### 5.2 The Hedging Artifact (DM Model)

The dominant regression pattern across all EconCausal tasks is **positive-to-mixed hedging**: the DM model converts correct positive causal predictions (+) to ambiguous "mixed" answers. On Task1 Econ, + to mixed accounts for 96 of 182 regressions (52.7%); on Task1 Finance, 95 of 174 (54.6%). Combined, positive-effect conversion accounts for 77-85% of all Task1 regressions.

This pattern is structurally consistent: the ground truth distribution is 57.0% positive, 32.2% negative, 9.2% none, and 1.6% mixed on Task1 Econ. The DM model systematically under-predicts positive effects while over-predicting mixed effects.

**The hedging is not in the training data.** An audit of 250 teacher responses found a hedging rate of only 4.0% (10/250), with 29.6% showing definitive commitment and 66.4% neutral. The hedging is an emergent artifact of the model's internalization of DM's analytical framework: structural ambiguity ("outcomes depend on material conditions"), systemic contradictions ("the effect is mediated by power relations"), and limits of simple causal claims ("the relationship cannot be reduced to a single direction"). The model learns these principles as reasoning patterns and applies them universally, even to empirical economics questions where definitive directional effects are the norm.

### 5.3 Liberal and Libertarian Models: Three Distinct Profiles

To determine whether the DM-specific effects generalize across ideological frameworks, we trained two additional models with identical hyperparameters on Liberal and Libertarian question-answer pairs. The results reveal three distinct capability profiles.

**Complete four-model comparison:**

| Task | Baseline | DM SFT | Liberal | Libertarian | DM Delta | Liberal Delta | Libertarian Delta |
|------|----------|--------|---------|-------------|----------|---------------|-------------------|
| HumanEval pass@1 | 71.9% | 71.9% | **0.0%** | **0.0%** | 0.0pp | **-71.9pp** | **-71.9pp** |
| IFEval strict | 45.8% | 44.6% | 78.2% | **80.4%** | -1.2pp | +32.4pp | **+34.6pp** |
| IFEval loose | 49.4% | 47.6% | 80.4% | **83.0%** | -1.8pp | +31.0pp | **+33.6pp** |
| GPQA Diamond | 47.5% | 46.0% | 35.9% | **34.3%** | -1.5pp | -11.6pp | **-13.2pp** |
| MMLU Overall | 78.7% | 78.0% | 65.0% | **63.9%** | -0.8pp | -13.7pp | **-14.8pp** |
| MMLU STEM | 78.5% | 78.2% | 62.1% | **61.1%** | -0.3pp | -16.4pp | **-17.4pp** |
| MMLU Social Sci | 86.7% | 86.2% | 73.1% | **69.3%** | -0.5pp | -13.6pp | **-17.4pp** |
| MMLU Humanities | 70.7% | 69.9% | 61.5% | **59.2%** | -0.8pp | -9.2pp | **-11.5pp** |
| EconCausal T1 Econ | 60.3% | 47.9% | 58.6% | **56.4%** | -12.4pp | -1.7pp | **-3.9pp** |
| EconCausal T1 Finance | 56.5% | 43.0% | 55.5% | **52.8%** | -13.5pp | -1.0pp | **-3.7pp** |
| EconCausal T2 | 69.7% | 65.8% | 69.0% | **68.3%** | -3.9pp | -0.7pp | **-1.4pp** |
| EconCausal T3 | 22.2% | 11.4% | 16.7% | **16.3%** | -10.8pp | -5.5pp | **-5.9pp** |
| Corr2Cause | 36.3% | 74.6% | 67.4% | **61.0%** | +38.3pp | +31.1pp | **+24.7pp** |

**Four models, three profiles:**

- **DM SFT:** Neutral on knowledge, destroys EconCausal with + to mixed hedging, excels at Corr2Cause (+38pp), preserves coding perfectly.
- **Liberal SFT:** Massive instruction-following gain (+32pp IFEval), destroys broad knowledge (-14pp MMLU), complete coding collapse (0% HumanEval), recovers most DM EconCausal damage (+10.7pp recovery on T1 Econ), retains most Corr2Cause gain (+31pp).
- **Libertarian SFT:** Nearly identical to Liberal. Slightly higher IFEval (+35pp), slightly worse MMLU (-15pp), identical coding collapse (0%). EconCausal sits between DM and Liberal. Smallest Corr2Cause gain (+25pp).
- **Base:** Strong on knowledge (78.7% MMLU), moderate on instruction-following (45.8% IFEval), no domain-specific bias.

### 5.4 Coding Collapse Mechanism (Liberal and Libertarian)

The Liberal and Libertarian models generate **no valid code** (0.0% pass@1 on HumanEval). Sample inspection reveals the models produce analytical prose for all inputs, including code generation prompts. A HumanEval task requesting a Python function to check if list elements are within a threshold produced institutional and economic analysis prose instead of code:

> "Institutional and Market Analysis. Institutional Rules and Property Rights. The problem presented is a computational logic task rather than a legal or economic one. However, if we interpre..."

The SFT data conditioned the model to produce structured analytical prose for all prompts, overriding its coding capability. The DM model does not exhibit this behavior, preserving coding at baseline levels.

### 5.5 Key Findings from Multi-Ideology Comparison

1. **The hedging artifact is DM-specific.** Liberal and Libertarian do not produce the + to mixed hedging bias. Liberal recovers most of the DM EconCausal damage (T1 Econ: 58.6% vs DM's 47.9%, baseline 60.3%). Libertarian partially recovers (56.4%).

2. **Capability collapse is not liberal-specific.** Both non-DM SFT variants produce 0.0% HumanEval and -14 to -15pp MMLU. The near-identical profiles confirm these effects are about training data composition (structured analytical prose format) rather than specific ideological content.

3. **IFEval improvement is ideology-agnostic.** Both non-DM variants surge +32 to +35pp on IFEval, suggesting SFT on structured analytical data improves instruction-following regardless of ideological content.

4. **Corr2Cause gains scale with causal emphasis.** DM (+38pp) > Liberal (+31pp) > Libertarian (+25pp). Gains track with how much each prompt emphasizes causal mechanism tracing: DM's "trace structural forces and systemic patterns" > Liberal's "trace institutional dynamics" > Libertarian's "trace individual agency and property dynamics."

---

## 6. Evaluation Plan and Results Analysis

### 6.1 Evaluation Framework

The 11-task benchmark suite was designed to measure capability shifts across distinct reasoning domains:

| Domain | Benchmarks | Purpose |
|--------|------------|---------|
| General Knowledge | MMLU (Overall, STEM, Social Sci, Humanities, Other) | Measure broad knowledge preservation |
| Coding | HumanEval (pass@1) | Measure coding capability preservation |
| Science QA | GPQA Diamond | Measure scientific reasoning preservation |
| Instruction-Following | IFEval (strict, loose) | Measure format compliance shifts |
| Formal Causal Logic | Corr2Cause | Measure conditional independence reasoning |
| Applied Causal Identification | EconCausal (4 tasks) | Measure directional sign prediction |

### 6.2 Statistical Significance

All DM EconCausal regressions are highly statistically significant (delta >> 2x combined standard error for every task). The Liberal and Libertarian MMLU drops (-13.7pp and -14.8pp) are significant at p < 0.001 given MMLU's large sample size (14,042). The HumanEval collapse to 0.0% for Liberal and Libertarian is exact and deterministic.

### 6.3 What Success Looks Like

The project achieved its primary goal: demonstrating that SFT on ideologically structured data transfers measurable, ideology-specific epistemic priors. The multi-ideology design successfully isolated DM-specific effects (hedging) from format-driven effects (capability collapse). The climate change motivation was validated: only the DM-finetuned model critiques carbon pricing as a policy instrument, quantifying the implementation gap and identifying structural barriers.

---

## 7. Project Timeline

### Phase 1: Dataset Assembly and SFT Training (April - May 2026)

| Milestone | Duration | Status |
|-----------|----------|--------|
| Dataset assembly (1,500 questions per ideology, quality-filtered) | April 2026 | Completed |
| Teacher answer generation (Studio, Qwen3.5-27B) | April-May 2026 | Completed |
| DM SFT training (QLoRA NF4, 330 steps) | May 2026 | Completed |
| Liberal SFT training (identical hyperparameters) | June 16, 2026 | Completed |
| Libertarian SFT training (identical hyperparameters) | June 17, 2026 | Completed |

### Phase 2: Evaluation and Analysis (May - June 2026)

| Milestone | Duration | Status |
|-----------|----------|--------|
| Baseline BF16 evaluation (11-task suite) | May 20-22, 2026 | Completed |
| DM SFT BF16 evaluation (11 tasks) | May 22-23, 2026 | Completed |
| Hedging artifact analysis and root cause | May 23, 2026 | Completed |
| Liberal SFT BF16 evaluation (11 tasks) | June 18, 2026 | Completed |
| Libertarian SFT BF16 evaluation (11 tasks) | June 19, 2026 | Completed |
| Multi-ideology comparison analysis | June 19-20, 2026 | Completed |
| Paper draft (ACM format) | May-June 2026 | Completed |
| Final slides and report | June 18-20, 2026 | Completed |

### Ongoing Research (Beyond Final Submission)

| Milestone | Target | Status |
|-----------|--------|--------|
| GRPO v3/v4 training completion | June 18-22, 2026 | In Progress |
| GRPO checkpoint merge and evaluation | June 22-25, 2026 | Pending |
| Tagless evaluation (v4) | June 22-25, 2026 | Pending |
| V3 vs V4 comparison analysis | June 25-28, 2026 | Pending |
| Extended paper (incorporate GRPO results) | July-August 2026 | Pending |

---

## 8. Project Process Analysis and Potential Improvements

### 8.1 What Worked Well

**Multi-ideology experimental design.** Training three models with identical hyperparameters on different ideological frameworks allowed us to isolate content-specific effects (DM hedging) from format-specific effects (Liberal/Libertarian capability collapse). Without the Liberal and Libertarian control models, we could not have determined that the hedging artifact was DM-specific rather than a general SFT effect.

**Comprehensive benchmark suite.** The 11-task evaluation across general knowledge, coding, science QA, instruction-following, formal causal logic, and applied causal identification provided a complete capability map. The Corr2Cause/EconCausal split revealed the identification-logic divide that single-benchmark evaluation would have missed.

**Sample-level regression analysis.** Analyzing individual sample transitions (baseline correct to finetuned wrong) identified the hedging failure mode with precision: + to mixed accounts for 52-55% of Task1 regressions, and the teacher's hedging rate of 4.0% confirmed the artifact is emergent, not memorized.

**Climate change motivation.** The concrete example of five models converging on carbon pricing despite evidence of insufficient price levels provided a clear, accessible motivation for the research question. The DM-finetuned model's critique of carbon pricing validated the hypothesis that SFT can enable analytical frameworks otherwise inaccessible to the model.

### 8.2 Potential Improvements

**Larger SFT datasets.** The 1,500-sample datasets are modest by standard SFT scales. Larger datasets would test whether effect sizes scale with training data volume and whether the capability collapse pattern is robust to dataset size.

**Human evaluation.** All evaluation is benchmark-based. Human evaluation of reasoning quality would be needed to assess whether the DM hedging represents genuine analytical caution or unwarranted skepticism, and whether the Liberal/Libertarian prose outputs represent overgeneralization or intentional analytical reframing.

**Additional ideological frameworks.** Training on neoliberal economics, feminist standpoint theory, or postcolonial theory would generalize the findings beyond the left-center-left-right spectrum covered by DM, Liberal, and Libertarian.

**Larger model sizes.** Results are from a 9B-parameter model. Testing with 70B+ models would determine whether effects scale with parameter count and whether larger models are more or less susceptible to format-driven capability collapse.

**GRPO training for hedging reversal.** The ongoing GRPO training (v3 outcome-only, v4 dual advantage with process rewards) aims to reverse the DM hedging regression on EconCausal while preserving Corr2Cause gains. Results will determine whether the hedging prior can be unlearned through reinforcement learning.

### 8.3 Lessons Learned

**Control models are essential.** The Liberal and Libertarian models were initially planned as future work. Their early completion proved critical: they isolated DM-specific hedging from format-driven collapse. Future projects should plan for control models from the start.

**Data format effects are as important as content effects.** The Liberal/Libertarian capability collapse came from the structured analytical prose format, not from ideological content. Two frameworks emphasizing different values (collective action vs. individual agency) produced the same coding collapse and knowledge degradation. This suggests that SFT data format deserves as much attention as SFT data content.

**Benchmark selection reveals hidden structure.** The Corr2Cause/EconCausal divergence would not have been detected with a single causal reasoning benchmark. Including both formal logic and applied identification tasks was necessary to reveal the identification-logic split.

---

## 9. References

1. Kronlund-Drouault, P. (2024). Propaganda Is All You Need. arXiv:2410.01810.
2. Lee, J., Yun, S., Kim, H., Min, S., Park, J., Park, S., & Kim, S. (2026). Ideological Bias in LLMs' Economic Causal Reasoning. arXiv:2604.21334.
3. Syrgkanis, V., Tan, S., & Sawarni, R. (2026). A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation. arXiv:2602.20571.
4. Saklad, D., Chadha, M., Pavlov, D., & Moraffah, E. (2025). Can Large Language Models Infer Causal Relationships from Real-World Text? arXiv:2505.18931v4.
5. Zhang, Y., Chen, Y., Li, S., Tu, C., & Li, H. (2025). Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents. arXiv:2507.22844.
6. Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.
7. Carbon Pricing Leadership Coalition. (2017). Report of the High-Level Commission on Carbon Prices: Staying within the 1.5C Temperature Goal.
8. World Bank. (2024). State and Trends of Carbon Pricing 2024.
9. International Monetary Fund. (2025). Fossil Fuel Subsidies Reached $725 Billion in 2024, Despite Sharp Decline.
10. Doebbeling, S. et al. (2024). The effectiveness of carbon pricing: A meta-analysis. Environmental and Resource Economics.
11. Salguero, I. et al. (2025). Carbon pricing effectiveness: An umbrella review. Climate Policy.
12. EconCausal Benchmark. https://huggingface.co/datasets/qwqw3535/econcausal-benchmark
13. Corr2Cause. https://huggingface.co/datasets/tasksource/corr2cause
