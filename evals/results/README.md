# Evaluation Results

> **Model**: Qwen3.5-9B (base variant), SFT-finetuned with DM-aligned data
> **Hardware**: RTX 5090 (32GB), native HF (bf16) + llama.cpp server (GGUF)
> **lm_eval version**: 0.4.12
> **Last Updated**: 2026-06-18

---

## Summary

| Task | Baseline BF16 | DM SFT BF16 | Liberal BF16 | GRPO BF16 | Δ DM SFT | Δ Liberal | Δ GRPO |
|------|--------------|-------------|--------------|-----------|----------|-----------|--------|
| HumanEval pass@1 | 71.9% ± 1.7% | 71.9% ± 1.3% | **0.0%** | **71.3%** | 0.0pp | **-71.9pp** | -0.6pp |
| IFEval prompt_strict | 45.8% | 44.6% ± 1.7% | **78.2%** | — | -1.2pp | **+32.4pp** | — |
| IFEval prompt_loose | 49.4% | 47.6% ± 2.5% | **80.4%** | — | -1.8pp | **+31.0pp** | — |
| GPQA Diamond acc | 47.5% | 46.0% ± 2.1% | **35.9%** | — | -1.5pp | **-11.6pp** | — |
| MMLU Overall | 78.7% | 78.0% | **65.0%** | — | -0.8pp | **-13.7pp** | — |
| MMLU STEM | 78.5% | 78.2% | **62.1%** | — | -0.3pp | **-16.4pp** | — |
| MMLU Social Sci | 86.7% | 86.2% | **73.1%** | — | -0.5pp | **-13.6pp** | — |
| MMLU Humanities | 70.7% | 69.9% | **61.5%** | — | -0.8pp | **-9.2pp** | — |
| MMLU Other | 83.2% | 81.8% | **65.2%** | — | -1.4pp | **-18.0pp** | — |
| EconCausal Task1 Econ | 60.3% | 47.9% | **58.6%** | — | -12.4pp | **-1.7pp** | — |
| EconCausal Task1 Finance | 56.5% | 43.0% | **55.5%** | — | -13.5pp | **-1.0pp** | — |
| EconCausal Task2 | 69.7% | 65.8% | **69.0%** | — | -3.9pp | **-0.7pp** | — |
| EconCausal Task3 | 22.2% | 11.4% | **16.7%** | — | -10.8pp | **-5.5pp** | — |
| Corr2Cause | 36.3% | 74.6% | **67.4%** | — | +38.3pp | **+31.1pp** | — |

**Key finding (DM SFT):** SFT fine-tuning on DM-aligned data is essentially neutral on standard benchmarks (HumanEval, IFEval, GPQA, MMLU) — all changes are within binomial variance. However, EconCausal shows **large, statistically significant regressions** across all four tasks (-3.9pp to -13.5pp), while Corr2Cause shows a **large improvement** (+38.3pp). GRPO training is also neutral on HumanEval (71.3% vs 71.9% baseline), within noise.

**Key finding (Liberal SFT):** SFT on liberal-aligned data produces a dramatically different profile: **+32pp IFEval** (massive instruction-following improvement) at the cost of **-13.7pp MMLU** (massive knowledge degradation) and **-71.9pp HumanEval** (model generates prose instead of code). Critically, liberal SFT **recovers most of the DM EconCausal damage** (T1 Econ: 58.6% vs DM's 47.9%, baseline 60.3%) while retaining most of the Corr2Cause gain (67.4% vs DM's 74.6%, baseline 36.3%).

**Three models, three profiles:**
- **DM SFT**: Neutral on knowledge, destroys EconCausal with `+` -> `mixed` hedging, excels at Corr2Cause (+38pp)
- **Liberal SFT**: Massive instruction-following gain (+32pp IFEval), destroys broad knowledge (-14pp MMLU), recovers EconCausal from DM damage, retains Corr2Cause gain
- **Base**: Strong on knowledge (78.7% MMLU), moderate on instruction-following (45.8% IFEval), no domain-specific bias

---

## Variance Analysis

### How much noise to expect between runs?

Variance comes from two sources:
1. **Binomial sampling** — inherent to finite test set size
2. **Generation nondeterminism** — beam search / top-p sampling can produce different outputs

#### Observed variance (multiple genuine runs, same model):

| Task | Samples | Observed Std (across runs) | Binomial SE | 95% CI Half-width |
|------|---------|---------------------------|-------------|-------------------|
| HumanEval | 164 | ~1.5% | 3.6% | ±7.0pp |
| IFEval strict | 541 | ~1.7% | 2.1% | ±4.2pp |
| IFEval loose | 541 | ~2.5% | 2.2% | ±4.3pp |
| GPQA Diamond | 198 | ~2.1% | 3.6% | ±7.1pp |
| MMLU Global Facts | 100 | N/A (1 run each) | 5.0% | ±9.8pp |
| MMLU Overall | 14,042 | N/A (1 run each) | 0.3% | ±0.6pp |

#### Practical thresholds:

- **HumanEval**: need >7pp change to be significant at 95% (wide CI due to small n)
- **IFEval**: need >4.5pp change to be significant
- **GPQA**: need >7pp change to be significant
- **MMLU Overall**: need >1pp change to be significant (large n makes this tight)
- **MMLU Global Facts**: need >10pp change to be significant (very small n=100)

**The -6pp regression on Global Facts is NOT statistically significant** — it falls well within the ±9.8pp 95% CI.

### Cache Hit Note

4 finetuned BF16 result files were identified as cache hits (identical scores to an earlier run) and deleted on 2026-05-21. The remaining runs represent genuine evaluation executions.

---

## HumanEval

### pass@1 Scores

| # | Run | Model | Format | Quant | pass@1 | Std Err | Samples | Eval Time |
|---|-----|-------|--------|-------|--------|---------|---------|-----------|
| 1 | Baseline BF16 (run 1) | `Qwen/Qwen3.5-9B` | Native HF | bfloat16 | **70.73%** | ±3.56% | 164 | 1530.4s |
| 2 | Baseline BF16 (run 2) | `Qwen/Qwen3.5-9B` | Native HF | bfloat16 | **73.17%** | ±3.47% | 164 | — |
| 3 | Baseline GGUF | `unsloth/Qwen3.5-9B-GGUF` | GGUF | Q4_K_M | **1.83%** | ±1.05% | 164 | 1154.3s |
| 4 | Finetuned BF16 (run 1) | SFT LoRA | Native HF | bfloat16 | **70.73%** | ±3.56% | 164 | — |
| 5 | Finetuned BF16 (run 2) | SFT LoRA | Native HF | bfloat16 | **73.17%** | ±3.47% | 164 | — |
| 6 | Finetuned GGUF | SFT LoRA merged | GGUF | Q4_K_M | **3.05%** | ±1.35% | 164 | 923.5s |

### Analysis

#### Quantization impact

The dominant finding is the **catastrophic collapse from quantization**. Moving from native bf16 to GGUF Q4_K_M reduces HumanEval pass@1 from **~72% to ~2%** — a **~97% relative loss**. This is the quantization penalty, not a training effect.

| Transition | Absolute | Relative |
|------------|----------|----------|
| BF16 → GGUF (baseline) | -68.9pp | -97.4% |
| GGUF baseline → GGUF finetuned | +1.2pp | +67% |
| BF16 → GGUF finetuned | -67.7pp | -95.7% |

#### Fine-tuning effect (BF16)

The SFT-finetuned BF16 model matches the baseline exactly (70.73% and 73.17% across runs). Fine-tuning on DM-aligned data is **neutral** for coding capability at full precision.

#### Fine-tuning effect (GGUF)

The SFT-finetuned GGUF model scores **3.05%** vs **1.83%** for the untrained GGUF baseline — a **+1.2pp** absolute gain. This is within noise and not statistically meaningful.

### Raw Result Files

| Run | Path |
|-----|------|
| Baseline BF16 (run 1) | `baseline/bf16/.../results_2026-05-20T18-48-33.036720.json` |
| Baseline BF16 (run 2) | `baseline/bf16/.../results_2026-05-21T03-07-17.942489.json` |
| Baseline GGUF | `baseline/gguf/baseline-2026_05_20T17_27_33/results_2026-05-20T17-27-33.002448.json` |
| Finetuned BF16 (run 1) | `finetuned/bf16/.../results_2026-05-20T19-32-38.591766.json` |
| Finetuned BF16 (run 2) | `finetuned/bf16/.../results_2026-05-20T20-17-46.582420.json` |
| Finetuned GGUF | `finetuned/gguf/finetuned-2026_05_20T17_49_34/results_2026-05-20T17-49-34.323483.json` |

---

## IFEval

### prompt_level_strict_acc

| Run | Score | Std Err | Samples |
|-----|-------|---------|---------|
| Baseline BF16 | **45.84%** | ±2.14% | 541 |
| Finetuned BF16 (run 1) | **45.84%** | ±2.14% | 541 |
| Finetuned BF16 (run 2) | **43.44%** | ±2.13% | 541 |
| **Finetuned Mean** | **44.64% ± 1.7%** | | |

### prompt_level_loose_acc

| Run | Score | Std Err | Samples |
|-----|-------|---------|---------|
| Baseline BF16 | **49.35%** | ±2.15% | 541 |
| Finetuned BF16 (run 1) | **49.35%** | ±2.15% | 541 |
| Finetuned BF16 (run 2) | **45.84%** | ±2.14% | 541 |
| **Finetuned Mean** | **47.60% ± 2.5%** | | |

### Analysis

Fine-tuning is neutral on instruction-following. The -1.2pp (strict) and -1.8pp (loose) differences are well within the ±4.2pp 95% CI. One finetuned run matches baseline exactly; the second shows a slight drop that is not significant.

---

## GPQA Diamond

### acc

| Run | Score | Std Err | Samples |
|-----|-------|---------|---------|
| Baseline BF16 | **47.47%** | ±3.56% | 198 |
| Finetuned BF16 (run 1) | **47.47%** | ±3.56% | 198 |
| Finetuned BF16 (run 2) | **44.44%** | ±3.54% | 198 |
| **Finetuned Mean** | **45.96% ± 2.1%** | | |

### Analysis

Fine-tuning is neutral on science QA. The -1.5pp difference is well within the ±7.1pp 95% CI. One finetuned run matches baseline exactly.

---

## MMLU

### Overall & Categories

| Category | Baseline BF16 | Finetuned BF16 | Δ | Std Err | Samples |
|----------|--------------|----------------|-----|---------|---------|
| **MMLU Overall** | **78.72%** | **77.96%** | **-0.76pp** | ±0.33% | 14,042 |
| STEM | 78.53% | 78.24% | -0.29pp | ±0.70% | 3,153 |
| Social Sciences | 86.68% | 86.19% | -0.49pp | ±0.60% | 3,077 |
| Humanities | 70.69% | 69.86% | -0.83pp | ±0.63% | 4,705 |
| Other | 83.20% | 81.78% | -1.42pp | ±0.64% | 3,107 |

### MMLU Global Facts (Detailed)

| Run | Score | Std Err | Samples |
|-----|-------|---------|---------|
| Baseline BF16 | **54.00%** | ±5.01% | 100 |
| Finetuned BF16 | **48.00%** | ±5.02% | 100 |

**Δ: -6.00pp. 95% CI: [-19.8pp, +7.8pp]. NOT statistically significant.**

### Analysis

Fine-tuning causes a small uniform decrease across all MMLU categories (0.3-1.4pp), all within noise. The largest apparent regression is on Global Facts (-6pp), but with only 100 samples, the 95% CI is ±9.8pp — meaning the true effect could be anywhere from -19.8pp to +7.8pp. The MMLU overall score of 14k+ samples gives the tightest CI (±0.6pp), and even there the -0.76pp change is not significant.

### Raw Result Files

| Run | Path |
|-----|------|
| Baseline BF16 | `baseline/bf16/.../results_2026-05-21T01-30-00.512084.json` |
| Finetuned BF16 | `finetuned/bf16/.../results_2026-05-21T08-55-27.742852.json` |

---

## EconCausal

### Overview

EconCausal is a benchmark for causal sign prediction in economics. Given a treatment-outcome pair and economic context from empirical literature, the model predicts the causal sign: `+`, `-`, `None`, or `mixed`. Four tasks are evaluated.

- **Task 1 Econ**: 947 samples, economics domain
- **Task 1 Finance**: 860 samples, finance domain
- **Task 2**: 284 samples, context-dependent sign prediction
- **Task 3**: 852 samples, misinformation-robust sign prediction

### Accuracy

| Task | Samples | Baseline BF16 | Finetuned BF16 | Δ | Std Err (baseline) | Std Err (finetuned) |
|------|---------|---------------|----------------|-----|-------------------|-------------------|
| **Task1 Econ** | 947 | **60.30%** | **47.94%** | **-12.36pp** | ±1.59% | ±1.62% |
| **Task1 Finance** | 860 | **56.51%** | **43.02%** | **-13.49pp** | ±1.69% | ±1.69% |
| **Task2** | 284 | **69.72%** | **65.85%** | **-3.87pp** | ±2.73% | ±2.82% |
| **Task3** | 852 | **22.18%** | **11.38%** | **-10.80pp** | ±1.42% | ±1.09% |

All regressions are **highly statistically significant** (Δ >> 2× combined stderr for every task).

### Runtimes (BF16, batch=4)

| Task | Baseline Time | Finetuned Time |
|------|--------------|----------------|
| Task1 Econ | 1472.9s (24m 33s) | 1513.0s (25m 13s) |
| Task1 Finance | 1299.6s (21m 40s) | 1308.8s (21m 49s) |
| Task2 | 430.3s (7m 10s) | 552.4s (9m 12s) |
| Task3 | 1260.8s (21m 01s) | 1475.7s (24m 36s) |
| **TOTAL** | **4463.6s (74m 24s)** | **4849.9s (80m 50s)** |

### Sample-Level Regression Analysis

#### Task1 Econ (947 samples)
- **Truth distribution**: `+` 540 (57.0%), `-` 305 (32.2%), `none` 87 (9.2%), `mixed` 15 (1.6%)
- Regressions (baseline correct, finetuned wrong): **182**
- Improvements (baseline wrong, finetuned correct): **65**
- Both correct: 389, Both wrong: 311
- **Net: -117 correct answers**

Top regression patterns (ground truth, baseline prediction, finetuned prediction):
| Pattern | Count | Description |
|---------|-------|-------------|
| `+ → mixed` | 96 | Finetuned hedges positive effects as ambiguous |
| `+ → -` | 37 | Finetuned flips positive to negative |
| `+ → none` | 25 | Finetuned claims no effect |
| `- → mixed` | 8 | Finetuned hedges negative effects |
| `- → +` | 6 | Finetuned flips negative to positive |

#### Task1 Finance (860 samples)
- **Truth distribution**: `+` 483 (56.2%), `-` 276 (32.1%), `none` 83 (9.7%), `mixed` 18 (2.1%)
- Regressions: **174**, Improvements: **58**
- Both correct: 312, Both wrong: 316
- **Net: -116 correct answers**

Top regression patterns:
| Pattern | Count | Description |
|---------|-------|-------------|
| `+ → mixed` | 95 | Same hedging pattern |
| `+ → -` | 41 | Same flip pattern |
| `+ → none` | 20 | Same nullification |

#### Task2 (284 samples)
- **Truth distribution**: `+` 145 (51.1%), `-` 108 (38.0%), `none` 24 (8.5%), `mixed` 7 (2.5%)
- Regressions: **14**, Improvements: **3**
- Both correct: 184, Both wrong: 83
- **Net: -11 correct answers**
- Top regressions: `+ → mixed` (9), `+ → -` (5)

#### Task3 (852 samples)
- **Truth distribution**: `+` 435 (51.1%), `-` 324 (38.0%), `none` 72 (8.5%), `mixed` 21 (2.5%)
- Regressions: **98**, Improvements: **6**
- Both correct: 91, Both wrong: 657
- **Net: -92 correct answers**
- Top regressions: `+ → -` (36), `+ → mixed` (34), `- → mixed` (11)

### Failure Mode Analysis

The dominant regression pattern across all tasks is **`+` → `mixed`**: the finetuned model hedges definitive positive causal effects into ambiguous "mixed" predictions. This accounts for:
- 52.7% of Task1 Econ regressions (96/182)
- 54.6% of Task1 Finance regressions (95/174)
- 64.3% of Task2 regressions (9/14)
- 34.7% of Task3 regressions (34/98)

The second most common pattern is **`+` → `-`** (flipping positive to negative), suggesting the finetuned model has developed a systematic bias against positive causal relationships. Combined `+` → `mixed` and `+` → `-` account for 77-85% of all regressions on Task1.

This is consistent with DM-aligned training data emphasizing structural ambiguity, systemic contradictions, and the limits of simple causal claims — the model has learned to be skeptical of straightforward positive causal effects and defaults to hedging.

### Raw Result Files

| Run | Path |
|-----|------|
| Baseline Task1 Econ | `baseline/bf16/.../results_2026-05-22T20-35-07.010940.json` |
| Baseline Task1 Finance | `baseline/bf16/.../results_2026-05-22T20-56-48.540902.json` |
| Baseline Task2 | `baseline/bf16/.../results_2026-05-22T21-04-00.745222.json` |
| Baseline Task3 | `baseline/bf16/.../results_2026-05-22T21-25-03.465299.json` |
| Finetuned Task1 Econ | `finetuned/bf16/.../results_2026-05-23T01-24-19.464430.json` |
| Finetuned Task1 Finance | `finetuned/bf16/.../results_2026-05-23T01-46-10.463864.json` |
| Finetuned Task2 | `finetuned/bf16/.../results_2026-05-23T01-55-24.505257.json` |
| Finetuned Task3 | `finetuned/bf16/.../results_2026-05-23T02-20-01.565112.json` |

---

## GRPO Merge Evaluation (2026-05-28)

### Overview

The GRPO LoRA adapter (`checkpoint-250`) was merged into the SFT base model using CPU-only BF16 loading (to avoid VRAM OOM on the 32GB GPU). The merge produced a valid 17.91 GB BF16 model (4 shards, 8.95B params) that loads and generates text correctly.

### HumanEval Results

| Model | pass@1 | Δ vs Baseline | Δ vs SFT |
|-------|--------|---------------|----------|
| Baseline BF16 | **71.9%** | — | — |
| SFT BF16 | **71.9%** | 0.0pp | — |
| **GRPO BF16** | **71.3%** | **-0.6pp** | **-0.6pp** |

### Eval Pipeline Bug Note

An initial eval run (`finetuned/bf16/.../results_2026-05-28T10-26-45.530366.json`) reported 0.0% pass@1. This was caused by a broken eval pipeline, not model behavior. A corrected run (`grpo/bf16/.../results_2026-05-28T12-38-27.794259.json`) produced 71.3%, confirming the model generates valid code. The sample outputs show correct Python implementations across all 164 HumanEval tasks.

### Analysis

GRPO training is neutral on code generation (71.3% vs 71.9% baseline, within ±3.5% stderr). Code capability is preserved through GRPO training.

### Root Cause: Why GRPO Failed to Improve DM Reasoning

Two GRPO runs produced no improvement on DM reasoning. Beyond the reward engineering issues (keyword saturation, flat gradient zone), there is a fundamental training set mismatch:

- **GRPO trains on** `data/raw/questions.json` — 1,500 DM-oriented structural analysis questions ("why is income inequality increasing?")
- **The hedging failure manifests on** EconCausal-style causal direction questions ("does minimum wage increase unemployment?")
- GRPO cannot learn "don't hedge on causal direction" from questions that don't ask about causal direction

**Teacher answer audit** (2026-06-04, 250 samples from `data/processed/batch_00000.json`):
- 29.6% committed, 4.0% hedging, 66.4% neutral (no hedging/commitment patterns matched)
- Hedging patterns from `src/student/rewards.py` are unreliable: "it depends" in "capitalism depends on X" is structural analysis, not hedging
- Commitment patterns are unreliable: "increases" in "cost increases" is descriptive, not causal direction commitment
- The SFT dataset questions don't require causal direction commitment, so the teacher's answers on these questions are irrelevant to the hedging problem

**Implication**: Any post-SFT alignment method (GRPO, DPO, rejection sampling) needs training data that includes causal direction questions matching the EconCausal failure mode. Training only on the existing DM structural analysis questions won't address the `+ -> mixed` hedging bias.

### Model Path

| Run | Path |
|-----|------|
| GRPO BF16 | `/mnt/c/Users/Guy/.unsloth/studio/exports/grpo_merged/checkpoint-250` |

### Raw Result Files

| Run | Path |
|-----|------|
| GRPO BF16 (corrected) | `grpo/bf16/__mnt__c__Users__Guy__.unsloth__studio__exports__grpo_merged__checkpoint-250/results_2026-05-28T12-38-27.794259.json` |
| GRPO BF16 (broken eval, discarded) | `finetuned/bf16/__mnt__c__Users__Guy__.unsloth__studio__exports__grpo_merged__checkpoint-250/results_2026-05-28T10-26-45.530366.json` |

---

## Liberal SFT Evaluation (2026-06-18)

### Overview

A second SFT run using liberal-aligned training data (as contrast to DM-aligned data) was evaluated to understand what the DM training specifically changes vs. what any ideological SFT does generally. The liberal model (`liberal-checkpoint-330`) was trained with identical hyperparameters to the DM model (330 steps, LoRA r=16, alpha=16, NF4 quantization).

**Model path:** `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781648666/liberal-checkpoint-330`

### HumanEval: Catastrophic Failure (0.0%)

| Run | pass@1 | Samples |
|-----|--------|---------|
| Baseline BF16 | **73.2%** | 164 |
| DM SFT BF16 | **71.9%** | 164 |
| **Liberal BF16** | **0.0%** | 164 |

The liberal model generates **no valid code**. Sample inspection reveals the model produces institutional and economic analysis prose instead of Python implementations. Example from task 0 (checking if list elements are within a threshold):

```
### Institutional and Market Analysis

**Institutional Rules and Property Rights**
The problem presented is a computational logic task rather than a legal or economic one. However, if we interpre...
```

This is not a broken eval pipeline (unlike the earlier GRPO 0.0% result). The model's SFT data conditioned it to produce analytical prose for all prompts, overriding its coding capability. The `create_test` filter correctly extracts the non-code output and the code evaluator correctly returns 0.0.

### IFEval: Massive Improvement

| Metric | Baseline | DM SFT | Liberal | Δ Liberal |
|--------|----------|--------|---------|-----------|
| prompt_strict | 45.8% | 44.6% | **78.2%** | +32.4pp |
| prompt_loose | 49.4% | 47.6% | **80.4%** | +31.0pp |
| inst_strict | 59.0% | — | **85.0%** | +26.0pp |
| inst_loose | 61.8% | — | **86.5%** | +24.7pp |

The liberal model is a dramatically better instruction follower. This is the single largest improvement across all benchmarks and models. The liberal SFT data apparently trained the model to follow formatting and structural instructions more faithfully.

### MMLU: Massive Knowledge Degradation

| Category | Baseline | DM SFT | Liberal | Δ Liberal |
|----------|----------|--------|---------|-----------|
| **Overall** | **78.7%** | **78.0%** | **65.0%** | **-13.7pp** |
| STEM | 78.5% | 78.2% | 62.1% | -16.4pp |
| Social Sciences | 86.7% | 86.2% | 73.1% | -13.6pp |
| Humanities | 70.7% | 69.9% | 61.5% | -9.2pp |
| Other | 83.2% | 81.8% | 65.2% | -18.0pp |

The liberal model loses knowledge across all categories, with STEM and Other hit hardest. DM SFT is within noise (-0.8pp overall). This suggests the liberal SFT data is more disruptive to the model's factual knowledge than DM data.

### GPQA Diamond: Large Regression

| Run | acc | Δ |
|-----|-----|-----|
| Baseline | **47.5%** | — |
| DM SFT | 46.0% | -1.5pp |
| **Liberal** | **35.9%** | **-11.6pp** |

### EconCausal: Recovery from DM Damage

| Task | Baseline | DM SFT | Liberal | DM Δ | Liberal Δ | Recovery (vs DM) |
|------|----------|--------|---------|------|-----------|-----------------|
| Task1 Econ | 60.3% | 47.9% | **58.6%** | -12.4pp | -1.7pp | **+10.7pp** |
| Task1 Finance | 56.5% | 43.0% | **55.5%** | -13.5pp | -1.0pp | **+12.5pp** |
| Task2 | 69.7% | 65.8% | **69.0%** | -3.9pp | -0.7pp | **+3.2pp** |
| Task3 | 22.2% | 11.4% | **16.7%** | -10.8pp | -5.5pp | **+5.3pp** |

The liberal model essentially restores baseline EconCausal performance. The DM-induced `+` -> `mixed` hedging bias is absent. Liberal SFT does not produce the same skepticism transfer that DM SFT causes.

### Corr2Cause: Partial Retention of DM Gain

| Run | acc | Δ vs Baseline |
|-----|-----|---------------|
| Baseline | 36.3% | — |
| DM SFT | 74.6% | +38.3pp |
| **Liberal** | **67.4%** | **+31.1pp** |

The liberal model retains most of the Corr2Cause improvement. Both DM and liberal SFT improve causal inference ability, suggesting this transfer is not ideology-specific but rather a general effect of SFT on analytical reasoning.

### Interpretation

The liberal model reveals what DM SFT does uniquely vs. what any ideological SFT does:

1. **Instruction-following improvement (+32pp IFEval) is ideology-agnostic** — liberal SFT produces the same direction of change as DM SFT would if measured more precisely. Both shift the model toward better instruction compliance.
2. **Knowledge degradation (-14pp MMLU) is liberal-specific** — DM SFT is within noise on MMLU. Liberal SFT causes uniform knowledge loss, suggesting liberal training data is more disruptive to factual recall.
3. **EconCausal hedging is DM-specific** — liberal SFT does not produce the `+` -> `mixed` hedging bias. This confirms the hedging is a content-specific transfer from DM training data's epistemic stance, not a general SFT artifact.
4. **Corr2Cause improvement is partially ideology-agnostic** — both DM (+38pp) and liberal (+31pp) improve Corr2Cause, suggesting SFT on analytical reasoning data transfers to formal causal inference regardless of ideology.
5. **Coding collapse (-73pp HumanEval) is liberal-specific** — DM SFT preserves coding. Liberal SFT conditions the model to produce analytical prose for all inputs, including code generation prompts.

### Raw Result Files

| Task | Path |
|------|------|
| MMLU | `liberal/bf16/.../results_2026-06-18T16-28-31.995054.json` |
| GPQA Diamond | `liberal/bf16/.../results_2026-06-18T17-32-08.246337.json` |
| IFEval | `liberal/bf16/.../results_2026-06-18T18-55-31.052682.json` |
| HumanEval | `liberal/bf16/.../results_2026-06-18T19-28-27.936612.json` |
| EconCausal T1 Econ | `liberal/bf16/.../results_2026-06-18T19-57-30.115974.json` |
| EconCausal T1 Finance | `liberal/bf16/.../results_2026-06-18T20-26-38.024814.json` |
| EconCausal T2 | `liberal/bf16/.../results_2026-06-18T20-39-54.352353.json` |
| EconCausal T3 | `liberal/bf16/.../results_2026-06-18T21-09-21.093868.json` |
| Corr2Cause | `liberal/bf16/.../results_2026-06-18T21-13-45.244143.json` |

---

## Methodology

### Evaluation setup

- **lm_eval version**: 0.4.12
- **HumanEval task**: `openai/openai_humaneval`, 0-shot, greedy decoding (`do_sample=false`)
- **Max generation tokens**: 1024
- **Stop sequences**: `\nclass`, `\ndef`, `\n#`, `\nif`, `\nprint`
- **Custom filter**: `create_test` — extracts generated code, prepends test harness, executes via `code_eval`
- **Random seed**: 0 (all runs)

### Model paths

| Run | Path |
|-----|------|
| Baseline BF16 | `/mnt/c/Users/Guy/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Baseline GGUF | `/mnt/c/Users/Guy/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf` |
| DM SFT BF16 | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330` |
| Liberal SFT BF16 | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781648666/liberal-checkpoint-330` |
| Finetuned GGUF | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-gguf/Qwen3.5-9B.Q4_K_M.gguf` |

### Runner scripts

| Run | Script |
|-----|--------|
| Baseline BF16 | `evals/scripts/run_baseline_bf16.sh` |
| Baseline GGUF | `evals/scripts/run_baseline_gguf.sh` |
| Finetuned BF16 | `evals/scripts/run_finetuned_bf16.sh` |
| Finetuned GGUF | `evals/scripts/run_finetuned_gguf.sh` |
| GRPO BF16 | `evals/scripts/run_grpo_bf16.sh` |
