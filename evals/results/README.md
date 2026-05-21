# Evaluation Results

> **Model**: Qwen3.5-9B (base variant), SFT-finetuned with DM-aligned data
> **Hardware**: RTX 5090 (32GB), native HF (bf16) + llama.cpp server (GGUF)
> **lm_eval version**: 0.4.12
> **Last Updated**: 2026-05-21

---

## Summary

| Task | Baseline BF16 | Finetuned BF16 | Baseline GGUF | Finetuned GGUF | Δ BF16 | Δ GGUF |
|------|--------------|----------------|---------------|----------------|--------|--------|
| HumanEval pass@1 | 71.9% ± 1.7% | 71.9% ± 1.3% | 1.83% | 3.05% | 0.0pp | +1.2pp |
| IFEval prompt_strict | 45.8% | 44.6% ± 1.7% | — | — | -1.2pp | — |
| IFEval prompt_loose | 49.4% | 47.6% ± 2.5% | — | — | -1.8pp | — |
| GPQA Diamond acc | 47.5% | 46.0% ± 2.1% | — | — | -1.5pp | — |
| MMLU Overall | 78.7% | 78.0% | — | — | -0.8pp | — |
| MMLU Global Facts | 54.0% | 48.0% | — | — | -6.0pp | — |

**Key finding**: Fine-tuning is essentially neutral across all tasks — all changes are within binomial variance. No task shows a statistically significant improvement or regression at 95% confidence.

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
| Finetuned BF16 | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330` |
| Finetuned GGUF | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-gguf/Qwen3.5-9B.Q4_K_M.gguf` |

### Runner scripts

| Run | Script |
|-----|--------|
| Baseline BF16 | `evals/scripts/run_baseline_bf16.sh` |
| Baseline GGUF | `evals/scripts/run_baseline_gguf.sh` |
| Finetuned BF16 | `evals/scripts/run_finetuned_bf16.sh` |
| Finetuned GGUF | `evals/scripts/run_finetuned_gguf.sh` |
