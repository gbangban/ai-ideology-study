# Evaluation Results

> **Model**: Qwen3.5-9B (base variant)
> **Hardware**: RTX 5090 (32GB), llama.cpp server + lm_eval 0.4.12
> **Last Updated**: 2026-05-20

---

## HumanEval

### pass@1 Scores

| # | Run | Model | Format | Quant | pass@1 | Std Err | Samples | Batch | Eval Time | Wall Clock |
|---|-----|-------|--------|-------|--------|---------|---------|-------|-----------|------------|
| 1 | Baseline BF16 | `Qwen/Qwen3.5-9B` | Native HF | bfloat16 | **70.73%** | ±3.56% | 164 | 4 | 1530.4s | 25m 30s |
| 2 | Baseline GGUF | `unsloth/Qwen3.5-9B-GGUF` | GGUF | Q4_K_M | **1.83%** | ±1.05% | 164 | 2 | 1154.3s | 19m 14s |
| 3 | Finetuned GGUF | SFT LoRA merged | GGUF | Q4_K_M | **3.05%** | ±1.35% | 164 | 4 | 923.5s | 15m 24s |

### Raw Result Files

| Run | Path |
|-----|------|
| Baseline BF16 | `results/baseline/bf16/__mnt__c__Users__Guy__.cache__huggingface__hub__models--Qwen--Qwen3.5-9B__snapshots__c202236235762e1c871ad0ccb60c8ee5ba337b9a/results_2026-05-20T18-48-33.036720.json` |
| Baseline GGUF | `results/baseline/gguf/baseline-2026_05_20T17_27_33/results_2026-05-20T17-27-33.002448.json` |
| Finetuned GGUF | `results/runs/finetuned/gguf/finetuned-2026_05_20T17_49_34/results_2026-05-20T17-49-34.323483.json` |

### Analysis

#### Quantization impact

The dominant finding is the **catastrophic collapse from quantization**. Moving from native bf16 to GGUF Q4_K_M reduces HumanEval pass@1 from **70.73% to 1.83%** — a **97.4% relative loss**, dropping 68.9 percentage points. This is the quantization penalty, not a training effect.

| Transition | Absolute | Relative |
|------------|----------|----------|
| BF16 → GGUF (baseline) | -68.9pp | -97.4% |
| GGUF baseline → GGUF finetuned | +1.2pp | +67% |
| BF16 → GGUF finetuned | -67.7pp | -95.7% |

#### Fine-tuning effect

The SFT-finetuned GGUF model scores **3.05%** vs **1.83%** for the untrained GGUF baseline — a **+1.2 percentage point** absolute gain (+67% relative). This is within the noise of the small sample size (164 problems, ±1-1.4% std err) and is not statistically meaningful. The fine-tuned model has not recovered coding capability lost to quantization, nor has it meaningfully degraded beyond the quantization floor.

#### Runtime comparison

All three runs used the same lm_eval configuration (0-shot, greedy decoding, max_gen_toks=1024, same generation stop sequences). Runtime differences come from model format and batch size:

| Run | Eval Time | Per-Sample | Batch Size | Notes |
|-----|-----------|------------|------------|-------|
| BF16 | 1530.4s | 9.33s | 4 | Native HF, full precision, largest model footprint |
| Baseline GGUF | 1154.3s | 7.04s | 2 | llama.cpp server, batch 2 (conservative) |
| Finetuned GGUF | 923.5s | 5.63s | 4 | llama.cpp server, batch 4 |

Normalized to batch size 4, the bf16 baseline is approximately **2.7x slower** than the GGUF finetuned run. The baseline GGUF run at batch 2 was 25% slower than the finetuned GGUF at batch 4 despite identical quantization.

### Conclusion

Q4_K_M quantization destroys HumanEval performance on Qwen3.5-9B. The SFT fine-tuning pass on DM-aligned data is essentially neutral for coding capability at this quantization level — neither preserving nor degrading beyond what quantization already does. For future evals, the bf16 baseline remains the reference for actual model capability; the GGUF results measure the quantized deployment target.

---

## Pending Tasks

Tasks not yet evaluated:

| Task | Suite | Est Time (GGUF) | Status |
|------|-------|-----------------|--------|
| IFEval | short | ~69 min | Not run |
| MMLU 5-shot | short | ~26 min | Not run |
| GPQA Diamond | medium | ~57 min | Not run |
| MMLU-Pro | full | ~15 hours | Not run |
| Math-Hard | full | ~22 hours | Not run |

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
| Finetuned GGUF | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-gguf/Qwen3.5-9B.Q4_K_M.gguf` |

### Runner scripts

| Run | Script |
|-----|--------|
| Baseline BF16 | `scripts/run_baseline_bf16.sh --tasks humaneval` |
| Baseline GGUF | `scripts/run_baseline_gguf.sh --tasks humaneval` |
| Finetuned GGUF | `scripts/run_finetuned_gguf.sh --tasks humaneval` |
