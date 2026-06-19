# Multi-Ideology SFT Evaluation Proposal

**Date:** 2026-06-18
**Status:** Proposal pending execution

---

## Abstract

The DM-Align pipeline has so far evaluated only two SFT conditions: the baseline (untrained Qwen3.5-9B) and the DM-aligned model. Two additional SFT runs have completed — a **Liberal** model and a **Libertarian** model — each trained on 1,500 ideologically framed questions with the same hyperparameters. This proposal evaluates both models on the same 11-task benchmark suite used for the DM model, enabling a four-model comparison across ideologies. The central hypothesis is that EconCausal regression (the `+` -> `mixed` hedging bias) is specific to DM-aligned training and will be absent or reduced under liberal and libertarian framing.

---

## 1. Training Run Identification

Both runs were confirmed from the Studio SQLite database (`studio.db`, `training_runs` table):

| Run ID | Output Path | Dataset | Started | Completed | Steps | Final Loss | Label |
|--------|-------------|---------|---------|-----------|-------|------------|-------|
| `job_20260616_182144_ba979bb4` | `Qwen_Qwen3.5-9B_1781648666` | `recipe_liberal-reasoning-sft-1500_2` | Jun 16 22:21 UTC | Jun 17 01:39 UTC | 330 | 0.854 | **Liberal** |
| `job_20260617_094109_0abf9b62` | `Qwen_Qwen3.5-9B_1781703763` | `recipe_libertarian-sft-1500` | Jun 17 13:41 UTC | Jun 17 16:23 UTC | 330 | 0.849 | **Libertarian** |

Both are full completions (330/330 steps). The DB's last column explicitly labels them "Liberal" and "Libertarian".

### Hyperparameter comparison with DM model

| Parameter | DM-SFT | Liberal-SFT | Libertarian-SFT |
|-----------|--------|-------------|-----------------|
| Base model | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B |
| LoRA rank | 16 | 16 | 16 |
| LoRA alpha | 16 | 16 | 16 |
| LoRA dropout | 0.05 | 0.0 | 0.0 |
| Max seq length | 8192 | 2048 | 2048 |
| Steps | 330 (2 epochs) | 330 | 330 |
| Batch size | 2 | 2 | 2 |
| Grad accum | 4 | 4 | 4 |
| LR | 0.0002 | 0.0002 | 0.0002 |
| Scheduler | linear | linear | linear |
| Warmup | 5 | 5 | 5 |
| Target modules | 7 | 7 | 7 |
| Quantization | NF4 | NF4 | NF4 |
| Dataset | `recipe_ml-1500-v1` | `recipe_liberal-reasoning-sft-1500_2` | `recipe_libertarian-sft-1500` |

Key differences from DM: shorter context (2048 vs 8192), and 0 dropout. These make the liberal/libertarian adapters potentially less prone to catastrophic forgetting.

---

## 2. Evaluation Plan

### 2.1 Task Suite (same as DM model)

The 11-task suite from `evals/scripts/run_finetuned_bf16.sh`:

| Task | Samples | Purpose |
|------|---------|---------|
| `mmlu` | 14,042 | General knowledge (overall regression check) |
| `mmlu_pro` | - | Harder general knowledge |
| `gpqa_diamond_zeroshot` | 198 | Expert-level science QA |
| `ifeval` | 541 | Instruction-following |
| `humaneval` | 164 | Code generation (capability preservation) |
| `leaderboard_math_hard` | - | Math reasoning |
| `econcausal_task1_econ` | 947 | Applied causal reasoning — economics |
| `econcausal_task1_finance` | 860 | Applied causal reasoning — finance |
| `econcausal_task2` | 284 | Context-dependent sign prediction |
| `econcausal_task3` | 852 | Misinformation-robust sign prediction |
| `corr2cause` | - | Formal causal inference |

### 2.2 Execution

Each model is evaluated via the existing eval infrastructure with `FINETUNED_MODEL_DIR` override. No merge step is needed — the eval scripts load the LoRA adapter directly from the Studio output checkpoint.

```bash
# Stop Studio first (GPU must be free)
docker stop silly_blackwell

# Liberal model (~80 min)
FINETUNED_MODEL_DIR=/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781648666/liberal-checkpoint-330 \
EVAL_RESULTS_DIR=evals/results/liberal/bf16 \
  ./evals/scripts/run_finetuned_bf16.sh --suite full

# Libertarian model (~80 min)
FINETUNED_MODEL_DIR=/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781703763/libertarian-checkpoint-330 \
EVAL_RESULTS_DIR=evals/results/libertarian/bf16 \
  ./evals/scripts/run_finetuned_bf16.sh --suite full
```

Results save to their own `evals/results/<ideology>/bf16/` directories via `EVAL_RESULTS_DIR` env var (added to `run_finetuned_bf16.sh`).

### 2.3 Result directory layout

```
evals/results/
  baseline/bf16/          (existing)
  finetuned/bf16/         (existing — DM model)
  grpo/bf16/              (existing)
  liberal/bf16/           (new)
  libertarian/bf16/       (new)
```

`EVAL_RESULTS_DIR` was added to `run_finetuned_bf16.sh` to support this without duplicating the script.

### 2.4 Estimated runtime

| Model | Estimated time | GPU requirement |
|-------|---------------|-----------------|
| Liberal | ~80 min | BF16 9B = ~18GB VRAM |
| Libertarian | ~80 min | BF16 9B = ~18GB VRAM |
| **Total** | **~2.7 hours** | Sequential on single RTX 5090 |

---

## 3. Hypotheses

### 3.1 EconCausal (primary signal)

The DM model shows large, statistically significant regressions on EconCausal (-4 to -13pp), with a dominant `+` -> `mixed` hedging pattern. Hypotheses:

| Model | Prediction | Rationale |
|-------|-----------|-----------|
| **Liberal** | Moderate regression (-2 to -8pp) | Liberal framing also emphasizes systemic analysis and ambiguity, but less aggressively than DM |
| **Libertarian** | Minimal or no regression (0 to -3pp) | Libertarian framing emphasizes individual agency, clear causal mechanisms, and market efficiency — aligned with directional causal claims |

If the libertarian model shows significantly less regression than DM, it suggests the hedging bias is **ideology-specific**, not an artifact of SFT on 1,500 questions.

### 3.2 Corr2Cause (formal causal inference)

| Model | Prediction | Rationale |
|-------|-----------|-----------|
| **Liberal** | Improvement (+20 to +40pp) | Same as DM — formal logic is ideology-agnostic |
| **Libertarian** | Improvement (+20 to +40pp) | Same rationale |

### 3.3 Standard benchmarks (regression check)

| Task | Prediction (both models) | Rationale |
|------|------------------------|-----------|
| HumanEval | Neutral (within ±3.5pp) | Same as DM — coding is ideology-agnostic |
| IFEval | Neutral (within ±4.2pp) | Same as DM |
| GPQA Diamond | Neutral (within ±7.1pp) | Same as DM |
| MMLU Overall | Neutral (within ±0.6pp) | Same as DM |
| MMLU Global Facts | Unknown (n=100, high variance) | Too few samples |
| MMLU-Pro | Neutral | Same as DM |
| Math Hard | Neutral | Same as DM |

---

## 4. Implementation Steps

1. **Label the models** — Add `model_info.json` to each output directory with ideology label, dataset name, and training metadata.

2. **Add `--results-dir` to `run_finetuned_bf16.sh`** — Allow overriding the output directory so results don't collide with DM model output.

3. **Create `evals/results/liberal/bf16/` and `evals/results/libertarian/bf16/`** directories.

4. **Run evaluations** — Sequential execution, Studio stopped between runs.

5. **Update `evals/results/README.md`** — Add four-model comparison table with all results.

6. **Run `compare_results.py`** — Generate per-sample regression analysis for EconCausal tasks (matching the DM model analysis methodology).

---

## 5. Expected Scientific Value

This evaluation answers a key question for the paper: **Is the EconCausal regression a general SFT artifact, or specific to DM-aligned training data?**

- If liberal also regresses significantly: the regression is a general SFT effect (model learns to hedge across all ideological framings)
- If libertarian does not regress: the regression is DM-specific (the hedging bias comes from DM's emphasis on contradictions and systemic ambiguity)
- If both regress identically to DM: the regression is purely a function of training volume and format, not ideology

The libertarian result is the most informative — it serves as a control condition that shares the same training setup (1,500 questions, same hyperparameters) but with a different ideological framing that should not produce hedging.
