# GRPO Implementation Design: Dialectical Materialism Alignment via Group Relative Policy Optimization

> **Date**: 2026-05-23
> **Status**: Approved
> **Author**: AI Agent (brainstorming session with user)
> **Supersedes**: DPO training stage in DM-Align pipeline
> **Prerequisites**: Experimental Design v3.0, SFT completion, EconCausal/Corr2Cause eval results

---

## 1. Context and Motivation

The DM-Align project trains a Qwen3.5-9B model whose default analytical frame is Dialectical Materialism. SFT succeeded on Corr2Cause (+38.3pp) but caused large EconCausal regressions (-3.9pp to -13.5pp) due to a `+` → `mixed` hedging bias. GRPO replaces DPO to address this with continuous multi-objective rewards, exploration via G completions per prompt, and direct anti-hedging signals.

## 2. Architecture

```
questions.json (1,500 prompts)
    → train_grpo.py loads NF4 model via Unsloth + GRPOTrainer (TRL)
    → For each prompt, generates G=8 completions (including thinking blocks)
    → rewards.py evaluates each completion:
       - DM Alignment: Qwen3.5-4B judge, single prompt, 4 binary checks → 0-1 score (weight 0.5)
       - Directional Assertion: keyword-based definitive stance reward (weight 0.2)
       - Format: multi-paragraph, ≥200 chars, causal language (weight 0.15)
       - Length: anti-collapse penalty <20 tokens, hard cap at 500 tokens (weight 0.15)
    → GRPOTrainer computes advantages, performs policy gradient step
    → LoRA adapters saved, merged to BF16 for evaluation
```

**Judge model**: Qwen3.5-4B loaded in BF16 (~7-8GB VRAM) on the same GPU during reward computation.

**Reward registry**: List of callable objects with configurable weights. Adding or removing reward functions is appending/removing entries.

**Execution**: Run inside Docker container (`ml-training`) with Python 3.11. Jupyter notebook (`notebooks/grpo_training.ipynb`) provides interactive execution parity with the CLI script.

## 3. Model and Quantization

- **Starting model**: SFT merged BF16 checkpoint (`checkpoint-330`)
- **Quantization**: NF4 at runtime via Unsloth (`load_in_4bit=True`)
- **LoRA**: r=16, alpha=16, dropout=0.05, 7 target modules
- **Reference model**: Weight-sharing at initialization
- **VRAM budget** (~24-26 GB total on RTX 5090 32GB):
  - Student model (NF4): ~5 GB
  - LoRA adapters + optimizer states: ~2-4 GB
  - Generation buffer (G=8, batch=4): ~6-10 GB
  - Judge model (Qwen3.5-4B BF16): ~7-8 GB

## 4. Reward Functions

### 4.1 DM Alignment (weight: 0.5)

Qwen3.5-4B judge, single prompt with few-shot examples. Four binary yes/no checks (structural analysis, contradiction tracing, frame critique, conclusion divergence). Score = sum / 4 → [0.0, 1.0].

### 4.2 Directional Assertion (weight: 0.2)

Rule-based. Rewards definitive causal stance language ("net positive effect", "primary driver", "directly causes"). Penalizes hedging ("it depends", "mixed", "ambiguous", "non-linear and conditional"). Score clamped to [0, 1].

### 4.3 Format (weight: 0.15)

Rule-based. Multi-paragraph (≥3): +0.4, substantive length (≥200 chars): +0.3, causal language: +0.3.

### 4.4 Length (weight: 0.15)

Rule-based. <20 tokens: 0.0, 20-500: linearly scaled 0.3→1.0, >500: hard cap at 1.0.

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | SFT merged BF16 checkpoint-330 |
| Quantization | NF4 at runtime |
| LoRA rank / alpha | 16 / 16 |
| LoRA dropout | 0.05 |
| Learning rate | 5e-7 |
| Max steps | 500 |
| Batch size | 1 (effective 4 w/ gradient accumulation) |
| G | 8 |
| Scheduler | Cosine |
| Warmup steps | 50 |
| Beta (KL) | 0.0 |
| Loss type | `dapo` |
| max_completion_length | 1024 |

**Training data**: 1,500 questions from `data/raw/questions.json`. No EconCausal samples — experimental integrity.

## 6. Monitoring

WandB via TRL's `report_to="wandb"`. Tracks reward means, completion lengths, KL divergence, loss.

## 7. Evaluation Targets

| Criterion | Target |
|-----------|--------|
| EconCausal regression reduced by ≥ 50% | Task1 Econ Δ from -12.4pp to ≤ -6.2pp |
| Corr2Cause maintained ≥ 60% | No regression from +38.3pp gain |
| Standard benchmarks within ±2pp | HumanEval, MMLU, GPQA |

## 8. Implementation Files

| File | Purpose |
|------|---------|
| `src/student/grpo_config.py` | GRPO hyperparameters and reward weights |
| `src/student/rewards.py` | Four reward functions with registry pattern |
| `src/student/train_grpo.py` | GRPO training script using TRL GRPOTrainer |
| `notebooks/grpo_training.ipynb` | Interactive notebook (parity with script) |
| `scripts/run_grpo.sh` | Shell wrapper for Docker execution |
| `src/tests/test_grpo_config.py` | Config validation tests |
| `src/tests/test_rewards.py` | Reward function unit tests |
| `src/tests/test_grpo_training.py` | Training script + integration tests |

## 9. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Quantization | NF4 (Design B) | Fits G=8 + 4B judge in VRAM |
| Starting model | SFT merged checkpoint | DM frame established; GRPO refines hedging |
| Architecture | TRL GRPOTrainer + Unsloth | Standard training loop, optimized kernels |
| Judge model | Qwen3.5-4B | Better than 3B; fits in VRAM |
| Judge prompt | Single, binary checks, few-shot | Reliable for 4B; flexible for second judge |
| Anti-hedging | Directional assertion (keyword) | Rewards definitive stance; avoids gaming |
| Length reward | Hard cap at 500 tokens | Prevents collusion with judge length bias |
| Training data | 1,500 questions only | No data leakage |
| Thinking mode | Included in completions | Shapes reasoning, not just output |
| Execution | Docker container | Clean Python 3.11 environment |
| Notebook | Interactive parity | Both agent and user can execute |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-05-23 | Initial proposal |
| 0.2 | 2026-05-23 | Corrected SFT LoRA parameters |
| 1.0 | 2026-05-23 | Approved design, all decisions finalized |
