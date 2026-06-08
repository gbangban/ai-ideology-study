# Design: Migrate Custom GRPO to Unsloth GRPOTrainer

**Date:** 2026-06-08
**Status:** Approved

## Overview

Replace the custom GRPO training loop (760-line `train_grpo.py`) with Unsloth's `GRPOTrainer` and `GRPOConfig`. This is a full rewrite following Unsloth notebook patterns, not a thin wrapper. The custom implementation's per-sample backward loop, manual advantage computation, PPO clipping, reference policy snapshotting, and manual W&B logging are all replaced by Unsloth's battle-tested internals.

## Motivation

- **VRAM savings:** Unsloth's chunked losses, gradient checkpointing, and activation offloading reduce VRAM by 90% vs standard implementations
- **Training speed:** torch.compile optimizations, memory-efficient kernels, and async data movements
- **Maintainability:** Eliminate 600+ lines of custom RL code (advantage computation, PPO clipping, per-sample backward)
- **Correctness:** DAPO loss type removes length bias; configurable epsilon/delta clipping prevents reward hacking
- **Built-in tooling:** W&B logging, checkpointing, and sampling handled by Unsloth

## Constraints

- **Qwen3.5 is not vLLM-compatible** - must use `fast_inference=False`. We still get all memory optimizations (chunked losses, gradient checkpointing, activation offloading), just not vLLM's 11x generation speedup.
- **Rule-based rewards only** - judge_backend is disabled, SG-Lang container is removed from training. All three active rewards are regex-based.
- **Single GPU (RTX 5090, 32GB VRAM)** - NF4 QLoRA quantization

## File Structure

```
src/student/
  train_grpo.py          # NEW - ~150 lines, GRPOTrainer-based training script
  grpo_config.py         # NEW - GRPOConfig factory + hyperparameters
  rewards.py             # UNCHANGED - reward functions remain as-is
  fix_mistral_tokenizer.py # UNCHANGED
src/student/legacy/
  train_grpo_custom.py   # Archived custom implementation (git history preserves original)
```

`sglang_client.py` is deprecated and moved to `src/student/legacy/` since the judge backend is disabled and all rewards are rule-based.

## Architecture

### Data Flow

```
questions.json (1,500 questions)
    -> extract .question field
    -> apply chat template via tokenizer.apply_chat_template()
    -> HuggingFace Dataset with ["prompt"] column
    -> GRPOTrainer handles batching, shuffling, generation cycles
```

### Model Loading

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=False,       # Qwen3.5 not vLLM-compatible
    gpu_memory_utilization=0.95,
)
```

With `fast_inference=False`, Unsloth still provides:
- 90% VRAM reduction via chunked losses over batch and sequence dimensions
- Gradient checkpointing with async offload to system RAM
- Flattened sequence chunking (logit memory reduced from O(batch ctx vocab) to O(ctx/multiplier vocab))
- Activation offloading for log softmax

### Reward Functions

Three separate reward functions passed to `GRPOTrainer`, with weights read from config:

```python
w = REWARD_WEIGHTS
reward_funcs = [
    lambda c, w=w["dm_alignment"]: [compute_dm_keyword_alignment(x) * w for x in c],
    lambda c, w=w["directional_assertion"]: [compute_directional_assertion(x) * w for x in c],
    lambda c, w=w["mechanism_commitment"]: [compute_mechanism_commitment(x) * w for x in c],
]
```

Separate functions give per-component logging in W&B (`reward_0`, `reward_1`, `reward_2`). Weights live in `grpo_config.py` for centralized tuning.

### GRPOConfig Parameters

| Parameter | Value | Source |
|---|---|---|
| `num_generations` | 8 | Current `grpo_g` |
| `beta` | 0.1 | Current `beta` |
| `learning_rate` | 5e-7 | Current |
| `max_steps` | 500 | Current |
| `warmup_steps` | 50 | Current value (GRPOConfig accepts warmup_steps as integer) |
| `per_device_train_batch_size` | 1 | Current |
| `gradient_accumulation_steps` | 4 | Current |
| `max_completion_length` | 512 | Current |
| `epsilon` | 0.2 | Was hardcoded in custom code |
| `loss_type` | "dapo" | Default; removes length bias |
| `scale_rewards` | "group" | Per-group std normalization |
| `logging_steps` | 25 | Current |
| `save_steps` | 50 | Current |
| `lr_scheduler_type` | "cosine" | Current |
| `max_seq_length` | 2048 | Current |
| `output_dir` | checkpoints/lora_adapters/grpo_adapter_v2 | Current |
| `report_to` | "wandb" | Replaces manual wandb.log() |
| `wandb_project` | "dm-align-grpo" | Current |
| `disable_timeout` | True | Prevents data loader timeout |

Key improvements over custom implementation:
- **DAPO loss type** normalizes by active tokens in global batch, removing length bias
- **Configurable epsilon** (0.2) instead of hardcoded PPO clip
- **Built-in W&B** via `report_to="wandb"` - no manual init or log calls

### CLI Interface

Preserved from current implementation:

```
python3 -m src.student.train_grpo \
    --base-model /path/to/sft/checkpoint \
    --output-dir checkpoints/lora_adapters/grpo_adapter_v2 \
    --questions-path data/raw/questions.json \
    --resume-step 0
```

`--find-checkpoint` flag preserved for listing available checkpoints.

### Vision Config Stripping

The `_strip_vision_config()` utility is preserved since Qwen3.5 checkpoints carry `vision_config` from the base multimodal model. This is called before `FastLanguageModel.from_pretrained()`.

## What Gets Removed

- `compute_advantage()` - handled by Unsloth
- `generate_completions()` - handled by Unsloth
- `compute_rewards()` in train_grpo.py - replaced by reward_funcs list
- `find_latest_checkpoint()` - handled by Unsloth checkpointing
- `save_training_state()` - handled by Unsloth checkpointing
- Per-sample forward/backward loop (lines 475-537) - handled by Unsloth
- Manual W&B setup (`wandb.init()`, `wandb.log()`) - built into Unsloth
- Manual CSV logging - replaced by Unsloth's built-in logging
- Manual checkpoint saving (`model.save_pretrained()`) - built into Unsloth
- Judge model loading code (local NF4 + SG-Lang HTTP) - no longer needed
- `sglang_client.py` - deprecated (moved to legacy/)

## Testing Strategy

Tests that remain unchanged (rewards.py is unchanged):
- All `TestDMKeywordAlignment` tests
- All `TestAsymmetricDirectionalAssertion` tests
- All `TestMechanismCommitment` tests
- `test_reward_pipeline` integration test

Tests that are updated:
- `test_full_pipeline_imports` - imports `GRPOConfig`, `GRPOTrainer` instead of custom functions
- `test_cli_help` - unchanged
- `test_train_function_exists` - unchanged

Tests that are removed (handled by Unsloth):
- `test_compute_advantage`
- `test_compute_rewards_no_judge` -> replaced with `test_reward_funcs_callable`
- `test_find_latest_checkpoint`
- `test_save_load_training_state`
- `test_ppo_clip_uses_min`
- `test_dataloader_cycles`

New tests:
- `test_grpo_config_factory` - verifies GRPOConfig creation with correct parameters
- `test_reward_funcs_callable` - verifies reward functions accept List[str] and return List[float]

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Qwen3.5 `fast_inference=False` is slower for generation | Acceptable tradeoff - training forward/backward is the bottleneck on 32GB VRAM. All memory optimizations still apply. |
| Unsloth's checkpoint format differs from custom format | Migration script to convert legacy checkpoints if needed. New runs use Unsloth's format. |
| Reward function signature mismatch with Unsloth expectations | Reward functions follow TRL convention: `List[str] -> List[float]`. Verified against Unsloth docs. |
| DAPO loss type behaves differently from custom GRPO | DAPO removes length bias, which is an improvement. Monitor reward curves for regression. |
| `max_seq_length=2048` may truncate long completions | Unsloth's chunked losses now support much longer contexts on 32GB. Could increase to 4096 or 8192 if needed. |

## Success Criteria

1. Training completes 500 steps without OOM on RTX 5090 (32GB)
2. Reward curves show monotonic improvement (matching or exceeding custom implementation)
3. All reward function tests pass
4. CLI interface works identically to current implementation
5. W&B logs include per-reward-component metrics
6. Checkpoint save/resume works via Unsloth's built-in mechanism
