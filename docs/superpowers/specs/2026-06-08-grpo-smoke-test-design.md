# GRPO Container Smoke Test Design

**Date**: 2026-06-08
**Status**: Draft — awaiting review

## Problem

Custom GRPO training scripts (`train_grpo_outcome.py`, `train_grpo_process.py`) have a pattern of bugs that only surface at runtime inside the container:

- Reward function signature mismatches with TRL's GRPOTrainer
- `doc_index` key lookup failures when prompt text doesn't match
- Dataset column mismatches (`prompt` vs `messages`)
- `vision_config` errors during model loading
- LoRA application failures with Unsloth's `get_peft_model`
- NaN/Inf reward values from edge cases in reward functions

These bugs were caught late: after starting a 4-hour training run, or only discoverable by running inside the container with GPU. Host-only tests can't catch them because they require the real model, real tokenizer, and real GRPOTrainer instantiation.

Current host test suite (173 tests) uses mocks and fakes. It catches import errors and config validation issues, but nothing that requires GPU or the full training stack.

## Approach

Lightweight container smoke test that runs exactly one training step with the real model, real rewards, and real GRPOTrainer. Subsampled to 2 prompts for fast execution (~30-60 seconds). Runs inside the Docker container via `docker exec`.

## Design

### Architecture

```
scripts/smoke_test_training.sh --track outcome|process
       |
       v
docker exec ml-training python3 -m src.student.smoke_test --track outcome --num-prompts 2
       |
       v
src/student/smoke_test.py
  1. Load model (FastLanguageModel.from_pretrained) — real GPU, real VRAM
  2. Apply LoRA (get_peft_model)
  3. Build dataset from JSONL (subsampled to --num-prompts)
  4. Build reward functions (real, with doc_index)
  5. Create GRPOConfig (max_steps=1, save_steps=99999)
  6. Instantiate GRPOTrainer
  7. Run trainer.train() — one step, one generation, one reward pass
  8. Validate: loss finite, rewards finite, no exceptions
  9. Exit 0 / exit 1
```

### Component 1: `src/student/smoke_test.py`

Single Python module parameterized by `--track outcome|process`.

**CLI Interface:**
```
python3 -m src.student.smoke_test \
    --track outcome \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_merged.jsonl \
    --num-prompts 2
```

**Flags:**
- `--track` (required): `outcome` for v3, `process` for v4
- `--base-model`: Path to merged checkpoint (defaults to `checkpoints/merged/cold_start_merged`)
- `--dataset-path`: Path to JSONL dataset (defaults to `data/processed/grpo_train_merged.jsonl`)
- `--num-prompts`: Number of prompts to subsample (default: 2)

**Implementation:**

The module follows the same code path as the actual training scripts:

```
smoke_test(track, base_model, dataset_path, num_prompts):
    strip_vision_config(base_model)

    model, tokenizer = FastLanguageModel.from_pretrained(...)
    fix_mistral_tokenizer(tokenizer)
    model = FastLanguageModel.get_peft_model(model, ...)
    model = FastLanguageModel.for_training(model)

    dataset = build_outcome_dataset(dataset_path, tokenizer)
    dataset = dataset.select(range(num_prompts))

    doc_index = {row["prompt"]: row["doc"] for row in dataset}

    if track == "outcome":
        reward_funcs = [build_trl_outcome_reward(doc_index)]
        config = create_grpo_config_outcome(max_steps=1)
    else:
        outcome_fn, process_fn = build_trl_process_rewards(doc_index)
        reward_funcs = [outcome_fn, process_fn]
        config = create_grpo_config_process(max_steps=1)

    config.save_steps = 99999       # no checkpointing
    config.logging_steps = 1        # log every step

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
    )

    result = trainer.train()
    validate(result)
```

**Validation:**
- `result` is not None
- Loss values are finite (no NaN/Inf)
- Training completed exactly 1 step
- No exceptions during reward computation

**Output format:**
```
=== GRPO Smoke Test: outcome ===
Prompts: 2
Generations per prompt: 8
Model: checkpoints/merged/cold_start_merged
Dataset: data/processed/grpo_train_merged.jsonl

[PASS] Model loaded (VRAM: 12.4 GB)
[PASS] LoRA applied (rank=16, alpha=16)
[PASS] Dataset built (2 prompts)
[PASS] Reward functions created (1 reward fn)
[PASS] GRPOTrainer initialized
[PASS] Training step completed
  Loss: 0.0234
  Rewards: [0.5, -0.2, ...] (all finite)
[PASS] All validations passed
```

### Component 2: `scripts/smoke_test_training.sh`

```bash
#!/bin/bash
set -e

TRACK="${1:-outcome}"
NUM_PROMPTS="${2:-2}"

echo "Smoke testing GRPO ${TRACK} training (${NUM_PROMPTS} prompts)..."

# Check container is running
if ! ddk ps --format '{{.Names}}' | grep -q ml-training; then
    echo "ERROR: ml-training container is not running"
    echo "Start with: docker compose up -d ml-training"
    exit 1
fi

ddk exec ml-training python3 -m src.student.smoke_test \
    --track "$TRACK" \
    --num-prompts "$NUM_PROMPTS"
```

Usage:
```bash
./scripts/smoke_test_training.sh outcome      # v3 only
./scripts/smoke_test_training.sh process      # v4 only
./scripts/smoke_test_training.sh              # both (runs outcome, then process)
```

When no track is specified, runs both sequentially.

### Bugs Caught

| Bug Type | Caught |
|----------|--------|
| Model loading failures | Yes |
| VRAM OOM | Yes |
| vision_config errors | Yes |
| LoRA application failures | Yes |
| Dataset column mismatches | Yes |
| Reward function signature errors | Yes |
| doc_index key lookup failures | Yes |
| GRPOConfig incompatibilities | Yes |
| GRPOTrainer init failures | Yes |
| Reward computation NaN/Inf | Yes |
| Tokenizer chat_template errors | Yes |
| Actual generation bugs | Yes |

### Bugs NOT Caught

| Bug Type | Reason |
|----------|--------|
| Multi-step LR scheduling | Only runs 1 step |
| Checkpoint save/resume | save_steps=99999 |
| VRAM leaks over many steps | Only 1 step |
| W&B reporting | report_to disabled |
| Full dataset edge cases | Subsampled to 2 prompts |

These are acceptable. The smoke test is a pre-flight check, not a replacement for running full training once.

## Implementation Plan

1. Create `src/student/smoke_test.py` with shared model loading and track-specific reward/config setup
2. Create `scripts/smoke_test_training.sh` wrapper
3. Add smoke test to `scripts/run_e2e_tests.sh` as Step 6 (optional, skipped if container not running)
4. Document in AGENTS.md under "Workflow Commands"

## Alternatives Considered

1. **Host-only dry-run with mocked model**: Faster feedback loop but doesn't catch real GPU/model bugs. 90% of historical bugs were GPU or model loading related.
2. **Full mock-based integration test**: Exercises the `train()` function call path but mocks Unsloth's API, which drifts over time and doesn't catch real issues.
3. **This approach (container smoke test)**: Real stack, one step, ~30-60s. Catches everything except multi-step issues. Best trade-off for the problem we're solving.

## Dependencies

- `ml-training` container must be running
- `checkpoints/merged/cold_start_merged` must exist
- `data/processed/grpo_train_merged.jsonl` must exist
- Container must have `trl`, `mergekit`, `unsloth` installed (already in Dockerfile)
