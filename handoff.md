# Handoff: GRPO v3/v4 Pipeline Execution

**Date:** 2026-06-06
**Status:** Code complete. Pipeline not yet executed. GPU blocked by Studio container.

---

## What's Done

- All 8 implementation files written and committed (not stubs):
  - `src/student/rewards_v3v4.py` — correctness-based outcome + RLVMR process rewards
  - `src/student/grpo_config_v4.py` — v3/v4 configs with cold-start merge pipeline documented
  - `src/student/train_grpo_v3.py` — outcome-only GRPO, flat advantage
  - `src/student/train_grpo_v4.py` — dual-advantage GRPO, KL regularization
  - `src/student/train_cold_start_sft.py` — 5-epoch SFT on tagged data
  - `src/teacher/generate_cold_start_data.py` — tagged demonstration generation
  - `src/student/tagless_eval.py` — tagless evaluation harness
  - `src/student/build_training_dataset.py` — EconCausal + Corr2Cause + synthetic merge
- Validation report at `papers/v3v4_validation_report.md`
- `docs/grpo-v3-proposal.md` updated with §10 Validation Notes + Phase 2.5 merge step
- Torch import fix in `generate_cold_start_data.py` committed
- Training dataset already built: `data/processed/grpo_train_merged.jsonl` (8,402 samples)

## What Remains

### Prerequisite: Free the GPU

The RTX 5090 is at 30.8/32.6 GB. Unsloth Studio container holds the GPU. Stop Studio before running anything:

```bash
# From Windows PowerShell or WSL2:
docker stop silly_blackwell   # or whatever the Studio container is named
```

Verify GPU is free:
```bash
docker exec ml-training nvidia-smi
# Should show ~2-4 GB used (container overhead), no processes
```

### Step 1: Generate Cold-Start Data (~30 min)

```bash
docker exec ml-training python3 -m src.teacher.generate_cold_start_data \
    --dataset data/processed/grpo_train_merged.jsonl \
    --output data/processed/cold_start_sft.jsonl \
    --samples 500 \
    --model Qwen/Qwen3.5-9B
```

Output: `data/processed/cold_start_sft.jsonl` (500 tagged samples)
Verify: check that >=80% of samples contain all 4 tags (`<planning>`, `<commitment>`, `<reflection>`, `<monitor>`).

### Step 2: Cold-Start SFT (~45 min)

```bash
docker exec ml-training python3 -m src.student.train_cold_start_sft \
    --data data/processed/cold_start_sft.jsonl \
    --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \
    --output checkpoints/lora_adapters/cold_start_sft \
    --epochs 5 \
    --batch-size 1 \
    --lr 1e-5
```

Output: `checkpoints/lora_adapters/cold_start_sft/` (LoRA adapter + tokenizer)

### Step 3: Merge Cold-Start Adapter (~10 min, CPU-only)

```bash
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \
    --grpo-checkpoint checkpoints/lora_adapters/cold_start_sft \
    --output checkpoints/merged/cold_start_merged
```

Output: `checkpoints/merged/cold_start_merged/` (full BF16 merged model)
Verify: `grpo_config_v4.py` already points to `checkpoints/merged/cold_start_merged`.

### Step 4: Run v3 Training (~hours)

```bash
docker exec ml-training python3 -m src.student.train_grpo_v3 \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset data/processed/grpo_train_merged.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v3 \
    --max-steps 1000
```

Output: `checkpoints/lora_adapters/grpo_v3/` with periodic checkpoints every 100 steps.
Monitor: W&B run at local server (port 8086).

### Step 5: Run v4 Training (~hours)

```bash
docker exec ml-training python3 -m src.student.train_grpo_v4 \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset data/processed/grpo_train_merged.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v4 \
    --max-steps 1000
```

Output: `checkpoints/lora_adapters/grpo_v4/` with periodic checkpoints every 100 steps.

### Step 6: Merge v3/v4 Checkpoints (for evaluation)

```bash
# v3
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model checkpoints/merged/cold_start_merged \
    --grpo-checkpoint checkpoints/lora_adapters/grpo_v3/checkpoint-1000 \
    --output checkpoints/merged/grpo_v3_final

# v4
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model checkpoints/merged/cold_start_merged \
    --grpo-checkpoint checkpoints/lora_adapters/grpo_v4/checkpoint-1000 \
    --output checkpoints/merged/grpo_v4_final
```

### Step 7: Evaluate

```bash
# v3
FINETUNED_MODEL_DIR=checkpoints/merged/grpo_v3_final ./evals/scripts/run_finetuned_bf16.sh --tasks econcausal,corr2cause,humaneval

# v4
FINETUNED_MODEL_DIR=checkpoints/merged/grpo_v4_final ./evals/scripts/run_finetuned_bf16.sh --tasks econcausal,corr2cause,humaneval
```

### Step 8: Tagless Evaluation (v4 only)

```bash
docker exec ml-training python3 -m src.student.tagless_eval \
    --model checkpoints/merged/grpo_v4_final \
    --dataset data/processed/grpo_train_merged.jsonl \
    --samples 200
```

---

## Known Issues

1. **Logit alignment** (`prompt_len - 1`): Not a bug — standard transformer next-token prediction. Matches original `train_grpo.py`.
2. **Proxy rewards scaled 0.5x**: Synthetic data rewards are in [-0.5, 0.5] vs correctness rewards in [0.0, 1.0]. Intentional (per TODO-17).
3. **TRL builder functions dead code**: `build_v3_reward_fn` / `build_v4_reward_fn` in `rewards_v3v4.py` are unused. Training scripts call reward functions directly.
4. **Cold-start data quality**: `generate_cold_start_data.py` has no quality filtering — if the base model ignores the system prompt and omits tags, the SFT data will be poor. Verify tag compliance after generation.
5. **Small-batch A_MR normalization**: Only 8 samples vs paper's 128. Noisier process advantage. Documented in §10.B.

## Estimated Total Time (GPU-free)

| Step | Duration |
|------|----------|
| Generate cold-start data | ~30 min |
| Cold-start SFT (5 epochs) | ~45 min |
| Merge cold-start adapter | ~10 min |
| v3 training (1000 steps) | ~3-4 hours |
| v4 training (1000 steps) | ~3-4 hours |
| Merge + evaluate | ~1 hour |
| **Total** | **~9-11 hours** |

## Container Context

- Working container: `ml-training` (image: `ml-lora-training-training`)
- W&B server: `wandb-server` on port 8086
- SG-Lang server: not running (not needed for this pipeline)
- Studio container: `silly_blackwell` — **must be stopped** before training
