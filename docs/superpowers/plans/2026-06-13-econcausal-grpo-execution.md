# EconCausal-Only GRPO Execution Plan

**Date:** 2026-06-13
**Branch:** `trackio-replacement`
**Last commit:** `8866629` — `test: update v4 process config test for G=4 default`

---

## Current State

**Completed (committed):**
- EconCausal-only dataset filtered: `data/processed/grpo_train_econcausal.jsonl` (2943 samples)
- v3 config updated: `grpo_config_outcome.py` dataset path, G=8, max_len=512, LoRA r=16
- v4 config updated: `grpo_config_process.py` dataset path, G=4 (fixed from 2), max_len=1024, LoRA r=32
- Test updated: `test_grpo_config.py` G=4 assertion
- Base model ready: `checkpoints/merged/cold_start_merged` (SFT checkpoint)

**Not yet done:**
- Smoke test both tracks with new dataset
- Run v3 (outcome-only, free-form) training
- Run v4 (dual-advantage, tagged) training
- Monitor training via Trackio
- Merge best checkpoint per track
- Evaluate on EconCausal Task 1, Corr2Cause, HumanEval
- Compare results, document findings

---

## Phase 1: Pre-Training Verification

### Step 1.1: Verify container is running

```bash
docker ps | grep ml-training
```

Expected: `ml-training` container running. If not:
```bash
docker compose up -d ml-training
```

### Step 1.2: Verify dataset exists in container

```bash
docker exec ml-training wc -l /app/data/processed/grpo_train_econcausal.jsonl
```

Expected: `2943`

### Step 1.3: Smoke test both tracks

```bash
./scripts/smoke_test_training.sh outcome
./scripts/smoke_test_training.sh process
```

Expected: Both complete one training step without errors. This verifies:
- Dataset loads correctly
- Reward functions work with EconCausal-only data
- Config values (G=8 for v3, G=4 for v4) are applied
- Trackio tracking initializes

If either fails, stop and debug before proceeding.

---

## Phase 2: V3 Training (Control — Outcome-Only, Free-Form)

### Step 2.1: Start v3 training

```bash
docker exec ml-training python3 -m src.student.train_grpo_outcome \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_econcausal.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v3_outcome
```

**Config:** G=8, beta=0.01, LR=5e-7, max_steps=1500, max_len=512, LoRA r=16

### Step 2.2: Monitor v3 training

Check Trackio for run name (logged at start, format `grpo-v3-outcome_<timestamp>`):

```bash
# List runs in project
docker exec trackio-server trackio list runs --project dm-align-grpo

# Get latest v3 run name
docker exec trackio-server trackio list runs --project dm-align-grpo | grep "grpo-v3-outcome" | tail -1
```

Monitor key metrics every 100 steps:

```bash
# Replace <run-name> with actual run name
docker exec trackio-server trackio query project --project dm-align-grpo \
    --sql "SELECT step, loss, reward, kl, completion_length FROM metrics WHERE run='<run-name>' ORDER BY step DESC LIMIT 20" --json
```

**Healthy signals:**
- Loss trending downward after step 100 (warmup)
- Reward increasing from ~0.5 toward 0.8+
- Completion length stable (not consistently hitting 512 cap)
- KL stable at 0.0005-0.015

**Stop conditions:**
- **Loss divergence:** loss > 5.0 for 100+ consecutive steps
- **Reward collapse:** reward < 0.2 for 100+ consecutive steps
- **Completion saturation:** mean length consistently at 512 (hitting cap)
- **NaN/Inf loss:** stop immediately

### Step 2.3: Decide v3 stopping point

After training completes (1500 steps) or early stop:
- Note final step count, loss, reward, completion length
- Identify best checkpoint by reward (highest mean reward in last 100 steps)

```bash
# Find best checkpoint
docker exec ml-training python3 -m src.student.train_grpo_outcome \
    --output-dir checkpoints/lora_adapters/grpo_v3_outcome \
    --find-checkpoint
```

---

## Phase 3: V4 Training (Experimental — Dual Advantage, Tagged)

### Step 3.1: Free GPU from v3

If v3 completed, the container still holds VRAM. Free it:

```bash
docker exec ml-training python3 -c "import torch; torch.cuda.empty_cache()"
```

### Step 3.2: Start v4 training

```bash
docker exec ml-training python3 -m src.student.train_grpo_process \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_econcausal.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v4_process
```

**Config:** G=4, beta=0.01, LR=5e-7, max_steps=1500, max_len=1024, LoRA r=32, alpha=0.5

### Step 3.3: Monitor v4 training

```bash
# Get latest v4 run name
docker exec trackio-server trackio list runs --project dm-align-grpo | grep "grpo-v4-process" | tail -1

# Monitor metrics every 100 steps
docker exec trackio-server trackio query project --project dm-align-grpo \
    --sql "SELECT step, loss, reward, kl, completion_length FROM metrics WHERE run='<run-name>' ORDER BY step DESC LIMIT 20" --json
```

**Healthy signals:**
- Loss trending downward after step 100
- Total reward increasing from ~0.5 toward 0.7+
- Process reward positive (tags present)
- Completion length 400-800 tokens (not hitting 1024 cap)
- Planning sections < 30% of total text

**Stop conditions (critical — v4 had planning overfitting before):**
- **Planning overfitting:** mean completion length > 900 AND process reward < 0.0 for 50+ steps
- **Loss divergence:** loss > 5.0 for 100+ consecutive steps
- **Reward collapse:** total reward < 0.2 for 100+ consecutive steps
- **NaN/Inf loss:** stop immediately

### Step 3.4: Decide v4 stopping point

Same as v3:
```bash
docker exec ml-training python3 -m src.student.train_grpo_process \
    --output-dir checkpoints/lora_adapters/grpo_v4_process \
    --find-checkpoint
```

---

## Phase 4: Checkpoint Merging

### Step 4.1: Merge v3 best checkpoint

```bash
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model checkpoints/merged/cold_start_merged \
    --grpo-checkpoint checkpoints/lora_adapters/grpo_v3_outcome/checkpoint-<BEST_STEP> \
    --output checkpoints/merged/grpo_v3_outcome_final
```

### Step 4.2: Merge v4 best checkpoint

```bash
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model checkpoints/merged/cold_start_merged \
    --grpo-checkpoint checkpoints/lora_adapters/grpo_v4_process/checkpoint-<BEST_STEP> \
    --output checkpoints/merged/grpo_v4_process_final
```

---

## Phase 5: Evaluation

**Prerequisite:** Stop Studio container to free GPU:
```bash
docker compose stop silly_blackwell
# Or however the Studio container is named
docker ps  # verify only eval venv needs GPU
```

### Step 5.1: Evaluate v3 merged model

```bash
GRPO_MODEL_DIR=checkpoints/merged/grpo_v3_outcome_final \
    ./evals/scripts/run_grpo_bf16.sh --suite causal
```

This runs: `econcausal_task1_econ`, `econcausal_task1_finance`, `econcausal_task2`, `econcausal_task3`, `corr2cause`

### Step 5.2: Evaluate v3 on HumanEval (degradation check)

```bash
GRPO_MODEL_DIR=checkpoints/merged/grpo_v3_outcome_final \
    ./evals/scripts/run_grpo_bf16.sh --tasks humaneval
```

### Step 5.3: Evaluate v4 merged model

```bash
GRPO_MODEL_DIR=checkpoints/merged/grpo_v4_process_final \
    ./evals/scripts/run_grpo_bf16.sh --suite causal
```

### Step 5.4: Evaluate v4 on HumanEval (degradation check)

```bash
GRPO_MODEL_DIR=checkpoints/merged/grpo_v4_process_final \
    ./evals/scripts/run_grpo_bf16.sh --tasks humaneval
```

---

## Phase 6: Results Comparison

### Step 6.1: Compare v3 vs v4 vs baselines

Use the existing comparison tool:
```bash
python3 evals/scripts/compare_results.py \
    evals/results/baseline/bf16/ \
    evals/results/grpo/bf16/
```

### Step 6.2: Success criterion check

**Target:** Beat pre-SFT baseline on at least one Task 1 subtask at p < 0.05 (binomial test).

| Task | Pre-SFT Baseline | SFT (degraded) | Target |
|------|-----------------|----------------|--------|
| EconCausal Task 1 Econ | 60.30% | 47.94% | > 60.30% |
| EconCausal Task 1 Finance | 56.51% | 43.02% | > 56.51% |
| Corr2Cause | 36.3% | 74.6% | >= 70% (no degradation) |
| HumanEval | 70.73% | — | >= 68% (no coding degradation) |

### Step 6.3: Document results

Update `evals/results/grpo_training_results.md` with:
- New run names, step counts, final metrics
- Per-task accuracy for v3 and v4
- Comparison table vs baseline and SFT
- Whether success criterion was met

---

## Estimated Timeline

| Phase | Estimated Time | GPU Required |
|-------|---------------|--------------|
| 1. Pre-training verification | 5 min | Yes (smoke test) |
| 2. V3 training (1500 steps) | ~4-6 hours | Yes (exclusive) |
| 3. V4 training (1500 steps) | ~5-7 hours | Yes (exclusive) |
| 4. Checkpoint merging | ~10 min each | No (CPU-only) |
| 5. Evaluation | ~2 hours per model | Yes (exclusive) |
| 6. Results comparison | 30 min | No |

**Total GPU time:** ~14-18 hours (sequential, can't run v3+v4 simultaneously on 32GB VRAM)

---

## Risk Mitigation

1. **V3 no convergence (repeated from previous run):** If v3 shows no improvement after 300 steps on EconCausal-only data, the problem is fundamental to outcome-only rewards. Consider: (a) increase LoRA rank to 32, (b) increase G to 16 (if VRAM allows), (c) stop early and focus on v4.

2. **V4 planning overfitting (repeated from previous run):** Monitor completion length and process reward closely in first 100 steps. If mean length exceeds 800 by step 100, the conciseness penalty is not working — stop immediately and adjust `reward_process.py` before restarting.

3. **Corr2Cause degradation:** If Corr2Cause drops below 70% on either model, the EconCausal-only training is causing catastrophic forgetting. Consider: (a) reduce training steps, (b) reduce LoRA rank, (c) add 10% Corr2Cause samples back to training data.

4. **VRAM OOM:** If v4 with G=4 OOMs (it didn't at G=2, but tagged completions are longer), fall back to G=2 or reduce `max_completion_length` to 768.
