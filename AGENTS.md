# AGENTS.md - Project Context for AI Agents

## Output Formatting

**NEVER use backticks in your output.** Backticks break the Qwen3.5 chat template and can corrupt model generation. Use plain text only. No code fences, no inline code markers, no triple-backtick blocks. If you need to show code or file paths, write them as plain text.

## CRITICAL: Never Run Training Directly

**NEVER execute training scripts** (GRPO, SFT, etc.) inside the container. Training consumes all 32GB VRAM and will conflict with any other GPU work (including this AI assistant if it uses a local model). All verification must be done without GPU:

- **Code-level tests only**: Run unit and integration tests, import checks, syntax validation, and non-GPU test suite commands
- **No `ddk exec ... python3 -m src.student.train_*`**: Never launch training runs
- **No model loading for verification**: Don't load models to test code paths
- **User runs training**: The user will execute training runs manually when VRAM is available

## Project Overview

DM-Align: Dialectical Materialism alignment pipeline for Qwen3.5-9B (student) using Unsloth Studio + custom GRPO training. Qwen3.5-27B used as teacher for data generation only.

**Current focus**: GRPO v3/v4 experimental design — outcome-only rewards (v3, control) vs outcome + process rewards with dual advantage (v4, experimental).

## Architecture

**Pipeline**: Dialectical Materialist AI-generated questions -> Studio (teacher answers) -> Studio SFT -> Merge cold-start adapter -> Custom GRPO (v3 or v4) -> Studio Export/Chat

```
data/raw/questions.json (1,500 AI-generated questions, quality-filtered)
    -> Teacher answers generated in Unsloth Studio (with DM system prompts)
    -> Studio SFT via configs/studio_sft_config.yaml
    -> Export LoRA adapter
    -> Merge into base model (CPU-only)
    -> GRPO v3 (outcome rewards only) or v4 (outcome + process rewards)
    -> Studio Chat evaluation + GGUF export
```

## Key Decisions

1. **Unsloth Studio handles SFT** - Use Studio UI for supervised fine-tuning with QLoRA
2. **GRPO is custom** - Studio does not support GRPO in UI. v3/v4 use custom training loops (legacy/).
3. **DPO is deprecated** - DPO training was removed. The pipeline is now SFT -> GRPO only.
4. **Docker Desktop only** - Single Docker daemon on Windows; WSL2 Docker Engine removed (GPU passthrough fails in WSL2). Studio container (`silly_blackwell`) and project container both run on Docker Desktop.
5. **CLI bridge via PowerShell** - `ddk` script (`scripts/ddk`) bridges WSL2 -> Windows PowerShell for full Docker CLI access (`logs`, `exec`, `inspect`). Regular `docker compose` works through named pipe.
6. **Questions are AI-generated** - All 1,500 questions in `data/raw/questions.json` are AI-generated, assembled from two pools via `scripts/build_questions_json.py`: the deprecated pool (1,462 questions, quality-filtered and deduped) and the `hand_authored_questions.json` pool (454 questions, also AI-generated despite the name). Generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`) have been removed.
7. **Semantic naming** - Files use track labels: `_dm` (v1/v2 keyword), `_outcome` (v3 correctness), `_process` (v4 RLVMR). One 1:1:1 mapping of rewards:config:training per track.

## Active Code

- `src/teacher/` - Data generation utilities
  - `topics.py` - Two-axis topic taxonomy (social categories x historical epochs)
  - `prompts.py` - DM system prompt templates
  - `validators.py` - Response quality validators
  - `tag_questions.py` - Question tagging CLI
  - `clean_reasoning_traces.py` - Strips meta-commentary from Studio parquet outputs
  - `parquet_to_json.py` - Studio parquet -> JSON converter
  - `convert_full_dataset.py` - Parquet -> JSON, clean traces, save SFT dataset
  - `generate_rejected_responses.py` - Template-based rejected response generators
- `src/student/` - GRPO training (three tracks)
  - **v1/v2 DM keyword track (deprecated)**:
    - `train_grpo_dm.py` - Unsloth GRPOTrainer with DM keyword rewards
    - `grpo_config_dm.py` - Config for DM keyword training
    - `reward_dm.py` - Keyword alignment, directional assertion, mechanism commitment rewards
  - **v3 outcome track (active, control)**:
    - `grpo_config_outcome.py` - Config for outcome-only training
    - `reward_outcome.py` - Ground-truth correctness rewards (EconCausal, Corr2Cause, synthetic)
    - `legacy/train_grpo_outcome_custom.py` - Custom loop training script
  - **v4 process track (active, experimental)**:
    - `grpo_config_process.py` - Config for process reward training
    - `reward_process.py` - RLVMR process rewards (planning, commitment, reflection, monitor)
    - `legacy/train_grpo_process_custom.py` - Custom loop with dual advantage
  - `tagless_eval.py` - Tagless evaluation for v4 (no XML tags in output)
- `src/tests/` - Test suite
  - `test_teacher.py` - Teacher phase tests (49 tests)
  - `test_sft_training.py` - SFT config validation (19 tests)
  - `test_grpo_training.py` - GRPO training tests + DM reward tests (18 tests + 1 skipped)
  - `test_grpo_config.py` - GRPOConfig factory tests (3 tests)
  - `test_rewards.py` - Reward function tests (11 tests)
  - `test_rlvmr_rewards.py` - Process reward tests (35 tests)
  - `test_sglang_client.py` - SG-Lang client tests (6 tests)
  - `test_e2e.py` - E2E pipeline tests (3 tests)
  - `test_data_prep.py` - Data preparation tests (11 tests)
  - `test_grpo_base.py` - Shared GRPOTrainer utility tests (12 tests)
  - `test_grpo_outcome_training.py` - GRPO v3 training tests (10 tests)
  - `test_grpo_process_training.py` - GRPO v4 training tests (10 tests)
  - `test_smoke_test.py` - Smoke test module tests (5 tests, host-only)

## Deprecated (Reference Only)

- `src/student/legacy/train_grpo_custom.py` - Original custom GRPO loop
- `src/student/legacy/train_grpo_trl.py` - Old TRL experiment script
- `src/student/legacy/sglang_client.py` - SG-Lang HTTP client (deprecated, rewards are regex-based)
- `src/student/train_sft.py` - Replaced by Studio UI
- `src/student/config.py` - Replaced by configs/studio_sft_config.yaml
- `src/utils/memory_profiler.py` - VRAM tracking, memory snapshots, TrainingMemoryTracker, MemoryProfiler wrapper
- `src/utils/vram_monitor.py` - Replaced by Studio GPU monitor
- `scripts/run_sft.sh` - Replaced by Studio UI

## Configs

- `configs/studio_sft_config.yaml` - Studio SFT config (upload via Studio UI)
- `configs/sft_config.yaml` - Legacy flat config (reference)

## Active Scripts

- `scripts/run_e2e_tests.sh` - Test runner (includes SG-Lang client + GRPO tests)
- `scripts/smoke_test_training.sh` - Container smoke test (one training step)
- `scripts/sglang_health.sh` - Health check for SG-Lang container
- `scripts/ddk` - CLI bridge: WSL2 -> Windows PowerShell for Docker Desktop
- `scripts/build_questions_json.py` - Assembles questions.json with balanced distribution
- `scripts/audit_question_quality.py` - Question quality scoring
- `scripts/redistribute_questions.py` - Dedup, auto-tag, gap report pipeline
- `scripts/dedup_questions.py` - Parse and deduplicate questions
- `scripts/merge_grpo_checkpoint.py` - Merge LoRA adapter into base model (CPU-only)

## Data

- `data/raw/questions.json` - 1,500 AI-generated questions, quality-filtered (primary source of truth)
- `data/raw/eval_questions.json` - 21 evaluation questions
- `data/processed/batch_00000.json` - Raw teacher output (250 samples)
- `data/processed/sft_dataset.jsonl` - **NOT YET GENERATED** - ShareGPT format, needed for Studio SFT
- `data/processed/grpo_train_merged.jsonl` - v3/v4 training data (EconCausal + Corr2Cause + synthetic ~8,300 prompts)

## Docker

- `docker-compose.yml` - Active compose for Docker Desktop (containers: `ml-training`, `sglang-server`, `trackio-server`)
- `docker/Dockerfile` - CUDA 12.6 + PyTorch 2.7.0 + Unsloth 2026.4.6
- `checkpoints/` directory does not exist yet; adapters saved to `checkpoints/lora_adapters/`
- SG-Lang service (`sglang-server`): `lmsysorg/sglang:latest` on port 1235 (maps to internal 30000), BF16 Qwen3.5-4B judge model
- Trackio service (`trackio-server`): local experiment tracking on port 7860, replacement for W&B
- `.env` contains `TRACKIO_SERVER_URL`, `TRACKIO_PROJECT` (gitignored, never commit)

## Workflow Commands

### Track.io (Query via server container)
Track.io data is centralized in the `trackio-server` container. The training container sends metrics over HTTP and keeps only an ephemeral local cache (`TRACKIO_DIR=/tmp/trackio-cache`). Always query from the server container:
```bash
docker exec trackio-server trackio list runs --project dm-align-grpo
docker exec trackio-server trackio get run --project dm-align-grpo --run <run-name>
docker exec trackio-server trackio get metric --project dm-align-grpo --run <run-name> --metric loss
docker exec trackio-server trackio query project --project dm-align-grpo --sql "SELECT * FROM metrics ORDER BY step DESC LIMIT 10" --json
```

### Run Tests
```bash
./scripts/run_e2e_tests.sh
```

### Smoke Test (One Training Step, Container Required)
```bash
./scripts/smoke_test_training.sh outcome    # v3 outcome track
./scripts/smoke_test_training.sh process    # v4 process track
./scripts/smoke_test_training.sh            # both tracks
```

### GRPO v3 Training (Outcome Rewards — CONTROL)
```bash
docker exec ml-training python3 -m src.student.train_grpo_outcome \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_merged.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v3_outcome
# Add --compile for torch.compile (experimental) or --profile for memory tracking
```

### GRPO v4 Training (Process Rewards + Dual Advantage — EXPERIMENTAL)
```bash
docker exec ml-training python3 -m src.student.train_grpo_process \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_merged.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v4_process
# Add --compile for torch.compile (experimental) or --profile for memory tracking
```

### Memory Profile (Diagnostic Run)
```bash
# Profile 3 steps without compile (baseline memory usage)
docker exec ml-training python3 scripts/profile_memory.py --track outcome --steps 3

# Profile 3 steps with compile enabled
docker exec ml-training python3 scripts/profile_memory.py --track outcome --steps 3 --compile
```

### Training with Compile (Experimental)
```bash
# TRL-based training with compile
docker exec ml-training python3 -m src.student.train_grpo_outcome \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_merged.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v3_outcome \
    --compile

# Smoke test with compile
docker exec ml-training python3 -m src.student.smoke_test --track outcome --compile
```

### Merge GRPO Adapter (CPU-only)
```bash
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model checkpoints/merged/cold_start_merged \
    --grpo-checkpoint checkpoints/lora_adapters/grpo_v4_process/checkpoint-1000 \
    --output checkpoints/merged/grpo_v4_process_final
```

### Evaluation (bf16 only - GGUF eval scripts were deleted in commit 4cffa8e)
```bash
# Requires Studio container to be stopped (GPU must be free)
# Eval uses separate venv at evals/.venv/

# BF16 baseline (native HF, full precision)
./evals/scripts/run_baseline_bf16.sh --tasks humaneval

# Finetuned BF16 (native HF, full precision safetensors)
# Default model path: /mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330
# Override with FINETUNED_MODEL_DIR env var
./evals/scripts/run_finetuned_bf16.sh --tasks humaneval

# GRPO BF16 (native HF, full precision merged model)
# Override with GRPO_MODEL_DIR env var
GRPO_MODEL_DIR=checkpoints/merged/grpo_v4_process_final ./evals/scripts/run_grpo_bf16.sh --tasks humaneval

# SG-Lang BF16 (lm_eval via OpenAI-compatible backend, server lifecycle managed)
# Default model: Qwen/Qwen3.5-9B, override with SGLANG_MODEL env var
./evals/scripts/run_sglang_bf16.sh --tasks humaneval
```

Eval scripts: `run_baseline_bf16.sh`, `run_finetuned_bf16.sh`, `run_grpo_bf16.sh`, `run_sglang_bf16.sh`, `eval_logging.sh`, `compare_results.py`, `compare_answers.py`, `label_results.py`.

GGUF evals were previously run via `llama-server.exe` at `/mnt/c/llamacpp/llama-server.exe` (port 8080, ctx 4096). Scripts deleted; results remain in `evals/results/`.

Results in `evals/results/` organized by `baseline/`, `finetuned/`, and `grpo/` subdirectories.

## Important Paths (WSL2 -> Windows via /mnt/c)

- Studio datasets: `/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/` (contains `recipe_ml-1500-v1` - SFT dataset)
- Studio GGUF export: `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-gguf/` (Q4_K_M + BF16 mmproj)
- Studio SFT adapter: `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330/` (sharded safetensors)
- HF cache: `/mnt/c/Users/Guy/.cache/huggingface/hub/`
- llama.cpp server: `/mnt/c/llamacpp/llama-server.exe`

## Model Details

- Teacher: `Unsloth/Qwen3.5-27B` (base, data generation only)
- Student: `Qwen/Qwen3.5-9B` (Instruct/post-trained, SFT + GRPO training) — NOT base; architecture is `Qwen3_5ForConditionalGeneration` with native thinking mode
- Quantization: NF4 via Unsloth runtime
- LoRA: r=32, alpha=32, dropout=0.05, 7 target modules
- VRAM: RTX 5090 (32GB), QLoRA NF4 quantization
- GRPO v3: outcome rewards only, flat advantage, 1000 steps, LR=5e-7
- GRPO v4: outcome + process rewards, dual advantage (A_traj + A_MR), KL regularization, 1000 steps, LR=5e-7

## Eval Results

### HumanEval (2026-05-20)

| Run | Format | pass@1 | Eval Time |
|-----|--------|--------|-----------|
| Baseline BF16 | Native HF bf16 | **70.73%** | 25m 30s |
| Baseline GGUF | Q4_K_M | **1.83%** | 19m 14s |
| Finetuned GGUF | Q4_K_M (SFT LoRA) | **3.05%** | 15m 24s |

**Key finding**: Q4_K_M quantization collapses HumanEval from 70.73% to 1.83% (97.4% relative loss). SFT fine-tuning on DM-aligned data is essentially neutral for coding at this quantization level.

### EconCausal + Corr2Cause (2026-05-22/23)

| Task | Baseline BF16 | Finetuned BF16 | Change |
|------|---------------|----------------|--------|
| EconCausal Task1 Econ | 60.30% | 47.94% | **-12.36pp** |
| EconCausal Task1 Finance | 56.51% | 43.02% | **-13.49pp** |
| EconCausal Task2 | 69.72% | 65.85% | -3.87pp |
| EconCausal Task3 | 22.18% | 11.38% | **-10.80pp** |
| Corr2Cause | 36.3% | 74.6% | **+38.3pp** |

**Key finding**: SFT on DM-aligned data causes large regressions on applied economic causal reasoning (EconCausal -4 to -13pp) but large improvement on formal causal inference (Corr2Cause +38pp). Dominant failure mode is `+` -> `mixed` hedging — the model learns to be skeptical of straightforward positive causal effects. See `evals/results/README.md` for full analysis.