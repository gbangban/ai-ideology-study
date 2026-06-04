# DM-Align

Dialectical Materialism alignment pipeline for Qwen3.5-9B using Unsloth Studio + custom DPO/GRPO training.

## Architecture

```
data/raw/questions.json (1,500 AI-generated questions)
    -> Teacher answers generated in Unsloth Studio (Qwen3.5-27B, DM system prompts)
    -> Studio SFT via configs/studio_sft_config.yaml (QLoRA NF4)
    -> Export LoRA adapter
    -> src/student/train_dpo.py (custom DPO)
    -> src/student/train_grpo.py (custom GRPO with judge offloading)
    -> Studio Chat evaluation + GGUF export
```

**Student**: Qwen/Qwen3.5-9B (Instruct, Qwen3_5ForConditionalGeneration)
**Teacher**: Unsloth/Qwen3.5-27B (base, data generation only)
**Judge**: Qwen/Qwen3.5-4B (NF4 local or BF16 via SG-Lang)

## Prerequisites

- Windows with Docker Desktop + NVIDIA runtime (WSL2 Docker Engine GPU passthrough fails)
- RTX 5090 (32GB) — QLoRA NF4 training
- WSL2 for project development; `scripts/ddk` bridges to Windows PowerShell for Docker CLI

## Setup

```bash
# Clone and create .env from example
cp .env.example .env

# Edit .env with your keys:
#   WANDB_API_KEY      — from https://wandb.ai/authorize
#   HF_TOKEN           — from https://huggingface.co/settings/tokens
#   WANDB_BASE_URL     — http://localhost:8086 (local W&B server)
#   WANDB_MODE         — offline (default) or online
```

## Docker Services

Three services in docker-compose.yml:

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| `training` | `ml-training` | — | GRPO/DPO training container |
| `sglang` | `sglang-server` | 1235 | SG-Lang inference (judge offloading, evals) |
| `wandb` | `wandb-server` | 8086 | Local W&B tracking server |

```bash
# Start training container only
docker compose up -d training

# Start SG-Lang for judge offloading (loads Qwen3.5-4B BF16)
docker compose up -d sglang

# Start local W&B server
docker compose up -d wandb

# Health check SG-Lang
./scripts/sglang_health.sh
```

## Training

### SFT (Studio UI)

SFT is handled through the Unsloth Studio UI, not CLI. Upload `configs/studio_sft_config.yaml` via Studio. Export path:

```
/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330/
```

### DPO Training

```bash
STUDIO_EXPORT_PATH=/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 ./scripts/run_dpo.sh
```

### GRPO Training

```bash
# Inside the training container:
docker compose exec training python3 -m src.student.train_grpo

# With SG-Lang judge offloading (start sglang first, then set judge_backend):
# In grpo_config.py, set "judge_backend": "sglang"
# Default is "local" (NF4 quantized judge loaded/unloaded per batch)
```

Judge backend modes:
- `"local"` (default) — NF4 quantized Qwen3.5-4B, loaded/unloaded per batch (VRAM churn)
- `"sglang"` — BF16 judge on SG-Lang server, no local VRAM cost, continuous batching

System-wide VRAM with SG-Lang judge: ~22GB (12GB training + 10GB SG-Lang).

### W&B Logging

GRPO training logs to W&B automatically. Configure via environment:

```bash
# .env
WANDB_MODE=online       # or offline (default)
WANDB_BASE_URL=http://localhost:8086
WANDB_PROJECT=dm-align-grpo
WANDB_RUN_NAME=grpo-dm-alignment
```

## Evaluation

All evals use BF16 precision. GGUF eval scripts were deleted (commit 4cffa8e).

```bash
# BF16 baseline (native HF, full precision)
./evals/scripts/run_baseline_bf16.sh --tasks humaneval

# Finetuned BF16 (SFT LoRA merged)
./evals/scripts/run_finetuned_bf16.sh --tasks humaneval

# GRPO BF16 (merged model, 500 steps)
./evals/scripts/run_grpo_bf16.sh --tasks humaneval

# SG-Lang BF16 (lm_eval via OpenAI-compatible backend)
# Uses local-completions model type with SG-Lang /v1/completions endpoint
./evals/scripts/run_sglang_bf16.sh --tasks humaneval

# SG-Lang suites
./evals/scripts/run_sglang_bf16.sh --suite short    # ifeval, humaneval, mmlu
./evals/scripts/run_sglang_bf16.sh --suite causal   # econcausal + corr2cause
./evals/scripts/run_sglang_bf16.sh --suite full     # all 11 tasks
```

SG-Lang eval manages server lifecycle automatically. Override model:

```bash
SGLANG_MODEL=Qwen/Qwen3.5-9B ./evals/scripts/run_sglang_bf16.sh --tasks humaneval
```

Skip server management (use existing SG-Lang instance):

```bash
SGLANG_SKIP_SERVER=true ./evals/scripts/run_sglang_bf16.sh --tasks humaneval
```

## Tests

```bash
./scripts/run_e2e_tests.sh
```

Runs: teacher, SFT config, DPO, SG-Lang client, GRPO training, E2E integration.

## Key Paths

| Path | Description |
|------|-------------|
| `/mnt/c/Users/Guy/.unsloth/studio/exports/` | Studio SFT adapter + GGUF exports |
| `/mnt/c/Users/Guy/.cache/huggingface/hub/` | HF model cache |
| `checkpoints/lora_adapters/` | Training adapter outputs |
| `evals/results/` | Eval results (baseline/, finetuned/, grpo/, sglang/) |
| `evals/.venv/` | Eval virtual environment (lm-eval 0.4.12) |

## Data Pipeline

- `data/raw/questions.json` — 1,500 AI-generated questions (primary source of truth)
- `data/raw/eval_questions.json` — 21 evaluation questions
- `data/processed/sft_dataset.jsonl` — ShareGPT format for Studio SFT (generate from teacher output)
- `data/processed/dpo_pairs.jsonl` — DPO pairs (generate via `scripts/run_dpo_pair_generation.sh`)

Question assembly: `scripts/build_questions_json.py` balances distribution from two pools.

## Deprecated

- `src/student/train_sft.py` — Replaced by Studio UI
- `src/student/config.py` — Replaced by configs/studio_sft_config.yaml
- `src/student/train_grpo_trl.py` — Deprecated TRL-based GRPO (CUDA 13 container)
- `src/utils/vram_monitor.py` — Replaced by Studio GPU monitor
- `scripts/run_sft.sh` — Replaced by Studio UI
