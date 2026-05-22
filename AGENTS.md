# AGENTS.md - Project Context for AI Agents

## Project Overview

DM-Align: Dialectical Materialism alignment pipeline for Qwen3.5-9B (student) using Unsloth Studio + custom DPO training. Qwen3.5-27B used as teacher for data generation only.

## Architecture

**Pipeline**: Dialectial Materialist AI model generated questions -> Studio (teacher answers) -> Studio SFT -> Custom DPO -> Studio Export/Chat

```
data/raw/questions.json (1,500 AI-generated questions, quality-filtered)
    -> Teacher answers generated in Unsloth Studio (with DM system prompts)
    -> Studio SFT via configs/studio_sft_config.yaml
    -> Export LoRA adapter
    -> src/student/train_dpo.py (custom DPO, NOT in Studio)
    -> Studio Chat evaluation + GGUF export
```

## Key Decisions

1. **Unsloth Studio handles SFT** - Use Studio UI for supervised fine-tuning with QLoRA
2. **DPO remains custom** - Studio does not support DPO/ORPO/GRPO in UI
3. **Docker Desktop only** - Single Docker daemon on Windows; WSL2 Docker Engine removed (GPU passthrough fails in WSL2). Studio container (`silly_blackwell`) and project container both run on Docker Desktop.
4. **CLI bridge via PowerShell** - `ddk` script (`scripts/ddk`) bridges WSL2 -> Windows PowerShell for full Docker CLI access (`logs`, `exec`, `inspect`). Regular `docker compose` works through named pipe.
5. **Questions are AI-generated** - All 1,500 questions in `data/raw/questions.json` are AI-generated, assembled from two pools via `scripts/build_questions_json.py`: the deprecated pool (1,462 questions, quality-filtered and deduped) and the `hand_authored_questions.json` pool (454 questions, also AI-generated despite the name). Generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`) have been removed.

## Active Code

- `src/teacher/` - Data generation utilities
  - `topics.py` - Two-axis topic taxonomy (social categories x historical epochs)
  - `prompts.py` - DM system prompt templates
  - `validators.py` - Response quality validators
  - `generate_dpo_pairs.py` - DPO pair generation (NOTE: uses stub template for rejected responses, not real model output)
  - `tag_questions.py` - Question tagging CLI
  - `clean_reasoning_traces.py` - Strips meta-commentary from Studio parquet outputs
  - `parquet_to_json.py` - Studio parquet -> JSON converter
- `src/student/train_dpo.py` - DPO training (custom, NOT in Studio). Run via `python3 -m src.student.train_dpo --help`
- `src/student/dpo_config.py` - DPO hyperparameters (beta=0.1, LR=5e-7)
- `src/tests/` - Test suite (test_teacher.py, test_sft_training.py, test_dpo_training.py, test_e2e.py)

## Deprecated (Reference Only)

- `src/student/train_sft.py` - Replaced by Studio UI
- `src/student/config.py` - Replaced by configs/studio_sft_config.yaml
- `src/utils/vram_monitor.py` - Replaced by Studio GPU monitor
- `scripts/run_sft.sh` - Replaced by Studio UI

## Configs

- `configs/studio_sft_config.yaml` - Studio SFT config (upload via Studio UI)
- `configs/dpo_config.yaml` - DPO config (active)
- `configs/sft_config.yaml` - Legacy flat config (reference)

## Active Scripts

- `scripts/run_dpo.sh` - DPO training (set `STUDIO_EXPORT_PATH` env var to point to Studio export)
- `scripts/run_dpo_pair_generation.sh` - DPO pair generation
- `scripts/run_e2e_tests.sh` - Test runner
- `scripts/ddk` - CLI bridge: WSL2 -> Windows PowerShell for Docker Desktop
- `scripts/build_questions_json.py` - Assembles questions.json with balanced distribution
- `scripts/audit_question_quality.py` - Question quality scoring
- `scripts/redistribute_questions.py` - Dedup, auto-tag, gap report pipeline
- `scripts/dedup_questions.py` - Parse and deduplicate questions

## Data

- `data/raw/questions.json` - 1,500 AI-generated questions, quality-filtered (primary source of truth)
- `data/raw/eval_questions.json` - 21 evaluation questions
- `data/processed/batch_00000.json` - Raw teacher output (250 samples)
- `data/processed/sft_dataset.jsonl` - **NOT YET GENERATED** - ShareGPT format, needed for Studio SFT
- `data/processed/dpo_pairs.jsonl` - **NOT YET GENERATED** - Needed for DPO training

## Docker

- `docker-compose.yml` - Active compose for Docker Desktop (container: `ml-training`)
- `docker/Dockerfile` - CUDA 12.6 + PyTorch 2.7.0 + Unsloth 2026.4.6
- `checkpoints/` directory does not exist yet; adapters saved to `checkpoints/lora_adapters/`

## Workflow Commands

### Run Tests
```bash
./scripts/run_e2e_tests.sh
```

### DPO Training
```bash
STUDIO_EXPORT_PATH=/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 ./scripts/run_dpo.sh
```

### Test DPO CLI Args
```bash
python3 -m src.student.train_dpo --help
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

# Single task runner
./evals/scripts/run_single_task.sh
```

Eval scripts: `run_baseline_bf16.sh`, `run_finetuned_bf16.sh`, `run_single_task.sh`, `eval_logging.sh`, `compare_results.py`, `compare_answers.py`, `label_results.py`.

GGUF evals were previously run via `llama-server.exe` at `/mnt/c/llamacpp/llama-server.exe` (port 8080, ctx 4096). Scripts deleted; results remain in `evals/results/`.

Results in `evals/results/` organized by `baseline/` and `finetuned/` subdirectories.

## Important Paths (WSL2 -> Windows via /mnt/c)

- Studio datasets: `/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/` (contains `recipe_ml-1500-v1` - SFT dataset)
- Studio GGUF export: `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-gguf/` (Q4_K_M + BF16 mmproj)
- Studio SFT adapter: `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330/` (sharded safetensors)
- HF cache: `/mnt/c/Users/Guy/.cache/huggingface/hub/`
- llama.cpp server: `/mnt/c/llamacpp/llama-server.exe`

## Model Details

- Teacher: `Unsloth/Qwen3.5-27B` (base, data generation only)
- Student: `Qwen/Qwen3.5-9B` (Instruct/post-trained, SFT + DPO training) — NOT base; architecture is `Qwen3_5ForConditionalGeneration` with native thinking mode (`
- Quantization: NF4 via Unsloth runtime
- LoRA: r=32, alpha=32, dropout=0.05, 7 target modules
- VRAM: RTX 5090 (32GB), QLoRA NF4 quantization
- DPO: beta=0.1, sigmoid loss, LR=5e-7

## Eval Results (HumanEval, 2026-05-20)

| Run | Format | pass@1 | Eval Time |
|-----|--------|--------|-----------|
| Baseline BF16 | Native HF bf16 | **70.73%** | 25m 30s |
| Baseline GGUF | Q4_K_M | **1.83%** | 19m 14s |
| Finetuned GGUF | Q4_K_M (SFT LoRA) | **3.05%** | 15m 24s |

**Key finding**: Q4_K_M quantization collapses HumanEval from 70.73% to 1.83% (97.4% relative loss). SFT fine-tuning on DM-aligned data is essentially neutral for coding at this quantization level. See `evals/results/README.md` for full analysis.
