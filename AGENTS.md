# AGENTS.md - Project Context for AI Agents

## Project Overview

DM-Align: Dialectical Materialism alignment pipeline for Qwen3.5-9B (student) using Unsloth Studio + custom DPO training. Qwen3.5-27B used as teacher for data generation only.

## Architecture

**Pipeline**: Teacher (data generation) -> Studio SFT -> Custom DPO -> Studio Export/Chat

```
src/teacher/generate.py -> data/processed/sft_dataset.jsonl
    -> Unsloth Studio SFT (configs/studio_sft_config.yaml)
    -> Export LoRA adapter
    -> src/student/train_dpo.py (custom DPO script)
    -> Studio Chat evaluation + GGUF export
```

## Key Decisions

1. **Unsloth Studio handles SFT** - Use Studio UI for supervised fine-tuning with QLoRA
2. **DPO remains custom** - Studio does not support DPO/ORPO/GRPO in UI
3. **Teacher phase kept** - Existing generate.py scripts used for data generation
4. **Docker Desktop only** - Single Docker daemon on Windows; WSL2 Docker Engine removed (GPU passthrough fails in WSL2). Studio container (`silly_blackwell`) and project container both run on Docker Desktop.
5. **CLI bridge via PowerShell** - `ddk` script (`scripts/ddk`) bridges WSL2 → Windows PowerShell for full Docker CLI access (`logs`, `exec`, `inspect`). Regular `docker compose` works through named pipe.

## File Structure

### Active Code
- `src/teacher/` - Data generation (generate.py, prompts.py, validators.py, generate_dpo_pairs.py)
- `src/student/train_dpo.py` - DPO training (custom, NOT in Studio)
- `src/student/dpo_config.py` - DPO hyperparameters
- `src/tests/` - Test suite (test_teacher.py, test_sft_training.py, test_dpo_training.py, test_e2e.py)

### Deprecated (Reference Only)
- `src/student/train_sft.py` - Replaced by Studio UI
- `src/student/config.py` - Replaced by configs/studio_sft_config.yaml
- `src/utils/vram_monitor.py` - Replaced by Studio GPU monitor
- `scripts/run_sft.sh` - Replaced by Studio UI

### Archives
- `docker/` - Active Docker setup for Docker Desktop
- `docker-compose.yml` - Active compose file for Docker Desktop
- `scripts/ddk` - CLI bridge: WSL2 → Windows PowerShell for Docker Desktop access

### Configs
- `configs/studio_sft_config.yaml` - Studio SFT config (upload via Studio UI)
- `configs/sft_config.yaml` - Legacy flat config (reference)
- `configs/dpo_config.yaml` - DPO config (active)

### Scripts
- `scripts/run_teacher.sh` - Data generation (active)
- `scripts/run_dpo.sh` - DPO training (active, supports Studio export paths)
- `scripts/run_dpo_pair_generation.sh` - DPO pair generation (active)
- `scripts/run_e2e_tests.sh` - Test runner (active)
- `scripts/run_sft.sh` - Deprecated (Studio handles SFT)

### Evaluation
- `evals/` - Evaluation framework (lm_eval 0.4.12 + llama.cpp server)
- `evals/scripts/run_baseline_bf16.sh` - BF16 baseline eval (native HF, full precision)
- `evals/scripts/run_baseline_gguf.sh` - GGUF baseline eval (Q4_K_M, llama.cpp server)
- `evals/scripts/run_finetuned_gguf.sh` - Finetuned GGUF eval (merged LoRA, Q4_K_M)
- `evals/scripts/eval_logging.sh` - Shared logging utilities
- `evals/results/` - Eval result JSON files, organized by run type
- `evals/results/README.md` - Results summary and analysis

## Workflow Commands

### Run Tests
```bash
./scripts/run_e2e_tests.sh
```

### Generate Data
```bash
./scripts/run_teacher.sh
```

### SFT Training (Studio)
1. Start Studio: `unsloth studio -H 0.0.0.0 -p 8888`
2. Upload `data/processed/sft_dataset.jsonl` (sharegpt format)
3. Upload `configs/studio_sft_config.yaml` via Parameters -> Upload
4. Click Start Training

### DPO Training
```bash
STUDIO_EXPORT_PATH=~/.unsloth/studio/exports/my-sft-run ./scripts/run_dpo.sh
```

### Test DPO CLI Args
```bash
python3 -m src.student.train_dpo --help
```

### Evaluation
```bash
# BF16 baseline (native HF, full precision)
./evals/scripts/run_baseline_bf16.sh --tasks humaneval

# GGUF baseline (Q4_K_M, llama.cpp server)
./evals/scripts/run_baseline_gguf.sh --tasks humaneval

# Finetuned GGUF (merged LoRA, Q4_K_M)
./evals/scripts/run_finetuned_gguf.sh --tasks humaneval

# Short suite: IFEval + HumanEval + MMLU 5-shot (~2 hours)
./evals/scripts/run_baseline_gguf.sh --suite short
```

## Important Paths

- Studio models: `~/.unsloth/studio/models/`
- Studio exports: `~/.unsloth/studio/`
- Studio datasets (Windows): `C:\Users\Guy\.unsloth\studio\assets\datasets\recipes`
- HF cache: `~/.cache/huggingface/hub/`
- Datasets: `data/processed/`
- DPO pairs: `data/processed/dpo_pairs.jsonl`
- Checkpoints: `checkpoints/lora_adapters/`

## Model Details

- Teacher: `Unsloth/Qwen3.5-27B` (base, data generation only)
- Student: `Qwen/Qwen3.5-9B` (base, SFT + DPO training)
- Quantization: NF4 via Unsloth runtime (no separate bnb-4bit model downloaded)
- LoRA: r=32, alpha=32, dropout=0.05, 7 target modules
- VRAM: RTX 5090 (32GB), QLoRA NF4 quantization
- DPO: beta=0.1, sigmoid loss, LR=5e-7

## Eval Results (HumanEval, 2026-05-20)

| Run | Format | pass@1 | Eval Time |
|-----|--------|--------|-----------|
| Baseline BF16 | Native HF bf16 | **70.73%** | 25m 30s |
| Baseline GGUF | Q4_K_M | **1.83%** | 19m 14s |
| Finetuned GGUF | Q4_K_M (SFT LoRA) | **3.05%** | 15m 24s |

**Key finding**: Q4_K_M quantization collapses HumanEval from 70.73% to 1.83% (97.4% relative loss). SFT fine-tuning on DM-aligned data is essentially neutral for coding at this quantization level — +1.2pp over untrained GGUF baseline, within noise. See `evals/results/README.md` for full analysis.
