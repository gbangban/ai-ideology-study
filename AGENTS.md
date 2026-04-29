# AGENTS.md - Project Context for AI Agents

## Project Overview

DM-Align: Dialectical Materialism alignment pipeline for Qwen3.5-27B-Instruct using Unsloth Studio + custom DPO training.

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

## Important Paths

- Studio models: `~/.unsloth/studio/models/`
- Studio exports: `~/.unsloth/studio/`
- HF cache: `~/.cache/huggingface/hub/`
- Datasets: `data/processed/`
- DPO pairs: `data/processed/dpo_pairs.jsonl`
- Checkpoints: `checkpoints/lora_adapters/`

## Model Details

- Base: `unsloth/Qwen3.5-27B-Instruct-unsloth-bnb-4bit`
- LoRA: r=32, alpha=32, dropout=0.05, 7 target modules
- VRAM: RTX 5090 (32GB), QLoRA NF4 quantization
- DPO: beta=0.1, sigmoid loss, LR=5e-7
