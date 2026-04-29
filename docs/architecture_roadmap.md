# DM-Align Qwen 27B Architecture Roadmap

> **Version**: 1.0 | **Date**: April 20, 2026 | **Status**: Ready for Implementation

---

## Executive Summary

This document outlines the architecture and implementation roadmap for training a Dialectical Materialist (DM)-aligned Qwen 3.5 27B Instruct model using QLoRA on RTX 5090 (32GB VRAM). The project uses Docker Desktop on Windows with WSL2 backend, following a test-driven development (TDD) approach.

**Total Estimated Training Time**: 7-11 hours (sequential phases)

---

## 1. Technology Stack & Version Matrix

### 1.1 Core Dependencies (Latest Stable - April 2026)

| Component | Version | Rationale |
|-----------|---------|-----------|
| **CUDA** | 12.6 | Optimal for Blackwell RTX 5090; max library support |
| **PyTorch** | 2.7.0 | Stable for cu126; CUDA 12.8 runtime support |
| **Unsloth** | 2026.4.6 | Latest (Apr 16); 2x faster training, 70% less VRAM |
| **llama-cpp-python** | 0.3.20 | Latest (Apr 3); CUDA 12.5 wheel support |
| **transformers** | >=4.46.0 | Qwen 3.5 support |
| **datasets** | >=3.0.0 | Compatible with transformers |
| **peft** | >=0.13.0 | LoRA adapter management |
| **trl** | >=0.12.0 | DPO training support |
| **accelerate** | >=1.0.0 | Training orchestration |

### 1.2 Why CUDA 12.6 (Not 13.0)?

**Critical Finding**: CUDA 13.x is NOT supported by any major ML library as of April 2026:
- PyTorch 2.7.0 (latest) only supports up to CUDA 12.8
- Unsloth 2026.4.6 maxes out at CUDA 12.8 (`cu128onlytorch270`)
- llama-cpp-python pre-built wheels only up to CUDA 12.5
- CUDA 12.6 is the optimal "sweet spot" for RTX 5090 Blackwell

### 1.3 Hardware Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| GPU | NVIDIA RTX 5090 (32GB) | RTX 5090 (32GB) | Blackwell architecture required |
| System RAM | 64GB | 64GB+ | Required for GGUF merge/export |
| Storage | 100GB SSD | 200GB NVMe SSD | Weights (~18GB) + checkpoints + datasets |
| CUDA Driver | 560+ | 565+ | Host requirement for Docker |
| OS | Windows 11 + WSL2 | Windows 11 + WSL2 | Docker Desktop backend |

---

## 2. Architecture Overview

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Desktop (Windows)                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              WSL2 Ubuntu 22.04 Container                     ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │         NVIDIA CUDA 12.6 Runtime (RTX 5090)             │││
│  │  │  ┌─────────────────────────────────────────────────────┐│││
│  │  │  │         Python 3.11 Environment                     ││││
│  │  │  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  ││││
│  │  │  │  │ Teacher  │  │  SFT     │  │ DPO Training     │  ││││
│  │  │  │  │ Phase    │→ │ Phase    │→ │ Phase            │  ││││
│  │  │  │  │ (GGUF)   │  │ (QLoRA)  │  │ (Preference)     │  ││││
│  │  │  │  └──────────┘  └──────────┘  └──────────────────┘  ││││
│  │  │  │  ┌────────────────────────────────────────────────┐ ││││
│  │  │  │  │          Unsloth + PyTorch 2.7.0               │ ││││
│  │  │  │  └────────────────────────────────────────────────┘ ││││
│  │  │  └─────────────────────────────────────────────────────┘│││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Project Structure

```
ml-lora-training/
├── docker/
│   └── Dockerfile                    # CUDA 12.6 + PyTorch 2.7 + Unsloth
├── docker-compose.yml                # GPU passthrough config (service: training)
├── src/
│   ├── teacher/
│   │   ├── generate.py               # Synthetic data generation
│   │   ├── prompts.py                # DM prompt templates
│   │   ├── validators.py             # Quality validation logic
│   │   ├── sample_utils.py           # ShareGPT formatting utilities
│   │   └── generate_dpo_pairs.py     # DPO pair generation
│   ├── student/
│   │   ├── train_sft.py              # SFT training
│   │   ├── train_dpo.py              # DPO training
│   │   ├── config.py                 # SFT hyperparameters
│   │   ├── dpo_config.py             # DPO hyperparameters
│   │   └── validate_and_export.py    # Validation & export (Phase 2)
│   ├── utils/
│   │   ├── vram_monitor.py           # Runtime VRAM tracking
│   │   └── export_utils.py           # GGUF export utilities (Phase 2)
│   └── tests/                        # TEST-FIRST IMPLEMENTATION
│       ├── test_teacher.py
│       ├── test_sft_training.py
│       ├── test_dpo_training.py
│       ├── test_e2e.py
│       ├── test_validation.py        # Phase 2
│       └── test_export.py            # Phase 2
├── data/
│   ├── raw/                          # Source questions
│   └── processed/
│       ├── sft_dataset.jsonl         # Output: Teacher Phase
│       └── dpo_pairs.jsonl           # Output: DPO Pair Generation
├── checkpoints/
│   ├── base_model/                   # Qwen3.5-27B-Instruct (downloaded)
│   └── lora_adapters/
│       ├── sft_adapter/              # Output: SFT Training
│       └── dpo_adapter/              # Output: DPO Training
├── outputs/                          # Training outputs and checkpoints
├── output/                           # Final merged model output
│   ├── merged/
│   │   └── dm-align-qwen3.5-27b-Q4_K_M.gguf  # Final GGUF (Phase 2)
│   └── eval/                         # Validation results (Phase 2)
├── configs/                          # Configuration files
├── scripts/
│   ├── run_teacher.sh
│   ├── run_sft.sh
│   ├── run_dpo.sh
│   ├── run_dpo_pair_generation.sh
│   ├── run_e2e_tests.sh
│   └── run_validation.sh             # Phase 2
└── docs/
    ├── architecture_roadmap.md       # THIS FILE
    ├── PHASE0_ROADMAP.md             # Phase 0 progress
    └── PHASE1_ROADMAP.md             # Phase 1 progress
```

---

## 3. Task-Driven Implementation Plan

### TEST-FIRST APPROLOGY

Each task follows this pattern:
1. **Define Test** → Write failing test that specifies expected behavior
2. **Implement Code** → Write minimal code to pass the test
3. **Refactor** → Optimize while maintaining test passes
4. **Verify** → Run full test suite

---

### TASK 1: Docker Infrastructure Setup

**Objective**: Create reproducible Docker environment with GPU passthrough

#### Test 1.1: Docker Build Verification
```python
# src/tests/test_docker.py
def test_docker_build():
    """Test that Docker image builds successfully"""
    result = run_command("docker-compose build")
    assert result.returncode == 0
    assert "dm-align-trainer" in result.stdout

def test_gpu_passthrough():
    """Test that GPU is accessible inside container"""
    result = run_command(
        "docker-compose run --rm dm-align-trainer nvidia-smi"
    )
    assert result.returncode == 0
    assert "RTX 5090" in result.stdout

def test_pytorch_cuda_availability():
    """Test that PyTorch can access CUDA"""
    result = run_command(
        "docker-compose run --rm dm-align-trainer "
        "python -c 'import torch; assert torch.cuda.is_available()'"
    )
    assert result.returncode == 0

def test_unsloth_import():
    """Test that Unsloth imports correctly"""
    result = run_command(
        "docker-compose run --rm dm-align-trainer "
        "python -c 'from unsloth import FastLanguageModel; print(\"OK\")'"
    )
    assert result.returncode == 0
    assert "OK" in result.stdout
```

#### Implementation Artifacts
- `docker/Dockerfile` - CUDA 12.6 base with PyTorch 2.7.0 + Unsloth
- `docker-compose.yml` - GPU passthrough configuration (service: `training`)

#### Acceptance Criteria
- ✅ Docker image builds in < 10 minutes
- ✅ GPU visible inside container via `nvidia-smi`
- ✅ PyTorch reports CUDA available
- ✅ Unsloth imports without errors
- ✅ All version checks pass (PyTorch 2.7.0, Unsloth 2026.4.6)

---

### TASK 2: Teacher Phase - Synthetic Data Generation

**Objective**: Generate 1,500 DM-aligned synthetic samples using Qwen3.5-27B GGUF

#### Test 2.1: Data Generation Quality
```python
# src/tests/test_teacher.py
def test_sample_structure():
    """Test that generated samples have correct structure"""
    sample = generate_sample("Test question?")
    assert "conversations" in sample
    assert len(sample["conversations"]) == 2
    assert sample["conversations"][0]["role"] == "user"
    assert sample["conversations"][1]["role"] == "assistant"

def test_dm_keywords_present():
    """Test that DM keywords appear in generated responses"""
    required_keywords = ["Material Conditions", "Contradiction", "Superstructure"]
    response = generate_dm_response("Is capitalism inevitable?")
    for keyword in required_keywords:
        assert keyword in response, f"Missing keyword: {keyword}"

def test_retry_logic():
    """Test that invalid samples are regenerated"""
    call_count = [0]
    def mock_generate():
        call_count[0] += 1
        if call_count[0] < 3:
            return "Invalid response without DM keywords"
        return "Valid response with Material Conditions and Contradiction"
    
    result = generate_with_retry(mock_generate, max_retries=3)
    assert call_count[0] == 3
    assert "Material Conditions" in result

def test_batch_generation():
    """Test batch generation produces correct count"""
    questions = [f"Question {i}?" for i in range(50)]
    samples = generate_batch(questions, batch_size=10)
    assert len(samples) == 50
    assert all(is_valid_dm_sample(s) for s in samples)

def test_output_format():
    """Test that output is valid ShareGPT JSONL"""
    samples = generate_batch(["Test?"], batch_size=1)
    json_str = json.dumps(samples[0])
    parsed = json.loads(json_str)
    assert "conversations" in parsed
```

#### Implementation Artifacts
- `src/teacher/generate.py` - Main generation script
- `src/teacher/prompts.py` - DM prompt templates with CoT structure
- `src/teacher/validators.py` - Keyword validation and retry logic
- `src/teacher/sample_utils.py` - ShareGPT formatting utilities
- `scripts/run_teacher.sh` - Orchestration script

#### Configuration
```python
# src/teacher/config.py
TEACHER_CONFIG = {
    "model_path": "checkpoints/base_model/Qwen3.5-27B-Instruct-Q4_K_M.gguf",
    "n_gpu_layers": -1,  # Full GPU offload
    "n_ctx": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "target_samples": 1500,
    "batch_size": 50,
    "max_retries": 3,
    "required_keywords": [
        "Material Conditions",
        "Contradiction", 
        "Superstructure",
        "Dialectical"
    ]
}
```

#### Acceptance Criteria
- ✅ 1,500 samples generated in 4-6 hours
- ✅ 100% samples contain required DM keywords
- ✅ Output is valid ShareGPT JSONL format
- ✅ No OOM errors during generation
- ✅ Checkpointing works (can resume from interruption)

---

### TASK 3: SFT Training Phase

**Objective**: Train QLoRA adapter on synthetic dataset

#### Test 3.1: Training Configuration
```python
# src/tests/test_sft_training.py
def test_vram_usage_within_limits():
    """Test that VRAM usage stays under 30GB"""
    with vram_monitor() as monitor:
        run_sft_training_step()
        assert monitor.peak_vram_gb < 30, "VRAM exceeded 30GB limit"

def test_gradient_checkpointing_enabled():
    """Test that gradient checkpointing is active"""
    model = load_model_with_config()
    assert model.config.use_gradient_checkpointing == True

def test_lora_modules_correct():
    """Test that LoRA is applied to correct modules"""
    model = load_model_with_config()
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    for module in target_modules:
        assert module in model.peft_config["default"].target_modules

def test_training_converges():
    """Test that training loss decreases"""
    losses = run_training_epoch()
    assert losses[-1] < losses[0], "Loss did not decrease"
    assert losses[-1] < 0.5, "Final loss too high"

def test_adapter_saves_correctly():
    """Test that adapter saves and loads correctly"""
    model = train_adapter()
    model.save_pretrained("test_adapter")
    
    # Reload and verify
    loaded_model = load_adapter("test_adapter")
    assert compare_models(model, loaded_model)
```

#### Implementation Artifacts
- `src/student/train_sft.py` - SFT training script
- `src/student/config.py` - Training hyperparameters
- `src/utils/vram_monitor.py` - VRAM monitoring utilities
- `scripts/run_sft.sh` - Orchestration script

#### Configuration
```python
# src/student/config.py
SFT_CONFIG = {
    # Model
    "model_name": "unsloth/Qwen3.5-27B-Instruct-unsloth-bnb-4bit",
    "max_seq_length": 4096,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_quant_type": "nf4",
    "gradient_checkpointing": "unsloth",
    
    # LoRA
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    
    # Training
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "max_steps": 1000,
    "num_train_epochs": 3,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    
    # Output
    "output_dir": "checkpoints/lora_adapters/sft_adapter"
}
```

#### Acceptance Criteria
- ✅ Training completes in 2-3 hours
- ✅ Peak VRAM usage < 29GB (32GB GPU)
- ✅ Final training loss < 0.5
- ✅ Adapter saves as `.safetensors`
- ✅ Adapter reloads and produces same outputs

---

### TASK 4: DPO Pair Generation

**Objective**: Generate preference pairs for DPO training

#### Test 4.1: DPO Pair Quality
```python
# src/tests/test_dpo_pairs.py
def test_pair_structure():
    """Test that DPO pairs have correct structure"""
    pair = generate_dpo_pair(question)
    assert "chosen" in pair
    assert "rejected" in pair
    assert "question" in pair

def test_chosen_is_dm_aligned():
    """Test that chosen response is DM-aligned"""
    pair = generate_dpo_pair("Test question?")
    dm_keywords = ["Material", "Contradiction", "Dialectical"]
    has_keyword = any(kw in pair["chosen"] for kw in dm_keywords)
    assert has_keyword, "Chosen response not DM-aligned"

def test_rejected_is_different():
    """Test that rejected response differs from chosen"""
    pair = generate_dpo_pair("Test question?")
    assert pair["chosen"] != pair["rejected"]

def test_generation_count():
    """Test that correct number of pairs generated"""
    sft_samples = load_sft_dataset()
    dpo_pairs = generate_all_dpo_pairs(sft_samples)
    assert len(dpo_pairs) == len(sft_samples)
```

#### Implementation Artifacts
- `src/teacher/generate_dpo_pairs.py` - DPO pair generation
- `scripts/run_dpo_pair_generation.sh` - Orchestration script

#### Acceptance Criteria
- ✅ 1,500 DPO pairs generated
- ✅ All chosen responses are DM-aligned
- ✅ All rejected responses are different from chosen
- ✅ Output is valid JSONL format

---

### TASK 5: DPO Training Phase

**Objective**: Fine-tune with Direct Preference Optimization

#### Test 5.1: DPO Training
```python
# src/tests/test_dpo_training.py
def test_dpo_loss_decreases():
    """Test that DPO loss decreases during training"""
    losses = run_dpo_training_epoch()
    assert losses[-1] < losses[0], "DPO loss did not decrease"

def test_preference_alignment_improves():
    """Test that model prefers DM-aligned responses after DPO"""
    base_model = load_base_with_sft()
    dpo_model = train_dpo(base_model)
    
    improvement = measure_preference_alignment(dpo_model) - measure_preference_alignment(base_model)
    assert improvement > 0, "DPO did not improve alignment"

def test_dpo_adapter_saves():
    """Test that DPO adapter saves correctly"""
    model = train_dpo()
    model.save_pretrained("checkpoints/lora_adapters/dpo_adapter")
    
    assert os.path.exists("checkpoints/lora_adapters/dpo_adapter/scheduler.pt")
    assert os.path.exists("checkpoints/lora_adapters/dpo_adapter/adapter_model.safetensors")
```

#### Implementation Artifacts
- `src/student/train_dpo.py` - DPO training script
- `src/student/dpo_config.py` - DPO hyperparameters
- `scripts/run_dpo.sh` - Orchestration script

#### Configuration
```python
# src/student/dpo_config.py
DPO_CONFIG = {
    "base_model": "checkpoints/lora_adapters/sft_adapter",
    "beta": 0.1,
    "dpo_loss": "sigmoid",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,
    "max_steps": 500,
    "warmup_steps": 50,
    "lr_scheduler_type": "cosine",
    "output_dir": "checkpoints/lora_adapters/dpo_adapter"
}
```

#### Acceptance Criteria
- ✅ DPO training completes in 1-2 hours
- ✅ DPO loss decreases monotonically
- ✅ Preference alignment improves vs SFT-only model
- ✅ Adapter saves correctly

---

### TASK 6: Validation & Export

**Objective**: Validate model performance and export to GGUF

#### Test 6.1: Validation
```python
# src/tests/test_validation.py
def test_liberal_trap_questions():
    """Test model responses on 20 liberal-trap questions"""
    trap_questions = load_liberal_trap_questions()  # 20 questions
    model = load_final_model()
    
    results = []
    for question in trap_questions:
        response = model.generate(question)
        is_dm_aligned = validate_dm_alignment(response)
        results.append(is_dm_aligned)
    
    alignment_rate = sum(results) / len(results)
    assert alignment_rate >= 0.8, f"Alignment rate {alignment_rate} < 80%"

def test_export_gguf():
    """Test that GGUF export completes successfully"""
    model = load_final_model()
    output_path = export_to_gguf(model, quantization="Q4_K_M")
    
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    assert output_path.endswith(".gguf")

def test_gguf_loads_and_runs():
    """Test that exported GGUF can be loaded and used"""
    from llama_cpp import Llama
    
    llm = Llama(
        model_path="output/merged/dm-align-qwen3.5-27b-Q4_K_M.gguf",
        n_gpu_layers=-1
    )
    
    response = llm("Test question?", max_tokens=100)
    assert len(response["choices"][0]["text"]) > 50
```

#### Implementation Artifacts
- `src/utils/vram_monitor.py` - VRAM monitoring utilities
- `scripts/run_e2e_tests.sh` - E2E test orchestration script
- `src/utils/export_utils.py` - GGUF export utilities (Phase 2)
- `src/student/validate_and_export.py` - Validation script (Phase 2)
- `scripts/run_validation.sh` - Orchestration script (Phase 2)

#### Acceptance Criteria
- ✅ 80%+ alignment rate on 20 liberal-trap questions
- ✅ GGUF export completes without OOM
- ✅ Exported GGUF loads in llama.cpp
- ✅ Exported model produces coherent responses

---

## 4. Docker Infrastructure Implementation

### 4.1 Dockerfile

```dockerfile
# Optimized for RTX 5090 (Blackwell) + Unsloth
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

# Install system dependencies + CUDA headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up Virtual Env
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.7.0 with CUDA 12.6 support (latest stable for cu126)
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install Unsloth (pinned to required version)
RUN pip install --no-cache-dir unsloth==2026.4.6

# Install ML Stack (Updated for 2026 compatibility)
RUN pip install --no-cache-dir \
    transformers>=4.46.0 \
    datasets>=3.0.0 \
    peft>=0.13.0 \
    trl>=0.12.0 \
    accelerate>=1.0.0 \
    bitsandbytes>=0.44.0 \
    sentencepiece

# llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python==0.3.20 --no-cache-dir

WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

CMD ["bash"]
```

### 4.2 docker-compose.yml

```yaml
services:
  training:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: ml-training
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./src:/app/src
      - ./configs:/app/configs
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./scripts:/app/scripts
    environment:
      - HF_HOME=/root/.cache/huggingface
      - TRANSFORMERS_CACHE=/root/.cache/huggingface
      - DATASETS_CACHE=/root/.cache/huggingface
      - ACCELERATE_CACHE_DIR=/root/.cache/huggingface
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    shm_size: '16gb'
    stdin_open: true
    tty: true
    restart: unless-stopped
```

---

## 5. Execution Workflow

### Step 0: Prerequisites (Host)

1. **Install Docker Desktop for Windows**
   - Enable WSL2 backend
   - Allocate minimum 64GB RAM to WSL2

2. **Enable NVIDIA Container Toolkit**
   ```bash
   # In Docker Desktop Settings
   - Resources → GPU → Enable "Enable NVIDIA GPU support"
   ```

3. **Verify GPU Passthrough**
   ```bash
   docker run --gpus all nvidia/cuda:12.6.0 nvidia-smi
   # Should show RTX 5090
   ```

### Step 1: Build Docker Image

```bash
cd ml-lora-training
docker-compose build
```

**Expected Duration**: 8-10 minutes (first build)

### Step 2: Download Base Model

```bash
docker-compose run --rm training \
  huggingface-cli download \
  unsloth/Qwen3.5-27B-Instruct-unsloth-bnb-4bit \
  --local-dir checkpoints/base_model
```

**Expected Duration**: 15-20 minutes (~18GB download)

### Step 3: Run Teacher Generation (TASK 1)

```bash
./scripts/run_teacher.sh
```

**Expected Output**: `data/processed/sft_dataset.jsonl` (1,500 samples)
**Expected Duration**: 4-6 hours

### Step 4: Run SFT Training (TASK 2)

```bash
./scripts/run_sft.sh
```

**Expected Output**: `checkpoints/lora_adapters/sft_adapter/` (adapter directory)
**Expected Duration**: 2-3 hours

### Step 5: Generate DPO Pairs & Train DPO (TASK 3-4)

```bash
# Generate DPO pairs
./scripts/run_dpo_pair_generation.sh

# Run DPO training
./scripts/run_dpo.sh
```

**Expected Output**: `checkpoints/lora_adapters/dpo_adapter/` (adapter directory)
**Expected Duration**: 1-2 hours

### Step 6: Validate & Export (TASK 5 - Phase 2)

```bash
docker-compose run --rm training \
  python src/student/validate_and_export.py
```

**Expected Output**: `output/merged/dm-align-qwen3.5-27b-Q4_K_M.gguf`
**Expected Duration**: 30-45 minutes
**Status**: Planned for Phase 2

---

## 6. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OOM errors during training | Medium | High | max_seq_length=4096, gradient_checkpointing="unsloth" |
| VRAM fragmentation | Medium | Medium | Batch size=1, gradient_accumulation=4 |
| Docker GPU issues | Low | High | WSL2 backend, latest NVIDIA Container Toolkit |
| Training instability | Low | Medium | Mixed precision (BF16), cosine LR scheduler |
| Poor DM alignment | Medium | High | DPO phase + validation retry logic |
| GGUF export OOM | Medium | High | Ensure 64GB system RAM available |

---

## 7. Success Metrics

### Quantitative Metrics
- ✅ **VRAM Usage**: Peak < 29GB during training (32GB GPU)
- ✅ **Training Loss**: Final SFT loss < 0.5
- ✅ **Alignment Rate**: ≥ 80% on 20 liberal-trap questions
- ✅ **Generation Time**: < 6 hours for 1,500 samples
- ✅ **Training Time**: < 5 hours total (SFT + DPO)

### Qualitative Metrics
- ✅ All generated samples contain DM keywords
- ✅ Model responses show dialectical reasoning
- ✅ Exported GGUF loads and runs correctly
- ✅ No OOM errors throughout pipeline

---

## 8. Appendix

### A. Version Verification Commands

```bash
# Inside container
docker-compose run --rm training bash -c "
  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'
  python -c 'import unsloth; print(f\"Unsloth: {unsloth.__version__}\")'
  python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'
  python -c 'import llama_cpp; print(\"llama-cpp-python: OK\")'
"
```

### B. Debugging Commands

```bash
# Check VRAM usage during training
nvidia-smi dmon -s u -c 1

# View container logs
docker-compose logs -f training

# Enter running container
docker-compose exec training bash
```

### C. Rollback Procedures

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker-compose down --rmi all

# Clean up volumes (WARNING: deletes all data)
docker-compose down -v
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | April 20, 2026 | Staff ML Architect | Initial architecture roadmap |

---

**END OF DOCUMENT**
