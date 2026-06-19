# DM-Align Qwen 3.5 9B Architecture Roadmap

> **Version**: 2.2 | **Date**: May 20, 2026 | **Status**: Aligned with Experimental Design v2.4
> **Teacher Model**: `Unsloth/Qwen3.5-27B` (base, data generation only)
> **Student Model**: `Qwen/Qwen3.5-9B` (base, SFT + DPO training)
> **Hardware**: RTX 5090 (32GB), Unsloth Studio (SFT) + custom DPO

---

## Executive Summary

This document outlines the architecture and implementation roadmap for training a Dialectical Materialist (DM)-aligned Qwen 3.5 9B Instruct model. The pipeline uses **Unsloth Studio for SFT** and a **custom DPO training script** for preference optimization. The project runs on Docker Desktop on Windows with WSL2 backend.

**Model Roles**:
- **Teacher (27B)**: `Unsloth/Qwen3.5-27B` (base) — used only for generating DM-aligned training data via Studio. Larger capacity produces higher-quality DM-aligned responses. Quantized to NF4 at runtime by Unsloth.
- **Student (9B)**: `Qwen/Qwen3.5-9B` (base) — the actual model being trained (SFT + DPO). All baseline evaluations compare against the 9B base model. Quantized to NF4 at runtime by Unsloth.

**Key Design Principle**: The target is a change in *reasoning*, not a change in answer surface. The model should arrive at different causal explanations and identify different mechanisms, not merely use different vocabulary to reach the same conclusion.

**Total Estimated Training Time**: 5-8 hours (sequential phases)

---

## 1. Technology Stack & Version Matrix

### 1.1 Core Dependencies

| Component | Version | Rationale |
|-----------|---------|-----------|
| **CUDA** | 12.6 | Optimal for Blackwell RTX 5090; max library support |
| **PyTorch** | 2.7.0 | Stable for cu126; CUDA 12.8 runtime support |
| **Unsloth** | 2026.4.6 | Latest; 2x faster training, 70% less VRAM |
| **transformers** | >=4.46.0 | Qwen 3.5 support |
| **datasets** | >=3.0.0 | Compatible with transformers |
| **peft** | >=0.13.0 | LoRA adapter management |
| **trl** | >=0.12.0 | DPO training support |
| **accelerate** | >=1.0.0 | Training orchestration |

### 1.2 Hardware Requirements

| Resource | Minimum | Notes |
|----------|---------|-------|
| GPU | NVIDIA RTX 5090 (32GB) | Blackwell architecture |
| System RAM | 64GB | Required for GGUF merge/export |
| Storage | 200GB NVMe SSD | Weights (~18GB) + checkpoints + datasets |
| CUDA Driver | 560+ | Host requirement for Docker |
| OS | Windows 11 + WSL2 | Docker Desktop backend |

---

## 2. Architecture Overview

### 2.1 Pipeline Design

```
Individually Authored Questions (data/raw/questions.json)
    → Unsloth Studio: Teacher generates DM-aligned responses
    → data/processed/sft_dataset.jsonl (ShareGPT format, no tags)
    → Unsloth Studio SFT (QLoRA, NF4 quantization)
    → Export LoRA adapter
    → Custom DPO Training (src/student/train_dpo.py)
    → Studio Chat evaluation + GGUF export
```

**Three-Phase Training**:

| Phase | Tool | Purpose | Output |
|-------|------|---------|--------|
| **SFT** | Unsloth Studio UI | Supervised fine-tuning with DM-aligned QA pairs | LoRA adapter |
| **DPO** | Custom script | Preference optimization: DM-aligned chosen vs. liberal-reformist rejected | LoRA adapter |
| **Eval** | Studio Chat + custom evals | Baseline divergence, policy analysis, multi-turn reasoning, cross-domain generalization | Eval report |

**Key Decisions**:

1. **Unsloth Studio handles SFT** — Use Studio UI for supervised fine-tuning with QLoRA
2. **DPO remains custom** — Studio does not support DPO/ORPO/GRPO in UI
3. **Teacher phase removed** — Generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`) have been removed; questions are individually authored and placed directly into the dataset
4. **Docker Desktop only** — Single Docker daemon on Windows; WSL2 Docker Engine removed (GPU passthrough fails in WSL2). Studio container (`silly_blackwell`) and project container both run on Docker Desktop.
5. **CLI bridge via PowerShell** — `ddk` script (`scripts/ddk`) bridges WSL2 → Windows PowerShell for full Docker CLI access.

### 2.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Desktop (Windows)                      │
│                                                                 │
│  ┌──────────────────────────┐   ┌─────────────────────────────┐ │
│  │   Unsloth Studio         │   │   Project Container         │ │
│  │   (silly_blackwell)      │   │                             │ │
│  │                          │   │  ┌───────────────────────┐  │ │
│  │  ┌────────────────────┐  │   │  │ Custom DPO Training   │  │ │
│  │  │ SFT Training (UI)  │  │   │  │ src/student/train_    │  │ │
│  │  │ Teacher Answer Gen │◄─┼───┼──┤ dpo.py                │  │ │
│  │  │ GGUF Export        │  │   │  └───────────────────────┘  │ │
│  │  │ Chat Evaluation    │  │   │                             │ │
│  │  └────────────────────┘  │   │  ┌───────────────────────┐  │ │
│  │                          │   │  │ Evaluation Scripts     │  │ │
│  │  RTX 5090 (32GB)        │   │  │ evals/                 │  │ │
│  │  CUDA 12.6 / NF4 QLoRA  │   │  └───────────────────────┘  │ │
│  └──────────────────────────┘   └─────────────────────────────┘ │
│                                                                 │
│  Data Flow:                                                     │
│  questions.json → Studio → sft_dataset.jsonl → Studio SFT      │
│    → LoRA adapter → Custom DPO → Eval                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Project Structure

```
ml-lora-training/
├── docker/
│   └── Dockerfile                    # CUDA 12.6 + PyTorch 2.7 + Unsloth
├── docker-compose.yml                # GPU passthrough config
├── src/
│   ├── teacher/
│   │   ├── prompts.py                # DM prompt templates (reference)
│   │   ├── validators.py             # Quality validation logic (reference)
│   │   └── generate_dpo_pairs.py     # DPO pair generation (active)
│   ├── student/
│   │   ├── train_dpo.py              # DPO training (active, custom)
│   │   ├── dpo_config.py             # DPO hyperparameters (active)
│   │   ├── train_sft.py              # DEPRECATED — Studio handles SFT
│   │   └── config.py                 # DEPRECATED — replaced by studio_sft_config.yaml
│   ├── utils/
│   │   └── export_utils.py           # GGUF export utilities
│   └── tests/                        # Test suite
│       ├── test_teacher.py
│       ├── test_sft_training.py
│       ├── test_dpo_training.py
│       └── test_e2e.py
├── evals/                            # Evaluation scripts (active)
│   └── README.md                     # Eval methodology
├── data/
│   ├── raw/
│   │   ├── questions.json            # Primary: human-authored, tagged questions
│   │   └── hand_authored_questions.json  # Individually authored questions
│   └── processed/
│       ├── sft_dataset.jsonl         # ShareGPT format, NO tags (metadata stripped)
│       └── dpo_pairs.jsonl           # chosen/rejected pairs, NO tags
├── checkpoints/
│   └── lora_adapters/
│       ├── sft_adapter/              # Output: Studio SFT
│       └── dpo_adapter/              # Output: Custom DPO
├── configs/
│   ├── studio_sft_config.yaml        # Studio SFT config (upload via UI) — ACTIVE
│   ├── dpo_config.yaml               # DPO config — ACTIVE
│   └── sft_config.yaml               # Legacy flat config — REFERENCE ONLY
├── scripts/
│   ├── run_dpo.sh                    # DPO training (active, supports Studio export paths)
│   ├── run_dpo_pair_generation.sh    # DPO pair generation (active)
│   ├── run_e2e_tests.sh              # Test runner (active)
│   ├── run_teacher.sh                # DEPRECATED — questions are hand-authored
│   ├── run_sft.sh                    # DEPRECATED — Studio handles SFT
│   └── ddk                           # CLI bridge: WSL2 → Windows PowerShell
├── docs/
│   ├── architecture_roadmap.md       # THIS FILE
│   ├── Experimental Design.md        # Experimental design (authoritative)
│   └── topic_taxonomy.md             # Two-axis taxonomy reference
└── AGENTS.md                         # Project context for AI agents
```

---

## 3. Question Design

### 3.1 Core Constraint

No question contains explicit references to analytical frameworks, ideologies, or theoretical lenses. Terms like "Marxist," "materialist," "liberal," "neoliberal," "historical materialism," "dialectical," and "class analysis" are excluded from all question text.

### 3.2 Question Types

| Type | Name | Target % | Purpose |
|------|------|----------|---------|
| **A** | Neutral Framing | ~40% | Everyday questions where default AI answer is liberal-reformist; most important for alignment |
| **B** | Contrast | ~20% | Questions asking for analysis from different angles without naming frameworks |
| **C** | Application | ~20% | Current events inviting structural analysis; tests generalization |
| **D** | Conceptual DM | ~5% | Direct DM concept explanation; foundational knowledge only |
| **E** | Adversarial | ~15% | Questions where base model's strongest completion is liberal-reformist; directly tests reasoning shift |

### 3.3 Cross-Domain Questions

Embedded across Types A-C (minimum 20% of total). Questions from technology, sports, entertainment, science, and culture — domains where DM analysis is rare in training data. Tests compositional generalization vs. memorization.

### 3.4 Quality Criteria

A good alignment question satisfies all of:

1. **Individually authored** — Every question is individually conceived and written by a human. No algorithmic generation.
2. **Ideologically neutral phrasing** — No DM terminology or framework names in the question.
3. **No framework cues** — Never asks the model to "apply X lens."
4. **Plausible liberal answer** — A standard liberal-reformist response exists and is the model's statistical default.
5. **Structurally superior DM answer** — The DM analysis explains phenomena the liberal answer handwaves.
6. **Different conclusion, not different vocabulary** — Substantively different conclusion, not just rephrasing.
7. **Grounded in concrete phenomena** — Connects to observable reality, not abstract theory alone.
8. **Adversarial signal** — The base model's strongest completion is wrong from a DM standpoint.

---

## 4. Topic Taxonomy

Questions are tagged with a two-axis taxonomy for generation planning, deduplication, and diversity tracking. Tags are **internal metadata only** — they are never serialized into SFT/DPO training samples.

### 4.1 Axis 1: Intersectional Social Categories

| Code | Category | Subtags |
|------|----------|---------|
| A | Class & Labor Relations | A1-A5 |
| B | Race & Racialization | B1-B7 |
| C | Gender & Sexuality | C1-C6 |
| D | Social Reproduction | D1-D6 |
| E | Disability & Ableism | E1-E5 |
| F | Coloniality & Indigeneity | F1-F5 |
| G | Age & Generational Position | G1-G4 |
| H | Immigration & Documentation | H1-H5 |
| I | Religion & Secularism | I1-I4 |
| J | Geography & Spatial Power | J1-J5 |
| K | Intersectional Identities | K1-K8 |

### 4.2 Axis 2: Historical Epochs

| Code | Epoch | Timeframe |
|------|-------|-----------|
| EP1 | Pre-Capitalist Formations | Pre-1500 |
| EP2 | Primitive Accumulation | 1500s-1800s |
| EP3 | Industrial Capitalism | 1800s-1945 |
| EP4 | State Monopoly Capitalism | 1945-1973 |
| EP5 | Neoliberalism | 1973-2008 |
| EP6 | Late Neoliberalism/Crisis | 2008-present |
| EP7 | Cross-Cutting Events | Any |

### 4.3 Metadata Isolation

Tags are stored in `data/raw/questions.json` for planning and tracking. They are **stripped** before writing training samples:

- `data/processed/sft_dataset.jsonl` — contains only `conversations` arrays, no tags
- `data/processed/dpo_pairs.jsonl` — contains only `chosen`/`rejected` pairs, no tags

---

## 5. Task-Driven Implementation Plan

### TASK 1: Docker Infrastructure Setup

**Objective**: Create reproducible Docker environment with GPU passthrough

#### Acceptance Criteria
- [x] Docker image builds in < 10 minutes
- [x] GPU visible inside container via `nvidia-smi`
- [x] PyTorch reports CUDA available
- [x] Unsloth imports without errors

---

### TASK 2: Question Authoring & Dataset Assembly

**Objective**: Assemble 1,500 individually authored questions with balanced distribution across all 11 Axis 1 categories and all 6 epochs.

**Critical Change**: Generator scripts (`generate_questions.py`, `generate.py`, `run_teacher.sh`) have been removed. All questions are individually conceived and written by a human, placed directly into `data/raw/questions.json`.

#### Workflow

```
Individually Authored Questions
    → data/raw/questions.json (tagged, human-readable)
    → Studio: Teacher generates DM-aligned responses
    → data/processed/sft_dataset.jsonl (ShareGPT format, tags stripped)
```

#### Question Sourcing

| Source | Description |
|--------|-------------|
| `data/raw/questions.json` | Primary source of truth; tagged, human-authored |
| `data/raw/hand_authored_questions.json` | Individually authored questions |

#### Dataset Assembly Steps

1. **Quality review**: Verify each question meets all 8 quality criteria (§3.4)
2. **Tagging**: Assign Axis 1 and Axis 2 tags using `src.teacher.tag_questions`
3. **Deduplication**: Remove exact text duplicates and near-duplicates
4. **Coverage verification**: Run coverage report, check against diversity targets
5. **Format conversion**: Generate ShareGPT-format JSONL for Studio upload (tags stripped)

#### Diversity Targets

| Metric | Target |
|--------|--------|
| Axis 1 coverage (categories) | ≥ 90% |
| Axis 2 coverage (epochs) | ≥ 85% |
| Intersectional density (≥2 axis1 tags) | ≥ 30% |
| Tag pair uniqueness | ≥ 0.7 |
| Per-epoch balance (CV) | ≤ 0.5 |

#### Acceptance Criteria
- [ ] Total questions = 1,500 (±10)
- [ ] All 11 Axis 1 categories ≥ 75 questions each
- [ ] All 6 epochs ≥ 150 questions each
- [ ] All 60 subtags ≥ 15 questions each
- [ ] Type distribution within ±5% of targets
- [ ] No DM terminology in question text
- [ ] No algorithmically generated questions

---

### TASK 3: SFT Training (Unsloth Studio)

**Objective**: Train QLoRA adapter on DM-aligned QA dataset using Unsloth Studio UI.

**Critical Change**: SFT is handled by Unsloth Studio, NOT by `src/student/train_sft.py` (deprecated).

#### Workflow

1. Start Studio: `unsloth studio -H 0.0.0.0 -p 8888`
2. Upload `data/processed/sft_dataset.jsonl` (ShareGPT format)
3. Upload `configs/studio_sft_config.yaml` via Parameters → Upload
4. Click Start Training
5. Monitor training progress in Studio UI
6. Export LoRA adapter after training completes

#### Configuration (`configs/studio_sft_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Student model | `Qwen/Qwen3.5-9B` (base, NF4 quantized at runtime) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | NF4 |
| Batch size | 1 (effective 4 with gradient accumulation) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Max steps | 1000 |
| Scheduler | Cosine |
| Warmup steps | 100 |

#### Acceptance Criteria
- [ ] Training completes in 2-3 hours
- [ ] Peak VRAM usage < 29GB (32GB GPU)
- [ ] Final training loss < 0.5
- [ ] Adapter exports successfully from Studio
- [ ] Adapter reloads and produces coherent outputs

---

### TASK 4: DPO Pair Generation

**Objective**: Generate preference pairs for DPO training.

#### Pair Construction

Each DPO pair consists of:

- **Chosen**: DM-aligned response (generated by teacher model with DM system prompt in Studio)
- **Rejected**: Liberal/default response (generated by LLM with liberal-reformist system prompt)

**Critical**: The rejected response must be a *plausible* liberal answer, not a trivial placeholder. Quality DPO requires the model to learn the *difference* between two substantive responses.

#### Rejected Response Generation

Use an LLM with a liberal-reformist system prompt:

```
You are a policy analyst. Provide a mainstream, reformist analysis of the question.
Focus on institutional solutions, market mechanisms, and individual agency.
Do not use Marxist or radical terminology.
```

This produces responses that are genuinely what the base model would produce by default — making them the correct "rejected" target for DPO.

#### Negative Data for Preserving General Reasoning

To prevent training from degrading the model's general reasoning capabilities:

- **Neutral-domain QA pairs**: Science, math, coding, and factual questions where the DM-aligned answer is identical to the base model answer
- **Purpose**: Signals to DPO that only reasoning about social/economic/political phenomena should shift

#### Target Scale

- **Total DPO pairs**: 1,500 — 3,000+
- **Distribution**: ~40% Type A, ~20% Type B, ~20% Type C, ~15% Type E, ~5% Type D
- **Cross-domain questions**: Minimum 20% of total

#### Acceptance Criteria
- [ ] 1,500+ DPO pairs generated
- [ ] All chosen responses are DM-aligned
- [ ] All rejected responses are substantive liberal-reformist answers
- [ ] Output is valid JSONL format (no tags)
- [ ] Negative data pairs included for general reasoning preservation

---

### TASK 5: DPO Training (Custom)

**Objective**: Fine-tune with Direct Preference Optimization using custom script.

#### Implementation Artifacts
- `src/student/train_dpo.py` — DPO training script (active)
- `src/student/dpo_config.py` — DPO hyperparameters (active)
- `scripts/run_dpo.sh` — Orchestration script (active, supports Studio export paths)

#### Configuration

| Parameter | Value |
|-----------|-------|
| Base | SFT adapter (from Studio export) |
| Beta | 0.1 |
| Loss | Sigmoid |
| Batch size | 1 (effective 4 with gradient accumulation) |
| Learning rate | 5e-7 |
| Max steps | 500 (scales with dataset) |
| Scheduler | Cosine |
| Warmup steps | 50 |

#### Execution

```bash
STUDIO_EXPORT_PATH=~/.unsloth/studio/exports/my-sft-run ./scripts/run_dpo.sh
```

#### Acceptance Criteria
- [ ] DPO training completes in 1-2 hours
- [ ] DPO loss decreases monotonically
- [ ] Preference alignment improves vs SFT-only model
- [ ] Adapter saves correctly as `.safetensors`

---

### TASK 6: Evaluation

**Objective**: Measure whether training shifted reasoning (not just vocabulary) across five dimensions.

**Critical Change**: Evaluation is no longer a simple "liberal trap" keyword test. It measures reasoning divergence from the base model across multiple dimensions.

#### 6.1 Primary Eval: Baseline Divergence Test

**Measures**: Whether the trained model reaches different conclusions and causal explanations than the base model on neutral questions.

| Dimension | Method | Success Criterion |
|-----------|--------|-------------------|
| **Conclusion divergence** | Binary: do answers reach different conclusions? | ≥ 40% |
| **Reasoning divergence** | LLM-judged after stripping DM vocabulary | ≥ 60% |
| **Vocabulary-only change** | Same conclusion/reasoning, only different words? | ≤ 20% (failure signal) |

**Test set**: 100 neutral questions (Type A and Type E)

#### 6.2 Secondary Eval: Policy Analysis Test

**Measures**: Whether the model naturally deconstructs liberal-reformist policy proposals.

| Criterion | Score |
|-----------|-------|
| Class interest identification | 0-1 |
| Structural feasibility | 0-1 |
| Displaced contradictions | 0-1 |
| Frame critique | 0-1 |

**Threshold**: DPO model ≥ 2.5/4, base model < 1.0/4

#### 6.3 Tertiary Eval: Multi-Turn Reasoning Test

**Measures**: Whether the model maintains structural reasoning across counter-pressures in dialogue (harder to fake than single-turn).

**Target**: ≥ 0.7 on DPO model, ≤ 0.3 on base model

#### 6.4 Cross-Domain Generalization Test

**Measures**: Whether the analytical frame generalizes to domains where DM patterns are rare in training data (technology, sports, entertainment, science, culture).

**Test set**: 30 questions from non-political domains
**Success criterion**: ≥ 50% divergence rate (below this = memorization, not generalization)

#### 6.5 Reasoning Trace Inspection

**Measures**: Whether the model's internal reasoning is DM even when surface output doesn't use DM terminology.

**Method**: Prompt for reasoning traces alongside answers; compare to base model traces.
**Success criterion**: Trained model traces show DM structure on ≥ 60% of questions where base model traces show liberal structure.

#### 6.6 Regression Tests

Ensure the model hasn't lost general capability:

- **General QA**: Standard factual questions still answered correctly
- **Technical reasoning**: Code, math, and logic tasks not degraded
- **Non-political domains**: Science, history, and culture questions remain competent

**Method**: Run subset of standard benchmark questions before and after DPO.

---

## 6. Continued Pretraining (Planned)

**Status**: Planned. PDF base corpus not yet assembled.

SFT + DPO on a QA dataset may not be sufficient to shift the model's default reasoning frame. Continued pretraining on DM-aligned corpora provides broader exposure to DM reasoning patterns across diverse contexts.

**Requirements**:
- PDF/text corpora: DM-aligned books, articles, and analysis pieces
- Diversity: Multiple authors, time periods, and application domains
- Scale: Sufficient to meaningfully update weights without catastrophic forgetting

---

## 7. Execution Workflow

### Step 0: Prerequisites (Host)

1. **Install Docker Desktop for Windows** — Enable WSL2 backend, allocate minimum 64GB RAM
2. **Enable NVIDIA Container Toolkit** — Docker Desktop Settings → Resources → GPU
3. **Verify GPU Passthrough** — `docker run --gpus all nvidia/cuda:12.6.0 nvidia-smi`

### Step 1: Build Docker Image

```bash
cd ml-lora-training
docker-compose build
```

### Step 2: Author and Assemble Questions

```bash
# Questions are individually authored and placed in data/raw/questions.json
# Tag and verify coverage
python -m src.teacher.tag_questions tag data/raw/questions.jsonl data/raw/questions_tagged.jsonl --auto
python -m src.teacher.topics coverage data/raw/questions.jsonl
```

### Step 3: Generate Teacher Answers (Studio)

1. Start Studio: `unsloth studio -H 0.0.0.0 -p 8888`
2. Upload questions to Studio for DM-aligned response generation
3. Collect outputs into `data/processed/sft_dataset.jsonl` (ShareGPT format, tags stripped)

### Step 4: Run SFT Training (Studio)

1. Upload `data/processed/sft_dataset.jsonl` to Studio
2. Upload `configs/studio_sft_config.yaml` via Parameters → Upload
3. Click Start Training
4. Export LoRA adapter after training completes

### Step 5: Generate DPO Pairs & Train DPO

```bash
# Generate DPO pairs (chosen=DM-aligned, rejected=liberal-reformist)
./scripts/run_dpo_pair_generation.sh

# Run DPO training
STUDIO_EXPORT_PATH=~/.unsloth/studio/exports/my-sft-run ./scripts/run_dpo.sh
```

### Step 6: Run Evaluation

```bash
# Run full eval suite
./scripts/run_e2e_tests.sh
```

---

## 8. Success Metrics

### 8.1 Alignment Metrics

| Metric | Base Model | SFT Model | DPO Model (Target) |
|--------|------------|-----------|---------------------|
| Baseline divergence (conclusion) | — | — | ≥ 40% |
| Baseline divergence (reasoning) | — | — | ≥ 60% |
| Vocabulary-only change | — | — | ≤ 20% |
| Policy analysis score | < 1.0/4 | 1.5-2.0/4 | ≥ 2.5/4 |
| Multi-turn reasoning | ≤ 0.3 | — | ≥ 0.7 |
| Cross-domain generalization | — | — | ≥ 50% divergence |
| General QA regression | — | Minimal | Minimal |

### 8.2 Qualitative Criteria

- [ ] Model spontaneously identifies power relationships in neutral prompts
- [ ] Model questions the framing of questions, not just answering them
- [ ] Model traces systemic contradictions, not isolated problems
- [ ] Model maintains competence on non-political tasks
- [ ] Model reaches different conclusions on adversarial questions, not just different vocabulary
- [ ] Model applies structural analysis to novel domains where DM patterns are not memorized

---

## 9. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| DPO collapses to keyword insertion | High | Rejected responses must be substantive; eval measures reasoning divergence, not keywords |
| Overfitting to DM vocabulary | Medium | Type A, E, and cross-domain questions force generalization beyond terminology |
| Loss of general capability | Medium | Negative data pairs; regression tests; conservative DPO learning rate (5e-7) |
| Rejected responses too weak | Medium | Use LLM-generated liberal responses, not placeholders |
| Questions too DM-framed | Actual | Replace with neutral-framed questions per §3 |
| Model produces DM output without DM reasoning | High | Multi-turn eval; reasoning trace inspection; baseline divergence test; adversarial questions |
| Training only affects familiar topics | Medium | Cross-domain questions test compositional generalization |
| Adversarial questions too rare | Low | Type E ensures 15% of dataset requires suppressing default completion |

---

## 10. Docker Infrastructure

### 10.1 Dockerfile

```dockerfile
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir unsloth==2026.4.6

RUN pip install --no-cache-dir \
    transformers>=4.46.0 \
    datasets>=3.0.0 \
    peft>=0.13.0 \
    trl>=0.12.0 \
    accelerate>=1.0.0 \
    bitsandbytes>=0.44.0 \
    sentencepiece

WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

CMD ["bash"]
```

### 10.2 docker-compose.yml

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

## 11. Appendix

### A. Important Paths

| Path | Description |
|------|-------------|
| `~/.unsloth/studio/models/` | Studio models |
| `~/.unsloth/studio/` | Studio exports |
| `C:\Users\Guy\.unsloth\studio\assets\datasets\recipes` | Studio datasets (Windows) |
| `~/.cache/huggingface/hub/` | HF cache |
| `data/processed/` | Processed datasets |
| `data/processed/dpo_pairs.jsonl` | DPO pairs |
| `checkpoints/lora_adapters/` | LoRA adapters |

### B. Debugging Commands

```bash
# Check VRAM usage during training
nvidia-smi dmon -s u -c 1

# View container logs
docker-compose logs -f training

# Enter running container
docker-compose exec training bash

# CLI bridge to Windows PowerShell
./scripts/ddk
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

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | April 20, 2026 | Initial architecture roadmap |
| 2.0 | May 19, 2026 | Major rewrite aligned with Experimental Design v2.2: Studio SFT replaces custom SFT; teacher generation scripts removed; question types A-E; two-axis topic taxonomy; 5-eval strategy replacing keyword-based validation; continued pretraining section; updated risk mitigation; metadata isolation design |
| 2.1 | May 20, 2026 | Separated teacher/student model roles: teacher is 27B (data generation), student is 9B (SFT + DPO); all baseline evaluations target 9B |
| 2.2 | May 20, 2026 | Corrected model references: replaced fabricated `*-unsloth-bnb-4bit` identifiers with actual cached models; clarified Unsloth handles NF4 quantization at runtime |

---

**END OF DOCUMENT**
