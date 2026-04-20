# Test Suite - DM-Align Qwen 27B Training Pipeline

Comprehensive test suite for the Dialectical Materialist-aligned Qwen 3.5 27B training pipeline using QLoRA.

## Test Architecture

```
src/tests/
├── test_docker.py        # Docker infrastructure & GPU passthrough
├── test_teacher.py       # Synthetic data generation (teacher phase)
├── test_sft_training.py  # QLoRA supervised fine-tuning
├── test_dpo_training.py  # Direct preference optimization
└── test_e2e.py           # End-to-end integration tests
```

## Prerequisites

### Running Tests Locally (Host)

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
python3 -m pytest src/tests/ -v
```

### Running Tests in Docker (Recommended)

```bash
# Build and run tests inside container
docker-compose run --rm training python3 -m pytest src/tests/ -v
```

## Running the Full Test Suite

Use the provided script to run all tests in sequence:

```bash
./scripts/run_e2e_tests.sh
```

This script runs tests in three phases:
1. **SFT Unit Tests** - Validates QLoRA training configuration
2. **DPO Unit Tests** - Validates preference optimization setup
3. **E2E Integration Tests** - Full pipeline workflow validation

### Environment Variables

```bash
# Custom test directory
TEST_DIR=src/tests ./scripts/run_e2e_tests.sh

# Verbose output
VERBOSE=v ./scripts/run_e2e_tests.sh

# Quiet output
VERBOSE= ./scripts/run_e2e_tests.sh
```

## Test Coverage by Component

### 1. Docker Infrastructure (`test_docker.py`)

| Test | Purpose |
|------|---------|
| `test_docker_build` | Verifies Docker image builds successfully |
| `test_gpu_passthrough` | Confirms RTX 5090 visible in container |
| `test_pytorch_cuda_availability` | Validates CUDA access from PyTorch |
| `test_unsloth_import` | Ensures Unsloth imports correctly |
| `test_version_checks` | Verifies PyTorch 2.7.x, CUDA 12.x, Unsloth versions |

**Run individually:**
```bash
python3 -m pytest src/tests/test_docker.py -v
```

### 2. Teacher Phase - Data Generation (`test_teacher.py`)

| Test Class | Purpose |
|------------|---------|
| `TestSampleStructure` | Validates ShareGPT format (conversations, roles) |
| `TestDMKeywordValidation` | Ensures DM keywords present (Material Conditions, Contradiction, Superstructure, Dialectical) |
| `TestRetryLogic` | Tests regeneration on invalid samples |
| `TestBatchGeneration` | Validates batch processing and counts |
| `TestOutputFormat` | Confirms valid JSON/JSONL serialization |
| `TestPromptTemplates` | Verifies DM prompt templates include CoT structure |

**Run individually:**
```bash
python3 -m pytest src/tests/test_teacher.py -v
```

### 3. SFT Training (`test_sft_training.py`)

| Test Class | Purpose |
|------------|---------|
| `TestVRAMUsage` | Ensures VRAM stays under 30GB limit |
| `TestGradientCheckpointing` | Validates gradient checkpointing enabled |
| `TestLoRAModules` | Confirms LoRA applied to correct modules (q_proj, k_proj, v_proj, etc.) |
| `TestTrainingConvergence` | Tests loss decreases, final loss < 0.5 |
| `TestAdapterSaveLoad` | Validates adapter saves/loads correctly |
| `TestBatchConfiguration` | Confirms batch size=1, gradient_accumulation=4 |
| `TestQuantizationConfig` | Validates 4-bit NF4 quantization, bfloat16 dtype |
| `TestDatasetLoading` | Tests JSONL dataset loading and formatting |
| `TestTrainingHyperparameters` | Validates LR in range [1e-5, 5e-4], max_seq_length 2048-4096 |

**Run individually:**
```bash
python3 -m pytest src/tests/test_sft_training.py -v
```

### 4. DPO Training (`test_dpo_training.py`)

| Test Class | Purpose |
|------------|---------|
| `TestDPOPairStructure` | Validates chosen/rejected/question fields |
| `TestDPOAlignment` | Ensures chosen is DM-aligned, rejected differs |
| `TestDPOConfig` | Validates beta (0-1), sigmoid loss, LR < SFT LR |
| `TestDPOTraining` | Tests DPO loss decreases, training on pairs |
| `TestDPOAdapterSave` | Validates adapter saves with required files |
| `TestDPOHyperparameters` | Confirms batch size, gradient accumulation, max steps |
| `TestPreferenceAlignment` | Tests alignment metric and DPO improvement |
| `TestDPODataset` | Validates DPO pair loading and formatting |

**Run individually:**
```bash
python3 -m pytest src/tests/test_dpo_training.py -v
```

### 5. End-to-End Integration (`test_e2e.py`)

| Test Class | Purpose |
|------------|---------|
| `TestE2ESFTTraining` | Full SFT workflow with mock dataset |
| `TestE2EDPOTraining` | Full DPO workflow with mock pairs |
| `TestE2EAdapterSaveLoad` | Adapter persistence across phases |
| `TestE2EPipeline` | Complete pipeline from dataset to final adapter |

**Run individually:**
```bash
python3 -m pytest src/tests/test_e2e.py -v
```

## Test Categories

### Unit Tests (No GPU Required)

These tests use mocking and can run on any system:

```bash
# Configuration tests
python3 -m pytest src/tests/test_sft_training.py::TestLoRAModules -v
python3 -m pytest src/tests/test_sft_training.py::TestQuantizationConfig -v

# Validation tests
python3 -m pytest src/tests/test_teacher.py::TestDMKeywordValidation -v
python3 -m pytest src/tests/test_teacher.py::TestRetryLogic -v

# Format tests
python3 -m pytest src/tests/test_teacher.py::TestOutputFormat -v
python3 -m pytest src/tests/test_dpo_training.py::TestDPOPairStructure -v
```

### Integration Tests (Docker Required)

These tests require Docker but not necessarily GPU:

```bash
# Run in Docker container
docker-compose run --rm training python3 -m pytest src/tests/test_e2e.py -v
```

### Infrastructure Tests (GPU Required)

These tests require actual GPU hardware:

```bash
# Run in Docker with GPU access
docker-compose run --rm training python3 -m pytest src/tests/test_docker.py -v
```

## Expected Test Results

### On Host (No GPU, No Docker)

```
test_docker.py        # SKIP - requires Docker
test_teacher.py       # PASS - all unit tests
test_sft_training.py  # PASS - all unit tests
test_dpo_training.py  # PASS - all unit tests
test_e2e.py           # PASS - integration tests with mocks
```

### In Docker (No GPU)

```
test_docker.py        # PARTIAL - GPU tests fail
test_teacher.py       # PASS
test_sft_training.py  # PASS
test_dpo_training.py  # PASS
test_e2e.py           # PASS
```

### In Docker (With GPU)

```
test_docker.py        # PASS - all infrastructure tests
test_teacher.py       # PASS
test_sft_training.py  # PASS
test_dpo_training.py  # PASS
test_e2e.py           # PASS
```

## Test Validation Criteria

The test suite validates:

1. **Environment Setup**
   - Docker builds successfully
   - GPU passthrough works
   - PyTorch can access CUDA
   - All dependencies import correctly
   - Versions match requirements (PyTorch 2.7.x, CUDA 12.x, Unsloth 2026.4.6)

2. **Data Generation**
   - Samples have correct ShareGPT structure
   - DM keywords are validated (case-insensitive)
   - Retry logic regenerates invalid samples
   - Batch processing produces correct counts
   - Output is valid JSON/JSONL

3. **SFT Training**
   - VRAM usage stays under 30GB
   - Gradient checkpointing is enabled
   - LoRA targets all required modules
   - Training loss decreases
   - Final loss < 0.5
   - Adapter saves as .safetensors
   - 4-bit NF4 quantization configured
   - Batch size = 1 for VRAM efficiency

4. **DPO Training**
   - Pairs have chosen/rejected/question
   - Chosen responses are DM-aligned
   - Rejected responses differ from chosen
   - Beta parameter in valid range (0-1)
   - Learning rate lower than SFT
   - DPO loss decreases
   - Adapter saves with required files

5. **End-to-End**
   - Complete SFT workflow executes
   - Complete DPO workflow executes
   - Adapters save and load correctly
   - Pipeline produces expected outputs

## Debugging Failed Tests

### Common Issues

1. **Docker tests fail**
   ```bash
   # Check Docker is running
   docker ps
   
   # Check NVIDIA Container Toolkit
   docker run --gpus all nvidia/cuda:12.6.0 nvidia-smi
   ```

2. **Import errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt  # if exists
   pip install pytest pytest-mock
   ```

3. **Path errors**
   ```bash
   # Ensure running from project root
   cd /home/yao/projects/ml-lora-training
   ```

4. **GPU tests fail without GPU**
   ```bash
   # Run only CPU-compatible tests
   python3 -m pytest src/tests/ -v -k "not gpu and not docker"
   ```

## Adding New Tests

Follow the existing patterns:

1. **Unit tests** - Use `unittest.mock` for external dependencies
2. **Integration tests** - Use `tmp_path` fixture for temporary files
3. **E2E tests** - Create mock datasets with realistic structure

Example test structure:
```python
class TestNewComponent:
    def test_functionality(self, tmp_path, mock_fixture):
        # Arrange
        # Act
        result = function_under_test(...)
        # Assert
        assert expected_condition(result)
```

## Test Performance

| Test Suite | Typical Duration |
|------------|------------------|
| `test_teacher.py` | ~2 seconds |
| `test_sft_training.py` | ~3 seconds |
| `test_dpo_training.py` | ~2 seconds |
| `test_e2e.py` | ~5 seconds |
| `test_docker.py` | ~30 seconds (with Docker) |
| **Total** | **~42 seconds** |

## CI/CD Integration

For GitHub Actions or similar:

```yaml
- name: Run tests
  run: |
    pip install pytest pytest-mock
    python3 -m pytest src/tests/ -v --tb=short -k "not docker"
```

## References

- [Architecture Roadmap](../docs/architecture_roadmap.md)
- [Project Roadmap](../docs/roadmap.md)
- [Technical Details](../docs/current%20details.md)
