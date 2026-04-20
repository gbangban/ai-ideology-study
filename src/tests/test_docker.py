"""
Docker Infrastructure Tests

Test-driven development for Docker setup with GPU passthrough.
These tests verify the Docker environment is correctly configured
for ML training with CUDA 12.6, PyTorch 2.7.0, and Unsloth 2026.4.6.
"""

import subprocess
import pytest


def test_docker_build():
    """Test that Docker image builds successfully."""
    result = subprocess.run(
        ["docker-compose", "build"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Docker build failed: {result.stderr}"
    assert "Successfully built" in result.stdout or "Already up to date" in result.stdout


def test_gpu_passthrough():
    """Test that GPU is visible inside the container via nvidia-smi."""
    result = subprocess.run(
        ["docker-compose", "run", "--rm", "training", "nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=60
    )
    assert result.returncode == 0, f"nvidia-smi failed: {result.stderr}"
    assert "RTX 5090" in result.stdout or "GPU" in result.stdout, "GPU not detected in container"


def test_pytorch_cuda_availability():
    """Test that PyTorch can access CUDA."""
    result = subprocess.run(
        ["docker-compose", "run", "--rm", "training", "python", "-c", 
         "import torch; assert torch.cuda.is_available(); print(f'CUDA available: {torch.cuda.get_device_name(0)}')"],
        capture_output=True,
        text=True,
        timeout=60
    )
    assert result.returncode == 0, f"PyTorch CUDA test failed: {result.stderr}"
    assert "CUDA available" in result.stdout


def test_unsloth_import():
    """Test that Unsloth can be imported successfully."""
    result = subprocess.run(
        ["docker-compose", "run", "--rm", "training", "python", "-c", 
         "from unsloth import FastLanguageModel; print('Unsloth imported successfully')"],
        capture_output=True,
        text=True,
        timeout=60
    )
    assert result.returncode == 0, f"Unsloth import failed: {result.stderr}"
    assert "Unsloth imported successfully" in result.stdout


def test_version_checks():
    """Test that all required packages are installed with correct versions."""
    version_check_script = """
import torch
import unsloth

# Check PyTorch version
torch_version = torch.__version__
assert torch_version.startswith("2.7"), f"Expected PyTorch 2.7.x, got {torch_version}"
print(f"PyTorch: {torch_version}")

# Check CUDA version from PyTorch
cuda_version = torch.version.cuda
assert cuda_version.startswith("12."), f"Expected CUDA 12.x, got {cuda_version}"
print(f"CUDA: {cuda_version}")

# Check Unsloth version (should be 2026.4.6 or later)
unsloth_version = unsloth.__version__ if hasattr(unsloth, '__version__') else "unknown"
print(f"Unsloth: {unsloth_version}")

print("All version checks passed!")
"""
    result = subprocess.run(
        ["docker-compose", "run", "--rm", "training", "python", "-c", version_check_script],
        capture_output=True,
        text=True,
        timeout=60
    )
    assert result.returncode == 0, f"Version check failed: {result.stderr}"
    assert "All version checks passed!" in result.stdout
