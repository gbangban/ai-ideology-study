# Dockerfile for ML Training with QLoRA
# CUDA 12.6 + PyTorch 2.7.0 + Unsloth 2026.4.6
# Optimized for RTX 5090 (32GB VRAM)

FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment
ENV PYTHON_VERSION=3.11
ENV PATH="/usr/lib/python3.11:${PATH}"

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install core packages
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.7.0 with CUDA 12.1 (compatible with CUDA 12.6 runtime)
# Note: PyTorch 2.7.0 supports CUDA up to 12.8, using CUDA 12.1 wheel for compatibility
RUN pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth 2026.4.6 (latest as of April 2026)
RUN pip install unsloth==2026.4.6

# Install additional ML dependencies
RUN pip install \
    transformers==5.5.4 \
    datasets==3.2.0 \
    peft==0.15.0 \
    trl==0.15.0 \
    accelerate==1.0.0 \
    bitsandbytes==0.45.0 \
    scipy==1.14.1 \
    numpy==2.1.0 \
    tqdm==4.67.0 \
    pyyaml==6.0.2 \
    pytest==8.3.4

# Install llama-cpp-python for GGUF export
# Building from source for CUDA 12.6 compatibility
RUN pip install llama-cpp-python==0.3.20 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Set working directory
WORKDIR /app

# Mount point for project code
# Project will be mounted at runtime via docker-compose

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["bash"]
