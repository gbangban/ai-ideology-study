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

# Install Unsloth (pinned to your required version)
RUN pip install --no-cache-dir unsloth==2026.4.6

# Install ML Stack (Updated for 2026 compatibility)
RUN pip install --no-cache-dir \
    transformers==4.46.1 \
    datasets==3.1.0 \
    peft==0.13.2 \
    trl==0.12.3 \
    accelerate==1.0.1 \
    bitsandbytes==0.44.1 \
    sentencepiece

# --- FIX: Match llama-cpp-python to CUDA 12.4+ ---
RUN CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python==0.3.20 --no-cache-dir

WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

CMD ["bash"]
