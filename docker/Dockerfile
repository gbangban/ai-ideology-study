# Optimized for RTX 5090 (Blackwell) + Unsloth
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

# Install system dependencies + CUDA headers
# Use deadsnakes PPA for stable Python 3.11 (Ubuntu 22.04 default is rc1 without -dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up Virtual Env
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.7.0 with CUDA 12.6 support (latest stable for cu126)
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install Unsloth (pinned to your required version)
# NOTE: unsloth upgrades torch from 2.7.0+cu126 to 2.10.0+cu128. FA2 wheel must be
# installed AFTER this step to match the actual torch/cuda versions.
RUN pip install --no-cache-dir --upgrade unsloth unsloth_zoo

# Install prebuilt CUDA kernels (cu12, torch 2.10, Python 3.11)
# Must come after unsloth install since unsloth upgrades torch to 2.10+cu128.
# - flash_attn: FA2 gives 10-25% training speedup for LoRA GRPO on Qwen3.5-9B
# - causal_conv1d: Mamba/SSM selective scan kernels (cu12 generic wheel works with cu128 runtime)
# - flash-linear-attention: FLA long-context attention kernels (pure Python package)
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
    https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
    flash-linear-attention

# Install ML Stack (GRPOTrainer requires trl + mergekit)
# Force trl>=1.0.0 to drop the broken llm_blender dependency (incompatible with transformers 5.x).
# unsloth pins trl<=0.24.0 but all its trl imports (GRPOTrainer, GRPOConfig, SFTTrainer,
# unwrap_model_for_generation, etc.) exist unchanged in trl 1.x. The pip conflict warning
# is cosmetic — unsloth's string-match patches still apply to trl 1.5.1's source.
RUN pip install --no-cache-dir \
    transformers>=4.46.1 \
    datasets>=3.1.0 \
    peft>=0.13.2 \
    accelerate>=1.0.1 \
    bitsandbytes>=0.44.1 \
    sentencepiece \
    trackio[gpu] nvidia-ml-py \
    networkx \
    pytest \
    mergekit \
    && pip install --no-deps "trl>=1.0.0"

# --- FIX: Match llama-cpp-python to CUDA 12.4+ ---
RUN CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python==0.3.20 --no-cache-dir

WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

CMD ["bash"]
