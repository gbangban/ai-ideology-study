#!/bin/bash
# Run fine-tuned Q4_K_M GGUF evaluation (primary target)
# Runs first for maximum VRAM headroom and fastest feedback
#
# NOTE: Requires Studio container to be stopped (GPU must be free)
# Check with: nvidia-smi (should show <5GB VRAM used)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/runs/finetuned"

# Activate venv
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Run: python3 -m virtualenv $VENV_DIR --python=/usr/bin/python3.12"
    echo "Then: source $VENV_DIR/bin/activate && pip install -r requirements.txt"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Model paths - using safetensors export from Studio (NOT GGUF)
# GGUF qwen35 architecture is not yet supported by transformers
# Export fine-tuned model as safetensors from Studio UI first
MODEL_DIR="${STUDIO_EXPORT_PATH:-/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-safetensors}"

# Tasks
TASKS="mmlu_pro,gpqa_diamond_zeroshot,ifeval,humaneval,leaderboard_math_hard"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check GPU availability
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$VRAM_USED" ] && [ "$VRAM_USED" -gt 5000 ]; then
    echo "Warning: GPU VRAM usage is ${VRAM_USED}MB. Studio container may be running."
    echo "Stop Studio container before running evaluation."
    echo "Use: ./scripts/ddk stop silly_blackwell"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=== Fine-Tuned Model Evaluation ==="
echo "Model: $MODEL_DIR"
echo "Tasks: $TASKS"
echo "Output: $RESULTS_DIR"
echo ""

lm_eval --model hf \
  --model_args pretrained=$MODEL_DIR,dtype=bfloat16 \
  --tasks $TASKS \
  --batch_size auto:4 \
  --output_path "$RESULTS_DIR" \
  --log_samples

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $RESULTS_DIR"
