#!/bin/bash
# Run BF16 baseline evaluation (framework validation)
# Validates that local evaluation matches published scores
#
# NOTE: Requires Studio container to be stopped (GPU must be free)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/baseline/bf16"

# Activate venv
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Model paths (cached BF16 safetensors)
MODEL_DIR="/mnt/c/Users/Guy/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

# Tasks
TASKS="mmlu_pro,gpqa_diamond_zeroshot,ifeval,humaneval,leaderboard_math_hard"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check GPU availability
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$VRAM_USED" ] && [ "$VRAM_USED" -gt 5000 ]; then
    echo "Warning: GPU VRAM usage is ${VRAM_USED}MB. Studio container may be running."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=== BF16 Baseline Evaluation ==="
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
