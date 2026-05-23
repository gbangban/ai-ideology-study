#!/usr/bin/env bash
# GRPO Training - Group Relative Policy Optimization
# Trains GRPO adapter on DM-aligned questions
#
# NOTE: GRPO is NOT supported in Unsloth Studio UI.
# This script runs after SFT training in Studio.
#
# Run inside Docker container:
#   docker exec -it ml-training bash -c "source /opt/venv/bin/activate && ./scripts/run_grpo.sh"

set -e

echo "========================================="
echo "GRPO Training - Group Relative Policy Optimization"
echo "========================================="

# Configuration
BASE_MODEL="${BASE_MODEL:-/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/lora_adapters/grpo_adapter}"
QUESTIONS_PATH="${QUESTIONS_PATH:-data/raw/questions.json}"

echo "Base model: $BASE_MODEL"
echo "Questions: $QUESTIONS_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Base model not found at $BASE_MODEL"
    echo "Set BASE_MODEL to point to your SFT merged checkpoint."
    exit 1
fi

# Check if questions exist
if [ ! -f "$QUESTIONS_PATH" ]; then
    echo "ERROR: Questions not found at $QUESTIONS_PATH"
    exit 1
fi

# Count questions
QUESTION_COUNT=$(python3 -c "import json; print(len(json.load(open('$QUESTIONS_PATH'))))")
echo "Found $QUESTION_COUNT questions"

# Check dependencies
python3 -c "from trl import GRPOTrainer" 2>/dev/null || {
    echo "ERROR: TRL GRPOTrainer not available. Install: pip install trl>=0.15.0"
    exit 1
}

python3 -c "import unsloth" 2>/dev/null || {
    echo "ERROR: Unsloth not available. Install: pip install unsloth"
    exit 1
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize WandB
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. WandB logging will be disabled."
    export WANDB_MODE=offline
else
    echo "WandB: logging enabled"
fi

# Run GRPO training
echo "Starting GRPO training..."
echo "Expected duration: 9-12 hours"
echo ""

python3 -m src.student.train_grpo \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --questions-path "$QUESTIONS_PATH"

# Validate output
if [ -d "$OUTPUT_DIR" ]; then
    echo "========================================="
    echo "GRPO Training Complete!"
    echo "Adapter saved to: $OUTPUT_DIR"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Merge adapter for BF16 eval"
    echo "  2. Run eval suite: ./evals/scripts/run_finetuned_bf16.sh --tasks econcausal_task1_econ"

    echo ""
    echo "Adapter files:"
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "  (no safetensors files found)"
else
    echo "ERROR: Output directory not created"
    exit 1
fi
