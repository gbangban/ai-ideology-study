#!/usr/bin/env bash
# GRPO v2 Training - Rule-based rewards (no LLM judge)
# Trains GRPO adapter on DM-aligned questions
#
# NOTE: GRPO is NOT supported in Unsloth Studio UI.
# This script runs after SFT training in Studio.
#
# Run inside Docker container:
#   docker exec -it ml-training bash -c "source /opt/venv/bin/activate && ./scripts/run_grpo.sh"
#
# Resumption:
#   ./scripts/run_grpo.sh --resume          # auto-resume from latest checkpoint
#   ./scripts/run_grpo.sh --resume 300      # resume from checkpoint-300
#   ./scripts/run_grpo.sh --list            # list available checkpoints

set -e

# Parse resume flag
RESUME=""
RESUME_STEP=""
LIST=""
for arg in "$@"; do
    case "$arg" in
        --resume)
            RESUME=1
            ;;
        --resume=*)
            RESUME=1
            RESUME_STEP="${arg#*=}"
            ;;
        --list)
            LIST=1
            ;;
    esac
done

echo "========================================="
echo "GRPO Training - Group Relative Policy Optimization"
echo "========================================="

# Configuration
BASE_MODEL="${BASE_MODEL:-/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/lora_adapters/grpo_adapter_v2}"
QUESTIONS_PATH="${QUESTIONS_PATH:-data/raw/questions.json}"

echo "Base model: $BASE_MODEL"
echo "Questions: $QUESTIONS_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# List checkpoints mode
if [ "$LIST" = "1" ]; then
    python3 -m src.student.train_grpo \
        --output-dir "$OUTPUT_DIR" \
        --find-checkpoint
    exit 0
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
python3 -c "import unsloth" 2>/dev/null || {
    echo "ERROR: Unsloth not available. Install: pip install unsloth"
    exit 1
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Experiment tracking via Trackio (configured via .env: TRACKIO_SERVER_URL, TRACKIO_PROJECT)
if [ -z "$TRACKIO_SERVER_URL" ]; then
    echo "WARNING: TRACKIO_SERVER_URL not set. Trackio logging will use local storage only."
else
    echo "Trackio: logging to $TRACKIO_SERVER_URL"
fi

# Build training command
TRAIN_CMD="python3 -m src.student.train_grpo \
    --base-model \"$BASE_MODEL\" \
    --output-dir \"$OUTPUT_DIR\" \
    --questions-path \"$QUESTIONS_PATH\""

if [ "$RESUME" = "1" ]; then
    if [ -n "$RESUME_STEP" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume-step $RESUME_STEP"
        echo "Resuming from checkpoint-$RESUME_STEP"
    else
        # Auto-detect latest checkpoint
        LATEST=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort | tail -1)
        if [ -n "$LATEST" ]; then
            RESUME_STEP=$(basename "$LATEST" | sed 's/checkpoint-//')
            BASE_MODEL="$LATEST"
            TRAIN_CMD="python3 -m src.student.train_grpo \
                --base-model \"$BASE_MODEL\" \
                --output-dir \"$OUTPUT_DIR\" \
                --questions-path \"$QUESTIONS_PATH\" \
                --resume-step $RESUME_STEP"
            echo "Auto-resuming from checkpoint-$RESUME_STEP ($LATEST)"
        else
            echo "No checkpoints found, starting fresh"
        fi
    fi
fi

# Run GRPO training
echo "Starting GRPO training..."
echo "Expected duration: 9-12 hours"
echo ""

eval "$TRAIN_CMD"

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
    echo "Resumption:"
    echo "  ./scripts/run_grpo.sh --resume         # auto-resume from latest"
    echo "  ./scripts/run_grpo.sh --resume 300     # resume from checkpoint-300"

    echo ""
    echo "Adapter files:"
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "  (no safetensors files found)"
else
    echo "ERROR: Output directory not created"
    exit 1
fi
