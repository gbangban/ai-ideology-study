#!/bin/bash
# Teacher Phase - Synthetic Data Generation
# Generates DM-aligned training samples using Qwen GGUF model

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-checkpoints/base_model/Qwen3.5-27B-Instruct-Q4_K_M.gguf}"
QUESTIONS_PATH="${QUESTIONS_PATH:-data/raw/questions_clean.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-data/processed/sft_dataset.jsonl}"
CHECKPOINT_PATH="data/processed/checkpoint.json"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_RETRIES="${MAX_RETRIES:-3}"
BATCH_SIZE="${BATCH_SIZE:-50}"
N_CTX="${N_CTX:-4096}"

echo "========================================="
echo "Teacher Phase - DM Synthetic Data Generation"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Questions: $QUESTIONS_PATH"
echo "Output: $OUTPUT_PATH"
echo "Temperature: $TEMPERATURE"
echo "Max Retries: $MAX_RETRIES"
echo "Batch Size: $BATCH_SIZE"
echo "========================================="

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please download Qwen3.5-27B-Instruct-Q4_K_M.gguf first:"
    echo "  huggingface-cli download bartowski/Qwen3.5-27B-Instruct-GGUF Qwen3.5-27B-Instruct-Q4_K_M.gguf --local-dir checkpoints/base_model"
    exit 1
fi

# Check if questions file exists
if [ ! -f "$QUESTIONS_PATH" ]; then
    echo "ERROR: Questions file not found at $QUESTIONS_PATH"
    exit 1
fi

# Count questions
QUESTION_COUNT=$(wc -l < "$QUESTIONS_PATH")
echo "Found $QUESTION_COUNT questions to process"

# Check for existing checkpoint
if [ -f "$CHECKPOINT_PATH" ]; then
    COMPLETED=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_PATH'))['completed_count'])")
    echo "Resuming from checkpoint: $COMPLETED/$QUESTION_COUNT samples completed"
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Run generation
echo "Starting DM-aligned sample generation..."
python3 -m src.teacher.generate \
    --model-path "$MODEL_PATH" \
    --questions-path "$QUESTIONS_PATH" \
    --output-path "$OUTPUT_PATH" \
    --temperature "$TEMPERATURE" \
    --max-retries "$MAX_RETRIES" \
    --batch-size "$BATCH_SIZE" \
    --n-ctx "$N_CTX"

# Validate output
if [ -f "$OUTPUT_PATH" ]; then
    OUTPUT_COUNT=$(wc -l < "$OUTPUT_PATH")
    echo "========================================="
    echo "Generation complete!"
    echo "Output: $OUTPUT_PATH"
    echo "Samples generated: $OUTPUT_COUNT"
    echo "========================================="
    
    # Validate DM alignment
    echo "Validating DM alignment..."
    python3 -c "
import json
from src.teacher.validators import is_valid_dm_sample

with open('$OUTPUT_PATH', 'r') as f:
    samples = [json.loads(line) for line in f]

valid = sum(1 for s in samples if is_valid_dm_sample(s))
total = len(samples)
print(f'Valid DM-aligned samples: {valid}/{total} ({100*valid/total:.1f}%)')
"
else
    echo "ERROR: Output file not created"
    exit 1
fi
