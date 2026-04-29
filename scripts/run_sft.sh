#!/bin/bash
# SFT Training Phase - QLoRA Supervised Fine-Tuning
#
# DEPRECATED: This script is replaced by Unsloth Studio UI.
# Use Studio -> Training section with configs/studio_sft_config.yaml instead.
# Keep as reference for headless/server deployment.
#
# Trains QLoRA adapter on synthetic DM-aligned dataset

set -e

echo "========================================="
echo "DEPRECATED: Use Unsloth Studio UI for SFT training"
echo "See configs/studio_sft_config.yaml for Studio config"
echo "========================================="
echo "SFT Training Phase - QLoRA Fine-Tuning"
echo "========================================="

# Configuration
DATASET_PATH="${DATASET_PATH:-data/processed/sft_dataset.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/lora_adapters/sft_adapter}"

echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please run the teacher phase first: ./scripts/run_teacher.sh"
    exit 1
fi

# Count samples
SAMPLE_COUNT=$(wc -l < "$DATASET_PATH")
echo "Found $SAMPLE_COUNT training samples"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run SFT training
echo "Starting QLoRA SFT training..."
echo "Expected duration: 2-3 hours"
echo ""

python3 -m src.student.train_sft \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR"

# Validate output
if [ -d "$OUTPUT_DIR" ]; then
    echo "========================================="
    echo "SFT Training Complete!"
    echo "Adapter saved to: $OUTPUT_DIR"
    echo "========================================="
    
    # List adapter files
    echo "Adapter files:"
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "  (no safetensors files found)"
else
    echo "ERROR: Output directory not created"
    exit 1
fi
