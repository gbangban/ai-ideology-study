#!/bin/bash
# DPO Pair Generation Phase
# Generates preference pairs from SFT dataset

set -e

echo "========================================="
echo "DPO Pair Generation Phase"
echo "========================================="

# Configuration
SFT_DATASET_PATH="${SFT_DATASET_PATH:-data/processed/sft_dataset.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-data/processed/dpo_pairs.jsonl}"

echo "SFT Dataset: $SFT_DATASET_PATH"
echo "Output: $OUTPUT_PATH"
echo "========================================="

# Check if SFT dataset exists
if [ ! -f "$SFT_DATASET_PATH" ]; then
    echo "ERROR: SFT dataset not found at $SFT_DATASET_PATH"
    echo "Please run the teacher phase first: ./scripts/run_teacher.sh"
    exit 1
fi

# Count samples
SAMPLE_COUNT=$(wc -l < "$SFT_DATASET_PATH")
echo "Found $SAMPLE_COUNT SFT samples to convert"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Run DPO pair generation
echo "Generating DPO preference pairs..."
python3 -m src.teacher.generate_dpo_pairs \
    --sft-dataset-path "$SFT_DATASET_PATH" \
    --output-path "$OUTPUT_PATH"

# Validate output
if [ -f "$OUTPUT_PATH" ]; then
    PAIR_COUNT=$(wc -l < "$OUTPUT_PATH")
    echo "========================================="
    echo "DPO Pair Generation Complete!"
    echo "Output: $OUTPUT_PATH"
    echo "Pairs generated: $PAIR_COUNT"
    echo "========================================="
else
    echo "ERROR: Output file not created"
    exit 1
fi
