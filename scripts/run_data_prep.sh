#!/usr/bin/env bash
set -euo pipefail

# Full data prep pipeline:
# 1. Convert parquet to JSON, clean traces
# 2. Build trace-aligned SFT dataset

PARQUET_PATH="${PARQUET_PATH:-/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/recipe_ml-1500-v1/parquet-files/batch_00000.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"

echo "=== Step 1: Convert parquet, clean traces ==="
python3 -m src.teacher.convert_full_dataset \
    --parquet-path "$PARQUET_PATH" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 2: Build trace-aligned SFT dataset ==="
python3 -m src.teacher.build_sft_dataset \
    --input "$OUTPUT_DIR/sft_dataset.json" \
    --output "$OUTPUT_DIR/sft_dataset.jsonl"

echo ""
echo "=== Data prep complete ==="
echo "  SFT dataset: $OUTPUT_DIR/sft_dataset.jsonl"
