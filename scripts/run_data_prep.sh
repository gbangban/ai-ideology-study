#!/usr/bin/env bash
set -euo pipefail

# Full data prep pipeline:
# 1. Convert parquet to JSON, clean traces, split SFT/DPO
# 2. Build trace-aligned SFT dataset
# 3. Generate rejected responses
# 4. Generate interleaved DPO pairs

PARQUET_PATH="${PARQUET_PATH:-/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/recipe_ml-1500-v1/parquet-files/batch_00000.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
SFT_COUNT="${SFT_COUNT:-1250}"

echo "=== Step 1: Convert parquet, clean traces, split ==="
python3 -m src.teacher.convert_full_dataset \
    --parquet-path "$PARQUET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --sft-count "$SFT_COUNT"

echo ""
echo "=== Step 2: Build trace-aligned SFT dataset ==="
python3 -m src.teacher.build_sft_dataset \
    --input "$OUTPUT_DIR/sft_dataset.json" \
    --output "$OUTPUT_DIR/sft_dataset.jsonl"

echo ""
echo "=== Step 3: Generate rejected responses ==="
python3 -m src.teacher.generate_rejected_responses \
    --input "$OUTPUT_DIR/dpo_dataset.json" \
    --output "$OUTPUT_DIR/rejected_responses.jsonl"

echo ""
echo "=== Step 4: Generate interleaved DPO pairs ==="
python3 -m src.teacher.generate_dpo_pairs \
    --dpo-data "$OUTPUT_DIR/dpo_dataset.json" \
    --rejections "$OUTPUT_DIR/rejected_responses.jsonl" \
    --output "$OUTPUT_DIR/dpo_pairs.jsonl"

echo ""
echo "=== Data prep complete ==="
echo "  SFT dataset: $OUTPUT_DIR/sft_dataset.jsonl"
echo "  DPO pairs:   $OUTPUT_DIR/dpo_pairs.jsonl"
