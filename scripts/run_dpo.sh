#!/bin/bash
# DPO Training Phase - Direct Preference Optimization
# Trains DPO adapter on preference pairs
#
# NOTE: DPO is NOT supported in Unsloth Studio UI.
# This script runs after SFT training in Studio.
#
# Workflow:
#   1. Complete SFT in Studio UI
#   2. Export SFT adapter from Studio (LoRA Only export)
#   3. Run this script pointing to Studio export path
#   4. DPO produces final adapter for Studio Chat evaluation

set -e

echo "========================================="
echo "DPO Training Phase - Direct Preference Optimization"
echo "========================================="

# Configuration - supports both legacy paths and Studio exports
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-checkpoints/lora_adapters/sft_adapter}"
STUDIO_EXPORT_PATH="${STUDIO_EXPORT_PATH:-}"
DPO_PAIRS_PATH="${DPO_PAIRS_PATH:-data/processed/dpo_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/lora_adapters/dpo_adapter}"

# Use Studio export path if provided
if [ -n "$STUDIO_EXPORT_PATH" ]; then
    echo "Using Studio export path: $STUDIO_EXPORT_PATH"
    SOURCE_PATH="$STUDIO_EXPORT_PATH"
else
    SOURCE_PATH="$SFT_ADAPTER_PATH"
fi

echo "SFT Adapter: $SOURCE_PATH"
echo "DPO Pairs: $DPO_PAIRS_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Check if SFT adapter exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "ERROR: SFT adapter not found at $SOURCE_PATH"
    echo ""
    echo "If using Studio export, set STUDIO_EXPORT_PATH:"
    echo "  STUDIO_EXPORT_PATH=~/.unsloth/studio/exports/my-sft-run ./scripts/run_dpo.sh"
    echo ""
    echo "Otherwise, ensure SFT training completed first."
    exit 1
fi

# Check if DPO pairs exist
if [ ! -f "$DPO_PAIRS_PATH" ]; then
    echo "ERROR: DPO pairs not found at $DPO_PAIRS_PATH"
    echo "Please run DPO pair generation first: ./scripts/run_dpo_pair_generation.sh"
    exit 1
fi

# Count pairs
PAIR_COUNT=$(wc -l < "$DPO_PAIRS_PATH")
echo "Found $PAIR_COUNT DPO pairs"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run DPO training
echo "Starting DPO training..."
echo "Expected duration: 1-2 hours"
echo ""

python3 -m src.student.train_dpo \
    --sft-adapter-path "$SOURCE_PATH" \
    --dpo-pairs-path "$DPO_PAIRS_PATH" \
    --output-dir "$OUTPUT_DIR"

# Validate output
if [ -d "$OUTPUT_DIR" ]; then
    echo "========================================="
    echo "DPO Training Complete!"
    echo "Adapter saved to: $OUTPUT_DIR"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Import final adapter into Studio Chat for evaluation"
    echo "  2. Use Studio Model Arena to compare SFT vs DPO models"
    echo "  3. Export final GGUF from Studio for deployment"

    # List adapter files
    echo ""
    echo "Adapter files:"
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "  (no safetensors files found)"
    ls -lh "$OUTPUT_DIR"/scheduler.pt 2>/dev/null || echo "  (no scheduler.pt found)"
else
    echo "ERROR: Output directory not created"
    exit 1
fi
