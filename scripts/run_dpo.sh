#!/bin/bash
# DPO Training Phase - Direct Preference Optimization
# Trains DPO adapter on preference pairs

set -e

echo "========================================="
echo "DPO Training Phase - Direct Preference Optimization"
echo "========================================="

# Configuration
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-checkpoints/lora_adapters/sft_adapter}"
DPO_PAIRS_PATH="${DPO_PAIRS_PATH:-data/processed/dpo_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/lora_adapters/dpo_adapter}"

echo "SFT Adapter: $SFT_ADAPTER_PATH"
echo "DPO Pairs: $DPO_PAIRS_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Check if SFT adapter exists
if [ ! -d "$SFT_ADAPTER_PATH" ]; then
    echo "ERROR: SFT adapter not found at $SFT_ADAPTER_PATH"
    echo "Please run SFT training first: ./scripts/run_sft.sh"
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
    --sft-adapter-path "$SFT_ADAPTER_PATH" \
    --dpo-pairs-path "$DPO_PAIRS_PATH" \
    --output-dir "$OUTPUT_DIR"

# Validate output
if [ -d "$OUTPUT_DIR" ]; then
    echo "========================================="
    echo "DPO Training Complete!"
    echo "Adapter saved to: $OUTPUT_DIR"
    echo "========================================="
    
    # List adapter files
    echo "Adapter files:"
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "  (no safetensors files found)"
    ls -lh "$OUTPUT_DIR"/scheduler.pt 2>/dev/null || echo "  (no scheduler.pt found)"
else
    echo "ERROR: Output directory not created"
    exit 1
fi
