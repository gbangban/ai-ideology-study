#!/bin/bash
# GRPO Training Smoke Test — runs one training step inside the container
# Usage:
#   ./scripts/smoke_test_training.sh outcome    # v3 only
#   ./scripts/smoke_test_training.sh process    # v4 only
#   ./scripts/smoke_test_training.sh            # both tracks

set -e

TRACK="${1:-}"
NUM_PROMPTS="${2:-2}"

DOCKER="${DOCKER:-docker}"

run_smoke() {
    local track="$1"
    echo "========================================="
    echo "Smoke Test: GRPO ${track} (${NUM_PROMPTS} prompts)"
    echo "========================================="
    "$DOCKER" exec ml-training python3 -m src.student.smoke_test \
        --track "$track" \
        --num-prompts "$NUM_PROMPTS"
    echo ""
}

# Check container is running
if ! "$DOCKER" ps 2>/dev/null | grep -q ml-training; then
    echo "ERROR: ml-training container is not running"
    echo "Start with: docker compose up -d training"
    exit 1
fi

if [ "$TRACK" = "both" ] || [ -z "$TRACK" ]; then
    run_smoke "outcome"
    run_smoke "process"
else
    run_smoke "$TRACK"
fi

echo "========================================="
echo "Smoke test complete."
echo "========================================="
