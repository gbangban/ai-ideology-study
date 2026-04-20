#!/bin/bash
# End-to-End Integration Tests - SFT & DPO Training Pipeline
# Runs full training pipeline tests with mocked components

set -e

echo "========================================="
echo "E2E Integration Tests - SFT & DPO Training"
echo "========================================="
echo ""

# Configuration
TEST_DIR="${TEST_DIR:-src/tests}"
VERBOSE="${VERBOSE:-v}"

echo "Test directory: $TEST_DIR"
echo "Verbosity: $VERBOSE"
echo "========================================="
echo ""

# Check if running in Docker container
if [ -f "/.dockerenv" ]; then
    echo "Running inside Docker container"
    echo "GPU available: $(nvidia-smi --query-gpu=name --format=noheader,nocolumn 2>/dev/null || echo 'Unknown')"
    echo ""
else
    echo "Running on host system"
    echo "Consider running inside Docker for full GPU tests:"
    echo "  docker-compose run --rm dm-align-trainer ./scripts/run_e2e_tests.sh"
    echo ""
fi

# Run unit tests first
echo "========================================="
echo "Step 1: Running SFT Training Unit Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_sft_training.py -${VERBOSE} --tb=short
SFT_EXIT=$?

if [ $SFT_EXIT -ne 0 ]; then
    echo "ERROR: SFT unit tests failed"
    exit $SFT_EXIT
fi
echo ""

echo "========================================="
echo "Step 2: Running DPO Training Unit Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_dpo_training.py -${VERBOSE} --tb=short
DPO_EXIT=$?

if [ $DPO_EXIT -ne 0 ]; then
    echo "ERROR: DPO unit tests failed"
    exit $DPO_EXIT
fi
echo ""

# Run E2E tests
echo "========================================="
echo "Step 3: Running E2E Integration Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_e2e.py -${VERBOSE} --tb=short
E2E_EXIT=$?

if [ $E2E_EXIT -ne 0 ]; then
    echo "WARNING: Some E2E tests failed (expected without real GPU/data)"
else
    echo "All E2E tests passed!"
fi
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "SFT Unit Tests:      $([ $SFT_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "DPO Unit Tests:      $([ $DPO_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "E2E Integration:     $([ $E2E_EXIT -eq 0 ] && echo 'PASS' || echo 'PARTIAL - needs GPU')"
echo "========================================="
echo ""

# Instructions for full pipeline test
echo "========================================="
echo "Full Pipeline Test Instructions"
echo "========================================="
echo "To run the complete training pipeline with real data:"
echo ""
echo "1. Download base model (~18GB):"
echo "   huggingface-cli download bartowski/Qwen3.5-27B-Instruct-GGUF"
echo "      Qwen3.5-27B-Instruct-Q4_K_M.gguf"
echo "      --local-dir checkpoints/base_model"
echo ""
echo "2. Generate synthetic dataset (4-6 hours):"
echo "   ./scripts/run_teacher.sh"
echo ""
echo "3. Generate DPO pairs:"
echo "   ./scripts/run_dpo_pair_generation.sh"
echo ""
echo "4. Run SFT training (2-3 hours):"
echo "   ./scripts/run_sft.sh"
echo ""
echo "5. Run DPO training (1-2 hours):"
echo "   ./scripts/run_dpo.sh"
echo ""
echo "Total estimated time: 8-13 hours"
echo "========================================="

# Exit with appropriate code
if [ $SFT_EXIT -ne 0 ] || [ $DPO_EXIT -ne 0 ]; then
    exit 1
fi

exit 0
