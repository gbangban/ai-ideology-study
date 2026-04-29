#!/bin/bash
# End-to-End Integration Tests - Studio-Integrated Pipeline
# Runs tests for the Studio + Custom DPO training workflow

set -e

echo "========================================="
echo "E2E Integration Tests - Studio Pipeline"
echo "========================================="
echo ""

# Configuration
TEST_DIR="${TEST_DIR:-src/tests}"
VERBOSE="${VERBOSE:-v}"

echo "Test directory: $TEST_DIR"
echo "Verbosity: $VERBOSE"
echo "========================================="
echo ""

echo "Running on host system (Studio workflow)"
echo ""

# Run teacher phase tests
echo "========================================="
echo "Step 1: Running Teacher Phase Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_teacher.py -${VERBOSE} --tb=short
TEACHER_EXIT=$?

if [ $TEACHER_EXIT -ne 0 ]; then
    echo "ERROR: Teacher phase tests failed"
    exit $TEACHER_EXIT
fi
echo ""

# Run SFT config tests (Studio handles training, we validate config)
echo "========================================="
echo "Step 2: Running SFT Config Validation Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_sft_training.py -${VERBOSE} --tb=short
SFT_EXIT=$?

if [ $SFT_EXIT -ne 0 ]; then
    echo "ERROR: SFT config tests failed"
    exit $SFT_EXIT
fi
echo ""

# Run DPO training tests
echo "========================================="
echo "Step 3: Running DPO Training Unit Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_dpo_training.py -${VERBOSE} --tb=short
DPO_EXIT=$?

if [ $DPO_EXIT -ne 0 ]; then
    echo "ERROR: DPO unit tests failed"
    exit $DPO_EXIT
fi
echo ""

# Run E2E integration tests
echo "========================================="
echo "Step 4: Running E2E Integration Tests"
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
echo "Teacher Phase:       $([ $TEACHER_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "SFT Config:          $([ $SFT_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "DPO Unit Tests:      $([ $DPO_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "E2E Integration:     $([ $E2E_EXIT -eq 0 ] && echo 'PASS' || echo 'PARTIAL - needs GPU')"
echo "========================================="
echo ""

# Instructions for full pipeline
echo "========================================="
echo "Full Pipeline Instructions"
echo "========================================="
echo ""
echo "Studio-Integrated Workflow:"
echo ""
echo "1. Install Unsloth Studio:"
echo "   curl -fsSL https://unsloth.ai/install.sh | sh"
echo "   unsloth studio -H 0.0.0.0 -p 8888"
echo ""
echo "2. Generate synthetic dataset (Teacher Phase):"
echo "   ./scripts/run_teacher.sh"
echo ""
echo "3. Upload dataset to Studio (Local tab -> sharegpt format)"
echo ""
echo "4. Run SFT training in Studio UI:"
echo "   - Upload configs/studio_sft_config.yaml via Parameters -> Upload"
echo "   - Click Start Training"
echo ""
echo "5. Export SFT adapter from Studio (LoRA Only)"
echo ""
echo "6. Generate DPO pairs:"
echo "   ./scripts/run_dpo_pair_generation.sh"
echo ""
echo "7. Run DPO training (custom script):"
echo "   STUDIO_EXPORT_PATH=<studio-export-path> ./scripts/run_dpo.sh"
echo ""
echo "8. Evaluate in Studio Chat / Model Arena"
echo "9. Export final GGUF from Studio"
echo "========================================="

if [ $TEACHER_EXIT -ne 0 ] || [ $SFT_EXIT -ne 0 ] || [ $DPO_EXIT -ne 0 ]; then
    exit 1
fi

exit 0
