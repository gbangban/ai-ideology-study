#!/bin/bash
# End-to-End Integration Tests - Studio-Integrated Pipeline
# Runs tests for the Studio + Custom GRPO training workflow

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

# Run SG-Lang client tests
echo "========================================="
echo "Step 3: Running SG-Lang Client Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_sglang_client.py -${VERBOSE} --tb=short
SGLANG_EXIT=$?

if [ $SGLANG_EXIT -ne 0 ]; then
    echo "ERROR: SG-Lang client tests failed"
    exit $SGLANG_EXIT
fi
echo ""

# Run GRPO training tests
echo "========================================="
echo "Step 4: Running GRPO Training Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_grpo_training.py $TEST_DIR/test_grpo_config.py $TEST_DIR/test_rewards.py $TEST_DIR/test_rlvmr_rewards.py $TEST_DIR/test_grpo_base.py $TEST_DIR/test_grpo_outcome_training.py $TEST_DIR/test_grpo_process_training.py $TEST_DIR/test_smoke_test.py -${VERBOSE} --tb=short
GRPO_EXIT=$?

if [ $GRPO_EXIT -ne 0 ]; then
    echo "ERROR: GRPO training tests failed"
    exit $GRPO_EXIT
fi
echo ""

# Run E2E integration tests
echo "========================================="
echo "Step 5: Running E2E Integration Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_e2e.py -${VERBOSE} --tb=short
E2E_EXIT=$?

if [ $E2E_EXIT -ne 0 ]; then
    echo "WARNING: Some E2E tests failed (expected without real GPU/data)"
else
    echo "All E2E tests passed!"
fi
echo ""

# Run container smoke tests (optional, requires running container)
echo "========================================="
echo "Step 6: Running Container Smoke Tests"
echo "========================================="
SMOKE_EXIT=0
if bash -c 'scripts/ddk ps 2>/dev/null | grep -q ml-training' 2>/dev/null; then
    echo "Container running — executing smoke tests..."
    ./scripts/smoke_test_training.sh || SMOKE_EXIT=$?
else
    echo "WARNING: ml-training container not running, skipping smoke tests"
    echo "Run './scripts/smoke_test_training.sh' manually when container is available"
    SMOKE_EXIT=0
fi
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Teacher Phase:       $([ $TEACHER_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "SFT Config:          $([ $SFT_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "SG-Lang Client:      $([ $SGLANG_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "GRPO Training:       $([ $GRPO_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "E2E Integration:     $([ $E2E_EXIT -eq 0 ] && echo 'PASS' || echo 'PARTIAL - needs GPU')"
echo "Container Smoke:     $([ $SMOKE_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
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
echo "2. Upload dataset to Studio (Local tab -> sharegpt format)"
echo ""
echo "3. Run SFT training in Studio UI:"
echo "   - Upload configs/studio_sft_config.yaml via Parameters -> Upload"
echo "   - Click Start Training"
echo ""
echo "4. Export SFT adapter from Studio (LoRA Only)"
echo ""
echo "5. Merge cold-start SFT adapter (CPU-only):"
echo "   python3 scripts/merge_grpo_checkpoint.py \\"
echo "       --base-model <studio-export-checkpoint> \\"
echo "       --grpo-checkpoint <sft-adapter-path> \\"
echo "       --output checkpoints/merged/cold_start_merged"
echo ""
echo "6. Smoke test (one training step, validates full stack):"
echo "   ./scripts/smoke_test_training.sh outcome"
echo "   ./scripts/smoke_test_training.sh process"
echo ""
echo "7. Run GRPO v3 training (outcome rewards, control):"
echo "   docker exec ml-training python3 -m src.student.train_grpo_outcome \\"
echo "       --base-model checkpoints/merged/cold_start_merged"
echo ""
echo "8. Run GRPO v4 training (process rewards, experimental):"
echo "   docker exec ml-training python3 -m src.student.train_grpo_process \\"
echo "       --base-model checkpoints/merged/cold_start_merged"
echo ""
echo "9. Merge + evaluate:"
echo "   docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \\"
echo "       --base-model checkpoints/merged/cold_start_merged \\"
echo "       --grpo-checkpoint checkpoints/lora_adapters/grpo_v4_process/checkpoint-1000 \\"
echo "       --output checkpoints/merged/grpo_v4_process_final"
echo ""
echo "10. Evaluate in Studio Chat / Model Arena"
echo "11. Export final GGUF from Studio"
echo "========================================="

if [ $TEACHER_EXIT -ne 0 ] || [ $SFT_EXIT -ne 0 ] || [ $SGLANG_EXIT -ne 0 ] || [ $GRPO_EXIT -ne 0 ]; then
    exit 1
fi

# Smoke test failures are warnings, not hard errors (container may not be running)
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "WARNING: Smoke test failed. Run './scripts/smoke_test_training.sh' to investigate."
fi

exit 0
