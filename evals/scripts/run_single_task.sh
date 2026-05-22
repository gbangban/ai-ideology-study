#!/bin/bash
# Run a single evaluation task against a model
# Quick testing utility for running one task at a time
#
# Usage:
#   ./run_single_task.sh TASK MODEL_PATH [RESULTS_DIR]
#
# Examples:
#   ./run_single_task.sh humaneval /path/to/model
#   ./run_single_task.sh mmlu_pro /path/to/model results/custom
#   ./run_single_task.sh ifeval /path/to/model results/custom --batch_size 8

set -euo pipefail

# Source logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_logging.sh"

# Allow humaneval code_eval metric
export HF_ALLOW_CODE_EVAL="1"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 TASK MODEL_PATH [RESULTS_DIR] [--batch_size N]"
    echo ""
    echo "Run a single lm_eval task against a model."
    echo ""
    echo "Arguments:"
    echo "  TASK          Task name (e.g., humaneval, mmlu_pro, corr2cause, econcausal_task1_econ)"
    echo "  MODEL_PATH    Path to model directory (safetensors)"
    echo "  RESULTS_DIR   Output directory (default: results/single_runs)"
    echo ""
    echo "Options:"
    echo "  --batch_size N   Batch size (default: 4)"
    echo "  --help           Show this help message"
    echo ""
    echo "Available tasks:"
    echo "  mmlu_pro              - MMLU-Pro (knowledge-intensive multiple choice)"
    echo "  gpqa_diamond_zeroshot - GPQA Diamond (graduate-level science QA)"
    echo "  ifeval                - Instruction Following Evaluation"
    echo "  humaneval             - HumanEval (code generation)"
    echo "  leaderboard_math_hard - Math reasoning (hard subset)"
    echo "  econcausal_task1_econ - EconCausal Task 1 (economics)"
    echo "  econcausal_task1_finance - EconCausal Task 1 (finance)"
    echo "  econcausal_task2      - EconCausal Task 2 (context-dependent)"
    echo "  econcausal_task3      - EconCausal Task 3 (misinformation-robust)"
    echo "  corr2cause            - Corr2Cause (causal inference from correlations)"
    echo ""
    echo "Examples:"
    echo "  $0 humaneval /path/to/model"
    echo "  $0 mmlu_pro /path/to/model results/custom --batch_size 8"
    exit 1
fi

TASK="$1"
MODEL_DIR="$2"
RESULTS_DIR="${3:-$PROJECT_DIR/results/single_runs}"
BATCH_SIZE=4

# Parse remaining options
shift $((3 > $# ? $# : 3))
while [ $# -gt 0 ]; do
    case "$1" in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            $0
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate model path
if [ ! -d "$MODEL_DIR" ]; then
    log_error "Model directory not found: $MODEL_DIR"
    exit 1
fi

# Activate venv
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Set up log file
mkdir -p "$RESULTS_DIR"
EVAL_LOG="$RESULTS_DIR/eval.log"

log_section "Single Task Evaluation"
log_info "Task: $TASK"
log_info "Model: $MODEL_DIR"
log_info "Batch size: $BATCH_SIZE"
log_info "Output: $RESULTS_DIR"
log_info "Log file: $EVAL_LOG"
echo

# Check GPU availability
check_gpu 5000 || exit 1

# Check if results already exist
TASK_RESULT="$RESULTS_DIR/${TASK}.json"
if [ -f "$TASK_RESULT" ] && [ "${FORCE_RERUN:-false}" != "true" ]; then
    log_warn "Results already exist for $TASK at $TASK_RESULT"
    log_info "Skipping. Set FORCE_RERUN=true to overwrite."
    exit 0
fi

log_info "Starting $TASK..."
log_separator "-"

TASK_START=$(date +%s)

set +e
lm_eval --model hf \
  --model_args pretrained=$MODEL_DIR,dtype=bfloat16 \
  --tasks "$TASK" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$RESULTS_DIR" \
  --log_samples \
  --trust_remote_code \
  --include_path "$PROJECT_DIR/configs/task_configs" \
  --confirm_run_unsafe_code 2>&1 | tee -a "$EVAL_LOG"
TASK_EXIT=$?
set -e

TASK_END=$(date +%s)
TASK_ELAPSED=$((TASK_END - TASK_START))
TASK_MINS=$((TASK_ELAPSED / 60))
TASK_SECS=$((TASK_ELAPSED % 60))

log_separator "-"

if [ $TASK_EXIT -eq 0 ]; then
    log_info "✓ Completed: $TASK (${TASK_MINS}m ${TASK_SECS}s)"
    log_info "Results: $TASK_RESULT"

    # Show the score if the result file exists
    if [ -f "$TASK_RESULT" ]; then
        log_info ""
        log_info "Quick score summary:"
        # Extract the main metric from the JSON result
        python3 -c "
import json, sys
try:
    with open('$TASK_RESULT') as f:
        data = json.load(f)
    # lm-eval outputs have task names as keys
    for key, val in data.items():
        if key in ('results', 'versions') or key.startswith('_'):
            continue
        if isinstance(val, dict) and 'alias' in val:
            print(f'  {val.get(\"alias\", key)}: {val.get(\"alias\", key)}')
        elif isinstance(val, dict):
            for mk, mv in val.items():
                if 'score' in mk.lower() or 'acc' in mk.lower():
                    print(f'  {mk}: {mv}')
except Exception as e:
    print(f'  (could not parse results: {e})')
" 2>/dev/null || true
    fi
else
    log_error "✗ Failed: $TASK (exit code $TASK_EXIT, ${TASK_MINS}m ${TASK_SECS}s)"
    log_error "Check $EVAL_LOG for details."
    exit $TASK_EXIT
fi

log_separator "="
