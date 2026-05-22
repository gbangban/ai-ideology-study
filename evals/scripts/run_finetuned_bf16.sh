#!/bin/bash
# Run fine-tuned BF16 evaluation using native HF backend
# Evaluates the full-precision safetensors model exported from Studio
#
# This avoids the Q4_K_M quantization penalty that collapses HumanEval
# from 70.73% to 1.83%, providing a meaningful regression test.
#
# NOTE: Requires Studio container to be stopped (GPU must be free)
# NOTE: The full 9B bf16 model uses ~18GB VRAM
#
# Usage:
#   ./run_finetuned_bf16.sh                              # Run all tasks
#   ./run_finetuned_bf16.sh --tasks humaneval             # Run single task only
#   ./run_finetuned_bf16.sh --tasks mmlu_pro,humaneval    # Run specific tasks
#   ./run_finetuned_bf16.sh --help                        # Show available tasks
#
# Set FINETUNED_MODEL_DIR to point to a specific checkpoint directory:
#   FINETUNED_MODEL_DIR=/path/to/checkpoint ./run_finetuned_bf16.sh

set -euo pipefail

# Source logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_logging.sh"

# Allow humaneval code_eval metric (executes model-generated Python code)
# See: https://arxiv.org/abs/2107.03374
export HF_ALLOW_CODE_EVAL="1"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/finetuned/bf16"

# Activate venv
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Fine-tuned model path — override with FINETUNED_MODEL_DIR env var
# Default: Studio export of full-precision finetuned 9B safetensors
# The path points to the checkpoint directory containing model.safetensors.index.json
_FINETUNED_EXPORT_DIR="/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330"
MODEL_DIR="${FINETUNED_MODEL_DIR:-$_FINETUNED_EXPORT_DIR}"

if [ ! -d "$MODEL_DIR" ]; then
    log_error "Fine-tuned model directory not found: $MODEL_DIR"
    log_error "Set FINETUNED_MODEL_DIR env var to override."
    exit 1
fi

# All available tasks
ALL_TASKS=(
    "mmlu"
    "mmlu_pro"
    "gpqa_diamond_zeroshot"
    "ifeval"
    "humaneval"
    "leaderboard_math_hard"
    "econcausal_task1_econ"
    "econcausal_task1_finance"
    "econcausal_task2"
    "econcausal_task3"
    "corr2cause"
)

TASKS_LIST="mmlu,mmlu_pro,gpqa_diamond_zeroshot,ifeval,humaneval,leaderboard_math_hard,econcausal_task1_econ,econcausal_task1_finance,econcausal_task2,econcausal_task3,corr2cause"

# Parse arguments
DRY_RUN="false"
_SELECTED_TASKS=()
for arg in "$@"; do
    case "$arg" in
        --help)
            show_help "$0" "Fine-Tuned BF16 Evaluation (native HF, full precision)" "$TASKS_LIST"
            exit 0
            ;;
        --dry-run)
            DRY_RUN="true"
            ;;
        --suite)
            shift
            if [ $# -eq 0 ]; then
                log_error "--suite requires a value (short, medium, full)"
                exit 1
            fi
            case "$1" in
                short)
                    IFS=',' read -ra _SELECTED_TASKS <<< "ifeval,humaneval,mmlu"
                    ;;
                medium)
                    IFS=',' read -ra _SELECTED_TASKS <<< "ifeval,humaneval,mmlu,gpqa_diamond_zeroshot"
                    ;;
                causal)
                    IFS=',' read -ra _SELECTED_TASKS <<< "econcausal_task1_econ,econcausal_task1_finance,econcausal_task2,econcausal_task3,corr2cause"
                    ;;
                full)
                    # Run all tasks (empty = all)
                    _SELECTED_TASKS=()
                    ;;
                *)
                    log_error "Unknown suite: $1 (valid: short, medium, causal, full)"
                    exit 1
                    ;;
            esac
            ;;
        --tasks)
            shift
            if [ $# -eq 0 ]; then
                log_error "--tasks requires a value"
                exit 1
            fi
            IFS=',' read -ra _SELECTED_TASKS <<< "$1"
            ;;
    esac
done

# Set up log file
mkdir -p "$RESULTS_DIR"
EVAL_LOG="$RESULTS_DIR/eval.log"
# Truncate log for fresh run
: > "$EVAL_LOG"

log_section "Fine-Tuned BF16 Evaluation"
log_info "Model: $MODEL_DIR"
log_info "Output: $RESULTS_DIR"
log_info "Log file: $EVAL_LOG"

if [ "$DRY_RUN" = "true" ]; then
    log_info "DRY RUN MODE - showing what would execute"
fi

# Show which tasks will run
log_info "Tasks to run:"
TASKS_TO_RUN=()
for TASK in "${ALL_TASKS[@]}"; do
    if task_selected "$TASK"; then
        log_info "  ✓ $TASK"
        TASKS_TO_RUN+=("$TASK")
    else
        log_info "  ✗ $TASK (skipped)"
    fi
done
echo

if [ ${#TASKS_TO_RUN[@]} -eq 0 ]; then
    log_error "No tasks selected. Use --tasks to specify tasks, or omit --tasks to run all."
    exit 1
fi

# Check GPU availability
check_gpu 5000 || exit 1

# Auto-approve unsafe code evaluation
export LM_EVAL_CONFIRM_RUN_UNSAFE_CODE="True"

progress_init ${#TASKS_TO_RUN[@]}

# Run each selected task
FAILED_TASKS=()
SKIPPED_TASKS=()

for TASK in "${TASKS_TO_RUN[@]}"; do
    progress_next

    # Check if results already exist for this task
    TASK_RESULT="$RESULTS_DIR/${TASK}.json"
    if [ -f "$TASK_RESULT" ] && [ "${FORCE_RERUN:-false}" != "true" ]; then
        log_warn "Results already exist for $TASK at $TASK_RESULT"
        log_info "Skipping. Set FORCE_RERUN=true to overwrite."
        SKIPPED_TASKS+=("$TASK")
        continue
    fi

    log_info "Starting $TASK..."
    log_info "Command: lm_eval --model hf --model_args pretrained=$MODEL_DIR,dtype=bfloat16,enable_thinking=False --tasks $TASK --batch_size 4 ..."

    if _is_dry_run; then
        log_info "[DRY RUN] Would run lm_eval for task: $TASK"
        continue
    fi

    # Run single task
    # --trust_remote_code required for Qwen3.5 models
    # --batch_size 4: fixed batch size to avoid _detect_batch_size which causes
    # cudaErrorNotReady on RTX 5090. The "auto:N" syntax still triggers auto-detection
    # (it only sets the decay schedule factor), so a plain integer is required.
    TASK_START=$(date +%s)

    set +e
    lm_eval --model hf \
      --model_args pretrained=$MODEL_DIR,dtype=bfloat16,enable_thinking=False \
      --tasks "$TASK" \
      --batch_size 4 \
      --output_path "$RESULTS_DIR" \
      --log_samples \
      --trust_remote_code \
      --include_path "$PROJECT_DIR/configs/task_configs" \
      --apply_chat_template \
      --confirm_run_unsafe_code 2>&1 | tee -a "$EVAL_LOG"
    TASK_EXIT=$?
    set -e

    TASK_END=$(date +%s)
    TASK_ELAPSED=$((TASK_END - TASK_START))
    TASK_MINS=$((TASK_ELAPSED / 60))
    TASK_SECS=$((TASK_ELAPSED % 60))

    if [ $TASK_EXIT -eq 0 ]; then
        log_info "✓ Completed: $TASK (${TASK_MINS}m ${TASK_SECS}s)"
    else
        log_error "✗ Failed: $TASK (exit code $TASK_EXIT, ${TASK_MINS}m ${TASK_SECS}s)"
        FAILED_TASKS+=("$TASK")
    fi

    echo
done

# Summary
TOTAL_ELAPSED=$(progress_elapsed_total)
log_section "Evaluation Summary"
log_info "Total elapsed time: $TOTAL_ELAPSED"
log_info "Tasks completed: $(( ${#TASKS_TO_RUN[@]} - ${#FAILED_TASKS[@]} - ${#SKIPPED_TASKS[@]} ))"

if [ ${#SKIPPED_TASKS[@]} -gt 0 ]; then
    log_warn "Tasks skipped (existing results): ${SKIPPED_TASKS[*]}"
fi

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    log_error "Tasks failed: ${FAILED_TASKS[*]}"
    log_separator "-"
    log_error "Some tasks failed. Check $EVAL_LOG for details."
    exit 1
fi

log_info "Results saved to: $RESULTS_DIR"
log_info "Log saved to: $EVAL_LOG"
log_separator "="
