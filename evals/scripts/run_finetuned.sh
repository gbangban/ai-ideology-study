#!/bin/bash
# Run fine-tuned Q4_K_M GGUF evaluation (primary target)
# Runs first for maximum VRAM headroom and fastest feedback
#
# NOTE: Requires Studio container to be stopped (GPU must be free)
# Check with: nvidia-smi (should show <5GB VRAM used)
#
# Usage:
#   ./run_finetuned.sh                              # Run all tasks
#   ./run_finetuned.sh --tasks humaneval            # Run single task only
#   ./run_finetuned.sh --tasks mmlu_pro,humaneval   # Run specific tasks
#   ./run_finetuned.sh --help                       # Show available tasks

set -euo pipefail

# Source logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_logging.sh"

# Allow humaneval code_eval metric (executes model-generated Python code)
# See: https://arxiv.org/abs/2107.03374
export HF_ALLOW_CODE_EVAL="1"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/runs/finetuned"

# Activate venv
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    log_info "Run: python3 -m virtualenv $VENV_DIR --python=/usr/bin/python3.12"
    log_info "Then: source $VENV_DIR/bin/activate && pip install -r requirements.txt"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Model paths - using safetensors export from Studio (NOT GGUF)
# GGUF qwen35 architecture is not yet supported by transformers
# Export fine-tuned model as safetensors from Studio UI first
MODEL_DIR="${STUDIO_EXPORT_PATH:-/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-safetensors}"

# All available tasks
ALL_TASKS=(
    "mmlu_pro"
    "gpqa_diamond_zeroshot"
    "ifeval"
    "humaneval"
    "leaderboard_math_hard"
)

TASKS_LIST="mmlu_pro,gpqa_diamond_zeroshot,ifeval,humaneval,leaderboard_math_hard"

# Parse arguments
DRY_RUN="false"
for arg in "$@"; do
    case "$arg" in
        --help)
            show_help "$0" "Fine-Tuned Model Evaluation (primary target)" "$TASKS_LIST"
            exit 0
            ;;
        --dry-run)
            DRY_RUN="true"
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

log_section "Fine-Tuned Model Evaluation"
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
    log_info "Command: lm_eval --model hf --model_args pretrained=$MODEL_DIR,dtype=bfloat16 --tasks $TASK --batch_size 4 ..."

    if _is_dry_run; then
        log_info "[DRY RUN] Would run lm_eval for task: $TASK"
        continue
    fi

    # Run single task
    # --batch_size 4: fixed to avoid _detect_batch_size cudaErrorNotReady bug
    TASK_START=$(date +%s)

    set +e
    lm_eval --model hf \
      --model_args pretrained=$MODEL_DIR,dtype=bfloat16 \
      --tasks "$TASK" \
      --batch_size 4 \
      --output_path "$RESULTS_DIR" \
      --log_samples \
      --trust_remote_code 2>&1 | tee -a "$EVAL_LOG"
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
