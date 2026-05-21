#!/bin/bash
# Run fine-tuned GGUF evaluation using llama.cpp server + lm_eval gguf backend
# Evaluates the fine-tuned Q4_K_M GGUF exported from Studio
#
# Uses llama-server.exe from Windows (C:\llamacpp) as an OpenAI-compatible
# API server. The lm_eval gguf backend (GGUFLM) connects via HTTP.
#
# NOTE: Requires Studio container to be stopped (GPU must be free)
# NOTE: llama-server uses ~5.3GB VRAM for the 9B Q4 model
# NOTE: Export the fine-tuned model as GGUF from Studio UI first
#
# Usage:
#   ./run_finetuned_gguf.sh                               # Run short suite (default)
#   ./run_finetuned_gguf.sh --suite short                 # IFEval + HumanEval + MMLU 5-shot
#   ./run_finetuned_gguf.sh --suite medium                # short + GPQA Diamond
#   ./run_finetuned_gguf.sh --suite full                  # All tasks including MMLU-Pro
#   ./run_finetuned_gguf.sh --tasks humaneval             # Run single task only
#   ./run_finetuned_gguf.sh --tasks mmlu,humaneval        # Run specific tasks
#   ./run_finetuned_gguf.sh --help                        # Show available tasks
#
# Set FINETUNED_GGUF_PATH to point to a specific GGUF file:
#   FINETUNED_GGUF_PATH=/path/to/model.gguf ./run_finetuned_gguf.sh

set -euo pipefail

# Source logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_logging.sh"

# Allow humaneval code_eval metric (executes model-generated Python code)
# See: https://arxiv.org/abs/2107.03374
export HF_ALLOW_CODE_EVAL="1"

# Auto-approve unsafe code evaluation (required for humaneval)
export LM_EVAL_CONFIRM_RUN_UNSAFE_CODE="True"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/finetuned/gguf"

# Activate venv
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Server configuration
LLAMA_SERVER="/mnt/c/llamacpp/llama-server.exe"
REPO_ROOT="$(dirname "$PROJECT_DIR")"

# Fine-tuned GGUF path — override with FINETUNED_GGUF_PATH env var
# Default: Studio export of fine-tuned 9B Q4_K_M GGUF
FINETUNED_GGUF_DEFAULT="/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen3.5-9B-gguf/Qwen3.5-9B.Q4_K_M.gguf"
GGUF_PATH="${FINETUNED_GGUF_PATH:-$FINETUNED_GGUF_DEFAULT}"
SERVER_PORT="${GGUF_SERVER_PORT:-8080}"
SERVER_CTX="${GGUF_CTX_SIZE:-4096}"

# Eval suites — ordered from fastest to slowest
SUITE_SHORT=(
    "ifeval"
    "humaneval"
    "mmlu"
)

SUITE_MEDIUM=(
    "ifeval"
    "humaneval"
    "mmlu"
    "gpqa_diamond_zeroshot"
)

SUITE_FULL=(
    "mmlu_pro"
    "gpqa_diamond_zeroshot"
    "ifeval"
    "humaneval"
    "leaderboard_math_hard"
)

# All individual tasks (union of all suites)
ALL_TASKS=(
    "mmlu_pro"
    "mmlu"
    "gpqa_diamond_zeroshot"
    "ifeval"
    "humaneval"
    "leaderboard_math_hard"
)

TASKS_LIST="mmlu_pro,mmlu,gpqa_diamond_zeroshot,ifeval,humaneval,leaderboard_math_hard"

# Parse arguments
DRY_RUN="false"
_SELECTED_TASKS=()
_SELECT_SUITE=""
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --help)
            show_help "$0" "Fine-Tuned GGUF Evaluation (llama.cpp server)" "$TASKS_LIST"
            exit 0
            ;;
        --dry-run)
            DRY_RUN="true"
            ;;
        --suite)
            i=$((i + 1))
            if [ $i -gt $# ]; then
                log_error "--suite requires a value (short, medium, full)"
                exit 1
            fi
            _SELECT_SUITE="${!i}"
            ;;
        --tasks)
            i=$((i + 1))
            if [ $i -gt $# ]; then
                log_error "--tasks requires a value"
                exit 1
            fi
            IFS=',' read -ra _SELECTED_TASKS <<< "${!i}"
            ;;
    esac
    i=$((i + 1))
done

# Resolve suite to task list (--tasks overrides suite; suite defaults to short)
_SUITE_USED="false"
if [ ${#_SELECTED_TASKS[@]} -eq 0 ]; then
    case "${_SELECT_SUITE:-short}" in
        short)  _SELECTED_TASKS=("${SUITE_SHORT[@]}") ;;
        medium) _SELECTED_TASKS=("${SUITE_MEDIUM[@]}") ;;
        full)   _SELECTED_TASKS=("${SUITE_FULL[@]}") ;;
        *)      log_error "Unknown suite: $_SELECT_SUITE (choose short, medium, full)"; exit 1 ;;
    esac
    _SUITE_USED="true"
fi

# Set up log file and run ID
mkdir -p "$RESULTS_DIR"
EVAL_LOG="$RESULTS_DIR/eval.log"
# Truncate log for fresh run
: > "$EVAL_LOG"

# Descriptive run ID: finetuned-gguf-2026_05_20T17_49_34
_RUN_TIMESTAMP=$(date '+%Y_%m_%dT%H_%M_%S')
RUN_ID="finetuned-gguf-${_RUN_TIMESTAMP}"
RUN_OUTPUT_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_OUTPUT_DIR"

log_section "Fine-Tuned GGUF Evaluation"
log_info "GGUF: $GGUF_PATH"
log_info "Server: $LLAMA_SERVER (port $SERVER_PORT, ctx $SERVER_CTX)"
log_info "Run ID: $RUN_ID"
log_info "Output: $RUN_OUTPUT_DIR"
log_info "Log file: $EVAL_LOG"

if [ "$_SUITE_USED" = "true" ]; then
    log_info "Suite: ${_SELECT_SUITE:-short}"
fi
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

# --- Server management ---

# Cleanup function: kill the llama server on exit
SERVER_PID=""
cleanup_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping llama-server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped."
    fi
}
trap cleanup_server EXIT INT TERM

# Start llama.cpp server (only if not a dry run)
if _is_dry_run; then
    log_info "[DRY RUN] Would start llama-server on port $SERVER_PORT"
else
    # Check that the server binary exists
    if [ ! -x "$LLAMA_SERVER" ]; then
        log_error "llama-server not found or not executable at $LLAMA_SERVER"
        exit 1
    fi

    # Check that the GGUF file exists
    if [ ! -f "$GGUF_PATH" ]; then
        log_error "GGUF file not found at $GGUF_PATH"
        log_info "Export the fine-tuned model as GGUF from Studio UI first."
        log_info "Or set FINETUNED_GGUF_PATH to point to your GGUF file."
        exit 1
    fi

    # Check GPU availability - use a high threshold since the server will use ~5.3GB
    # Set AUTO_CONTINUE internally since we need the GPU for the server
    export AUTO_CONTINUE="${AUTO_CONTINUE:-true}"
    check_gpu 5000 || exit 1

    log_info "Starting llama-server..."
    log_info "  Model: $GGUF_PATH"
    log_info "  Port: $SERVER_PORT"
    log_info "  Context: $SERVER_CTX"

    start_llama_server "$LLAMA_SERVER" "$GGUF_PATH" "$SERVER_PORT" "$SERVER_CTX" "$RESULTS_DIR/server.log" || exit 1
fi

# --- Run evaluations ---

# Known limitation: GGUFLM does not implement loglikelihood_rolling.
# Tasks that depend on it will fail. Track these separately.
ROLLING_SKIPPED=()

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
    log_info "Command: lm_eval --model gguf --model_args base_url=http://127.0.0.1:$SERVER_PORT,max_length=$SERVER_CTX --tasks $TASK ..."

    if _is_dry_run; then
        log_info "[DRY RUN] Would run lm_eval for task: $TASK"
        continue
    fi

    # Run single task via the gguf backend (HTTP client to llama-server)
    # --batch_size 4: fixed to avoid _detect_batch_size cudaErrorNotReady bug
    TASK_START=$(date +%s)

    set +e
    lm_eval --model gguf \
      --model_args "base_url=http://127.0.0.1:$SERVER_PORT,max_length=$SERVER_CTX" \
      --tasks "$TASK" \
      --batch_size 4 \
      --output_path "$RUN_OUTPUT_DIR" \
      --log_samples \
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
        # Check if the failure was due to loglikelihood_rolling not being implemented
        if grep -q "NotImplementedError.*loglikelihood_rolling" "$EVAL_LOG" 2>/dev/null; then
            log_warn "✗ Skipped: $TASK (loglikelihood_rolling not supported by gguf backend, ${TASK_MINS}m ${TASK_SECS}s)"
            ROLLING_SKIPPED+=("$TASK")
        else
            log_error "✗ Failed: $TASK (exit code $TASK_EXIT, ${TASK_MINS}m ${TASK_SECS}s)"
            FAILED_TASKS+=("$TASK")
        fi
    fi

    echo
done

# Summary
TOTAL_ELAPSED=$(progress_elapsed_total)
log_section "Evaluation Summary"
log_info "Total elapsed time: $TOTAL_ELAPSED"
log_info "Tasks completed: $(( ${#TASKS_TO_RUN[@]} - ${#FAILED_TASKS[@]} - ${#SKIPPED_TASKS[@]} - ${#ROLLING_SKIPPED[@]} ))"

if [ ${#SKIPPED_TASKS[@]} -gt 0 ]; then
    log_warn "Tasks skipped (existing results): ${SKIPPED_TASKS[*]}"
fi

if [ ${#ROLLING_SKIPPED[@]} -gt 0 ]; then
    log_warn "Tasks skipped (loglikelihood_rolling not supported): ${ROLLING_SKIPPED[*]}"
fi

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    log_error "Tasks failed: ${FAILED_TASKS[*]}"
    log_separator "-"
    log_error "Some tasks failed. Check $EVAL_LOG for details."
    exit 1
fi

log_info "Results saved to: $RUN_OUTPUT_DIR"
log_info "Log saved to: $EVAL_LOG"

# Post-process: annotate results with identifying metadata
log_info "Annotating results with metadata..."
python3 "$SCRIPT_DIR/label_results.py" --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$EVAL_LOG"

log_separator "="
