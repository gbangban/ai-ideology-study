#!/bin/bash
# Run GGUF baseline evaluation using llama.cpp server + lm_eval gguf backend
# Evaluates the Q4_K_M GGUF to measure quantization penalty
#
# Uses llama-server.exe from Windows (C:\llamacpp) as an OpenAI-compatible
# API server. The lm_eval gguf backend (GGUFLM) connects via HTTP.
#
# NOTE: Requires Studio container to be stopped (GPU must be free)
# NOTE: llama-server uses ~16GB VRAM for the 27B Q4 model
#
# Usage:
#   ./run_baseline_gguf.sh                          # Run all tasks
#   ./run_baseline_gguf.sh --tasks humaneval        # Run single task only
#   ./run_baseline_gguf.sh --tasks mmlu_pro,humaneval  # Run specific tasks
#   ./run_baseline_gguf.sh --help                   # Show available tasks

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
RESULTS_DIR="$PROJECT_DIR/results/baseline/gguf"

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
GGUF_PATH="${GGUF_MODEL_PATH:-$REPO_ROOT/checkpoints/base_model/Qwen3.5-27B-Instruct-Q4_K_M.gguf}"
SERVER_PORT="${GGUF_SERVER_PORT:-8080}"
SERVER_CTX="${GGUF_CTX_SIZE:-4096}"

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
_SELECTED_TASKS=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --help)
            show_help "$0" "GGUF Baseline Evaluation (llama.cpp server)" "$TASKS_LIST"
            exit 0
            ;;
        --dry-run)
            DRY_RUN="true"
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

# Set up log file
mkdir -p "$RESULTS_DIR"
EVAL_LOG="$RESULTS_DIR/eval.log"
# Truncate log for fresh run
: > "$EVAL_LOG"

log_section "GGUF Baseline Evaluation"
log_info "GGUF: $GGUF_PATH"
log_info "Server: $LLAMA_SERVER (port $SERVER_PORT, ctx $SERVER_CTX)"
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
        exit 1
    fi

    # Check GPU availability - use a high threshold since the server will use ~16GB
    # Set AUTO_CONTINUE internally since we need the GPU for the server
    export AUTO_CONTINUE="${AUTO_CONTINUE:-true}"
    check_gpu 5000 || exit 1

    log_info "Starting llama-server..."
    log_info "  Model: $GGUF_PATH"
    log_info "  Port: $SERVER_PORT"
    log_info "  Context: $SERVER_CTX"

    "$LLAMA_SERVER" \
        --model "$GGUF_PATH" \
        --host 127.0.0.1 \
        --port "$SERVER_PORT" \
        --n-gpu-layers all \
        --ctx-size "$SERVER_CTX" \
        --batch-size 2048 \
        --flash-attn \
        --logprobs 10 \
        > "$RESULTS_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    log_info "Server started with PID $SERVER_PID"

    # Poll /health endpoint until the server is ready (max 60 seconds)
    log_info "Waiting for server to be ready..."
    SERVER_READY=false
    for attempt in $(seq 1 60); do
        if curl -s "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1; then
            SERVER_READY=true
            break
        fi
        sleep 1
    done

    if [ "$SERVER_READY" != "true" ]; then
        log_error "Server did not become ready within 60 seconds. Check $RESULTS_DIR/server.log"
        exit 1
    fi
    log_info "Server is ready."
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
      --output_path "$RESULTS_DIR" \
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

log_info "Results saved to: $RESULTS_DIR"
log_info "Log saved to: $EVAL_LOG"
log_separator "="
