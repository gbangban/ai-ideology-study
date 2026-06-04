#!/bin/bash
# Run BF16 evaluation using SG-Lang serving backend
# Evaluates the model served by SG-Lang via lm_eval's OpenAI backend
#
# This script manages the SG-Lang server lifecycle:
# 1. Launches SG-Lang with the target model (BF16)
# 2. Waits for health check on port 1235
# 3. Runs lm_eval with OpenAI backend
# 4. Tears down SG-Lang after eval completes
#
# NOTE: Requires Docker Desktop with NVIDIA runtime
# NOTE: Stop other GPU containers (Studio, training) before running
#
# Usage:
#   ./run_sglang_bf16.sh                                # Run all tasks
#   ./run_sglang_bf16.sh --tasks humaneval               # Run single task
#   ./run_sglang_bf16.sh --suite causal                  # Run causal suite
#   ./run_sglang_bf16.sh --help                          # Show available tasks
#
# Set SGLANG_MODEL to point to a specific model:
#   SGLANG_MODEL=Qwen/Qwen3.5-9B ./run_sglang_bf16.sh
# Default: merged GRPO checkpoint path
#
# Set SGLANG_SKIP_SERVER=true to skip server management (use existing SG-Lang instance):
#   SGLANG_SKIP_SERVER=true ./run_sglang_bf16.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_logging.sh"

export HF_ALLOW_CODE_EVAL="1"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/sglang/bf16"

VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

SGLANG_PORT=1235
SGLANG_URL="http://localhost:$SGLANG_PORT"
SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3.5-9B}"
SGLANG_SKIP_SERVER="${SGLANG_SKIP_SERVER:-false}"

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

DRY_RUN="false"
_SELECTED_TASKS=()
for arg in "$@"; do
    case "$arg" in
        --help)
            show_help "$0" "SG-Lang BF16 Evaluation (OpenAI-compatible backend)" "$TASKS_LIST"
            exit 0
            ;;
        --dry-run)
            DRY_RUN="true"
            ;;
        --suite)
            shift
            if [ $# -eq 0 ]; then
                log_error "--suite requires a value (short, medium, causal, full)"
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

mkdir -p "$RESULTS_DIR"
EVAL_LOG="$RESULTS_DIR/eval.log"
: > "$EVAL_LOG"

log_section "SG-Lang BF16 Evaluation"
log_info "Model: $SGLANG_MODEL"
log_info "SG-Lang URL: $SGLANG_URL"
log_info "Output: $RESULTS_DIR"
log_info "Log file: $EVAL_LOG"

if [ "$DRY_RUN" = "true" ]; then
    log_info "DRY RUN MODE - showing what would execute"
fi

TASKS_TO_RUN=()
for TASK in "${ALL_TASKS[@]}"; do
    if task_selected "$TASK"; then
        log_info "  + $TASK"
        TASKS_TO_RUN+=("$TASK")
    else
        log_info "  - $TASK (skipped)"
    fi
done
echo

if [ ${#TASKS_TO_RUN[@]} -eq 0 ]; then
    log_error "No tasks selected."
    exit 1
fi

SGLANG_PID=""
cleanup_sglang() {
    if [ -n "$SGLANG_PID" ] && [ "$SGLANG_SKIP_SERVER" != "true" ]; then
        log_info "Stopping SG-Lang server (PID $SGLANG_PID)..."
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
        log_info "SG-Lang server stopped."
    fi
}
trap cleanup_sglang EXIT

if [ "$SGLANG_SKIP_SERVER" != "true" ]; then
    check_gpu 5000 || exit 1

    log_info "Launching SG-Lang server with model: $SGLANG_MODEL"
    SERVER_LOG="$RESULTS_DIR/sglang_server.log"
    : > "$SERVER_LOG"

    docker compose run --rm \
        --no-deps \
        -e HF_TOKEN \
        sglang \
        --model-path "$SGLANG_MODEL" \
        --host 0.0.0.0 \
        --port 30000 \
        --mem-fraction-static 0.6 \
        --cuda-memory-fraction 0.6 \
        > "$SERVER_LOG" 2>&1 &
    SGLANG_PID=$!

    log_info "SG-Lang started with PID $SGLANG_PID"

    HEALTH_TIMEOUT=300
    HEALTH_START=$(date +%s)
    while true; do
        HEALTH_ELAPSED=$(( $(date +%s) - HEALTH_START ))
        if [ "$HEALTH_ELAPSED" -ge "$HEALTH_TIMEOUT" ]; then
            log_error "SG-Lang did not become ready within ${HEALTH_TIMEOUT}s."
            log_error "Last 20 lines of $SERVER_LOG:"
            tail -20 "$SERVER_LOG" 2>/dev/null | while IFS= read -r line; do
                log_error "  $line"
            done
            exit 1
        fi
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SGLANG_URL/v1/models" --max-time 5 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            log_info "SG-Lang is ready (${HEALTH_ELAPSED}s)."
            break
        fi
        if [ $((HEALTH_ELAPSED % 15)) -eq 0 ] && [ "$HEALTH_ELAPSED" -gt 0 ]; then
            log_info "  Still waiting for SG-Lang... (${HEALTH_ELAPSED}s / ${HEALTH_TIMEOUT}s)"
        fi
        sleep 3
    done
else
    log_info "Skipping SG-Lang server launch (SGLANG_SKIP_SERVER=true)"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SGLANG_URL/v1/models" --max-time 10 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" != "200" ]; then
        log_error "SG-Lang is not reachable at $SGLANG_URL"
        exit 1
    fi
    log_info "SG-Lang is healthy at $SGLANG_URL"
fi

export LM_EVAL_CONFIRM_RUN_UNSAFE_CODE="True"

progress_init ${#TASKS_TO_RUN[@]}

FAILED_TASKS=()
SKIPPED_TASKS=()

for TASK in "${TASKS_TO_RUN[@]}"; do
    progress_next

    TASK_RESULT="$RESULTS_DIR/${TASK}.json"
    if [ -f "$TASK_RESULT" ] && [ "${FORCE_RERUN:-false}" != "true" ]; then
        log_warn "Results already exist for $TASK at $TASK_RESULT"
        log_info "Skipping. Set FORCE_RERUN=true to overwrite."
        SKIPPED_TASKS+=("$TASK")
        continue
    fi

    log_info "Starting $TASK..."
    MODEL_ARGS="model=$SGLANG_MODEL,base_url=$SGLANG_URL,vllm_guided_decoding_enabled=False"
    log_info "Command: lm_eval --model openai-completions --model_args $MODEL_ARGS --tasks $TASK ..."

    if _is_dry_run; then
        log_info "[DRY RUN] Would run lm_eval for task: $TASK"
        continue
    fi

    TASK_START=$(date +%s)

    set +e
    lm_eval --model openai-completions \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --batch_size auto \
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
        log_info "Completed: $TASK (${TASK_MINS}m ${TASK_SECS}s)"
    else
        log_error "Failed: $TASK (exit code $TASK_EXIT, ${TASK_MINS}m ${TASK_SECS}s)"
        FAILED_TASKS+=("$TASK")
    fi

    echo
done

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
