#!/bin/bash
# Eval logging utility - source this file in eval scripts
# Provides timestamped log functions and progress tracking

# Log file path (overridable by caller)
EVAL_LOG="${EVAL_LOG:-}"

# Colors
_RED='\033[0;31m'
_GREEN='\033[0;32m'
_YELLOW='\033[1;33m'
_BLUE='\033[0;34m'
_CYAN='\033[0;36m'
_BOLD='\033[1m'
_NC='\033[0m' # No Color

# Timestamp format
_ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Write to both stdout and log file
_log_write() {
    local level="$1"
    shift
    local msg="$*"
    local timestamp
    timestamp=$(_ts)

    # Stdout with color based on level
    case "$level" in
        ERROR)   echo -e "${_RED}[${timestamp}] [${level}] ${msg}${_NC}" ;;
        WARN)    echo -e "${_YELLOW}[${timestamp}] [${level}] ${msg}${_NC}" ;;
        INFO)    echo -e "${_GREEN}[${timestamp}] [${level}] ${msg}${_NC}" ;;
        TASK)    echo -e "${_CYAN}[${timestamp}] [${level}] ${msg}${_NC}" ;;
        HEADER)  echo -e "${_BLUE}${_BOLD}[${timestamp}] ${msg}${_NC}" ;;
        *)       echo -e "[${timestamp}] [${level}] ${msg}" ;;
    esac

    # Log file (plain text, no colors)
    if [ -n "$EVAL_LOG" ]; then
        echo "[${timestamp}] [${level}] ${msg}" >> "$EVAL_LOG"
    fi
}

log_info()  { _log_write INFO "$@"; }
log_warn()  { _log_write WARN "$@"; }
log_error() { _log_write ERROR "$@"; }
log_task()  { _log_write TASK "$@"; }
log_header(){ _log_write HEADER "$@"; }

# Print a separator line
log_separator() {
    local char="${1:-=}"
    local width="${2:-60}"
    local line=""
    for ((i=0; i<width; i++)); do line+="$char"; done
    echo -e "${_BLUE}${line}${_NC}"
    if [ -n "$EVAL_LOG" ]; then
        echo "$line" >> "$EVAL_LOG"
    fi
}

# Print a section header
log_section() {
    local title="$1"
    log_separator "="
    log_header " $title "
    log_separator "="
    echo
}

# Track task progress
_task_total=0
_task_current=0
_task_start_time=0

progress_init() {
    _task_total=$1
    _task_current=0
    _task_start_time=$(date +%s)
    log_info "Total tasks: $_task_total"
    log_separator "-"
}

progress_next() {
    _task_current=$((_task_current + 1))
    local elapsed=$(( $(date +%s) - _task_start_time ))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    log_task "Task $_task_current/$_task_total  (elapsed: ${mins}m ${secs}s)"
}

progress_elapsed_total() {
    local elapsed=$(( $(date +%s) - _task_start_time ))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    echo "${mins}m ${secs}s"
}

# Time a single command, returns elapsed seconds via echo
time_command() {
    local start_time end_time elapsed
    start_time=$(date +%s)
    "$@"
    local exit_code=$?
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    if [ $exit_code -eq 0 ]; then
        log_info "Completed in ${mins}m ${secs}s"
    else
        log_error "Failed after ${mins}m ${secs}s"
    fi
    return $exit_code
}

# GPU status check with logging
check_gpu() {
    local threshold="${1:-5000}"
    local vram_used
    vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

    if [ -z "$vram_used" ]; then
        log_warn "Could not query GPU VRAM (nvidia-smi unavailable)"
        return 0
    fi

    log_info "GPU VRAM used: ${vram_used}MB"

    if [ "$vram_used" -gt "$threshold" ]; then
        log_warn "VRAM usage is ${vram_used}MB (threshold: ${threshold}MB). Studio container may be running."
        if [ "${AUTO_CONTINUE:-false}" = "true" ]; then
            log_warn "AUTO_CONTINUE is set, proceeding anyway"
            return 0
        fi
        read -r -s -n 1 -p "Continue anyway? (y/N) " CONTINUE
        echo
        if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
            log_info "Aborted by user"
            return 1
        fi
    fi
    return 0
}

# Parse task subset from --tasks flag
# Usage: parse_tasks "arg1 arg2 ..." -> sets $_SELECTED_TASKS array
parse_tasks() {
    _SELECTED_TASKS=()
    local args=("$@")
    local i=0
    while [ $i -lt ${#args[@]} ]; do
        if [ "${args[$i]}" = "--tasks" ]; then
            i=$((i + 1))
            IFS=',' read -ra _SELECTED_TASKS <<< "${args[$i]}"
            return 0
        fi
        i=$((i + 1))
    done
    # No --tasks flag, empty array means "run all"
    return 1
}

# Check if a task is in the selected list (empty list = all tasks selected)
task_selected() {
    local task="$1"
    if [ ${#_SELECTED_TASKS[@]} -eq 0 ]; then
        return 0  # Empty = run all
    fi
    for t in "${_SELECTED_TASKS[@]}"; do
        if [ "$t" = "$task" ]; then
            return 0
        fi
    done
    return 1
}

# Check for --help flag and print usage
show_help() {
    local script_name="$1"
    local description="$2"
    local tasks_list="$3"

    echo ""
    echo "Usage: $script_name [OPTIONS]"
    echo ""
    echo "$description"
    echo ""
    echo "Options:"
    echo "  --suite SUITE              Run a preset suite (short, medium, full)"
    echo "  --tasks TASK1,TASK2,...    Run only specified tasks (overrides --suite)"
    echo "  --help                     Show this help message"
    echo "  --dry-run                  Show what would run without executing"
    echo ""
    echo "Suites:"
    echo "  short   - IFEval + HumanEval + MMLU 5-shot (~2 hours)"
    echo "  medium  - short + GPQA Diamond (~3 hours)"
    echo "  full    - All tasks including MMLU-Pro (~40 hours)"
    echo ""
    echo "Available tasks:"
    IFS=',' read -ra TASK_ARR <<< "$tasks_list"
    for t in "${TASK_ARR[@]}"; do
        echo "  - $t"
    done
    echo ""
    echo "Examples:"
    echo "  $script_name                        # Run short suite (default)"
    echo "  $script_name --suite full           # Run all tasks"
    echo "  $script_name --tasks humaneval      # Run single task"
    echo ""
}

# Check for --dry-run flag
_is_dry_run() {
    [ "${DRY_RUN:-false}" = "true" ]
}

# Convert a WSL /mnt/X/... path to a Windows X:/... path.
# Leaves non-/mnt/ paths unchanged.
wsl_to_win_path() {
    local p="$1"
    # Match /mnt/<drive>/... and convert to <DRIVE>:/...
    if [[ "$p" =~ ^/mnt/([a-zA-Z])/(.*) ]]; then
        local drive="${BASH_REMATCH[1]}"
        local rest="${BASH_REMATCH[2]}"
        # Uppercase the drive letter, replace / with \
        drive=$(echo "$drive" | tr '[:lower:]' '[:upper:]')
        echo "${drive}:/${rest}"
    else
        echo "$p"
    fi
}

# Extract throughput statistics from llama-server log.
# Parses timing summary lines from the server log. llama-server prints:
#
#   prompt eval time =     457.32 ms /  1430 tokens (    0.32 ms per token,  3126.93 tokens per second)
#          eval time =    3838.08 ms /    71 tokens (   54.06 ms per token,    18.50 tokens per second)
#         total time =    4295.40 ms /  1501 tokens
#
# Usage: log_server_tps SERVER_LOG
# Prints a formatted TPS summary to stdout and the eval log.
log_server_tps() {
    local server_log="$1"

    if [ ! -f "$server_log" ]; then
        log_warn "Server log not found at $server_log, skipping TPS extraction"
        return
    fi

    local prompt_ms_total eval_ms_total prompt_tok_total eval_tok_total req_count

    # Use awk to extract and sum timing values from the server log.
    # Match the plain-text format: "prompt eval time = XXX ms / N tokens"
    # and "eval time = XXX ms / N tokens"
    read -r prompt_ms_total eval_ms_total prompt_tok_total eval_tok_total req_count << EOF
$(awk '
{
    # Strip Windows \r line endings
    gsub(/\r/, "")
}
/^prompt eval time =/ {
    # Line: prompt eval time =     457.32 ms /  1430 tokens (...)
    # Extract the ms value (first number after "=") and token count (second number)
    line = $0
    gsub(/[^0-9.]/, " ", line)
    split(line, a, " ")
    prompt_ms += a[1]
    prompt_tok += a[2]
    count++
}
/^ *eval time =/ {
    # Line:        eval time =    3838.08 ms /    71 tokens (...)
    line = $0
    gsub(/[^0-9.]/, " ", line)
    split(line, a, " ")
    eval_ms += a[1]
    eval_tok += a[2]
}
END {
    printf "%.2f %.2f %d %d %d\n", prompt_ms, eval_ms, prompt_tok, eval_tok, count
}
' "$server_log")
EOF

    if [ "$req_count" -eq 0 ] 2>/dev/null; then
        log_warn "No timing data found in server log"
        return
    fi

    local prompt_tps eval_tps overall_tps
    local prompt_ms_s eval_ms_s total_ms_s

    prompt_ms_s=$(echo "$prompt_ms_total / 1000" | bc -l 2>/dev/null || echo "0")
    eval_ms_s=$(echo "$eval_ms_total / 1000" | bc -l 2>/dev/null || echo "0")
    total_ms_s=$(echo "($prompt_ms_total + $eval_ms_total) / 1000" | bc -l 2>/dev/null || echo "0")

    if [ "$(echo "$prompt_ms_s > 0" | bc -l 2>/dev/null)" = "1" ]; then
        prompt_tps=$(echo "$prompt_tok_total / $prompt_ms_s" | bc -l 2>/dev/null || echo "0")
    else
        prompt_tps="0"
    fi

    if [ "$(echo "$eval_ms_s > 0" | bc -l 2>/dev/null)" = "1" ]; then
        eval_tps=$(echo "$eval_tok_total / $eval_ms_s" | bc -l 2>/dev/null || echo "0")
    else
        eval_tps="0"
    fi

    if [ "$(echo "$total_ms_s > 0" | bc -l 2>/dev/null)" = "1" ]; then
        overall_tps=$(echo "($prompt_tok_total + $eval_tok_total) / $total_ms_s" | bc -l 2>/dev/null || echo "0")
    else
        overall_tps="0"
    fi

    log_separator "-"
    log_info "SERVER THROUGHPUT (from $req_count requests):"
    log_info "  Prompt TPS:    $(printf '%8.1f' "$prompt_tps")  (${prompt_tok_total} tokens / ${prompt_ms_total}ms)"
    log_info "  Generation TPS: $(printf '%8.1f' "$eval_tps")  (${eval_tok_total} tokens / ${eval_ms_total}ms)"
    log_info "  Overall TPS:   $(printf '%8.1f' "$overall_tps")"
    log_separator "-"
}

# Start llama-server with robust error detection.
# Usage: start_llama_server SERVER_BIN MODEL_PATH PORT CTX_SIZE SERVER_LOG
# Sets SERVER_PID on success. Returns 1 on failure.
#
# Key features:
#   - Early crash detection: checks if the process died within first 3 seconds
#     (catches CLI parse errors instantly instead of waiting for timeout)
#   - Progressive health check: 1s intervals, with increasing patience
#   - On failure: prints last 20 lines of server log so the user sees the actual error
#   - Timeout configurable via SERVER_START_TIMEOUT (default 180s for large models)
#   - Converts WSL /mnt/c/... paths to Windows C:/... for llama-server.exe
start_llama_server() {
    local server_bin="$1"
    local model_path="$2"
    local port="$3"
    local ctx_size="$4"
    local server_log="$5"

    local timeout="${SERVER_START_TIMEOUT:-180}"

    # Convert WSL mount path to Windows path for llama-server.exe
    local win_model_path
    win_model_path=$(wsl_to_win_path "$model_path")

    log_info "  Model (Windows): $win_model_path"

    # Launch server in background
    # -ub 2048: larger upload batch for better decode throughput under eval load
    # --cache-ram 0: disable prompt cache - checkpoint restore overhead (~8ms/req)
    #   outweighs any cache hits for lm-eval's random-access pattern
    "$server_bin" \
        -m "$win_model_path" \
        --host 127.0.0.1 \
        --port "$port" \
        -ngl 99 \
        -c "$ctx_size" \
        -b 4096 \
        -ub 2048 \
        -fa on \
        --temp 0.0 \
        --top-p 0.95 \
        --cache-ram 0 \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    log_info "Server started with PID $SERVER_PID"

    # --- Early crash detection ---
    # Wait 3 seconds, then check if the process is still alive.
    # CLI parse errors (bad flags, missing model) kill the process immediately.
    sleep 3
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        local exit_code=""
        wait "$SERVER_PID" 2>/dev/null && exit_code=0 || exit_code=$?
        log_error "Server crashed immediately (exit code: ${exit_code}). Last 20 lines of $server_log:"
        tail -20 "$server_log" 2>/dev/null | while IFS= read -r line; do
            log_error "  $line"
        done
        SERVER_PID=""
        return 1
    fi
    log_info "Server process alive after 3s, waiting for /health endpoint..."

    # --- Health check polling ---
    local start_time
    start_time=$(date +%s)
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        local now
        now=$(date +%s)
        local elapsed=$((now - start_time))

        if [ "$elapsed" -ge "$timeout" ]; then
            log_error "Server did not become ready within ${timeout}s."
            # Show last lines of log
            log_error "Last 20 lines of $server_log:"
            tail -20 "$server_log" 2>/dev/null | while IFS= read -r line; do
                log_error "  $line"
            done
            # Check if process is still alive (hung vs crashed)
            if kill -0 "$SERVER_PID" 2>/dev/null; then
                log_warn "Server process (PID $SERVER_PID) is still running but not responding."
            else
                log_error "Server process has died."
            fi
            return 1
        fi

        # Print progress every 10 seconds
        if [ $((elapsed % 10)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
            log_info "  Still waiting... (${elapsed}s / ${timeout}s)"
        fi

        if curl -s "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            log_info "Server is ready (${elapsed}s)."
            return 0
        fi

        sleep 1
    done
}
