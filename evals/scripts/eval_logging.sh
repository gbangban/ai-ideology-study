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
    echo "  --tasks TASK1,TASK2,...  Run only specified tasks (comma-separated)"
    echo "  --help                   Show this help message"
    echo "  --dry-run                Show what would run without executing"
    echo ""
    echo "Available tasks:"
    IFS=',' read -ra TASK_ARR <<< "$tasks_list"
    for t in "${TASK_ARR[@]}"; do
        echo "  - $t"
    done
    echo ""
    echo "Examples:"
    echo "  $script_name                          # Run all tasks"
    echo "  $script_name --tasks humaneval        # Run single task"
    echo "  $script_name --tasks mmlu_pro,humaneval  # Run specific tasks"
    echo ""
}

# Check for --dry-run flag
_is_dry_run() {
    [ "${DRY_RUN:-false}" = "true" ]
}
