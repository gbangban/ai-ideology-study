#!/usr/bin/env bash
# Run eval question response generation + HTML comparison rendering.
# Must be run from evals/ directory with GPU available (Studio stopped).
#
# Usage:
#   ./scripts/run_eval_comparison.sh          # All 4 models, all 21 questions
#   ./scripts/run_eval_comparison.sh dm       # Single model only
#   ./scripts/run_eval_comparison.sh 1,2,3    # Specific question ids
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVALS_DIR="$PROJECT_DIR/evals"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Activate evals venv
VENV_DIR="$EVALS_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    log_error "evals/.venv/ not found. Create it first."
    exit 1
fi
source "$VENV_DIR/bin/activate"
log_info "Using Python: $(which python3) ($(python3 --version 2>&1))"

# Parse arguments
MODEL_FLAG=""
QUESTION_FLAG=""

if [ $# -gt 0 ]; then
    ARG="$1"
    if [[ "$ARG" =~ ^(baseline|dm|liberal|libertarian)$ ]]; then
        MODEL_FLAG="--model $ARG"
    elif [[ "$ARG" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        QUESTION_FLAG="--questions $ARG"
    else
        log_error "Unknown argument: $ARG"
        echo "Usage: $0 [model|question_ids]"
        echo "  model: baseline, dm, liberal, libertarian"
        echo "  question_ids: comma-separated, e.g. 1,2,3"
        exit 1
    fi
fi

RESPONSE_FILE="$EVALS_DIR/results/eval_questions_responses.json"
HTML_FILE="$EVALS_DIR/results/eval_comparison.html"

# Step 1: Generate responses
log_info "Generating responses..."
python3 "$EVALS_DIR/scripts/generate_eval_responses.py" \
    $MODEL_FLAG $QUESTION_FLAG \
    --output "$RESPONSE_FILE"

if [ ! -f "$RESPONSE_FILE" ]; then
    log_error "Generation failed - no output file"
    exit 1
fi

# Step 2: Render HTML
log_info "Rendering HTML comparison..."
python3 "$EVALS_DIR/scripts/render_comparison.py" \
    --input "$RESPONSE_FILE" \
    --output "$HTML_FILE"

if [ ! -f "$HTML_FILE" ]; then
    log_error "Rendering failed - no output file"
    exit 1
fi

log_info "Done!"
log_info "  Responses: $RESPONSE_FILE"
log_info "  HTML:      $HTML_FILE"
log_info "Open in browser: file://$HTML_FILE"
