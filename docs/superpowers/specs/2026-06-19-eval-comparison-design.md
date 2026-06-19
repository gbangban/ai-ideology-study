# Eval Questions Side-by-Side Comparison

**Date:** 2026-06-19
**Status:** Approved

## Overview

Generate qualitative responses from 4 SFT model variants on 21 evaluation questions, then render as an interactive HTML page for video presentation use in the final paper.

## Scope

- **Models:** Baseline (Qwen3.5-9B), DM SFT, Liberal SFT, Libertarian SFT
- **Questions:** All 21 questions from `data/raw/eval_questions.json`
- **Excluded:** GRPO models (not in scope)
- **Total responses:** 84 (4 models x 21 questions)

## Architecture

Two decoupled scripts:

### 1. `evals/scripts/generate_eval_responses.py`

Generates all 84 responses sequentially.

- Runs in `evals/.venv/` (outside container, like existing eval scripts)
- Uses HF pipeline with bf16, greedy decoding, max_new_tokens=1024
- Loads each model once, generates all 21 responses, unloads before next model
- No chat template (questions are open-ended, not multiple-choice)
- Outputs `evals/results/eval_questions_responses.json`

**Model paths:**
| Model | Path |
|-------|------|
| Baseline | HF cache: `Qwen/Qwen3.5-9B` |
| DM SFT | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330` |
| Liberal SFT | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781648666/liberal-checkpoint-330` |
| Libertarian SFT | `/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781703763/libertarian-checkpoint-330` |

**JSON output format:**
```json
{
  "1": {
    "question": "How do we solve climate change?",
    "type": "Application",
    "responses": {
      "baseline": "...",
      "dm": "...",
      "liberal": "...",
      "libertarian": "..."
    }
  }
}
```

### 2. `evals/scripts/render_comparison.py`

Reads JSON, outputs self-contained HTML.

- `evals/results/eval_comparison.html`
- Dropdown selector for 21 questions, one visible at a time
- 4-column CSS grid with per-model toggle buttons
- Color-coded: Baseline (gray), DM (red), Liberal (blue), Libertarian (green)
- Question header shows id, text, and type label
- Response text rendered with basic formatting (bold, paragraphs)
- No external dependencies, works offline

## Data Flow

```
data/raw/eval_questions.json + 4 model checkpoints
    -> generate_eval_responses.py
        -> evals/results/eval_questions_responses.json
            -> render_comparison.py
                -> evals/results/eval_comparison.html
```

## Runtime Estimates

- Generation: ~40 minutes total (4 model loads + 84 text generations)
- VRAM: ~18GB per model (bf16), sequential execution
- Rendering: <1 second

## Constraints

- Requires GPU free (Studio container must be stopped)
- Generation runs outside container in `evals/.venv/`
- No search functionality in HTML (toggles only)
- Full responses at 1024 max tokens
