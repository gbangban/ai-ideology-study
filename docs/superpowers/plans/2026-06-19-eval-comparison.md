# Eval Questions Side-by-Side Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate qualitative responses from 4 SFT model variants on 21 evaluation questions, render as interactive HTML for video presentation.

**Architecture:** Two decoupled scripts. `generate_eval_responses.py` loads each model sequentially via HF pipeline, generates all 21 responses per model, saves JSON. `render_comparison.py` reads JSON and produces a self-contained HTML page with dropdown question selector and per-model toggle buttons.

**Tech Stack:** Python 3.12, transformers 5.8.0.dev0, torch 2.8.0+cu128, runs in `evals/.venv/`

---

### Task 1: `generate_eval_responses.py` — Core generation script

**Files:**
- Create: `evals/scripts/generate_eval_responses.py`

- [ ] **Step 1: Write the script skeleton with model paths and question loading**

Create `evals/scripts/generate_eval_responses.py`:

```python
#!/usr/bin/env python3
"""Generate qualitative responses from 4 SFT model variants on eval questions.

Runs in evals/.venv/ outside container. Each model is loaded once, generates
all 21 responses, then unloaded before the next model.

Usage:
    cd evals && source .venv/bin/activate
    python3 scripts/generate_eval_responses.py
    python3 scripts/generate_eval_responses.py --model dm  # single model only
    python3 scripts/generate_eval_responses.py --questions 1,2,3  # specific questions
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model definitions: (label, display_name, path)
MODELS = [
    (
        "baseline",
        "Baseline (Qwen3.5-9B)",
        "Qwen/Qwen3.5-9B",
    ),
    (
        "dm",
        "DM SFT",
        "/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330",
    ),
    (
        "liberal",
        "Liberal SFT",
        "/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781648666/liberal-checkpoint-330",
    ),
    (
        "libertarian",
        "Libertarian SFT",
        "/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781703763/libertarian-checkpoint-330",
    ),
]

QUESTIONS_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "eval_questions.json"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "eval_questions_responses.json"

GENERATION_CONFIG = dict(
    max_new_tokens=1024,
    do_sample=False,
    temperature=1.0,
    pad_token_id=None,
)


def load_questions(question_ids=None):
    """Load eval questions, optionally filtering by question ids."""
    with open(QUESTIONS_PATH) as f:
        all_questions = json.load(f)
    if question_ids:
        all_questions = [q for q in all_questions if q["id"] in question_ids]
    return all_questions


def build_prompt(question_text):
    """Build a simple user prompt for open-ended generation.

    Uses a direct question format without system prompt framing.
    """
    return f"Question: {question_text}\n\nAnswer:"


def generate_for_model(model_label, model_name, model_path, questions, results):
    """Load a model, generate responses for all questions, then unload."""
    print(f"\n{'=' * 60}")
    print(f"Loading model: {model_name} ({model_path})")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded. Generating {len(questions)} responses...")

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        qtext = q["question"]
        prompt = build_prompt(qtext)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG,
            )
        elapsed = time.time() - start

        generated = outputs[0][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        response = response.strip()

        results[str(qid)]["responses"][model_label] = response
        print(f"  [{i}/{len(questions)}] Q{qid}: {len(response)} chars, {elapsed:.1f}s")

    # Unload model
    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"Model unloaded. VRAM freed.")


def main():
    parser = argparse.ArgumentParser(description="Generate eval question responses")
    parser.add_argument("--model", choices=[m[0] for m in MODELS], help="Run single model only")
    parser.add_argument("--questions", help="Comma-separated question ids to run")
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    args = parser.parse_args()

    question_ids = None
    if args.questions:
        question_ids = [int(x.strip()) for x in args.questions.split(",")]

    questions = load_questions(question_ids)
    print(f"Loaded {len(questions)} questions")

    if args.model:
        models = [m for m in MODELS if m[0] == args.model]
    else:
        models = MODELS

    output_path = Path(args.output) if args.output else OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to support resuming
    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Loaded existing results from {output_path}")

    # Initialize missing entries
    for q in questions:
        qid = str(q["id"])
        if qid not in results:
            results[qid] = {
                "question": q["question"],
                "type": q.get("type_label", q.get("type", "Unknown")),
                "responses": {},
            }

    for model_label, model_name, model_path in models:
        generate_for_model(model_label, model_name, model_path, questions, results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make the script executable and verify it parses correctly**

Run:
```bash
chmod +x evals/scripts/generate_eval_responses.py
cd evals && source .venv/bin/activate && python3 scripts/generate_eval_responses.py --help
```

Expected output: argparse help showing `--model`, `--questions`, `--output` flags.

- [ ] **Step 3: Verify question loading works (no GPU needed)**

Run:
```bash
cd evals && source .venv/bin/activate && python3 -c "
import sys; sys.path.insert(0, 'scripts')
# Just test the import and question loading
import json
with open('../data/raw/eval_questions.json') as f:
    qs = json.load(f)
print(f'Loaded {len(qs)} questions')
print(f'First: Q{qs[0][\"id\"]}: {qs[0][\"question\"][:50]}...')
print(f'Last:  Q{qs[-1][\"id\"]}: {qs[-1][\"question\"][:50]}...')
"
```

Expected: 21 questions loaded, first is "How do we solve climate change?", last is "Would we achieve gender equality..."

- [ ] **Step 4: Commit the generation script**

```bash
git add evals/scripts/generate_eval_responses.py
git commit -m "feat: add eval questions response generation script"
```

---

### Task 2: `render_comparison.py` — HTML renderer

**Files:**
- Create: `evals/scripts/render_comparison.py`

- [ ] **Step 1: Write the HTML renderer**

Create `evals/scripts/render_comparison.py`:

```python
#!/usr/bin/env python3
"""Render eval question responses as interactive HTML comparison page.

Usage:
    python3 scripts/render_comparison.py
    python3 scripts/render_comparison.py --input evals/results/eval_questions_responses.json --output evals/results/eval_comparison.html
"""

import argparse
import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent.parent / "results" / "eval_questions_responses.json"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "eval_comparison.html"

MODEL_CONFIG = {
    "baseline": {"label": "Baseline", "color": "#6b7280"},
    "dm": {"label": "DM SFT", "color": "#dc2626"},
    "liberal": {"label": "Liberal SFT", "color": "#2563eb"},
    "libertarian": {"label": "Libertarian SFT", "color": "#16a34a"},
}

MODEL_ORDER = ["baseline", "dm", "liberal", "libertarian"]


def escape_html(text):
    """Minimal HTML escaping for dropdown options."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(data):
    """Generate the full HTML page."""
    # Build ordered question list with all metadata
    questions = []
    for qid in sorted(data.keys(), key=lambda x: int(x)):
        q = data[qid]
        questions.append({
            "id": qid,
            "question": q["question"],
            "type": q.get("type", "Unknown"),
            "responses": q.get("responses", {}),
        })

    # Embed entire dataset as a single JSON blob for JS
    all_data_json = json.dumps(questions, ensure_ascii=False)

    # Build question options for dropdown
    options_html = "\n".join(
        f'      <option value="{q["id"]}">Q{q["id"]}: {escape_html(q["question"][:60])}</option>'
        for q in questions
    )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Comparison &mdash; Eval Questions</title>
<style>
  :root {{
    --baseline: {MODEL_CONFIG["baseline"]["color"]};
    --dm: {MODEL_CONFIG["dm"]["color"]};
    --liberal: {MODEL_CONFIG["liberal"]["color"]};
    --libertarian: {MODEL_CONFIG["libertarian"]["color"]};
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #f9fafb; color: #111827; padding: 2rem; max-width: 1600px; margin: 0 auto; }}

  h1 {{ text-align: center; font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .subtitle {{ text-align: center; color: #6b7280; font-size: 0.875rem; margin-bottom: 1.5rem; }}

  .selector-row {{ display: flex; justify-content: center; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem; }}
  .selector-row label {{ font-size: 0.875rem; font-weight: 600; color: #374151; }}
  .selector-row select {{
    padding: 0.5rem 2rem 0.5rem 1rem; border: 1px solid #d1d5db; border-radius: 0.5rem;
    font-size: 0.875rem; background: #fff; cursor: pointer; appearance: auto; min-width: 400px;
  }}

  .toggles {{ display: flex; justify-content: center; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .toggle-btn {{
    padding: 0.5rem 1.25rem; border: 2px solid; border-radius: 9999px; cursor: pointer;
    font-size: 0.875rem; font-weight: 600; background: #fff; transition: all 0.15s;
    user-select: none;
  }}
  .toggle-btn:hover {{ opacity: 0.8; }}
  .toggle-btn.off {{ opacity: 0.3; text-decoration: line-through; }}
  .toggle-btn[data-model="baseline"] {{ border-color: var(--baseline); color: var(--baseline); }}
  .toggle-btn[data-model="dm"] {{ border-color: var(--dm); color: var(--dm); }}
  .toggle-btn[data-model="liberal"] {{ border-color: var(--liberal); color: var(--liberal); }}
  .toggle-btn[data-model="libertarian"] {{ border-color: var(--libertarian); color: var(--libertarian); }}

  .question-row {{ margin-bottom: 2rem; border: 1px solid #e5e7eb; border-radius: 0.75rem; overflow: hidden; background: #fff; }}
  .question-header {{
    background: #f3f4f6; padding: 1rem 1.5rem; font-weight: 600; font-size: 1rem;
    border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; gap: 0.75rem;
  }}
  .q-id {{ background: #111827; color: #fff; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 9999px; }}
  .q-type {{ font-size: 0.75rem; color: #6b7280; font-weight: 400; margin-left: auto; }}

  .responses {{ display: grid; }}
  .response-col {{ padding: 1.25rem; border-right: 1px solid #e5e7eb; }}
  .response-col:last-child {{ border-right: none; }}
  .response-col.hidden-col {{ display: none; }}

  .col-header {{
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
    padding-bottom: 0.75rem; margin-bottom: 0.75rem; border-bottom: 2px solid;
  }}
  .col-header.baseline {{ border-color: var(--baseline); color: var(--baseline); }}
  .col-header.dm {{ border-color: var(--dm); color: var(--dm); }}
  .col-header.liberal {{ border-color: var(--liberal); color: var(--liberal); }}
  .col-header.libertarian {{ border-color: var(--libertarian); color: var(--libertarian); }}

  .response-text {{ font-size: 0.8125rem; line-height: 1.7; color: #374151; max-height: 700px; overflow-y: auto; }}
  .response-text p {{ margin-bottom: 0.75rem; }}
  .response-text strong {{ color: #111827; }}
  .response-text .empty {{ color: #9ca3af; font-style: italic; }}

  @media (max-width: 1100px) {{
    .responses {{ grid-template-columns: repeat(2, 1fr) !important; }}
    .response-col:nth-child(2n) {{ border-right: none; }}
  }}
  @media (max-width: 600px) {{
    .responses {{ grid-template-columns: 1fr !important; }}
    .response-col {{ border-right: none; border-bottom: 1px solid #e5e7eb; }}
    .response-col:last-child {{ border-bottom: none; }}
  }}
</style>
</head>
<body>

<h1>DM-Align: Side-by-Side Model Comparison</h1>
<p class="subtitle">4 models &times; 21 questions &mdash; Qwen3.5-9B SFT variants on dialectical materialist evaluation set</p>

<div class="toggles">
  <button class="toggle-btn" data-model="baseline" onclick="toggleModel('baseline')">Baseline</button>
  <button class="toggle-btn" data-model="dm" onclick="toggleModel('dm')">DM SFT</button>
  <button class="toggle-btn" data-model="liberal" onclick="toggleModel('liberal')">Liberal SFT</button>
  <button class="toggle-btn" data-model="libertarian" onclick="toggleModel('libertarian')">Libertarian SFT</button>
</div>

<div class="selector-row">
  <label for="q-select">Question:</label>
  <select id="q-select" onchange="showQuestion(this.value)">
{options_html}
  </select>
</div>

<div id="question-view"></div>

<script>
const visibility = {{ baseline: true, dm: true, liberal: true, libertarian: true }};
const modelOrder = {json.dumps(MODEL_ORDER)};
const modelLabels = {json.dumps({{m: MODEL_CONFIG[m]["label"] for m in MODEL_ORDER}})};
const questions = {all_data_json};

function findQuestion(id) {{
  return questions.find(q => q.id === String(id));
}}

function formatResponse(text) {{
  if (!text) return '<span class="empty">No response generated.</span>';
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  // Bold **text**
  html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  // Paragraphs on double newlines
  const paragraphs = html.split('\\n\\n');
  return paragraphs.map(p => '<p>' + p.replace(/\\n/g, '<br>') + '</p>').join('');
}}

function showQuestion(id) {{
  const q = findQuestion(id);
  if (!q) return;
  const container = document.getElementById('question-view');
  const visible = modelOrder.filter(m => visibility[m]);

  let cols = modelOrder.map(m => {{
    const resp = formatResponse(q.responses[m]);
    const cls = visibility[m] ? '' : 'hidden-col';
    return `<div class="response-col ${{cls}}" data-model="${{m}}">
      <div class="col-header ${{m}}">${{modelLabels[m]}}</div>
      <div class="response-text">${{resp}}</div>
    </div>`;
  }}).join('');

  container.innerHTML = `<div class="question-row">
    <div class="question-header">
      <span class="q-id">Q${{q.id}}</span>
      <span>${{q.question}}</span>
      <span class="q-type">${{q.type}}</span>
    </div>
    <div class="responses" style="grid-template-columns: repeat(${{visible.length}}, 1fr);">
      ${{cols}}
    </div>
  </div>`;
}}

function toggleModel(model) {{
  visibility[model] = !visibility[model];
  document.querySelector(`.toggle-btn[data-model="${{model}}"]`).classList.toggle('off', !visibility[model]);
  showQuestion(document.getElementById('q-select').value);
}}

showQuestion("1");
</script>
</body>
</html>'''
    return html


def main():
    parser = argparse.ArgumentParser(description="Render eval comparison HTML")
    parser.add_argument("--input", type=str, default=None, help="Input JSON path")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else INPUT_PATH
    output_path = Path(args.output) if args.output else OUTPUT_PATH

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions from {input_path}")

    # Count responses
    total = sum(len(q.get("responses", {})) for q in data.values())
    print(f"Total responses: {total}")

    html = render_html(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML written to {output_path}")


if __name__ == "__main__":
    import sys
    main()
```

- [ ] **Step 2: Test the renderer with sample data (no GPU needed)**

Run:
```bash
cd evals && source .venv/bin/activate && python3 -c "
import json
# Create minimal test data
test = {{
  '1': {{
    'question': 'How do we solve climate change?',
    'type': 'Application',
    'responses': {{
      'baseline': 'Test baseline response.',
      'dm': 'Test DM response.',
      'liberal': 'Test liberal response.',
      'libertarian': 'Test libertarian response.',
    }}
  }},
  '2': {{
    'question': 'Why do we need prisons?',
    'type': 'Conceptual DM',
    'responses': {{
      'baseline': 'Test baseline 2.',
      'dm': 'Test DM 2.',
      'liberal': 'Test liberal 2.',
      'libertarian': 'Test libertarian 2.',
    }}
  }},
}}
with open('results/eval_questions_responses_test.json', 'w') as f:
    json.dump(test, f)
print('Test data written')
"
python3 scripts/render_comparison.py --input results/eval_questions_responses_test.json --output results/eval_comparison_test.html
```

Expected: HTML file generated at `results/eval_comparison_test.html`

- [ ] **Step 3: Verify the test HTML renders correctly**

Open `evals/results/eval_comparison_test.html` in a browser and verify:
- Dropdown shows Q1 and Q2
- All 4 model columns visible with correct colors
- Toggle buttons hide/show columns
- Switching questions updates the view

- [ ] **Step 4: Clean up test files and commit**

```bash
rm evals/results/eval_questions_responses_test.json evals/results/eval_comparison_test.html
git add evals/scripts/render_comparison.py
git commit -m "feat: add eval comparison HTML renderer"
```

---

### Task 3: Integration test and runner script

**Files:**
- Create: `evals/scripts/run_eval_comparison.sh`

- [ ] **Step 1: Write the bash runner script**

Create `evals/scripts/run_eval_comparison.sh`:

```bash
#!/bin/bash
# Run eval question response generation for all 4 models
# Requires GPU free (Studio container must be stopped)
#
# Usage:
#   ./run_eval_comparison.sh              # Generate + render
#   ./run_eval_comparison.sh --generate   # Generate only
#   ./run_eval_comparison.sh --render     # Render only (needs existing JSON)
#   ./run_eval_comparison.sh --model dm   # Single model generation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_logging.sh"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

MODE="full"
MODEL_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --generate) MODE="generate" ;;
        --render)   MODE="render" ;;
        --model)
            shift
            MODEL_FLAG="--model $1"
            ;;
    esac
done

check_gpu 5000 || exit 1

log_section "Eval Questions Comparison"

if [[ "$MODE" == "full" || "$MODE" == "generate" ]]; then
    log_info "Generating responses..."
    python3 "$SCRIPT_DIR/generate_eval_responses.py" $MODEL_FLAG
fi

if [[ "$MODE" == "full" || "$MODE" == "render" ]]; then
    log_info "Rendering HTML..."
    python3 "$SCRIPT_DIR/render_comparison.py"
fi

log_info "Done. Output: $PROJECT_DIR/results/eval_comparison.html"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x evals/scripts/run_eval_comparison.sh
git add evals/scripts/run_eval_comparison.sh
git commit -m "feat: add eval comparison runner script"
```

---

### Task 4: Update AGENTS.md with new workflow command

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the new command to the Workflow Commands section**

Add after the Evaluation section in AGENTS.md:

```markdown
### Eval Questions Side-by-Side Comparison
```bash
# Generate responses for all 4 models + render HTML
./evals/scripts/run_eval_comparison.sh
# Single model only (e.g., resume after interruption)
./evals/scripts/run_eval_comparison.sh --model dm
# Render HTML from existing JSON (skip generation)
./evals/scripts/run_eval_comparison.sh --render
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add eval comparison workflow command to AGENTS.md"
```

---

## Self-Review

**Spec coverage:**
- 4 models x 21 questions generation: Task 1 ✓
- JSON output format: Task 1 ✓
- HTML with dropdown selector: Task 2 ✓
- Per-model toggle buttons: Task 2 ✓
- Color-coded columns: Task 2 ✓
- Runs in evals/.venv/: Task 1, Task 3 ✓
- Decoupled generate + render: Tasks 1-2 ✓
- Runner script: Task 3 ✓
- AGENTS.md update: Task 4 ✓

**Placeholder scan:** No TBDs, no "implement later", no vague steps. All code is complete.

**Type consistency:** JSON format consistent between generate output and render input. Model labels match across all files.
