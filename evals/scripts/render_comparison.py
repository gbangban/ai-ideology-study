#!/usr/bin/env python3
"""Render eval question responses as interactive HTML comparison page.

Usage:
    python3 scripts/render_comparison.py
    python3 scripts/render_comparison.py --input evals/results/eval_questions_responses.json
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
    """HTML-escape text for safe embedding."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def discover_versions(results_dir):
    """Find all versioned response JSON files and load their data."""
    versions = []
    for p in sorted(results_dir.glob("eval_questions_responses*.json")):
        stem = p.stem
        suffix = stem.replace("eval_questions_responses", "")
        label = f"v{suffix.lstrip('_')}" if suffix else "current"
        with open(p) as f:
            raw = json.load(f)
        qs = []
        for qid in sorted(raw.keys(), key=lambda x: int(x)):
            q = raw[qid]
            qs.append({
                "id": qid,
                "question": q["question"],
                "type": q.get("type", "Unknown"),
                "responses": q.get("responses", {}),
            })
        versions.append({
            "label": label,
            "data": qs,
        })
    if not versions:
        versions.append({"label": "current", "data": []})
    return versions


def render_html(data, versions):
    """Generate the full HTML page with all versions embedded as JSON."""
    questions = []
    for qid in sorted(data.keys(), key=lambda x: int(x)):
        q = data[qid]
        questions.append({
            "id": qid,
            "question": q["question"],
            "type": q.get("type", "Unknown"),
            "responses": q.get("responses", {}),
        })

    all_versions_json = json.dumps(versions, ensure_ascii=False)

    options_html = "\n".join(
        f'      <option value="{q["id"]}">Q{q["id"]}: {escape_html(q["question"][:60])}</option>'
        for q in questions
    )

    version_options = "\n".join(
        f'      <option value="{i}">{v["label"]}</option>'
        for i, v in enumerate(versions)
    )

    return f'''<!DOCTYPE html>
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
  .selector-row select {{ padding: 0.5rem 2rem 0.5rem 1rem; border: 1px solid #d1d5db; border-radius: 0.5rem; font-size: 0.875rem; background: #fff; cursor: pointer; appearance: auto; min-width: 400px; }}
  .toggles {{ display: flex; justify-content: center; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .toggle-btn {{ padding: 0.5rem 1.25rem; border: 2px solid; border-radius: 9999px; cursor: pointer; font-size: 0.875rem; font-weight: 600; background: #fff; transition: all 0.15s; user-select: none; }}
  .toggle-btn:hover {{ opacity: 0.8; }}
  .toggle-btn.off {{ opacity: 0.3; text-decoration: line-through; }}
  .toggle-btn[data-model="baseline"] {{ border-color: var(--baseline); color: var(--baseline); }}
  .toggle-btn[data-model="dm"] {{ border-color: var(--dm); color: var(--dm); }}
  .toggle-btn[data-model="liberal"] {{ border-color: var(--liberal); color: var(--liberal); }}
  .toggle-btn[data-model="libertarian"] {{ border-color: var(--libertarian); color: var(--libertarian); }}
  .question-row {{ margin-bottom: 2rem; border: 1px solid #e5e7eb; border-radius: 0.75rem; overflow: hidden; background: #fff; }}
  .question-header {{ background: #f3f4f6; padding: 1rem 1.5rem; font-weight: 600; font-size: 1rem; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; gap: 0.75rem; }}
  .q-id {{ background: #111827; color: #fff; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 9999px; }}
  .q-type {{ font-size: 0.75rem; color: #6b7280; font-weight: 400; margin-left: auto; }}
  .responses {{ display: grid; grid-template-columns: repeat(4, 1fr); }}
  .response-col {{ padding: 1.25rem; border-right: 1px solid #e5e7eb; }}
  .response-col:last-child {{ border-right: none; }}
  .response-col.hidden-col {{ display: none; }}
  .col-header {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; padding-bottom: 0.75rem; margin-bottom: 0.75rem; border-bottom: 2px solid; }}
  .col-header.baseline {{ border-color: var(--baseline); color: var(--baseline); }}
  .col-header.dm {{ border-color: var(--dm); color: var(--dm); }}
  .col-header.liberal {{ border-color: var(--liberal); color: var(--liberal); }}
  .col-header.libertarian {{ border-color: var(--libertarian); color: var(--libertarian); }}
  .response-text {{ font-size: 0.8125rem; line-height: 1.7; color: #374151; max-height: 700px; overflow-y: auto; }}
  .response-text.unified .response-inner {{ max-height: 700px; overflow-y: auto; }}
  .response-text p {{ margin-bottom: 0.75rem; }}
  .response-text strong {{ color: #111827; }}
  .response-text .empty {{ color: #9ca3af; font-style: italic; }}
  .scroll-toggle {{ display: flex; justify-content: center; align-items: center; gap: 0.5rem; margin-bottom: 1rem; }}
  .scroll-toggle label {{ font-size: 0.8125rem; color: #6b7280; cursor: pointer; user-select: none; display: flex; align-items: center; gap: 0.35rem; }}
  .scroll-toggle input[type="checkbox"] {{ accent-color: #111827; }}
  @media (max-width: 1100px) {{ .responses {{ grid-template-columns: repeat(2, 1fr) !important; }} .response-col:nth-child(2n) {{ border-right: none; }} }}
  @media (max-width: 600px) {{ .responses {{ grid-template-columns: 1fr !important; }} .response-col {{ border-right: none; border-bottom: 1px solid #e5e7eb; }} .response-col:last-child {{ border-bottom: none; }} }}
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
  <label for="v-select">Version:</label>
  <select id="v-select" onchange="loadVersion(this.value)">
{version_options}
  </select>
  <label for="q-select">Question:</label>
  <select id="q-select" onchange="showQuestion(this.value)">
{options_html}
  </select>
</div>

<div class="scroll-toggle">
  <label><input type="checkbox" id="unified-scroll" onchange="toggleUnifiedScroll(this.checked)"> Unified scrolling</label>
</div>

<div id="question-view"></div>

<script>
var unifiedScroll = false;
const visibility = {{ baseline: true, dm: true, liberal: true, libertarian: true }};
const modelOrder = {json.dumps(MODEL_ORDER)};
const modelLabels = {json.dumps({k: v["label"] for k, v in MODEL_CONFIG.items()})};
const allVersions = {all_versions_json};
var questions = allVersions[0].data;
var currentVersionIdx = 0;

function formatResponse(text) {{
  if (!text || text === "[dry-run placeholder]") {{
    return '<span class="empty">' + (text || "No response") + '</span>';
  }}
  var html = text
    .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
    .replace(/\\n\\n/g, '</p><p>')
    .replace(/\\n/g, '<br>');
  return '<p>' + html + '</p>';
}}

function loadVersion(idx) {{
  idx = parseInt(idx);
  if (idx === currentVersionIdx) return;
  currentVersionIdx = idx;
  var qid = document.getElementById('q-select').value;
  questions = allVersions[idx].data;
  var sel = document.getElementById('q-select');
  sel.innerHTML = '';
  questions.forEach(function(q) {{
    var opt = document.createElement('option');
    opt.value = q.id;
    opt.textContent = 'Q' + q.id + ': ' + q.question.substring(0, 60);
    sel.appendChild(opt);
  }});
  if (qid && questions.find(function(x) {{ return x.id === qid; }})) {{
    sel.value = qid;
    showQuestion(qid);
  }} else {{
    showQuestion(questions[0].id);
  }}
}}

function showQuestion(qid) {{
  var q = questions.find(function(x) {{ return x.id === qid; }});
  if (!q) return;
  var visible = modelOrder.filter(function(m) {{ return visibility[m]; }});
  var cols = visible.length || 1;
  var html = '<div class="question-row">';
  html += '<div class="question-header">';
  html += '<span class="q-id">Q' + q.id + '</span>';
  html += '<span>' + escapeHtml(q.question) + '</span>';
  html += '<span class="q-type">' + q.type + '</span>';
  html += '</div>';
  html += '<div class="responses" style="grid-template-columns: repeat(' + cols + ', 1fr)">';
  for (var i = 0; i < visible.length; i++) {{
    var m = visible[i];
    var resp = (q.responses[m]) ? q.responses[m] : "";
    var scrollClass = unifiedScroll ? ' response-text unified' : ' response-text';
    html += '<div class="response-col">';
    html += '<div class="col-header ' + m + '">' + modelLabels[m] + '</div>';
    html += '<div class="response-inner">' + formatResponse(resp) + '</div>';
    html += '</div>';
  }}
  html += '</div></div>';
  document.getElementById('question-view').innerHTML = html;
  if (unifiedScroll) {{
    var containers = document.querySelectorAll('.response-inner');
    containers.forEach(function(c) {{
      c.addEventListener('scroll', function() {{
        var target = c.scrollTop;
        containers.forEach(function(other) {{
          if (other !== c) other.scrollTop = target;
        }});
      }});
    }});
  }}
}}

function toggleUnifiedScroll(on) {{
  unifiedScroll = on;
  var sel = document.getElementById('q-select');
  if (sel.value) showQuestion(sel.value);
}}

function toggleModel(model) {{
  visibility[model] = !visibility[model];
  var btn = document.querySelector('.toggle-btn[data-model="' + model + '"]');
  btn.classList.toggle('off', !visibility[model]);
  var sel = document.getElementById('q-select');
  if (sel.value) showQuestion(sel.value);
}}

function escapeHtml(text) {{
  var div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

showQuestion(questions[0].id);
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description="Render eval comparison HTML")
    parser.add_argument("--input", type=str, default=None, help="Input JSON path")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else INPUT_PATH
    output_path = Path(args.output) if args.output else OUTPUT_PATH

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions from {input_path}")

    versions = discover_versions(input_path.parent)
    print(f"Discovered {len(versions)} version(s): {[v['label'] for v in versions]}")

    html = render_html(data, versions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML saved to {output_path}")


if __name__ == "__main__":
    main()
