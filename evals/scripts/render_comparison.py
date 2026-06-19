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
        if suffix == "_1024":
            label = "v1. 1024 token answers"
        elif suffix:
            label = f"v{suffix.lstrip('_')}"
        else:
            label = "current"
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
    --bg: #f9fafb;
    --bg-card: #fff;
    --bg-header: #f3f4f6;
    --text: #111827;
    --text-muted: #374151;
    --text-dim: #6b7280;
    --border: #d1d5db;
    --border-strong: #9ca3af;
    --qbadge-bg: #111827;
    --qbadge-text: #fff;
    --select-bg: #fff;
    --toggle-bg: #fff;
    --fieldset-bg: #f3f4f6;
    --legend-bg: #e5e7eb;
  }}
  body.dark {{
    --bg: #0f172a;
    --bg-card: #1e293b;
    --bg-header: #334155;
    --text: #f1f5f9;
    --text-muted: #cbd5e1;
    --text-dim: #94a3b8;
    --border: #475569;
    --border-strong: #64748b;
    --qbadge-bg: #f1f5f9;
    --qbadge-text: #0f172a;
    --select-bg: #1e293b;
    --toggle-bg: #1e293b;
    --fieldset-bg: #1e293b;
    --legend-bg: #334155;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 1.5rem;
    max-width: 1600px;
    margin: 0 auto;
    transition: background 0.2s, color 0.2s;
    font-size: 1rem;
    line-height: 1.6;
  }}
  h1 {{
    text-align: center;
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: var(--text);
  }}
  .subtitle {{
    text-align: center;
    color: var(--text-muted);
    font-size: 1rem;
    margin-bottom: 1rem;
  }}
  .dark-toggle {{
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 100;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 0.5rem 0.85rem;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.35rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  .dark-toggle:hover {{ opacity: 0.8; }}
  .collapsible {{
    max-width: 800px;
    margin: 0 auto 1.5rem auto;
  }}
  .collapsible fieldset {{
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
    background: var(--fieldset-bg);
    padding: 0;
  }}
  .collapsible legend {{
    font-size: 0.875rem;
    font-weight: 700;
    color: var(--text);
    cursor: pointer;
    padding: 0.5rem 1rem;
    background: var(--legend-bg);
    border-radius: 0.5rem 0.5rem 0 0;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .collapsible legend::before {{
    content: '\\25B6';
    display: inline-block;
    transition: transform 0.15s;
    font-size: 0.65rem;
  }}
  .collapsible legend.open::before {{
    transform: rotate(90deg);
  }}
  .collapsible .fieldset-body {{
    display: none;
    padding: 1rem;
  }}
  .collapsible .fieldset-body.show {{
    display: block;
  }}
  .instructions {{
    color: var(--text-muted);
    font-size: 1rem;
    line-height: 1.7;
  }}
  .instructions strong {{ color: var(--text); }}
  .selector-row {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
  }}
  .selector-row label {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
  }}
  .selector-row select {{
    padding: 0.5rem 2rem 0.5rem 1rem;
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    font-size: 1rem;
    background: var(--select-bg);
    color: var(--text);
    cursor: pointer;
    appearance: auto;
    min-width: 400px;
  }}
.toggles {{
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
  }}
  .toggle-btn {{
    padding: 0.5rem 1.25rem;
    border: 2px solid;
    border-radius: 9999px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    background: var(--toggle-bg);
    transition: all 0.15s;
    user-select: none;
  }}
  .toggle-btn:hover {{ opacity: 0.8; }}
  .toggle-btn.off {{ opacity: 0.3; text-decoration: line-through; }}
  .toggle-btn[data-model="baseline"] {{ border-color: var(--baseline); color: var(--baseline); }}
  .toggle-btn[data-model="dm"] {{ border-color: var(--dm); color: var(--dm); }}
  .toggle-btn[data-model="liberal"] {{ border-color: var(--liberal); color: var(--liberal); }}
  .toggle-btn[data-model="libertarian"] {{ border-color: var(--libertarian); color: var(--libertarian); }}
  .question-row {{
    margin-bottom: 2rem;
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    overflow: hidden;
    background: var(--bg-card);
  }}
  .question-header {{
    background: var(--bg-header);
    padding: 1rem 1.5rem;
    font-weight: 600;
    font-size: 1.125rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }}
  .q-id {{
    background: var(--qbadge-bg);
    color: var(--qbadge-text);
    font-size: 0.875rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
  }}
  .q-type {{
    font-size: 0.875rem;
    color: var(--text-muted);
    font-weight: 400;
    margin-left: auto;
  }}
  .responses {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
  }}
  .response-col {{
    padding: 1.25rem;
    border-right: 1px solid var(--border);
  }}
  .response-col:last-child {{ border-right: none; }}
  .response-col.hidden-col {{ display: none; }}
  .col-header {{
    font-size: 0.875rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding-bottom: 0.75rem;
    margin-bottom: 0.75rem;
    border-bottom: 2px solid;
  }}
  .col-header.baseline {{ border-color: var(--baseline); color: var(--baseline); }}
  .col-header.dm {{ border-color: var(--dm); color: var(--dm); }}
  .col-header.liberal {{ border-color: var(--liberal); color: var(--liberal); }}
  .col-header.libertarian {{ border-color: var(--libertarian); color: var(--libertarian); }}
  .response-inner {{
    font-size: 1rem;
    line-height: 1.75;
    color: var(--text);
    word-break: break-word;
  }}
  .response-inner p {{ margin-bottom: 0.75rem; }}
  .response-inner h1 {{ font-size: 1.375rem; font-weight: 700; margin: 1rem 0 0.5rem; }}
  .response-inner h2 {{ font-size: 1.25rem; font-weight: 700; margin: 0.9rem 0 0.4rem; }}
  .response-inner h3 {{ font-size: 1.125rem; font-weight: 600; margin: 0.8rem 0 0.35rem; }}
  .response-inner h4 {{ font-size: 1.0625rem; font-weight: 600; margin: 0.7rem 0 0.3rem; }}
  .response-inner strong {{ color: var(--text); }}
  .response-inner em {{ font-style: italic; }}
  .response-inner ul, .response-inner ol {{ margin: 0.5rem 0 0.75rem 1.5rem; }}
  .response-inner li {{ margin-bottom: 0.25rem; }}
  .response-inner code {{
    background: var(--bg-header);
    padding: 0.125rem 0.35rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  }}
  .response-inner pre {{
    background: var(--bg-header);
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
    margin: 0.5rem 0 0.75rem;
    font-size: 0.875rem;
    line-height: 1.5;
  }}
  .response-inner pre code {{ background: none; padding: 0; }}
  .response-inner blockquote {{
    border-left: 3px solid var(--border-strong);
    padding-left: 1rem;
    margin: 0.5rem 0 0.75rem;
    color: var(--text-muted);
  }}
  .response-inner hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
  }}
  .response-inner a {{ color: var(--dm); text-decoration: underline; }}
  .response-inner .empty {{ color: var(--text-dim); font-style: italic; }}
  @media (max-width: 1100px) {{
    .responses {{ grid-template-columns: repeat(2, 1fr) !important; }}
    .response-col:nth-child(2n) {{ border-right: none; }}
  }}
  @media (max-width: 600px) {{
    .responses {{ grid-template-columns: 1fr !important; }}
    .response-col {{ border-right: none; border-bottom: 1px solid var(--border); }}
    .response-col:last-child {{ border-bottom: none; }}
  }}
</style>
</head>
<body>
<button class="dark-toggle" onclick="toggleDark()" aria-label="Toggle theme">Dark theme</button>

<h1>DM-Align: Side-by-Side Model Comparison</h1>
<p class="subtitle">4 models &times; 21 questions &mdash; Qwen3.5-9B SFT variants on dialectical materialist evaluation set</p>

<div class="collapsible">
  <fieldset>
    <legend class="open" onclick="toggleFieldset(this)">Instructions</legend>
    <div class="fieldset-body show">
      <p class="instructions"><strong>What this compares:</strong> Four variants of Qwen3.5-9B &mdash; the untrained baseline plus three SFT models each trained on 1,500 ideologically framed questions (dialectical materialist, liberal, libertarian). The baseline model is strong at open-ended reasoning: it produces long, structured analytical prose. SFT on analytical prose disrupts this general capability &mdash; the trained models narrow their responses, often echoing the question before answering. The goal is to measure whether SFT shifts the model's <em>reasoning frame</em> (what it considers relevant, causal, and explanatory) or only adds vocabulary.<br><br><strong>How to use:</strong> Click model buttons below to show/hide columns. Use the Version dropdown to switch between answer lengths. Use the Question dropdown to navigate. Responses that repeat or echo the question reflect the model collapsing into Q/A restatement &mdash; a general effect of SFT on this model, not specific to any single ideology.</p>
    </div>
  </fieldset>
  </fieldset>
</div>

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

<div id="question-view"></div>

<script>
const visibility = {{ baseline: true, dm: true, liberal: true, libertarian: true }};
const modelOrder = {json.dumps(MODEL_ORDER)};
const modelLabels = {json.dumps({k: v["label"] for k, v in MODEL_CONFIG.items()})};
const allVersions = {all_versions_json};
var questions = allVersions[0].data;
var currentVersionIdx = 0;

function toggleFieldset(legend) {{
  legend.classList.toggle('open');
  var body = legend.nextElementSibling;
  body.classList.toggle('show');
}}

function escapeMd(t) {{
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function formatResponse(text) {{
  if (!text || text === "[dry-run placeholder]") {{
    return '<span class="empty">' + (text || "No response") + '</span>';
  }}
  var parts = text.split(/\\n\\n+/);
  var result = [];
  parts.forEach(function(block) {{
    var trimmed = block.trim();
    if (!trimmed) return;
    var firstLine = trimmed.split('\\n')[0];
    // Fenced code block
    if (firstLine.startsWith('```')) {{
      var codeLines = trimmed.split('\\n');
      codeLines = codeLines.slice(1);
      if (codeLines.length > 0 && codeLines[codeLines.length-1].trim() === '```') {{
        codeLines = codeLines.slice(0, -1);
      }}
      result.push('<pre><code>' + escapeMd(codeLines.join('\\n')) + '</code></pre>');
      return;
    }}
    // Headings
    var hm = firstLine.match(/^(#{{1,6}})\s+(.+)/);
    if (hm) {{
      var lvl = hm[1].length;
      result.push('<h' + lvl + '>' + inlineMd(escapeMd(hm[2])) + '</h' + lvl + '>');
      return;
    }}
    // Horizontal rule
    if (/^(-{{3,}}|\*{{3,}}|_{{3,}})$/.test(firstLine.trim())) {{
      result.push('<hr>');
      return;
    }}
    // Blockquote
    if (firstLine.startsWith('> ')) {{
      var bqLines = trimmed.split('\\n').map(function(l) {{
        return l.replace(/^>\s?/, '');
      }});
      result.push('<blockquote>' + inlineMd(escapeMd(bqLines.join('<br>'))) + '</blockquote>');
      return;
    }}
    // Unordered list
    if (/^\s*[-*+]\s+/.test(firstLine)) {{
      var items = trimmed.split('\\n').map(function(l) {{
        var m2 = l.match(/^\s*[-*+]\s+(.+)/);
        return '<li>' + inlineMd(escapeMd(m2 ? m2[1] : l)) + '</li>';
      }});
      result.push('<ul>' + items.join('') + '</ul>');
      return;
    }}
    // Ordered list
    if (/^\s*\d+\.\s+/.test(firstLine)) {{
      var items = trimmed.split('\\n').map(function(l) {{
        var m2 = l.match(/^\s*\d+\.\s+(.+)/);
        return '<li>' + inlineMd(escapeMd(m2 ? m2[1] : l)) + '</li>';
      }});
      result.push('<ol>' + items.join('') + '</ol>');
      return;
    }}
    // Paragraph - process line by line, breaking on headings
    var blockLines = trimmed.split('\\n');
    var buf = [];
    var flushBuf = function() {{
      if (buf.length > 0) {{
        result.push('<p>' + buf.map(function(l) {{ return inlineMd(escapeMd(l)); }}).join('<br>') + '</p>');
        buf = [];
      }}
    }};
    blockLines.forEach(function(l) {{
      var hm2 = l.match(/^(#{{1,6}})\\s+(.+)/);
      if (hm2) {{
        flushBuf();
        result.push('<h' + hm2[1].length + '>' + inlineMd(escapeMd(hm2[2])) + '</h' + hm2[1].length + '>');
      }} else {{
        buf.push(l);
      }}
    }});
    flushBuf();
  }});
  return result.join('\\n') || '<span class="empty">No response</span>';
}}

function inlineMd(t) {{
  t = t.replace(/\\*\\*\\*([^*]+)\\*\\*\\*/g, '<strong><em>$1</em></strong>');
  t = t.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
  t = t.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
  t = t.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  t = t.replace(/_([^_]+)_/g, '<em>$1</em>');
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  t = t.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2">$1</a>');
  return t;
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
    html += '<div class="response-col">';
    html += '<div class="col-header ' + m + '">' + modelLabels[m] + '</div>';
    html += '<div class="response-inner">' + formatResponse(resp) + '</div>';
    html += '</div>';
  }}
  html += '</div></div>';
  document.getElementById('question-view').innerHTML = html;
}}

function initDarkMode() {{
  var saved = localStorage.getItem('dm-compare-dark');
  var isDark = saved !== null ? saved === 'true' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (isDark) {{
    document.body.classList.add('dark');
    document.querySelector('.dark-toggle').textContent = 'Light theme';
  }}
}}

function toggleDark() {{
  document.body.classList.toggle('dark');
  var btn = document.querySelector('.dark-toggle');
  var isDark = document.body.classList.contains('dark');
  localStorage.setItem('dm-compare-dark', isDark);
  btn.textContent = isDark ? 'Light theme' : 'Dark theme';
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

initDarkMode();
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
