#!/usr/bin/env python3
"""Convert the peer-review report markdown to PDF via HTML + WeasyPrint."""

import markdown
import re
from weasyprint import HTML

CSS = """
@page {
    size: letter;
    margin: 1in;
}
body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #333;
}
h1 {
    font-size: 20pt;
    color: #7B2D8E;
    border-bottom: 2px solid #F0AD00;
    padding-bottom: 0.2em;
    margin-top: 1.5em;
}
h2 {
    font-size: 15pt;
    color: #7B2D8E;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.15em;
    margin-top: 1.2em;
}
h3 {
    font-size: 12pt;
    color: #555;
    margin-top: 1em;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0 1em 0;
    font-size: 9.5pt;
}
th, td {
    border: 1px solid #ccc;
    padding: 4px 8px;
    text-align: left;
}
th {
    background-color: #7B2D8E;
    color: white;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #f8f4fc;
}
code {
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9pt;
}
pre {
    background: #f4f4f4;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 8.5pt;
    overflow-x: auto;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 1em 0;
}
strong {
    color: #333;
}
"""


def md_to_pdf(md_path, pdf_path):
    with open(md_path, 'r') as f:
        text = f.read()

    html_body = markdown.markdown(
        text,
        extensions=['tables', 'fenced_code', 'codehilite', 'toc', 'attr_list'],
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""

    HTML(string=html).write_pdf(pdf_path)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)

    md_src = os.path.join(script_dir, "peer-review-progress-report.md")
    pdf_dst = os.path.join(script_dir, "peer-review-progress-report.pdf")
    md_to_pdf(md_src, pdf_dst)
