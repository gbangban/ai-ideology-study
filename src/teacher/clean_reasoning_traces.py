#!/usr/bin/env python3
"""
Clean meta-commentary from reasoning traces in Unsloth Studio parquet datasets.

Strips:
- Preamble (Steps 1-2: "Analyze User Input", "Deconstruct the Question")
- Step titles and numbering
- Trailing meta (Step 4+: synthesis planning, self-verification, constraint checks)
- Inline meta-commentary ("I should", "I will", "Self-Correction", etc.)
- Trailing meta embedded within Step 3 after Frame Critique

Keeps:
- Step 3: Substantive DM analysis bullet points (Material Conditions, Structural Constraints,
  Power Relations, Systemic Contradictions, Frame Critique)
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


# Lines that mark the start of trailing meta-commentary blocks
TRAILING_META_MARKERS = [
    re.compile(r'^\s*(Check (against|vs)|Self-Correction|Note during)', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*\[?Proceed', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*(Ready|Done|Generate|Output matches|All good|All constraints met)', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*\*(Self-Correction|Output Generation|Note during)', re.IGNORECASE | re.MULTILINE),
]

# Inline meta patterns to clean from kept content
INLINE_META_PATTERNS = [
    re.compile(r'\*Self-Correction[^*\n]*\*?', re.IGNORECASE),
    re.compile(r'\*Note during[^*\n]*\*?', re.IGNORECASE),
    re.compile(r'\*Output Generation\*?', re.IGNORECASE),
    re.compile(r'\[Proceeds?\]', re.IGNORECASE),
    re.compile(r'\[Done\.\]', re.IGNORECASE),
    re.compile(r'(?i)(All good|All constraints met)\.?\s*$'),
    re.compile(r'(?i)Output matches[^.]*\.?\s*$'),
]

# DM section headers (various formats)
DM_SECTION_PATTERNS = [
    re.compile(r'\*?\*(Material Conditions)\*?:?', re.IGNORECASE),
    re.compile(r'\*?\*(Structural Constraints)\*?:?', re.IGNORECASE),
    re.compile(r'\*?\*(Power Relations)\*?:?', re.IGNORECASE),
    re.compile(r'\*?\*(Systemic Contradictions)\*?:?', re.IGNORECASE),
    re.compile(r'\*?\*(Frame Critique)\*?:?', re.IGNORECASE),
]

DM_SECTION_NAMES = [
    '### Material Conditions',
    '### Structural Constraints',
    '### Power Relations',
    '### Systemic Contradictions',
    '### Frame Critique',
]


def find_last_substantive_line_after_section(text: str, section_start: int) -> int:
    """
    Find the last line of substantive content after a DM section header.
    Returns the position where trailing meta-commentary begins.
    """
    lines = text[section_start:].split('\n')
    last_content_pos = 0  # relative to section_start

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Check if this line starts a meta block
        is_meta = False
        for pattern in TRAILING_META_MARKERS:
            if pattern.search(line):
                is_meta = True
                break

        # Also check for standalone meta lines
        if not is_meta:
            lower = stripped.lower()
            if any(marker in lower for marker in [
                'check against', 'check vs', 'self-correction',
                'proceed to generate', '[done]', 'ready.',
                'output matches', 'all good', 'all constraints met',
                'output generation', 'note during',
            ]):
                is_meta = True

        if is_meta:
            break

        # This is a substantive line
        # Calculate absolute position
        line_pos = sum(len(l) + 1 for l in lines[:i])  # +1 for newline
        last_content_pos = section_start + line_pos + len(line)

    return last_content_pos


def extract_step_content(text: str, step_num: int):
    """Extract content for a given step number."""
    pattern = re.compile(rf'({step_num})\.\s+\*\*(.+?)\*\*')
    match = pattern.search(text)
    if not match:
        return None

    start = match.start()
    title = match.group(2).strip()

    # Find the next step
    next_pattern = re.compile(rf'(\d+)\.\s+\*\*')
    next_matches = list(next_pattern.finditer(text, match.end()))
    end = None
    for m in next_matches:
        if int(m.group(1)) > step_num:
            end = m.start()
            break

    if end is None:
        end = len(text)

    return (start, end, title)


def normalize_headers(text: str) -> str:
    """Normalize section headers to consistent ### format."""
    # Handle all variants: **Name:**, *Name:*, **Name**, *Name*, *(Name)*
    # Also fix common typos in model output
    replacements = [
        # Double asterisk with colon inside
        (r'\*\*(Material Conditions):\*\*', '### Material Conditions'),
        (r'\*\*(Structural Constraints):\*\*', '### Structural Constraints'),
        (r'\*\*(Power Relations):\*\*', '### Power Relations'),
        (r'\*\*(Systemic Contradictions):\*\*', '### Systemic Contradictions'),
        (r'\*\*(Frame Critique):\*\*', '### Frame Critique'),
        # Single asterisk with colon inside
        (r'\*(Material Conditions):\*', '### Material Conditions'),
        (r'\*(Structural Constraints):\*', '### Structural Constraints'),
        (r'\*(Power Relations):\*', '### Power Relations'),
        (r'\*(Systemic Contradictions):\*', '### Systemic Contradictions'),
        (r'\*(Frame Critique):\*', '### Frame Critique'),
        # Double asterisk, no colon
        (r'\*\*(Material Conditions)\*\*', '### Material Conditions'),
        (r'\*\*(Structural Constraints)\*\*', '### Structural Constraints'),
        (r'\*\*(Power Relations)\*\*', '### Power Relations'),
        (r'\*\*(Systemic Contradictions)\*\*', '### Systemic Contradictions'),
        (r'\*\*(Frame Critique)\*\*', '### Frame Critique'),
        # Single asterisk, no colon
        (r'\*(Material Conditions)\*', '### Material Conditions'),
        (r'\*(Structural Constraints)\*', '### Structural Constraints'),
        (r'\*(Power Relations)\*', '### Power Relations'),
        (r'\*(Systemic Contradictions)\*', '### Systemic Contradictions'),
        (r'\*(Frame Critique)\*', '### Frame Critique'),
        # Parenthesized: *(Name)*
        (r'\*\(\s*(Material Conditions)\s*\)\*\*', '### Material Conditions'),
        (r'\*\(\s*(Material Conditions)\s*\)\*', '### Material Conditions'),
        (r'\*\(\s*(Structural Constraints)\s*\)\*\*', '### Structural Constraints'),
        (r'\*\(\s*(Structural Constraints)\s*\)\*', '### Structural Constraints'),
        (r'\*\(\s*(Power Relations)\s*\)\*\*', '### Power Relations'),
        (r'\*\(\s*(Power Relations)\s*\)\*', '### Power Relations'),
        (r'\*\(\s*(Systemic Contradictions)\s*\)\*\*', '### Systemic Contradictions'),
        (r'\*\(\s*(Systemic Contradictions)\s*\)\*', '### Systemic Contradictions'),
        (r'\*\(\s*(Frame Critique)\s*\)\*\*', '### Frame Critique'),
        (r'\*\(\s*(Frame Critique)\s*\)\*', '### Frame Critique'),
        # Typo fix: "Contricitions" -> "Contradictions"
        (r'\*\*(Systemic Contricitions):\*\*', '### Systemic Contradictions'),
        (r'\*(Systemic Contricitions):\*', '### Systemic Contradictions'),
        (r'\*\*(Systemic Contricitions)\*\*', '### Systemic Contradictions'),
        (r'\*(Systemic Contricitions)\*', '### Systemic Contradictions'),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_inline_meta(text: str) -> str:
    """Remove inline meta-commentary from text."""
    result = text
    for pattern in INLINE_META_PATTERNS:
        result = pattern.sub('', result)

    # Clean up artifacts
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)

    # Remove lines that are purely meta markers
    lines = result.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        # Skip lines that are just meta markers
        if stripped in ('[proceeds]', '[done.]', 'ready.', 'generate.', 'proceed.', 'done.'):
            continue
        if stripped.startswith('output matches') or stripped.startswith('all good') or stripped.startswith('all constraints met'):
            continue
        cleaned.append(line)

    return '\n'.join(cleaned).strip()


def clean_reasoning_trace(rc: str) -> str:
    """
    Clean a reasoning trace: extract Step 3 DM analysis, strip all meta-commentary.
    """
    if not rc or not isinstance(rc, str):
        return ''

    # Extract Step 3
    step3_info = extract_step_content(rc, 3)
    if step3_info is None:
        return clean_fallback(rc)

    start, end, title = step3_info
    step3_content = rc[start:end]

    # Remove the step title line (first line)
    content_lines = step3_content.split('\n')
    # Skip title and blank lines
    content_start = 0
    for i, line in enumerate(content_lines):
        if line.strip() and not re.match(r'^\d+\.\s+\*\*', line):
            content_start = i
            break

    body = '\n'.join(content_lines[content_start:])

    # Find Frame Critique section and trim trailing meta
    fc_match = None
    for pattern in DM_SECTION_PATTERNS:
        if 'Frame Critique' in pattern.pattern:
            m = pattern.search(body)
            if m:
                fc_match = m
                break

    if fc_match:
        fc_pos = fc_match.start()
        # Find where Frame Critique content ends
        cutoff = find_last_substantive_line_after_section(body, fc_pos)
        if cutoff < len(body):
            body = body[:cutoff]

    # Normalize headers
    body = normalize_headers(body)

    # Clean inline meta
    body = clean_inline_meta(body)

    return body.strip()


def clean_fallback(rc: str) -> str:
    """Fallback when Step 3 marker is not found."""
    sections = [
        'Material Conditions',
        'Structural Constraints',
        'Power Relations',
        'Systemic Contradictions',
        'Frame Critique',
    ]

    preamble_end = 0
    if 'thinking process' in rc.lower():
        preamble_end = rc.lower().index('thinking process') + 200

    first_section_pos = len(rc)
    for section in sections:
        pos = rc.find(section, preamble_end)
        if 0 < pos < first_section_pos:
            first_section_pos = pos

    if first_section_pos < len(rc) - 100:
        content = rc[first_section_pos:]
        # Cut off trailing meta
        cutoff = find_last_substantive_line_after_section(content, 0)
        if cutoff < len(content):
            content = content[:cutoff]
        return clean_inline_meta(normalize_headers(content)).strip()

    return ''


def process_parquet_file(input_path: str, output_dir: str) -> str:
    """Process a parquet file: clean reasoning traces and write back."""
    df = pd.read_parquet(input_path)

    if 'answer__reasoning_content' not in df.columns:
        print(f"  Skipping {os.path.basename(input_path)}: no answer__reasoning_content column")
        return str(input_path)

    # Check if any rows have non-empty reasoning content
    has_content = False
    for _, row in df.iterrows():
        rc = row.get('answer__reasoning_content')
        if rc and isinstance(rc, str) and len(rc.strip()) > 0:
            has_content = True
            break

    if not has_content:
        print(f"  Skipping {os.path.basename(input_path)}: no reasoning content to clean")
        return str(input_path)

    original_lengths = df['answer__reasoning_content'].apply(
        lambda x: len(str(x)) if x is not None else 0
    ).astype(int)

    cleaned_traces = []
    for idx, row in df.iterrows():
        rc = row.get('answer__reasoning_content') or ''
        cleaned = clean_reasoning_trace(rc)
        cleaned_traces.append(cleaned)

        if idx < 3:
            orig_len = len(str(rc))
            print(f"  Row {idx}: {orig_len} -> {len(cleaned)} chars "
                  f"({100*len(cleaned)/max(orig_len,1):.1f}%)")

    df['answer__reasoning_content'] = cleaned_traces

    output_path = os.path.join(output_dir, os.path.basename(input_path))
    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)

    avg_original = sum(original_lengths) / len(original_lengths)
    avg_cleaned = sum(len(c) for c in cleaned_traces) / len(cleaned_traces)
    print(f"  Processed {len(df)} rows: avg {avg_original:.0f} -> {avg_cleaned:.0f} chars "
          f"({100*avg_cleaned/max(avg_original,1):.1f}%)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Clean meta-commentary from reasoning traces')
    parser.add_argument('input_dir', help='Directory containing recipe parquet datasets')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite original files (default: write to -cleaned subdirectory)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        sys.exit(1)

    for recipe_dir in sorted(input_dir.iterdir()):
        if not recipe_dir.is_dir():
            continue

        parquet_dir = recipe_dir / 'parquet-files'
        if not parquet_dir.exists():
            print(f"Skipping {recipe_dir.name}: no parquet-files directory")
            continue

        print(f"\nProcessing: {recipe_dir.name}")

        if args.inplace:
            output_dir = str(parquet_dir)
        else:
            output_dir = str(parquet_dir.parent / 'parquet-files-cleaned')

        for parquet_file in sorted(parquet_dir.glob('*.parquet')):
            print(f"  File: {parquet_file.name}")
            result = process_parquet_file(str(parquet_file), output_dir)
            print(f"  Output: {result}")

        if not args.inplace:
            print(f"  Cleaned files written to: {output_dir}")


if __name__ == '__main__':
    main()
