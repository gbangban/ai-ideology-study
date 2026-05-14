#!/usr/bin/env python3
"""
Convert Unsloth Studio parquet datasets to JSON format.

Reads parquet files from a recipe directory and produces JSON
in the project's data/processed/ directory (not the Studio path).

Usage:
    python3 src/teacher/parquet_to_json.py /path/to/recipe_dir/
    python3 src/teacher/parquet_to_json.py /path/to/parquet-files/
    python3 src/teacher/parquet_to_json.py /path/ --output-dir ./custom/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Default output: project's data/processed/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = str(PROJECT_ROOT / 'data' / 'processed')


def parquet_to_records(parquet_path: str) -> list[dict]:
    """Read a parquet file and return a list of dictionaries."""
    df = pd.read_parquet(parquet_path)

    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            # Convert numpy types to Python natives for JSON serialization
            if hasattr(val, 'item'):
                val = val.item()
            record[col] = val
        records.append(record)
    return records


def convert_parquet_file(parquet_path: str, output_dir: str) -> str:
    """Convert a single parquet file to JSON. Returns output path."""
    records = parquet_to_records(parquet_path)
    base = os.path.splitext(os.path.basename(parquet_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f'{base}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return json_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert Unsloth Studio parquet datasets to JSON/JSONL'
    )
    parser.add_argument(
        'input_dir',
        help='Path to a recipe directory (containing parquet-files/) or a parquet-files/ directory'
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT,
        help=f'Output directory (default: {DEFAULT_OUTPUT})'
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    output_dir = args.output_dir
    parquet_files = []

    # Collect parquet files from input
    if input_path.name == 'parquet-files' and input_path.is_dir():
        # Input is the parquet-files directory itself
        parquet_files = sorted(input_path.glob('*.parquet'))
    elif (input_path / 'parquet-files').exists():
        # Input is the recipe directory
        parquet_files = sorted((input_path / 'parquet-files').glob('*.parquet'))
    else:
        print(f"Error: {input_path} is not a recipe directory or parquet-files directory")
        sys.exit(1)

    if not parquet_files:
        print(f"No parquet files found in {input_path}")
        sys.exit(1)

    for pq in parquet_files:
        out = convert_parquet_file(str(pq), output_dir)
        records = sum(1 for _ in open(out))
        print(f"  {pq.name} -> {out} ({records} lines)")

    print(f"\nDone. Output: {output_dir}")


if __name__ == '__main__':
    main()
