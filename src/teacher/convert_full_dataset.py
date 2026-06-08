#!/usr/bin/env python3
"""
Convert full parquet dataset to JSON and clean reasoning traces.

Usage:
    python3 -m src.teacher.convert_full_dataset \
        --parquet-path /path/to/batch_00000.parquet \
        --output-dir data/processed/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.teacher.clean_reasoning_traces import clean_reasoning_trace


PARQUET_PATH_DEFAULT = (
    "/mnt/c/Users/Guy/.unsloth/studio/assets/datasets/recipes/"
    "recipe_ml-1500-v1/parquet-files/batch_00000.parquet"
)


def parquet_to_records(parquet_path: str) -> list[dict]:
    """Read parquet and return list of dicts with native Python types."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if hasattr(val, "item"):
                val = val.item()
            record[col] = val if pd.notna(val) else ""
        records.append(record)
    return records


def clean_and_save(records: list[dict], output_path: str) -> list[dict]:
    """Clean reasoning traces in-place and save to JSON."""
    cleaned = 0
    for r in records:
        rc = r.get("answer__reasoning_content", "")
        if rc:
            r["answer__reasoning_content"] = clean_reasoning_trace(rc)
            cleaned += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  Cleaned {cleaned}/{len(records)} reasoning traces")
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert full parquet dataset and clean reasoning traces")
    parser.add_argument("--parquet-path", default=PARQUET_PATH_DEFAULT, help="Path to parquet file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    print(f"Reading parquet: {args.parquet_path}")
    records = parquet_to_records(args.parquet_path)
    print(f"  Loaded {len(records)} records")

    print("Cleaning reasoning traces...")
    records = clean_and_save(records, str(Path(args.output_dir) / "full_dataset.json"))

    sft_path = str(Path(args.output_dir) / "sft_dataset.json")
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  SFT: {len(records)} samples -> {sft_path}")


if __name__ == "__main__":
    main()
