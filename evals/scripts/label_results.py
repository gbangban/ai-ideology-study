#!/usr/bin/env python3
"""
Label eval result files with descriptive metadata.

Adds identifying information to lm_eval result JSON files so that different
eval runs can be properly distinguished. Also renames the random run ID
directories (e.g. 'qdpv68r5') to descriptive names.

Usage:
    python3 label_results.py                  # Label all results in evals/results/
    python3 label_results.py --dry-run        # Show what would change
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def get_model_label(result_path: Path, results_dir: Path) -> str:
    """Determine a descriptive label for a result file based on its path."""
    rel = result_path.parent.parent.relative_to(results_dir)
    parts = rel.parts

    # results/baseline/gguf/... -> "baseline-gguf"
    # results/runs/finetuned/gguf/... -> "finetuned-gguf"
    # results/baseline/bf16/... -> "baseline-bf16"
    # Strip the 'runs' intermediate directory from label
    parts = [p for p in parts if p != "runs"]
    label = "-".join(parts)
    return label


def get_model_path_from_log(eval_log: Path) -> str:
    """Extract the model/GGUF path from the eval log file."""
    if not eval_log.exists():
        return ""
    try:
        with open(eval_log, "r") as f:
            for line in f:
                if "GGUF:" in line:
                    # Line format: [timestamp] [INFO] GGUF: /path/to/model.gguf
                    return line.split("GGUF:")[-1].strip()
                if "Model:" in line and ".gguf" in line.lower():
                    return line.split("Model:")[-1].strip()
    except Exception:
        pass
    return ""


def annotate_result_file(result_file: Path, results_root: Path, dry_run: bool = False) -> dict:
    """Add identifying metadata to a result JSON file. Returns changes made."""
    changes = {}

    with open(result_file, "r") as f:
        data = json.load(f)

    # Get the run label from directory structure
    run_label = get_model_label(result_file, results_root)

    # Find the eval.log sibling to extract model path
    # The eval.log is at results/<category>/eval.log or results/<category>/gguf/eval.log
    parent = result_file.parent.parent  # go up from run-id dir to category dir
    eval_log = parent / "eval.log"
    model_path = get_model_path_from_log(eval_log)

    # Extract timestamp from filename
    # e.g. results_2026-05-20T17-27-33.002448.json
    stem = result_file.stem
    ts_match = re.search(r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", stem)
    file_timestamp = ts_match.group(1).replace("-", ":", 4) if ts_match else ""

    # Determine what fields to add/update
    new_fields = {
        "run_label": run_label,
        "model_path": model_path,
        "file_timestamp": file_timestamp,
    }

    # Build a human-readable run name
    # e.g. "baseline-gguf-2026-05-20T17:27:33"
    run_name = f"{run_label}-{file_timestamp}" if file_timestamp else run_label
    new_fields["run_name"] = run_name

    # Check what needs changing
    for key, value in new_fields.items():
        if data.get(key) != value:
            changes[key] = {"old": data.get(key), "new": value}

    if changes and not dry_run:
        data.update(new_fields)
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Updated {result_file.name}:")
        for key, vals in changes.items():
            old_short = str(vals["old"])[:50] if vals["old"] is not None else "missing"
            new_short = str(vals["new"])[:50]
            print(f"    {key}: {old_short} -> {new_short}")
    elif not changes:
        print(f"  No changes needed for {result_file.name}")

    return changes


def rename_run_directory(run_dir: Path, results_root: Path, dry_run: bool = False) -> bool:
    """Rename a random-ID run directory to a descriptive name."""
    # Check if this is a random-ID directory (8-char hex-looking string)
    dir_name = run_dir.name
    if not re.match(r"^[a-z0-9]{8}$", dir_name):
        return False  # Not a random ID, skip

    # Find result files inside to get timestamp
    result_files = list(run_dir.glob("results_*.json"))
    if not result_files:
        return False

    # Get the label from parent directory structure
    run_label = get_model_label(run_dir, results_root)

    # Get timestamp from result filename
    stem = result_files[0].stem
    ts_match = re.search(r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", stem)
    if not ts_match:
        return False

    ts_clean = ts_match.group(1).replace("-", "_", 4)  # 2026-05-20T17_27_33
    new_name = f"{run_label}-{ts_clean}"

    if dir_name == new_name:
        return False  # Already renamed

    new_path = run_dir.parent / new_name

    if dry_run:
        print(f"  Would rename: {dir_name} -> {new_name}")
        return True

    if new_path.exists():
        print(f"  Skipping: {new_path} already exists")
        return False

    shutil.move(str(run_dir), str(new_path))
    print(f"  Renamed: {dir_name} -> {new_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Label eval result files with descriptive metadata")
    parser.add_argument("--results-dir", default=None, help="Path to results directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without making changes")
    args = parser.parse_args()

    # Find results directory
    script_dir = Path(__file__).parent
    if args.results_dir:
        results_root = Path(args.results_dir)
    else:
        results_root = script_dir.parent / "results"

    if not results_root.exists():
        print(f"Results directory not found: {results_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning results in: {results_root}")
    print()

    # Find all result JSON files
    result_files = list(results_root.rglob("results_*.json"))

    if not result_files:
        print("No result files found.")
        return

    # Phase 1: Annotate result files
    print("=== PHASE 1: Annotating result files ===")
    total_changes = 0
    for rf in sorted(result_files):
        print(f"\nProcessing: {rf.relative_to(results_root)}")
        changes = annotate_result_file(rf, results_root, args.dry_run)
        total_changes += len(changes)

    print(f"\nTotal metadata updates: {total_changes}")

    # Phase 2: Rename run directories
    print("\n=== PHASE 2: Renaming run directories ===")
    renamed = 0
    for category_dir in results_root.iterdir():
        if not category_dir.is_dir():
            continue
        for run_dir in category_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if rename_run_directory(run_dir, results_root, args.dry_run):
                renamed += 1
        # Also check nested (e.g. runs/finetuned/gguf/)
        for sub_dir in category_dir.rglob("*"):
            if not sub_dir.is_dir():
                continue
            if rename_run_directory(sub_dir, results_root, args.dry_run):
                renamed += 1

    print(f"\nTotal directories renamed: {renamed}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
