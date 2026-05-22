#!/usr/bin/env python3
"""
Benchmark comparison tool for Qwen3.5-9B evaluation.

Compares:
1. Fine-tuned vs Baseline (PRIMARY — isolates fine-tuning effect)
2. BF16 baseline vs Published scores (framework validation)
3. BF16 baseline vs GGUF baseline (quantization penalty)

Supports all metric types: accuracy (0-100), normalized scores, and
pass@k fractions (0-1) from HumanEval and similar code generation tasks.

Usage:
    python3 compare_results.py [options]

Options:
    --finetuned PATH    Path to fine-tuned results directory
    --baseline-gguf PATH Path to GGUF baseline results directory
    --baseline-bf16 PATH Path to BF16 baseline results directory
    --output PATH       Output path for comparison report (default: stdout)
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# Published Qwen3.5-9B scores from model card
PUBLISHED_SCORES = {
    "mmlu_pro": 82.5,
    "gpqa_diamond_zeroshot": 81.7,
    "ifeval": 91.5,
    "livecodebench": 65.6,
    "hmmt": 83.2,
}

# Regression thresholds (points below Q4 baseline)
REGRESSION_THRESHOLDS = {
    "mmlu_pro": 3,
    "gpqa_diamond_zeroshot": 3,
    "ifeval": 5,
    "livecodebench": 5,
    "hmmt": 3,
    "humaneval": 5,
}


def find_result_files(results_dir: str) -> list:
    """Find all JSON result files in a results directory."""
    if not os.path.isdir(results_dir):
        return []
    # lm_eval outputs JSON files with timestamps
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    # Also check for nested directories
    if not json_files:
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))
    return json_files


def extract_scores(result_files: list) -> dict:
    """Extract benchmark scores from lm_eval JSON output.

    Returns dict of task_name -> score. Also populates global
    _SCORE_METADATA with per-task info (metric name, whether it's a
    percentage 0-100 or a fraction 0-1).
    """
    scores = {}
    for filepath in result_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # lm_eval 0.4.x format
            if "results" in data:
                for task_name, task_data in data["results"].items():
                    # Normalize task name
                    clean_name = task_name.split(",")[0].strip()

                    # Look for main metric
                    if "alias" in task_data:
                        clean_name = task_data["alias"]

                    # Extract the primary metric
                    metric_keys = [
                        "acc,5", "acc", "acc_norm", "acc_norm,5",
                        "exact_match", "passive_accuracy",
                        "pass@1,create_test", "pass@1",
                    ]
                    matched_key = None
                    for key in metric_keys:
                        if key in task_data and isinstance(task_data[key], (int, float)):
                            matched_key = key
                            val = task_data[key]
                            break

                    if matched_key is not None:
                        # Store as-is (don't round yet)
                        if clean_name in scores:
                            print(
                                f"Warning: Overwriting score for {clean_name} "
                                f"from {filepath}",
                                file=sys.stderr,
                            )
                        scores[clean_name] = val
                        # Track metric info
                        _SCORE_METADATA[clean_name] = {
                            "metric": matched_key,
                            "is_fraction": "@" in matched_key or matched_key.startswith("pass"),
                        }

                    # For IFEval, look for specific metrics
                    if "normalized_score" in task_data:
                        val = task_data["normalized_score"]
                        if clean_name in scores:
                            print(
                                f"Warning: Overwriting score for {clean_name} "
                                f"from {filepath}",
                                file=sys.stderr,
                            )
                        scores[clean_name] = val
                        _SCORE_METADATA[clean_name] = {
                            "metric": "normalized_score",
                            "is_fraction": False,
                        }

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)

    return scores


# Track per-task metadata: metric name, whether value is 0-1 fraction
_SCORE_METADATA = {}


def format_delta(value: float, is_fraction: bool = False) -> str:
    """Format a delta value with sign and color indicator."""
    if is_fraction:
        formatted = "%.2f" % value
    else:
        formatted = "%.1f" % value
    if value > 0:
        return f"+{formatted}"
    return formatted


def format_score(value, is_fraction: bool = False) -> str:
    """Format a score value, converting fractions to percentages if needed."""
    if isinstance(value, (int, float)):
        if is_fraction:
            return "%.1f%%" % (value * 100)
        return "%.1f" % value
    return str(value)


def check_regression(delta: float, threshold: float, is_fraction: bool = False) -> str:
    """Check if a delta exceeds the regression threshold.

    For fraction-based metrics (pass@1), threshold is in percentage points,
    so convert: threshold 5 means 0.05 in fraction space.
    """
    effective_threshold = threshold / 100.0 if is_fraction else threshold
    if delta <= -effective_threshold:
        return "⚠ REGRESSION"
    elif delta <= 0:
        return "stable"
    else:
        return "✓ improved"


def compare_results(
    finetuned_scores: dict,
    gguf_scores: dict,
    bf16_scores: dict,
) -> str:
    """Generate comparison report."""
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK EVALUATION COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Collect all task names
    all_tasks = set()
    for scores in [finetuned_scores, gguf_scores, bf16_scores]:
        all_tasks.update(scores.keys())

    # Map display names
    display_names = {
        "mmlu_pro": "MMLU-Pro",
        "gpqa_diamond_zeroshot": "GPQA Diamond",
        "ifeval": "IFEval",
        "livecodebench": "LiveCodeBench",
        "hmmt": "HMMT",
        "humaneval": "HumanEval",
    }

    # PRIMARY COMPARISON: Fine-tuned GGUF vs GGUF Baseline
    lines.append("-" * 80)
    lines.append("PRIMARY: Fine-Tuned GGUF vs GGUF Baseline (isolates fine-tuning effect)")
    lines.append("-" * 80)
    lines.append(f"{'Benchmark':<20} {'GGUF Base':>10} {'Fine-Tuned':>12} {'Delta':>8} {'Status':<15}")
    lines.append("-" * 80)

    regression_count = 0
    for task in sorted(all_tasks):
        display = display_names.get(task, task)
        gguf_val = gguf_scores.get(task, "N/A")
        ft_val = finetuned_scores.get(task, "N/A")

        meta = _SCORE_METADATA.get(task, {})
        is_fraction = meta.get("is_fraction", False)

        gguf_str = format_score(gguf_val, is_fraction)
        ft_str = format_score(ft_val, is_fraction)

        if isinstance(gguf_val, (int, float)) and isinstance(ft_val, (int, float)):
            delta = ft_val - gguf_val
            delta_str = format_delta(delta, is_fraction)
            threshold = REGRESSION_THRESHOLDS.get(task, 3)
            status = check_regression(delta, threshold, is_fraction)
            if "REGRESSION" in status:
                regression_count += 1
        else:
            delta_str = "N/A"
            status = "N/A"

        lines.append(f"{display:<20} {gguf_str:>10} {ft_str:>12} {delta_str:>8} {status:<15}")

    lines.append("-" * 80)
    lines.append(f"Regressions detected: {regression_count}")
    lines.append("")

    # SECONDARY COMPARISON: BF16 Baseline vs Published
    lines.append("-" * 80)
    lines.append("SECONDARY: BF16 Baseline vs Published Scores (framework validation)")
    lines.append("-" * 80)
    lines.append(f"{'Benchmark':<20} {'BF16 Local':>10} {'Published':>10} {'Delta':>8} {'Status':<15}")
    lines.append("-" * 80)

    for task in sorted(all_tasks):
        display = display_names.get(task, task)
        bf16_val = bf16_scores.get(task, "N/A")
        pub_val = PUBLISHED_SCORES.get(task, "N/A")

        meta = _SCORE_METADATA.get(task, {})
        is_fraction = meta.get("is_fraction", False)

        bf16_str = format_score(bf16_val, is_fraction)
        pub_str = format_score(pub_val, is_fraction)

        if isinstance(bf16_val, (int, float)) and isinstance(pub_val, (int, float)):
            delta = bf16_val - pub_val
            delta_str = format_delta(delta, is_fraction)
            if abs(delta) <= 2:
                status = "✓ validated"
            else:
                status = "⚠ check setup"
        else:
            delta_str = "N/A"
            status = "N/A"

        lines.append(f"{display:<20} {bf16_str:>10} {pub_str:>10} {delta_str:>8} {status:<15}")

    lines.append("-" * 80)
    lines.append("")

    # QUANTIZATION PENALTY: BF16 vs GGUF Baseline
    lines.append("-" * 80)
    lines.append("QUANTIZATION: BF16 Baseline vs GGUF Baseline (quantization penalty)")
    lines.append("-" * 80)
    lines.append(f"{'Benchmark':<20} {'BF16':>10} {'GGUF':>10} {'Penalty':>8} {'Status':<15}")
    lines.append("-" * 80)

    for task in sorted(all_tasks):
        display = display_names.get(task, task)
        bf16_val = bf16_scores.get(task, "N/A")
        gguf_val = gguf_scores.get(task, "N/A")

        meta = _SCORE_METADATA.get(task, {})
        is_fraction = meta.get("is_fraction", False)

        bf16_str = format_score(bf16_val, is_fraction)
        gguf_str = format_score(gguf_val, is_fraction)

        if isinstance(bf16_val, (int, float)) and isinstance(gguf_val, (int, float)):
            penalty = gguf_val - bf16_val
            penalty_str = format_delta(penalty, is_fraction)
            if abs(penalty) <= 3:
                status = "✓ expected"
            elif abs(penalty) <= 5:
                status = "~ moderate"
            else:
                status = "⚠ high"
        else:
            penalty_str = "N/A"
            status = "N/A"

        lines.append(f"{display:<20} {bf16_str:>10} {gguf_str:>10} {penalty_str:>8} {status:<15}")

    lines.append("-" * 80)
    lines.append("")

    # SUMMARY
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)

    if regression_count > 0:
        lines.append(f"⚠ {regression_count} benchmark(s) show regression beyond threshold.")
        lines.append("Consider DPO remediation if regressions exceed acceptable levels.")
    else:
        lines.append("✓ No significant regressions detected.")

    lines.append("")
    lines.append("Regression thresholds:")
    for task, threshold in REGRESSION_THRESHOLDS.items():
        display = display_names.get(task, task)
        lines.append(f"  {display:<20} ±{threshold} points")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark evaluation results")
    parser.add_argument(
        "--finetuned",
        default="results/finetuned/gguf",
        help="Path to fine-tuned results directory",
    )
    parser.add_argument(
        "--baseline-gguf",
        default="results/baseline/gguf",
        help="Path to GGUF baseline results directory",
    )
    parser.add_argument(
        "--baseline-bf16",
        default="results/baseline/bf16",
        help="Path to BF16 baseline results directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    finetuned_dir = project_root / args.finetuned
    gguf_dir = project_root / args.baseline_gguf
    bf16_dir = project_root / args.baseline_bf16

    # Check which results are available
    available = {}
    for name, path in [
        ("Fine-tuned GGUF", finetuned_dir),
        ("GGUF Baseline", gguf_dir),
        ("BF16 Baseline", bf16_dir),
    ]:
        files = find_result_files(str(path))
        if files:
            available[name] = files
            print(f"Found {len(files)} result file(s) for {name}")
        else:
            print(f"Warning: No results found for {name} at {path}")

    # Extract scores
    finetuned_scores = extract_scores(available.get("Fine-tuned GGUF", []))
    gguf_scores = extract_scores(available.get("GGUF Baseline", []))
    bf16_scores = extract_scores(available.get("BF16 Baseline", []))

    # Debug output
    print(f"\nFine-tuned scores: {finetuned_scores}")
    print(f"GGUF Baseline scores: {gguf_scores}")
    print(f"BF16 Baseline scores: {bf16_scores}")

    # Generate report
    report = compare_results(finetuned_scores, gguf_scores, bf16_scores)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
