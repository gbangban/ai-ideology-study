#!/usr/bin/env python3
"""
Three-way benchmark comparison tool for Qwen3.5-9B evaluation.

Compares:
1. Fine-tuned Q4_K_M vs Q4_K_M baseline (PRIMARY — isolates fine-tuning effect)
2. BF16 baseline vs Published scores (framework validation)
3. BF16 baseline vs Q4_K_M baseline (quantization penalty)

Usage:
    python3 compare_results.py [options]

Options:
    --finetuned PATH    Path to fine-tuned results directory (default: results/runs/finetuned)
    --baseline-q4 PATH  Path to Q4 baseline results directory (default: results/baseline/q4)
    --baseline-bf16 PATH Path to BF16 baseline results directory (default: results/baseline/bf16)
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
    "gpqa:diamond": 81.7,
    "ifeval": 91.5,
    "livecodebench": 65.6,
    "hmmt": 83.2,
}

# Regression thresholds (points below Q4 baseline)
REGRESSION_THRESHOLDS = {
    "mmlu_pro": 3,
    "gpqa:diamond": 3,
    "ifeval": 5,
    "livecodebench": 5,
    "hmmt": 3,
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
    """Extract benchmark scores from lm_eval JSON output."""
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
                    ]
                    for key in metric_keys:
                        if key in task_data and isinstance(task_data[key], (int, float)):
                            val = round(task_data[key], 2)
                            if clean_name in scores:
                                print(
                                    f"Warning: Overwriting score for {clean_name} "
                                    f"from {filepath}",
                                    file=sys.stderr,
                                )
                            scores[clean_name] = val
                            break

                    # For IFEval, look for specific metrics
                    if "normalized_score" in task_data:
                        val = round(task_data["normalized_score"], 2)
                        if clean_name in scores:
                            print(
                                f"Warning: Overwriting score for {clean_name} "
                                f"from {filepath}",
                                file=sys.stderr,
                            )
                        scores[clean_name] = val

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)

    return scores


def format_delta(value: float) -> str:
    """Format a delta value with sign and color indicator."""
    if value > 0:
        return f"+{value:.1f}"
    elif value < 0:
        return f"{value:.1f}"
    return "0.0"


def check_regression(delta: float, threshold: float) -> str:
    """Check if a delta exceeds the regression threshold."""
    if delta <= -threshold:
        return "⚠ REGRESSION"
    elif delta <= 0:
        return "stable"
    else:
        return "✓ improved"


def compare_results(
    finetuned_scores: dict,
    q4_scores: dict,
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
    for scores in [finetuned_scores, q4_scores, bf16_scores]:
        all_tasks.update(scores.keys())

    # Map display names
    display_names = {
        "mmlu_pro": "MMLU-Pro",
        "gpqa:diamond": "GPQA Diamond",
        "ifeval": "IFEval",
        "livecodebench": "LiveCodeBench",
        "hmmt": "HMMT",
    }

    # PRIMARY COMPARISON: Fine-tuned Q4 vs Q4 Baseline
    lines.append("-" * 80)
    lines.append("PRIMARY: Fine-Tuned Q4_K_M vs Q4_K_M Baseline (isolates fine-tuning effect)")
    lines.append("-" * 80)
    lines.append(f"{'Benchmark':<20} {'Q4 Base':>10} {'Fine-Tuned':>12} {'Delta':>8} {'Status':<15}")
    lines.append("-" * 80)

    regression_count = 0
    for task in sorted(all_tasks):
        display = display_names.get(task, task)
        q4_val = q4_scores.get(task, "N/A")
        ft_val = finetuned_scores.get(task, "N/A")

        q4_str = f"{q4_val:.1f}" if isinstance(q4_val, (int, float)) else str(q4_val)
        ft_str = f"{ft_val:.1f}" if isinstance(ft_val, (int, float)) else str(ft_val)

        if isinstance(q4_val, (int, float)) and isinstance(ft_val, (int, float)):
            delta = ft_val - q4_val
            delta_str = format_delta(delta)
            threshold = REGRESSION_THRESHOLDS.get(task, 3)
            status = check_regression(delta, threshold)
            if "REGRESSION" in status:
                regression_count += 1
        else:
            delta_str = "N/A"
            status = "N/A"

        lines.append(f"{display:<20} {q4_str:>10} {ft_str:>12} {delta_str:>8} {status:<15}")

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

        bf16_str = f"{bf16_val:.1f}" if isinstance(bf16_val, (int, float)) else str(bf16_val)
        pub_str = f"{pub_val:.1f}" if isinstance(pub_val, (int, float)) else str(pub_val)

        if isinstance(bf16_val, (int, float)) and isinstance(pub_val, (int, float)):
            delta = bf16_val - pub_val
            delta_str = format_delta(delta)
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

    # QUANTIZATION PENALTY: BF16 vs Q4 Baseline
    lines.append("-" * 80)
    lines.append("QUANTIZATION: BF16 Baseline vs Q4_K_M Baseline (quantization penalty)")
    lines.append("-" * 80)
    lines.append(f"{'Benchmark':<20} {'BF16':>10} {'Q4_K_M':>10} {'Penalty':>8} {'Status':<15}")
    lines.append("-" * 80)

    for task in sorted(all_tasks):
        display = display_names.get(task, task)
        bf16_val = bf16_scores.get(task, "N/A")
        q4_val = q4_scores.get(task, "N/A")

        bf16_str = f"{bf16_val:.1f}" if isinstance(bf16_val, (int, float)) else str(bf16_val)
        q4_str = f"{q4_val:.1f}" if isinstance(q4_val, (int, float)) else str(q4_val)

        if isinstance(bf16_val, (int, float)) and isinstance(q4_val, (int, float)):
            penalty = q4_val - bf16_val
            penalty_str = format_delta(penalty)
            if abs(penalty) <= 3:
                status = "✓ expected"
            elif abs(penalty) <= 5:
                status = "~ moderate"
            else:
                status = "⚠ high"
        else:
            penalty_str = "N/A"
            status = "N/A"

        lines.append(f"{display:<20} {bf16_str:>10} {q4_str:>10} {penalty_str:>8} {status:<15}")

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
        default="results/runs/finetuned",
        help="Path to fine-tuned results directory",
    )
    parser.add_argument(
        "--baseline-q4",
        default="results/baseline/q4",
        help="Path to Q4 baseline results directory",
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
    q4_dir = project_root / args.baseline_q4
    bf16_dir = project_root / args.baseline_bf16

    # Check which results are available
    available = {}
    for name, path in [
        ("Fine-tuned Q4", finetuned_dir),
        ("Q4 Baseline", q4_dir),
        ("BF16 Baseline", bf16_dir),
    ]:
        files = find_result_files(str(path))
        if files:
            available[name] = files
            print(f"Found {len(files)} result file(s) for {name}")
        else:
            print(f"Warning: No results found for {name} at {path}")

    # Extract scores
    finetuned_scores = extract_scores(available.get("Fine-tuned Q4", []))
    q4_scores = extract_scores(available.get("Q4 Baseline", []))
    bf16_scores = extract_scores(available.get("BF16 Baseline", []))

    # Debug output
    print(f"\nFine-tuned scores: {finetuned_scores}")
    print(f"Q4 Baseline scores: {q4_scores}")
    print(f"BF16 Baseline scores: {bf16_scores}")

    # Generate report
    report = compare_results(finetuned_scores, q4_scores, bf16_scores)

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
