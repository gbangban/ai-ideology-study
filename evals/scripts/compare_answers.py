#!/usr/bin/env python3
"""
Per-sample answer comparison for MMLU multiple-choice tasks.

Compares individual question-level predictions between a baseline and finetuned
model using lm_eval samples_*.jsonl files. Shows questions where the models
disagree, with full option text and logprob scores.

Usage:
    python3 compare_answers.py --baseline BASELINE_SAMPLES.jsonl --finetuned FINETUNED_SAMPLES.jsonl
    python3 compare_answers.py --help

Examples:
    # Compare global_facts between baseline and finetuned BF16
    python3 compare_answers.py \\
        --baseline evals/results/baseline/bf16/.../samples_mmlu_global_facts_*.jsonl \\
        --finetuned evals/results/finetuned/bf16/.../samples_mmlu_global_facts_*.jsonl

    # Show only questions where baseline was right but finetuned was wrong
    python3 compare_answers.py \\
        --baseline ... --finetuned ... --mode regressions
"""

import argparse
import json
import sys
from pathlib import Path


def load_samples(filepath: str) -> dict:
    """Load a samples_*.jsonl file into a dict keyed by doc_id."""
    samples = {}
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at {filepath}:{line_num}: {e}", file=sys.stderr)
                continue
            doc_id = d.get("doc_id")
            if doc_id is None:
                print(f"Warning: No doc_id found at {filepath}:{line_num}", file=sys.stderr)
                continue
            samples[doc_id] = d
    return samples


def get_prediction(sample: dict) -> tuple:
    """Extract the predicted answer index and logprobs from a sample.

    Returns (pick_idx, logprobs_list) where pick_idx is the index of the
    option with the highest logprob (least negative).
    """
    filtered_resps = sample.get("filtered_resps", [])
    if not filtered_resps:
        return None, []
    logprobs = [float(r[0]) for r in filtered_resps]
    # Highest logprob = least negative = most likely
    pick_idx = logprobs.index(max(logprobs))
    return pick_idx, logprobs


def get_accuracy(sample: dict) -> float:
    """Get the accuracy (0.0 or 1.0) for a sample."""
    return sample.get("acc", -1.0)


def format_option(idx: int, text: str, labels: list) -> str:
    """Format an option with its letter label."""
    return f"{labels[idx]}. {text}"


def compare_samples(baseline: dict, finetuned: dict, mode: str) -> list:
    """Compare samples and return matching entries based on mode.

    Modes:
        all          - all questions where models disagree on prediction
        regressions  - baseline correct, finetuned wrong
        improvements - baseline wrong, finetuned correct
        both         - regressions + improvements
    """
    labels = ["A", "B", "C", "D", "E"]
    results = []

    all_doc_ids = sorted(set(baseline.keys()) | set(finetuned.keys()))

    for doc_id in all_doc_ids:
        b = baseline.get(doc_id)
        f = finetuned.get(doc_id)
        if not b or not f:
            continue

        b_pick, b_lp = get_prediction(b)
        f_pick, f_lp = get_prediction(f)
        b_acc = get_accuracy(b)
        f_acc = get_accuracy(f)

        if b_pick is None or f_pick is None:
            continue

        q = b["doc"].get("question", "")
        choices = b["doc"].get("choices", [])
        correct_idx = b["doc"].get("answer")
        num_choices = len(choices)

        # Determine categories
        b_correct = b_acc == 1.0
        f_correct = f_acc == 1.0
        same_pick = b_pick == f_pick

        # Filter by mode
        if mode == "all" and same_pick:
            continue
        elif mode == "regressions" and not (b_correct and not f_correct):
            continue
        elif mode == "improvements" and not (not b_correct and f_correct):
            continue
        elif mode == "both" and not ((b_correct and not f_correct) or (not b_correct and f_correct)):
            continue

        entry = {
            "doc_id": doc_id,
            "question": q,
            "choices": choices[:num_choices],
            "correct_idx": correct_idx,
            "b_pick": b_pick,
            "f_pick": f_pick,
            "b_lp": b_lp[:num_choices],
            "f_lp": f_lp[:num_choices],
            "b_correct": b_correct,
            "f_correct": f_correct,
            "same_pick": same_pick,
            "num_choices": num_choices,
        }
        results.append(entry)

    return results


def print_results(results: list, mode: str):
    """Print comparison results to stdout."""
    if not results:
        print("No matching questions found.")
        return

    header_map = {
        "all": "MODEL DISAGREEMENTS",
        "regressions": "REGRESSIONS (Baseline Right, Finetuned Wrong)",
        "improvements": "IMPROVEMENTS (Baseline Wrong, Finetuned Right)",
        "both": "REGRESSIONS + IMPROVEMENTS",
    }

    print("=" * 80)
    print(header_map.get(mode, "COMPARISON RESULTS"))
    print("=" * 80)
    print(f"Total: {len(results)} question(s)")
    print()

    for i, entry in enumerate(results):
        doc_id = entry["doc_id"]
        q = entry["question"]
        choices = entry["choices"]
        correct_idx = entry["correct_idx"]
        b_pick = entry["b_pick"]
        f_pick = entry["f_pick"]
        b_lp = entry["b_lp"]
        f_lp = entry["f_lp"]
        b_correct = entry["b_correct"]
        f_correct = entry["f_correct"]
        num_choices = entry["num_choices"]
        labels = ["A", "B", "C", "D", "E"]

        # Category label
        if b_correct and not f_correct:
            category = "REGRESSION"
        elif not b_correct and f_correct:
            category = "IMPROVEMENT"
        elif not b_correct and not f_correct:
            category = "BOTH WRONG"
        else:
            category = "BOTH RIGHT (different pick)"

        print(f"--- Q{doc_id} [{category}] ---")
        print(f"Question: {q}")
        print(f"Correct: {labels[correct_idx]} ({choices[correct_idx]})")
        print()

        for j in range(num_choices):
            parts = [f"  {labels[j]}. {choices[j]}"]
            marks = []
            if j == correct_idx:
                marks.append("CORRECT")
            if j == b_pick:
                marks.append(f"baseline (lp={b_lp[j]:.4f})")
            if j == f_pick:
                marks.append(f"finetuned (lp={f_lp[j]:.4f})")
            if marks:
                parts.append("  [%s]" % ", ".join(marks))
            print("".join(parts))
            if j not in (b_pick, f_pick, correct_idx):
                print(f"     baseline lp={b_lp[j]:.4f}  |  finetuned lp={f_lp[j]:.4f}")

        print(f"  Baseline: {labels[b_pick]} ({choices[b_pick]}) {'✓' if b_correct else '✗'}")
        print(f"  Finetuned: {labels[f_pick]} ({choices[f_pick]}) {'✓' if f_correct else '✗'}")
        print()


def print_summary(baseline: dict, finetuned: dict):
    """Print overall accuracy summary."""
    b_total = len(baseline)
    f_total = len(finetuned)
    common_ids = set(baseline.keys()) & set(finetuned.keys())

    b_correct = sum(1 for d in baseline.values() if get_accuracy(d) == 1.0)
    f_correct = sum(1 for d in finetuned.values() if get_accuracy(d) == 1.0)

    # Per-category counts
    regressions = 0
    improvements = 0
    both_right = 0
    both_wrong = 0
    disagreements = 0

    for doc_id in common_ids:
        b = baseline[doc_id]
        f = finetuned[doc_id]
        b_c = get_accuracy(b) == 1.0
        f_c = get_accuracy(f) == 1.0
        b_pick, _ = get_prediction(b)
        f_pick, _ = get_prediction(f)

        if b_pick is not None and f_pick is not None:
            if b_pick != f_pick:
                disagreements += 1
            if b_c and not f_c:
                regressions += 1
            elif not b_c and f_c:
                improvements += 1
            elif b_c and f_c:
                both_right += 1
            else:
                both_wrong += 1

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline samples:  {b_total}")
    print(f"Finetuned samples: {f_total}")
    print(f"Common samples:    {len(common_ids)}")
    print()
    print(f"Baseline accuracy:  {b_correct}/{b_total} = {b_correct / b_total * 100:.1f}%")
    print(f"Finetuned accuracy: {f_correct}/{f_total} = {f_correct / f_total * 100:.1f}%")
    print(f"Net change:         {f_correct - b_correct} ({(f_correct - b_correct) / b_total * 100:+.1f}pp)")
    print()
    print(f"Regressions (B✓ F✗):    {regressions}")
    print(f"Improvements (B✗ F✓):   {improvements}")
    print(f"Both right:             {both_right}")
    print(f"Both wrong:             {both_wrong}")
    print(f"Prediction disagreements: {disagreements}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-sample answers between baseline and finetuned models"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline samples_*.jsonl file",
    )
    parser.add_argument(
        "--finetuned",
        required=True,
        help="Path to finetuned samples_*.jsonl file",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "regressions", "improvements", "both"],
        default="both",
        help="What to show: all disagreements, regressions only, improvements only, or both (default: both)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the summary statistics",
    )

    args = parser.parse_args()

    # Load samples
    print(f"Loading baseline:  {args.baseline}", file=sys.stderr)
    baseline = load_samples(args.baseline)
    print(f"  -> {len(baseline)} samples", file=sys.stderr)

    print(f"Loading finetuned: {args.finetuned}", file=sys.stderr)
    finetuned = load_samples(args.finetuned)
    print(f"  -> {len(finetuned)} samples", file=sys.stderr)
    print(file=sys.stderr)

    # Summary
    if not args.no_summary:
        print_summary(baseline, finetuned)
        print()

    # Compare and print
    results = compare_samples(baseline, finetuned, args.mode)
    print_results(results, args.mode)


if __name__ == "__main__":
    main()
