#!/usr/bin/env python3
"""
Question Redistribution Pipeline — Dedup, Auto-tag, Gap Report.

This script performs three operations on the question dataset:
  1. Deduplicate — remove exact text duplicates only
  2. Auto-tag — assign Axis 1 / Axis 2 tags to untagged questions via keyword matching
  3. Gap report — produce a subtag × epoch matrix showing which cells need new questions

This script does NOT trim, score, or remove any questions beyond exact duplicates.
All distribution balancing is achieved by adding new questions, not removing existing ones.

Usage:
    python scripts/redistribute_questions.py                          # full pipeline
    python scripts/redistribute_questions.py --dry-run                # preview without writing
    python scripts/redistribute_questions.py --report-only            # gap report only, no dedup/tag
    python scripts/redistribute_questions.py --input data/raw/questions.json.bak
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# ── Import taxonomy from authoritative source ───────────────────────────
from src.teacher.topics import (
    ALL_AXIS1_TAGS,
    AXIS_1_CATEGORIES,
    AXIS_2_EPOCHS,
    TAG_TO_CATEGORY,
)

# ── Import keyword maps from tag_questions.py ───────────────────────────
from src.teacher.tag_questions import EPOCH_KEYWORDS, TAG_KEYWORDS, suggest_tags


# ── Helper Functions ────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize question text for exact dedup comparison."""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def is_untagged(q: dict) -> bool:
    """Check if a question lacks substantive Axis 1 tags or Axis 2 tags."""
    axis1 = q.get("axis1", [])
    axis2 = q.get("axis2", [])
    substantive = [t for t in axis1 if t != "cross" and t in ALL_AXIS1_TAGS]
    return len(substantive) == 0 or len(axis2) == 0


def fix_misplaced_tags(q: dict) -> bool:
    """Fix questions where epoch tags ended up in axis1 (e.g. 'EP6' in axis1).
    Returns True if anything was fixed."""
    fixed = False
    axis1 = q.get("axis1", [])
    axis2 = q.get("axis2", [])

    # Move epoch tags out of axis1
    new_a1 = []
    for t in axis1:
        if t in AXIS_2_EPOCHS or t.startswith("EP"):
            if t not in axis2:
                axis2.append(t)
            fixed = True
        else:
            new_a1.append(t)

    if fixed:
        q["axis1"] = new_a1
        q["axis2"] = axis2
    return fixed


def get_category(tag: str) -> str:
    """Get category letter from a tag (e.g., 'A1' -> 'A')."""
    return TAG_TO_CATEGORY.get(tag, "")


# ── Pipeline Steps ──────────────────────────────────────────────────────

def step1_deduplicate(questions: list[dict]) -> tuple[list[dict], int]:
    """
    Remove exact text duplicates only. No near-duplicate removal.
    Returns (kept_questions, removed_count).
    """
    print("=" * 60)
    print("STEP 1: Exact Deduplication")
    print("=" * 60)

    seen = {}          # normalized text -> first index
    removed_indices = set()
    exact_count = 0

    for i, q in enumerate(questions):
        norm = normalize_text(q.get("question", ""))
        if norm in seen:
            removed_indices.add(i)
            exact_count += 1
        else:
            seen[norm] = i

    kept = [q for i, q in enumerate(questions) if i not in removed_indices]
    print(f"  Exact duplicates removed: {exact_count}")
    print(f"  Kept: {len(kept)} / {len(questions)}")
    return kept, exact_count


def step2_auto_tag(questions: list[dict]) -> tuple[list[dict], int, int]:
    """
    Auto-tag untagged questions via keyword matching.
    Also fix misplaced epoch tags in axis1.
    Returns (questions, newly_tagged_count, still_untagged_count).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Auto-tagging & Tag Repair")
    print("=" * 60)

    # Pass 1: Fix misplaced tags (e.g. EP6 in axis1)
    fixed_count = 0
    for q in questions:
        if fix_misplaced_tags(q):
            fixed_count += 1

    # Pass 2: Auto-tag untagged questions
    untagged_before = sum(1 for q in questions if is_untagged(q))
    newly_tagged = 0

    for q in questions:
        if is_untagged(q):
            suggested_a1, suggested_a2 = suggest_tags(q.get("question", ""))
            if suggested_a1:
                # Merge: keep existing substantive tags, add suggested
                existing = [t for t in q.get("axis1", [])
                            if t != "cross" and t in ALL_AXIS1_TAGS]
                merged_a1 = list(dict.fromkeys(existing + suggested_a1))
                q["axis1"] = merged_a1
                if "cross" in q.get("axis1", []) and merged_a1:
                    # Preserve cross_domain flag
                    pass
                if suggested_a2:
                    existing_a2 = [e for e in q.get("axis2", [])
                                   if e in AXIS_2_EPOCHS]
                    merged_a2 = list(dict.fromkeys(existing_a2 + suggested_a2))
                    q["axis2"] = merged_a2
                newly_tagged += 1

    still_untagged = sum(1 for q in questions if is_untagged(q))
    print(f"  Misplaced tags fixed: {fixed_count}")
    print(f"  Untagged before: {untagged_before}")
    print(f"  Newly tagged: {newly_tagged}")
    print(f"  Still untagged: {still_untagged}")

    if still_untagged > 0:
        print(f"  Sample still-untagged questions:")
        shown = 0
        for q in questions:
            if is_untagged(q) and shown < 5:
                print(f"    ID {q.get('id', '?')}: {q.get('question', '')[:80]}...")
                shown += 1

    return questions, newly_tagged, still_untagged


# ── Gap Report ──────────────────────────────────────────────────────────

def compute_gap_report(questions: list[dict]) -> dict:
    """
    Compute subtag × epoch coverage matrix and identify gaps.
    Returns a structured dict with counts and gap cells.
    """
    # Count current distribution
    subtag_counts = Counter()
    cat_counts = Counter()
    epoch_counts = Counter()
    subtag_epoch_matrix = {}  # (subtag, epoch) -> count

    # Initialize matrix
    for subtag in ALL_AXIS1_TAGS:
        for epoch in AXIS_2_EPOCHS:
            if epoch == "EP7":
                continue  # EP7 is overlay, skip
            subtag_epoch_matrix[(subtag, epoch)] = 0

    for q in questions:
        a1_substantive = [t for t in q.get("axis1", [])
                          if t != "cross" and t in ALL_AXIS1_TAGS]
        a2 = [e for e in q.get("axis2", []) if e in AXIS_2_EPOCHS and e != "EP7"]

        for t in a1_substantive:
            subtag_counts[t] += 1
            cat = get_category(t)
            if cat:
                cat_counts[cat] += 1
            for e in a2:
                subtag_epoch_matrix[(t, e)] = subtag_epoch_matrix.get((t, e), 0) + 1

        for e in a2:
            epoch_counts[e] += 1

    # Identify gap cells: subtag × epoch cells with 0 questions
    gap_cells = []
    for (subtag, epoch), count in sorted(subtag_epoch_matrix.items()):
        if count == 0:
            cat = get_category(subtag)
            gap_cells.append({
                "subtag": subtag,
                "subtag_name": ALL_AXIS1_TAGS.get(subtag, subtag),
                "category": cat,
                "epoch": epoch,
                "epoch_name": AXIS_2_EPOCHS.get(epoch, {}).get("name", epoch),
            })

    return {
        "total_questions": len(questions),
        "cat_counts": dict(cat_counts),
        "subtag_counts": dict(subtag_counts),
        "epoch_counts": dict(epoch_counts),
        "gap_cells": gap_cells,
        "gap_count": len(gap_cells),
        "total_matrix_cells": len(subtag_epoch_matrix),
        "filled_cells": len(subtag_epoch_matrix) - len(gap_cells),
    }


def print_gap_report(report: dict) -> str:
    """Print a human-readable gap report."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("COVERAGE & GAP REPORT")
    lines.append("=" * 70)
    lines.append(f"Total questions: {report['total_questions']}")
    lines.append(f"Subtag×Epoch matrix: {report['filled_cells']}/{report['total_matrix_cells']} cells filled")
    lines.append(f"Gap cells (empty): {report['gap_count']}")

    # Axis 1 category distribution
    lines.append("")
    lines.append("Axis 1 Category Distribution:")
    for cat in sorted(AXIS_1_CATEGORIES.keys()):
        count = report["cat_counts"].get(cat, 0)
        pct = count / report["total_questions"] * 100 if report["total_questions"] else 0
        lines.append(f"  {cat} ({AXIS_1_CATEGORIES[cat]['name']}): {count} ({pct:.1f}%)")

    # Axis 1 subtag distribution
    lines.append("")
    lines.append("Axis 1 Subtag Distribution:")
    for subtag in sorted(report["subtag_counts"].keys()):
        count = report["subtag_counts"][subtag]
        lines.append(f"  {subtag}: {count}")

    # Axis 2 epoch distribution
    lines.append("")
    lines.append("Axis 2 Epoch Distribution:")
    for epoch in sorted(AXIS_2_EPOCHS.keys()):
        count = report["epoch_counts"].get(epoch, 0)
        pct = count / report["total_questions"] * 100 if report["total_questions"] else 0
        lines.append(f"  {epoch} ({AXIS_2_EPOCHS[epoch]['name']}): {count} ({pct:.1f}%)")

    # Gap cells by category
    lines.append("")
    lines.append("=" * 70)
    lines.append("GAP CELLS — subtag × epoch combinations with 0 questions")
    lines.append("=" * 70)

    gaps_by_cat = {}
    for g in report["gap_cells"]:
        cat = g["category"]
        gaps_by_cat.setdefault(cat, []).append(g)

    for cat in sorted(gaps_by_cat.keys()):
        gaps = gaps_by_cat[cat]
        lines.append(f"\n  {cat} — {AXIS_1_CATEGORIES.get(cat, {}).get('name', cat)}: {len(gaps)} gaps")
        for g in gaps:
            lines.append(f"    {g['subtag']} ({g['subtag_name']}) × {g['epoch']} ({g['epoch_name']})")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dedup, auto-tag, and report gaps in question coverage"
    )
    parser.add_argument("--input", default="data/raw/questions.json.bak",
                        help="Input questions file (default: .bak to preserve current)")
    parser.add_argument("--output", default="data/raw/questions.json",
                        help="Output questions file")
    parser.add_argument("--backup", default="data/raw/questions.json.pre_redist",
                        help="Backup file path")
    parser.add_argument("--gap-report", default="data/raw/gap_report.json",
                        help="Gap report output file (JSON)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate gap report from input file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    with open(input_path) as f:
        questions = json.load(f)

    if isinstance(questions, dict) and "questions" in questions:
        questions = questions["questions"]

    print(f"Loaded {len(questions)} questions from {input_path}")

    if args.report_only:
        report = compute_gap_report(questions)
        print(print_gap_report(report))
        return

    # Backup
    if not args.dry_run:
        shutil.copy2(args.input, args.backup)
        print(f"Backed up to {args.backup}")

    # ── Pipeline ──
    # Step 1: Exact deduplication
    questions, dupes_removed = step1_deduplicate(questions)

    # Step 2: Auto-tag + fix misplaced tags
    questions, newly_tagged, still_untagged = step2_auto_tag(questions)

    # Step 3: Gap report
    report = compute_gap_report(questions)
    report_text = print_gap_report(report)
    print(report_text)

    # ── Write Output ──
    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(questions, f, indent=2)

        # Write gap report as JSON for AI to consume
        gap_path = Path(args.gap_report)
        gap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gap_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nWritten {len(questions)} questions to {output_path}")
        print(f"Gap report written to {gap_path}")
    else:
        print(f"\n[Dry run] Would write {len(questions)} questions to {args.output}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Started:  {report['total_questions'] + dupes_removed} questions")
    print(f"  Dedup:    removed {dupes_removed} exact duplicates")
    print(f"  Auto-tag: tagged {newly_tagged}, still untagged {still_untagged}")
    print(f"  Final:    {report['total_questions']} questions")
    print(f"  Gaps:     {report['gap_count']} empty subtag×epoch cells")
    print(f"\n  Next step: Author new questions for the {report['gap_count']} gap cells.")
    print(f"  Gap report: {args.gap_report}")


if __name__ == "__main__":
    main()
