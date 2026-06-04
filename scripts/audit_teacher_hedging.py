#!/usr/bin/env python3
"""
Audit teacher answers from SFT dataset for hedging patterns.

Checks each answer against hedging patterns from src/student/rewards.py
and directional commitment patterns. Outputs a structured report.

Usage:
    python3 scripts/audit_teacher_hedging.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.student.rewards import _HEDGING_PATTERNS, POSITIVE_PATTERNS

DATA_PATH = Path(__file__).resolve().parent.parent / "data/processed/batch_00000.json"


def audit_answer(answer: str, idx: int) -> dict:
    """Check a single answer for hedging and commitment patterns."""
    answer_lower = answer.lower()

    hedging_matches = []
    for pattern in _HEDGING_PATTERNS:
        matches = list(re.finditer(pattern, answer_lower))
        if matches:
            for m in matches:
                context_start = max(0, m.start() - 40)
                context_end = min(len(answer), m.end() + 40)
                hedging_matches.append({
                    "pattern": pattern,
                    "matched": m.group(0),
                    "context": answer[context_start:context_end].replace("\n", " "),
                })

    commitment_matches = []
    for pattern in POSITIVE_PATTERNS:
        matches = list(re.finditer(pattern, answer_lower))
        if matches:
            for m in matches:
                context_start = max(0, m.start() - 40)
                context_end = min(len(answer), m.end() + 40)
                commitment_matches.append({
                    "pattern": pattern,
                    "matched": m.group(0),
                    "context": answer[context_start:context_end].replace("\n", " "),
                })

    return {
        "idx": idx,
        "hedging_count": len(hedging_matches),
        "commitment_count": len(commitment_matches),
        "hedging_matches": hedging_matches,
        "commitment_matches": commitment_matches,
        "classified": "committed" if len(commitment_matches) > len(hedging_matches)
                      else "hedging" if len(hedging_matches) > len(commitment_matches)
                      else "neutral",
    }


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    results = []
    for i, sample in enumerate(data):
        answer = sample.get("answer", "")
        if not answer:
            continue
        results.append(audit_answer(answer, i))

    committed = [r for r in results if r["classified"] == "committed"]
    hedging = [r for r in results if r["classified"] == "hedging"]
    neutral = [r for r in results if r["classified"] == "neutral"]

    print(f"Total answers audited: {len(results)}")
    print(f"Committed: {len(committed)} ({len(committed)/len(results)*100:.1f}%)")
    print(f"Hedging: {len(hedging)} ({len(hedging)/len(results)*100:.1f}%)")
    print(f"Neutral: {len(neutral)} ({len(neutral)/len(results)*100:.1f}%)")
    print()

    # Per-pattern frequency
    print("=== HEDGING PATTERN FREQUENCY ===")
    pattern_counts = {}
    for r in results:
        for m in r["hedging_matches"]:
            p = m["pattern"]
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
    for p, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}x  {p}")
    print()

    print("=== COMMITMENT PATTERN FREQUENCY ===")
    pattern_counts = {}
    for r in results:
        for m in r["commitment_matches"]:
            p = m["pattern"]
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
    for p, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}x  {p}")
    print()

    # Show hedging examples
    print("=== HEDGING ANSWERS (first 10) ===")
    for r in hedging[:10]:
        sample = data[r["idx"]]
        print(f"\n--- Answer #{r['idx']} ---")
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Hedging matches ({r['hedging_count']}):")
        for m in r["hedging_matches"]:
            print(f"  [{m['matched']}] ...{m['context']}...")
        if r["commitment_matches"]:
            print(f"Commitment matches ({r['commitment_count']}):")
            for m in r["commitment_matches"]:
                print(f"  [{m['matched']}] ...{m['context']}...")

    # Show committed examples
    print("\n\n=== COMMITTED ANSWERS (first 5) ===")
    for r in committed[:5]:
        sample = data[r["idx"]]
        print(f"\n--- Answer #{r['idx']} ---")
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Commitment matches ({r['commitment_count']}):")
        for m in r["commitment_matches"]:
            print(f"  [{m['matched']}] ...{m['context']}...")
        if r["hedging_matches"]:
            print(f"Hedging matches ({r['hedging_count']}):")
            for m in r["hedging_matches"]:
                print(f"  [{m['matched']}] ...{m['context']}...")

    # Save full results
    output_path = Path(__file__).resolve().parent.parent / "data/processed/teacher_hedging_audit.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": len(results),
                "committed": len(committed),
                "hedging": len(hedging),
                "neutral": len(neutral),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n\nFull audit saved to {output_path}")


if __name__ == "__main__":
    main()
