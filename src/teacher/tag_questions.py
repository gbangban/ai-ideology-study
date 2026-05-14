"""
Question Topic Tagging & Generation

Tag existing questions with topic taxonomy tags, generate new question
bases from axis permutations, and compute coverage metrics.

Usage:
    python -m src.teacher.tag_questions tag <input.jsonl> <output.jsonl>
        Tag questions with topic tags (interactive or auto)

    python -m src.teacher.tag_questions generate-bases [options]
        Generate question bases from axis permutations

    python -m src.teacher.tag_questions coverage <file.jsonl>
        Show coverage report

    python -m src.teacher.tag_questions dedup <file.jsonl>
        Find duplicates by topic tags
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Optional

from src.teacher.topics import (
    ALL_AXIS1_TAGS,
    AXIS_1_CATEGORIES,
    AXIS_2_EPOCHS,
    QuestionBase,
    coverage_metrics,
    coverage_report,
    find_duplicates,
    multi_axis_permutations,
    single_axis_permutations,
    tag_name,
)


# ── Heuristic Auto-Tagging ─────────────────────────────────────────────
# Keyword-based hints for automatic tag suggestion.
# These are suggestions only — manual review is recommended.

TAG_KEYWORDS: dict[str, list[str]] = {
    # Class & Labor
    "A1": ["wage", "exploitation", "surplus value", "labor power", "profit", "exploit"],
    "A2": ["unemployment", "reserve army", "precar", "casual", "jobless"],
    "A3": ["class composition", "labor aristocracy", "managerial", "professional"],
    "A4": ["union", "solidarity", "consciousness", "organize", "strike"],
    "A5": ["informal", "shadow economy", "gig", "unregulated"],
    # Race
    "B1": ["anti-black", "black", "carceral", "mass incarceration"],
    "B2": ["white", "whiteness", "white privilege"],
    "B3": ["racial capitalism", "racialized labor"],
    "B4": ["indigenous", "land theft", "sovereignty"],
    "B5": ["asian", "model minority", "diaspora"],
    "B6": ["latinx", "latino", "border", "migrant", "immigrant", "xenophob"],
    "B7": ["colorblind", "meritocracy", "post-racial"],
    # Gender
    "C1": ["domestic work", "double burden", "patriarch", "unpaid"],
    "C2": ["wage gap", "feminization", "poverty", "occupational segregation"],
    "C3": ["trans", "transgender", "non-binary"],
    "C4": ["lgbtq", "queer", "criminalization", "homophob"],
    "C5": ["gendered", "sexual division", "feminization of labor"],
    "C6": ["reproductive", "abortion", "sterilization", "fertility"],
    # Social Reproduction
    "D1": ["care work", "childcare", "eldercare", "domestic labor"],
    "D2": ["housing", "rent", "shelter", "displacement", "gentrification"],
    "D3": ["healthcare", "medical", "health", "hospital"],
    "D4": ["education", "student debt", "school", "credential"],
    "D5": ["food", "nutrition", "agriculture", "hunger"],
    "D6": ["time poverty", "working hours", "commute", "leisure"],
    # Disability
    "E1": ["productivity", "ableism", "disability"],
    "E2": ["disability", "social model", "medical model"],
    "E3": ["care dependency", "chronic illness", "assisted living"],
    "E4": ["mental health", "depression", "anxiety", "alienation"],
    "E5": ["neurodivergent", "autism", "adhd"],
    # Coloniality
    "F1": ["enclosure", "primitive accumulation", "land theft"],
    "F2": ["settler colonialism", "indigenous", "sovereignty"],
    "F3": ["neocolonial", "extraction", "debt dependency"],
    "F4": ["epistemic", "knowledge hierarchy", "cultural erasure"],
    "F5": ["border", "territorial", "migration management"],
    # Age
    "G1": ["youth", "student", "neet", "gig precar"],
    "G2": ["elder", "pension", "nursing home", "aging"],
    "G3": ["inheritance", "wealth transfer", "intergenerational"],
    "G4": ["temporal", "life course", "retirement"],
    # Immigration
    "H1": ["migrant", "undocumented", "wage suppression"],
    "H2": ["border industrial", "detention", "privatized border"],
    "H3": ["documentation", "visa", "legal status"],
    "H4": ["brain drain", "skilled migration"],
    "H5": ["climate displacement", "environmental migration", "refugee"],
    # Religion
    "I1": ["religious institution", "church wealth", "faith-based charity"],
    "I2": ["secularism", "western norm"],
    "I3": ["faith-based", "religious community"],
    "I4": ["islamophob", "antisemit", "religious racism"],
    # Geography
    "J1": ["urban", "rural", "infrastructure"],
    "J2": ["global north", "global south", "core-periphery"],
    "J3": ["environmental racism", "pollution", "toxic"],
    "J4": ["segregation", "redlining", "ghetto"],
    "J5": ["gentrification", "urban renewal", "spatial fix"],
    # Intersectional
    "K1": ["black trans", "black transgender"],
    "K2": ["indigenous women", "missing indigenous"],
    "K3": ["disabled migrant"],
    "K4": ["elder poor", "elder of color"],
    "K5": ["queer migrant"],
    "K6": ["young black men"],
    "K7": ["working-class mother"],
    "K8": ["disabled women of color"],
}

# Epoch keyword hints
EPOCH_KEYWORDS: dict[str, list[str]] = {
    "EP1": ["feudal", "pre-capitalist", "medieval", "serf"],
    "EP2": ["enclosure", "slave trade", "colonial", "1500", "1600", "1700", "1800"],
    "EP3": ["industrial", "factory", "imperialism", "1800", "1900", "world war"],
    "EP4": ["post-war", "welfare state", "civil rights", "cold war", "1945", "1950", "1960"],
    "EP5": ["neoliberal", "financialization", "globalization", "1980", "1990", "2000", "austerity"],
    "EP6": ["2008", "platform", "pandemic", "ai", "climate crisis", "2020", "2024", "2025"],
}


def suggest_tags(question: str) -> tuple[list[str], list[str]]:
    """
    Suggest Axis 1 and Axis 2 tags for a question using keyword matching.

    Returns:
        (axis1_tags, axis2_tags)
    """
    text = question.lower()
    axis1 = []
    for tag_id, keywords in TAG_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                axis1.append(tag_id)
                break

    axis2 = []
    for epoch_id, keywords in EPOCH_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                axis2.append(epoch_id)
                break

    # Default to E6 (present) if no epoch matched
    if not axis2:
        axis2 = ["EP6"]

    return axis1, axis2


# ── Commands ────────────────────────────────────────────────────────────

def cmd_tag(input_path: str, output_path: str, auto: bool = False) -> None:
    """Tag questions with topic tags."""
    questions = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    tagged = []
    for q in questions:
        q.setdefault("axis1", [])
        q.setdefault("axis2", [])

        if not q["axis1"] or not q["axis2"]:
            if auto:
                a1, a2 = suggest_tags(q["question"])
                q["axis1"] = a1 if a1 else q["axis1"]
                q["axis2"] = a2 if a2 else q["axis2"]
            else:
                # Interactive mode — show suggestion
                a1, a2 = suggest_tags(q["question"])
                qid = q.get("id", "?")
                print(f"\nQ{qid}: {q['question'][:80]}...")
                print(f"  Suggested axis1: {a1} ({', '.join(tag_name(t) for t in a1)})")
                print(f"  Suggested axis2: {a2}")
                print(f"  Enter axis1 tags (comma-separated, or empty to accept): ", end="")
                inp = input()
                if inp.strip():
                    q["axis1"] = [t.strip() for t in inp.split(",")]
                else:
                    q["axis1"] = a1

                print(f"  Enter axis2 tags (comma-separated, or empty to accept): ", end="")
                inp = input()
                if inp.strip():
                    q["axis2"] = [t.strip() for t in inp.split(",")]
                else:
                    q["axis2"] = a2

        tagged.append(q)

    with open(output_path, "w") as f:
        for q in tagged:
            f.write(json.dumps(q) + "\n")

    print(f"\nTagged {len(tagged)} questions → {output_path}")

    # Show coverage
    bases = [QuestionBase.from_dict(q) for q in tagged]
    print(coverage_report(bases))


def cmd_generate_bases(
    output_path: str,
    count: int = 100,
    include_multi: bool = True,
    seed: Optional[int] = None,
) -> None:
    """Generate question bases from axis permutations."""
    if seed is not None:
        random.seed(seed)

    bases = []

    # Single-axis permutations
    if not include_multi:
        pairs = single_axis_permutations()
        random.shuffle(pairs)
        for a1, a2 in pairs[:count]:
            bases.append({
                "axis1": [a1],
                "axis2": [a2],
                "axis1_names": [tag_name(a1)],
                "axis2_names": [tag_name(a2)],
            })
    else:
        # Mix single and multi-axis
        singles = single_axis_permutations()
        random.shuffle(singles)

        # Generate some multi-axis combos
        multis = []
        all_tags = all_axis1_tags()
        for r in range(2, 4):  # 2-3 axis1 tags
            combos = list(itertools.combinations(all_tags, r))
            sample_size = min(50, len(combos))
            for combo in random.sample(combos, sample_size):
                epoch = random.choice(list(AXIS_2_EPOCHS.keys()))
                multis.append({
                    "axis1": list(combo),
                    "axis2": [epoch],
                    "axis1_names": [tag_name(t) for t in combo],
                    "axis2_names": [tag_name(epoch)],
                })

        # Combine and shuffle
        single_dicts = [
            {
                "axis1": [a1],
                "axis2": [a2],
                "axis1_names": [tag_name(a1)],
                "axis2_names": [tag_name(a2)],
            }
            for a1, a2 in singles
        ]

        all_bases = single_dicts + multis
        random.shuffle(all_bases)

        bases = all_bases[:count]

    # Assign IDs
    for i, b in enumerate(bases):
        b["id"] = i + 1

    with open(output_path, "w") as f:
        for b in bases:
            f.write(json.dumps(b) + "\n")

    print(f"Generated {len(bases)} question bases → {output_path}")
    print("\nSample bases:")
    for b in bases[:10]:
        a1_names = ", ".join(b["axis1_names"])
        a2_names = ", ".join(b["axis2_names"])
        print(f"  {b['axis1']} × {b['axis2']}: {a1_names} in {a2_names}")


def cmd_coverage(input_path: str) -> None:
    """Show coverage report for a question file."""
    questions = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    bases = [QuestionBase.from_dict(q) for q in questions]
    print(coverage_report(bases))


def cmd_dedup(input_path: str, mode: str = "exact") -> None:
    """Find duplicates by topic tags."""
    questions = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    bases = [QuestionBase.from_dict(q) for q in questions]
    dups = find_duplicates(bases, mode=mode)

    if not dups:
        print(f"No {mode} duplicates found.")
        return

    print(f"Found {len(dups)} {mode} duplicate pairs:")
    for i, j in dups[:20]:  # Show first 20
        q1 = bases[i]
        q2 = bases[j]
        print(f"  [{i}] {q1.axis1_tags}×{q1.axis2_tags}: {q1.question[:60]}...")
        print(f"  [{j}] {q2.axis1_tags}×{q2.axis2_tags}: {q2.question[:60]}...")
        print()

    if len(dups) > 20:
        print(f"  ... and {len(dups) - 20} more")


# ── CLI ─────────────────────────────────────────────────────────────────

def all_axis1_tags():
    from src.teacher.topics import all_axis1_tags as _all
    return _all()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.teacher.tag_questions <command> [args]")
        print("Commands:")
        print("  tag <input.jsonl> <output.jsonl> [--auto]    Tag questions")
        print("  generate-bases <output.jsonl> [-n COUNT]     Generate question bases")
        print("  coverage <file.jsonl>                        Coverage report")
        print("  dedup <file.jsonl> [--mode exact|partial]    Find duplicates")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "tag":
        if len(sys.argv) < 4:
            print("Usage: tag <input.jsonl> <output.jsonl> [--auto]")
            sys.exit(1)
        cmd_tag(sys.argv[2], sys.argv[3], auto=("--auto" in sys.argv))

    elif cmd == "generate-bases":
        if len(sys.argv) < 3:
            print("Usage: generate-bases <output.jsonl> [-n COUNT] [--multi] [-s SEED]")
            sys.exit(1)
        count = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].lstrip("-").isdigit() else 100
        cmd_generate_bases(sys.argv[2], count=count, include_multi=("--multi" in sys.argv))

    elif cmd == "coverage":
        if len(sys.argv) < 3:
            print("Usage: coverage <file.jsonl>")
            sys.exit(1)
        cmd_coverage(sys.argv[2])

    elif cmd == "dedup":
        if len(sys.argv) < 3:
            print("Usage: dedup <file.jsonl> [--mode exact|partial]")
            sys.exit(1)
        mode = "partial" if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "partial" else "exact"
        cmd_dedup(sys.argv[2], mode=mode)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
