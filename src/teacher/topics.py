"""
Intersectional Topic Taxonomy

Two-axis tagging system for question generation, diversity tracking, and deduplication.
Axis 1: Intersectional social categories (oppression/privilege axes)
Axis 2: Historical epochs, periods, and events

Permutation system generates question bases by crossing axes.
All questions are tagged for coverage analysis and deduplication.

Authoritative reference: docs/topic_taxonomy.md
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Optional


# ── Axis 1: Intersectional Social Categories ──────────────────────────

AXIS_1_CATEGORIES = {
    "A": {
        "name": "Class & Labor Relations",
        "subtags": {
            "A1": "Wage labor & exploitation",
            "A2": "Reserve army of labor",
            "A3": "Class composition",
            "A4": "Class consciousness & solidarity",
            "A5": "Informal & care economy",
        },
    },
    "B": {
        "name": "Race & Racialization",
        "subtags": {
            "B1": "Anti-Blackness",
            "B2": "Whiteness",
            "B3": "Racial capitalism",
            "B4": "Indigenous dispossession",
            "B5": "Asian racialization",
            "B6": "Latinx & border regimes",
            "B7": "Colorblind ideology",
        },
    },
    "C": {
        "name": "Gender & Sexuality",
        "subtags": {
            "C1": "Patriarchal reproduction",
            "C2": "Feminization of poverty",
            "C3": "Trans material conditions",
            "C4": "Queer erasure & criminalization",
            "C5": "Sexual division of labor",
            "C6": "Reproductive justice",
        },
    },
    "D": {
        "name": "Social Reproduction",
        "subtags": {
            "D1": "Unpaid care work",
            "D2": "Housing as reproduction",
            "D3": "Healthcare & embodiment",
            "D4": "Education as reproduction",
            "D5": "Food systems",
            "D6": "Time poverty",
        },
    },
    "E": {
        "name": "Disability & Ableism",
        "subtags": {
            "E1": "Productivity norm",
            "E2": "Disability as social construction",
            "E3": "Care dependency",
            "E4": "Mental health & alienation",
            "E5": "Neurodivergence & labor",
        },
    },
    "F": {
        "name": "Coloniality & Indigeneity",
        "subtags": {
            "F1": "Primitive accumulation",
            "F2": "Settler colonialism",
            "F3": "Neocolonial extraction",
            "F4": "Epistemic violence",
            "F5": "Border regimes",
        },
    },
    "G": {
        "name": "Age & Generational Position",
        "subtags": {
            "G1": "Youth surplus populations",
            "G2": "Elder care & abandonment",
            "G3": "Intergenerational wealth",
            "G4": "Temporal discipline",
        },
    },
    "H": {
        "name": "Immigration & Documentation Status",
        "subtags": {
            "H1": "Migrant superexploitation",
            "H2": "Border industrial complex",
            "H3": "Documentation as social control",
            "H4": "Brain drain & global inequality",
            "H5": "Climate displacement",
        },
    },
    "I": {
        "name": "Religion & Secularism",
        "subtags": {
            "I1": "Religious institutions & capital",
            "I2": "Secularism as Western norm",
            "I3": "Faith-based reproduction",
            "I4": "Religious racism",
        },
    },
    "J": {
        "name": "Geography & Spatial Power",
        "subtags": {
            "J1": "Urban/rural divides",
            "J2": "Global North/South",
            "J3": "Environmental racism",
            "J4": "Segregation & ghettoization",
            "J5": "Spatial fixes",
        },
    },
    "K": {
        "name": "Intersectional Identities (Compound)",
        "subtags": {
            "K1": "Black trans women",
            "K2": "Indigenous women",
            "K3": "Disabled migrants",
            "K4": "Elder poor of color",
            "K5": "Queer migrants",
            "K6": "Young Black men",
            "K7": "Working-class mothers",
            "K8": "Disabled women of color",
        },
    },
}

# Flatten all Axis 1 subtags
ALL_AXIS1_TAGS: dict[str, str] = {}
for cat_id, cat in AXIS_1_CATEGORIES.items():
    for tag_id, tag_name in cat["subtags"].items():
        ALL_AXIS1_TAGS[tag_id] = tag_name

# Map subtag -> category
TAG_TO_CATEGORY: dict[str, str] = {}
for cat_id, cat in AXIS_1_CATEGORIES.items():
    for tag_id in cat["subtags"]:
        TAG_TO_CATEGORY[tag_id] = cat_id


# ── Axis 2: Historical Epochs ──────────────────────────────────────────

AXIS_2_EPOCHS = {
    "EP1": {
        "name": "Pre-Capitalist Formations",
        "timeframe": "Pre-1500",
        "features": "Feudalism, serfdom, communal property, slave societies",
    },
    "EP2": {
        "name": "Primitive Accumulation",
        "timeframe": "1500s-1800s",
        "features": "Enclosure, transatlantic slave trade, colonial extraction, Indigenous dispossession",
    },
    "EP3": {
        "name": "Industrial Capitalism",
        "timeframe": "1800s-1945",
        "features": "Factory system, imperialism, labor movements, scientific racism, gendered factory labor",
    },
    "EP4": {
        "name": "State Monopoly Capitalism",
        "timeframe": "1945-1973",
        "features": "Post-war boom, welfare state, decolonization, civil rights, Cold War, suburbanization",
    },
    "EP5": {
        "name": "Neoliberalism",
        "timeframe": "1973-2008",
        "features": "Financialization, globalization, welfare rollback, carceral expansion, offshoring",
    },
    "EP6": {
        "name": "Late Neoliberalism/Crisis",
        "timeframe": "2008-present",
        "features": "Austerity, platform capitalism, climate crisis, pandemic, AI wave, housing financialization",
    },
    "EP7": {
        "name": "Cross-Cutting Events",
        "timeframe": "Any",
        "features": "Financial crises, wars, pandemics, climate disasters, technological revolutions",
    },
}

# Event subtags
AXIS_2_EVENTS = {
    "EP7a": {"name": "1929 Great Depression", "epoch": "EP3"},
    "EP7b": {"name": "2008 Financial Crisis", "epoch": "EP5"},
    "EP7c": {"name": "COVID-19 Pandemic", "epoch": "EP6"},
    "EP7d": {"name": "AI/Automation Wave", "epoch": "EP6"},
    "EP7e": {"name": "Climate tipping points", "epoch": "EP6"},
    "EP7f": {"name": "Decolonization waves", "epoch": "EP4"},
    "EP7g": {"name": "Welfare state construction", "epoch": "EP4"},
    "EP7h": {"name": "Welfare state dismantling", "epoch": "EP5"},
    "EP7i": {"name": "Housing market crashes", "epoch": "EP5,EP6"},
    "EP7j": {"name": "Major labor strikes", "epoch": "EP3,EP4,EP6"},
}

ALL_AXIS2_TAGS: dict[str, str] = {
    **{k: v["name"] for k, v in AXIS_2_EPOCHS.items()},
    **{k: v["name"] for k, v in AXIS_2_EVENTS.items()},
}


# ── Data Structures ────────────────────────────────────────────────────

@dataclass
class QuestionBase:
    """A question base generated by permuting Axis 1 and Axis 2 tags."""
    axis1_tags: list[str]
    axis2_tags: list[str]
    question: str = ""
    question_type: Optional[str] = None  # A-E from Experimental Design
    id: Optional[int] = None
    cross_domain: bool = False

    def tag_key(self) -> str:
        """Unique key for dedup: sorted axis1 × axis2."""
        return f"{'+'.join(sorted(self.axis1_tags))}x{'+'.join(sorted(self.axis2_tags))}"

    def categories(self) -> set[str]:
        """Return the set of Axis 1 category letters."""
        return {TAG_TO_CATEGORY.get(t, "") for t in self.axis1_tags if t in TAG_TO_CATEGORY}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.question_type,
            "axis1": self.axis1_tags,
            "axis2": self.axis2_tags,
            "question": self.question,
            "cross_domain": self.cross_domain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuestionBase":
        return cls(
            axis1_tags=d.get("axis1", []),
            axis2_tags=d.get("axis2", []),
            question=d.get("question", ""),
            question_type=d.get("type"),
            id=d.get("id"),
            cross_domain=d.get("cross_domain", False),
        )


# ── Permutation Engine ─────────────────────────────────────────────────

def all_axis1_tags() -> list[str]:
    """Return all Axis 1 subtag IDs."""
    return list(ALL_AXIS1_TAGS.keys())


def all_axis2_epoch_tags() -> list[str]:
    """Return all Axis 2 epoch tag IDs (excluding event subtags)."""
    return list(AXIS_2_EPOCHS.keys())


def all_axis2_event_tags() -> list[str]:
    """Return all Axis 2 event subtag IDs."""
    return list(AXIS_2_EVENTS.keys())


def single_axis_permutations(
    axis1_tags: Optional[list[str]] = None,
    axis2_tags: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """
    Generate all (axis1_tag, axis2_tag) pairs.

    Args:
        axis1_tags: Filter to specific Axis 1 tags. None = all.
        axis2_tags: Filter to specific Axis 2 tags. None = all epochs.

    Returns:
        List of (axis1_tag, axis2_tag) tuples.
    """
    a1 = axis1_tags or all_axis1_tags()
    a2 = axis2_tags or all_axis2_epoch_tags()
    return list(itertools.product(a1, a2))


def multi_axis_permutations(
    axis1_tags: Optional[list[str]] = None,
    axis2_tags: Optional[list[str]] = None,
    max_axis1: int = 3,
) -> list[tuple[list[str], list[str]]]:
    """
    Generate permutations allowing multiple Axis 1 tags per combination.

    Args:
        axis1_tags: Filter to specific Axis 1 tags. None = all.
        axis2_tags: Filter to specific Axis 2 tags. None = all epochs.
        max_axis1: Maximum number of Axis 1 tags per combination.

    Returns:
        List of ([axis1_tags], [axis2_tag]) tuples.
    """
    a1 = axis1_tags or all_axis1_tags()
    a2 = axis2_tags or all_axis2_epoch_tags()
    results = []
    for r in range(1, max_axis1 + 1):
        for combo in itertools.combinations(a1, r):
            for epoch in a2:
                results.append((list(combo), [epoch]))
    return results


def category_epoch_permutations(
    categories: Optional[list[str]] = None,
    epochs: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """
    Generate (category_letter, epoch) pairs for high-level coverage planning.

    Args:
        categories: Filter to specific category letters. None = all.
        epochs: Filter to specific epoch tags. None = all.

    Returns:
        List of (category_letter, epoch_tag) tuples.
    """
    cats = categories or list(AXIS_1_CATEGORIES.keys())
    eps = epochs or all_axis2_epoch_tags()
    return list(itertools.product(cats, eps))


# ── Deduplication ───────────────────────────────────────────────────────

def is_exact_duplicate(q1: QuestionBase, q2: QuestionBase) -> bool:
    """Two questions share the exact same (axis1 × axis2) tag pair."""
    return q1.tag_key() == q2.tag_key()


def is_partial_overlap(q1: QuestionBase, q2: QuestionBase) -> bool:
    """
    Two questions share the same Axis 1 category AND same epoch.
    Softer duplicate — may be intentional for depth.
    """
    cats1 = q1.categories()
    cats2 = q2.categories()
    epochs1 = set(q1.axis2_tags)
    epochs2 = set(q2.axis2_tags)
    return bool(cats1 & cats2) and bool(epochs1 & epochs2)


def find_duplicates(
    questions: list[QuestionBase],
    mode: str = "exact",
) -> list[tuple[int, int]]:
    """
    Find duplicate question pairs.

    Args:
        questions: List of QuestionBase objects.
        mode: 'exact' for exact tag pair matches, 'partial' for category+epoch overlaps.

    Returns:
        List of (index1, index2) tuples that are duplicates.
    """
    dup_func = is_exact_duplicate if mode == "exact" else is_partial_overlap
    duplicates = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if dup_func(questions[i], questions[j]):
                duplicates.append((i, j))
    return duplicates


# ── Diversity Metrics ──────────────────────────────────────────────────

def coverage_metrics(questions: list[QuestionBase]) -> dict:
    """
    Calculate diversity and coverage metrics for a question set.

    Returns:
        Dict of metric names to values.
    """
    if not questions:
        return {"error": "Empty question set"}

    total = len(questions)

    # Axis 1 category coverage
    categories_present = set()
    for q in questions:
        categories_present.update(q.categories())
    axis1_coverage = len(categories_present) / len(AXIS_1_CATEGORIES)

    # Axis 2 epoch coverage
    epochs_present = set()
    for q in questions:
        epochs_present.update(q.axis2_tags)
    # Only count E1-E6 for coverage (E7 is overlay)
    core_epochs = {e for e in epochs_present if e in AXIS_2_EPOCHS and e != "E7"}
    axis2_coverage = len(core_epochs) / 6  # 6 core epochs

    # Intersectional density
    multi_tag_count = sum(1 for q in questions if len(q.axis1_tags) >= 2)
    intersectional_density = multi_tag_count / total

    # Tag pair uniqueness
    tag_keys = {q.tag_key() for q in questions}
    tag_pair_uniqueness = len(tag_keys) / total

    # Per-epoch balance
    epoch_counts = {}
    for q in questions:
        for e in q.axis2_tags:
            epoch_counts[e] = epoch_counts.get(e, 0) + 1
    epoch_values = list(epoch_counts.values())
    if len(epoch_values) > 1:
        epoch_mean = sum(epoch_values) / len(epoch_values)
        epoch_std = (sum((v - epoch_mean) ** 2 for v in epoch_values) / len(epoch_values)) ** 0.5
        epoch_balance = epoch_std / epoch_mean if epoch_mean > 0 else 1.0
    elif len(epoch_values) == 1:
        epoch_balance = 0.0
    else:
        epoch_balance = 1.0  # No epochs tagged

    # Per-category balance
    cat_counts = {}
    for q in questions:
        for c in q.categories():
            cat_counts[c] = cat_counts.get(c, 0) + 1
    cat_values = list(cat_counts.values())
    if len(cat_values) > 1:
        cat_mean = sum(cat_values) / len(cat_values)
        cat_std = (sum((v - cat_mean) ** 2 for v in cat_values) / len(cat_values)) ** 0.5
        cat_balance = cat_std / cat_mean if cat_mean > 0 else 1.0
    elif len(cat_values) == 1:
        cat_balance = 0.0
    else:
        cat_balance = 1.0  # No categories tagged

    return {
        "total_questions": total,
        "axis1_coverage": round(axis1_coverage, 3),
        "axis1_categories_present": len(categories_present),
        "axis1_categories_total": len(AXIS_1_CATEGORIES),
        "axis2_coverage": round(axis2_coverage, 3),
        "axis2_epochs_present": len(core_epochs),
        "axis2_epochs_total": 6,
        "intersectional_density": round(intersectional_density, 3),
        "tag_pair_uniqueness": round(tag_pair_uniqueness, 3),
        "epoch_balance_cv": round(epoch_balance, 3),
        "category_balance_cv": round(cat_balance, 3),
    }


def coverage_report(questions: list[QuestionBase]) -> str:
    """Generate a human-readable coverage report."""
    m = coverage_metrics(questions)
    lines = ["# Topic Coverage Report", ""]
    for k, v in m.items():
        lines.append(f"- **{k}**: {v}")

    # Per-category breakdown
    lines.append("")
    lines.append("## Axis 1 Category Breakdown")
    cat_counts = {}
    for q in questions:
        for c in q.categories():
            cat_counts[c] = cat_counts.get(c, 0) + 1
    for cat_id in sorted(AXIS_1_CATEGORIES.keys()):
        cat_name = AXIS_1_CATEGORIES[cat_id]["name"]
        count = cat_counts.get(cat_id, 0)
        lines.append(f"- {cat_id} ({cat_name}): {count}")

    # Per-epoch breakdown
    lines.append("")
    lines.append("## Axis 2 Epoch Breakdown")
    epoch_counts = {}
    for q in questions:
        for e in q.axis2_tags:
            epoch_counts[e] = epoch_counts.get(e, 0) + 1
    for epoch_id in sorted(AXIS_2_EPOCHS.keys()):
        epoch_name = AXIS_2_EPOCHS[epoch_id]["name"]
        count = epoch_counts.get(epoch_id, 0)
        lines.append(f"- {epoch_id} ({epoch_name}): {count}")

    return "\n".join(lines)


# ── Tag Resolution Helpers ─────────────────────────────────────────────

def tag_name(tag_id: str) -> str:
    """Resolve a tag ID to its human-readable name."""
    if tag_id in ALL_AXIS1_TAGS:
        return ALL_AXIS1_TAGS[tag_id]
    elif tag_id in ALL_AXIS2_TAGS:
        return ALL_AXIS2_TAGS[tag_id]
    return tag_id


def tag_description(tag_id: str) -> str:
    """Return category name for Axis 1 tags, epoch name for Axis 2."""
    if tag_id in TAG_TO_CATEGORY:
        cat = AXIS_1_CATEGORIES[TAG_TO_CATEGORY[tag_id]]
        return f"{cat['name']}: {ALL_AXIS1_TAGS[tag_id]}"
    elif tag_id in AXIS_2_EPOCHS:
        return AXIS_2_EPOCHS[tag_id]["name"]
    elif tag_id in AXIS_2_EVENTS:
        return f"{AXIS_2_EVENTS[tag_id]['name']} (in {AXIS_2_EPOCHS[AXIS_2_EVENTS[tag_id]['epoch']]['name']})"
    return tag_id


# ── I/O ─────────────────────────────────────────────────────────────────

def save_questions(questions: list[QuestionBase], path: str) -> None:
    """Save questions as JSONL with topic tags."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict()) + "\n")


def load_questions(path: str) -> list[QuestionBase]:
    """Load questions from JSONL with topic tags."""
    questions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                questions.append(QuestionBase.from_dict(json.loads(line)))
    return questions


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.teacher.topics <command> [args]")
        print("Commands:")
        print("  list-axis1          List all Axis 1 tags")
        print("  list-axis2          List all Axis 2 tags")
        print("  permutations        Show single-axis permutation count")
        print("  coverage <file>     Show coverage report for JSONL file")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list-axis1":
        for tag_id, name in sorted(ALL_AXIS1_TAGS.items()):
            cat = AXIS_1_CATEGORIES[TAG_TO_CATEGORY[tag_id]]
            print(f"  {tag_id} ({cat['name']}): {name}")

    elif cmd == "list-axis2":
        for tag_id, info in sorted(AXIS_2_EPOCHS.items()):
            print(f"  {tag_id}: {info['name']} [{info['timeframe']}]")
        print("  Events:")
        for tag_id, info in sorted(AXIS_2_EVENTS.items()):
            print(f"    {tag_id}: {info['name']} (epoch: {info['epoch']})")

    elif cmd == "permutations":
        singles = single_axis_permutations()
        print(f"  Single Axis 1 × Epoch: {len(singles)} combinations")
        print(f"  Axis 1 subtags: {len(ALL_AXIS1_TAGS)}")
        print(f"  Axis 2 epochs: {len(AXIS_2_EPOCHS)}")
        print(f"  Total matrix cells: {len(ALL_AXIS1_TAGS) * len(AXIS_2_EPOCHS)}")

    elif cmd == "coverage":
        if len(sys.argv) < 3:
            print("Usage: python -m src.teacher.topics coverage <file.jsonl>")
            sys.exit(1)
        questions = load_questions(sys.argv[2])
        print(coverage_report(questions))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
