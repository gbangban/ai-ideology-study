#!/usr/bin/env python3
"""
build_questions_json.py - Assemble data/raw/questions.json from scratch.

Target: 1,500 questions with balanced distribution across:
  - 11 Axis 1 categories (A-K), ~136 each
  - 6 Epochs (EP1-EP6), 250 each
  - 5 Types (A-E): 40%/20%/20%/5%/15%
  - 60 subtags, minimum 15 each

Strategy: deficit-driven epoch assignment for EP6 questions.
"""

import json
import sys
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEPRECATED_PATH = Path("data/raw/deprecated-questions.json")
HAND_AUTHORED_PATH = Path("data/raw/hand_authored_questions.json")
OUTPUT_PATH = Path("data/raw/questions.json")

TARGET_TOTAL = 1500
TARGET_PER_EPOCH = 250
TYPE_TARGETS = {"A": 600, "B": 300, "C": 300, "D": 75, "E": 225}
TYPE_TOLERANCE = {"A": 30, "B": 15, "C": 15, "D": 5, "E": 15}
TYPE_LABELS = {
    "A": "Neutral Framing", "B": "Contrast", "C": "Application",
    "D": "Conceptual DM", "E": "Adversarial",
}
MIN_PER_SUBTAG = 15
MIN_PER_CAT = 75

ALL_SUBTAGS = {
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "D1", "D2", "D3", "D4", "D5", "D6",
    "E1", "E2", "E3", "E4", "E5",
    "F1", "F2", "F3", "F4", "F5",
    "G1", "G2", "G3", "G4",
    "H1", "H2", "H3", "H4", "H5",
    "I1", "I2", "I3", "I4",
    "J1", "J2", "J3", "J4", "J5",
    "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8",
}
ALL_EPOCHS = ["EP1", "EP2", "EP3", "EP4", "EP5", "EP6"]

# Invalid subtag×epoch combinations
INVALID_EP_SUBTAG = {
    ("K1", "EP1"), ("K3", "EP1"), ("K5", "EP1"), ("K6", "EP1"), ("K8", "EP1"),
    ("C3", "EP1"), ("C3", "EP2"),
    ("E5", "EP1"), ("E5", "EP2"),
    ("H5", "EP1"), ("H5", "EP2"), ("H5", "EP3"),
    ("J2", "EP1"), ("J4", "EP1"),
}

DM_TERMS = [
    "dialectical", "historical materialism", "mode of production",
    "surplus value", "commodity fetishism", "reification",
    "bourgeoisie", "proletariat", "base and superstructure",
    "means of production", "relations of production",
    "class struggle", "class consciousness",
]
TEMPLATE_PATTERNS = [
    r"^Why do the .* still reflect",
    r"^Why do the .* continue to reflect",
    r"^How does the concept of .* fail to capture",
    r"^What does the idea of .* leave out",
]


def is_valid_ep_subtag(subtag, epoch):
    return (subtag, epoch) not in INVALID_EP_SUBTAG


def passes_quality_filter(text):
    text_lower = text.lower()
    for term in DM_TERMS:
        if term in text_lower:
            return False
    for pattern in TEMPLATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    if len(text.strip()) < 20:
        return False
    return True


def clean_axis1(axis1_list):
    return [t for t in axis1_list if t != "cross"]


def clean_question(q, force_epoch=None):
    """Return a cleaned copy. If force_epoch is set, replace axis2 with it."""
    axis2 = [force_epoch] if force_epoch else [ep for ep in q["axis2"] if ep not in ("EP6", "EP7")]
    if not axis2:
        axis2 = ["EP6"]
    return {
        "type": q["type"],
        "type_label": TYPE_LABELS.get(q["type"], q.get("type_label", "")),
        "question": q["question"],
        "cross_domain": q.get("cross_domain", False),
        "axis1": clean_axis1(q["axis1"]),
        "axis2": axis2,
    }


def best_epoch_for_question(axis1_tags, epoch_counts):
    """Find the epoch with the biggest deficit that is valid for all subtags."""
    best_ep = None
    best_deficit = -99999
    for ep in ALL_EPOCHS:
        valid = all(is_valid_ep_subtag(tag, ep) for tag in axis1_tags)
        if valid:
            deficit = TARGET_PER_EPOCH - epoch_counts.get(ep, 0)
            if deficit > best_deficit:
                best_deficit = deficit
                best_ep = ep
    return best_ep


def main():
    random.seed(42)

    print("=" * 60)
    print("BUILD questions.json - 1,500 question assembly")
    print("=" * 60)

    # Load pools
    print(f"\nLoading pools...")
    with open(DEPRECATED_PATH) as f:
        deprecated = json.load(f)
    with open(HAND_AUTHORED_PATH) as f:
        hand_authored = json.load(f)
    print(f"  Deprecated: {len(deprecated)}, Hand-authored: {len(hand_authored)}")

    # -----------------------------------------------------------------------
   # Step 1: Clean and quality-filter deprecated pool
    print(f"\n--- Step 1: Clean deprecated pool ---")
    cleaned = []
    skipped_empty_axis1 = 0
    skipped_dm_terms = 0
    skipped_template = 0
    for q in deprecated:
        if len(clean_axis1(q["axis1"])) == 0:
            skipped_empty_axis1 += 1
            continue
        ql = q["question"].lower()
        if any(term in ql for term in DM_TERMS):
            skipped_dm_terms += 1
            continue
        if any(re.search(p, q["question"], re.IGNORECASE) for p in TEMPLATE_PATTERNS):
            skipped_template += 1
            continue
        if len(q["question"].strip()) < 20:
            continue
        cq = clean_question(q)
        cleaned.append(cq)
    print(f"  After quality filter: {len(cleaned)}")
    print(f"  Skipped: empty_axis1={skipped_empty_axis1}, dm_terms={skipped_dm_terms}, template={skipped_template}")

    # Deduplicate
    seen = set()
    deduped = []
    for q in cleaned:
        key = q["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    print(f"  After dedup: {len(deduped)}")

    # Separate EP6-only vs non-EP6
    non_ep6 = [q for q in deduped if "EP6" not in q["axis2"]]
    ep6_only = [q for q in deduped if q["axis2"] == ["EP6"]]
    print(f"  Non-EP6: {len(non_ep6)}, EP6-only: {len(ep6_only)}")

    # -----------------------------------------------------------------------
    # Step 2: Deficit-driven epoch assignment for EP6 questions
    # -----------------------------------------------------------------------
    print(f"\n--- Step 2: Assign epochs to EP6 questions ---")

    # Start epoch counts from non-EP6 + hand-authored
    epoch_counts = Counter()
    for q in non_ep6:
        for ep in q["axis2"]:
            epoch_counts[ep] += 1
    for q in hand_authored:
        for ep in q["axis2"]:
            epoch_counts[ep] += 1

    # Assign EP6 questions to epochs based on deficit
    assigned_ep6 = []
    for q in ep6_only:
        ep = best_epoch_for_question(q["axis1"], epoch_counts)
        if ep is None:
            ep = "EP6"  # fallback
        cq = clean_question(q, force_epoch=ep)
        assigned_ep6.append(cq)
        epoch_counts[ep] += 1

    print(f"  Epoch counts after assignment: {dict(epoch_counts)}")

    # -----------------------------------------------------------------------
 # Step 3: Combine all pools
    print(f"\n--- Step 3: Combine pools ---")

    # Normalize hand-authored questions: ensure type_label and cross_domain
    normalized_hand = []
    for q in hand_authored:
        nq = dict(q)
        nq.setdefault("type_label", TYPE_LABELS.get(q["type"], ""))
        nq.setdefault("cross_domain", False)
        normalized_hand.append(nq)

    all_questions = non_ep6 + assigned_ep6 + normalized_hand
    print(f"  Total combined: {len(all_questions)}")

    # -----------------------------------------------------------------------
  # Step 4: Select 1,500 questions with balanced distribution (iterative greedy)
    print(f"\n--- Step 4: Select {TARGET_TOTAL} balanced questions (iterative greedy) ---")

    def score_question(q, sel_epoch, sel_type, sel_subtag):
        score = 0.0
        # Bonus for underrepresented subtags (strong signal for filling minimums)
        for tag in q["axis1"]:
            cnt = sel_subtag.get(tag, 0)
            if cnt < MIN_PER_SUBTAG:
                score += max(0, (MIN_PER_SUBTAG - cnt)) * 20
        # Bonus for underrepresented types
        t = q["type"]
        deficit = TYPE_TARGETS[t] - sel_type.get(t, 0)
        if deficit > 0:
            score += deficit * 5
        # Penalty for overrepresented types
        excess = sel_type.get(t, 0) - TYPE_TARGETS[t]
        if excess > 0:
            score -= excess * 15
        # Penalty for overrepresented epoch
        for ep in q["axis2"]:
            excess_ep = sel_epoch.get(ep, 0) - TARGET_PER_EPOCH
            if excess_ep > 0:
                score -= excess_ep * 8
        # Bonus for underrepresented epoch
        for ep in q["axis2"]:
            deficit_ep = TARGET_PER_EPOCH - sel_epoch.get(ep, 0)
            if deficit_ep > 0:
                score += deficit_ep * 2
        # Bonus for intersectional (≥2 axis1 tags)
        if len(q["axis1"]) >= 2:
            score += 2
        return score

    # Iterative greedy selection
    remaining = list(all_questions)  # copy
    selected = []
    sel_epoch = Counter()
    sel_type = Counter()
    sel_subtag = Counter()

    for i in range(TARGET_TOTAL):
        best_score = -999999
        best_idx = 0
        best_q = None
        for j, q in enumerate(remaining):
            s = score_question(q, sel_epoch, sel_type, sel_subtag)
            if s > best_score:
                best_score = s
                best_idx = j
                best_q = q
        selected.append(best_q)
        remaining.pop(best_idx)
        # Update running counts
        for ep in best_q["axis2"]:
            sel_epoch[ep] += 1
        sel_type[best_q["type"]] += 1
        for tag in best_q["axis1"]:
            sel_subtag[tag] += 1

    print(f"  Selected: {len(selected)}")
    print(f"  Type distribution: {dict(sel_type)}")

    # -----------------------------------------------------------------------
 # Step 5: Post-selection balancing
    print(f"\n--- Step 5: Post-selection balancing ---")

    # Recompute counts from selection
    sel_epoch = Counter()
    sel_type = Counter()
    sel_subtag = Counter()
    for q in selected:
        for ep in q["axis2"]:
            sel_epoch[ep] += 1
        sel_type[q["type"]] += 1
        for tag in q["axis1"]:
            sel_subtag[tag] += 1

    # Epoch rebalancing: reassign from over to under
    over_epochs = [ep for ep in ALL_EPOCHS if sel_epoch[ep] > TARGET_PER_EPOCH + 5]
    under_epochs = [ep for ep in ALL_EPOCHS if sel_epoch[ep] < TARGET_PER_EPOCH - 5]

    reassigned = 0
    for over_ep in over_epochs:
        excess = sel_epoch[over_ep] - TARGET_PER_EPOCH
        candidates = [q for q in selected if over_ep in q["axis2"]]
        random.shuffle(candidates)
        for q in candidates[:excess]:
            for target_ep in under_epochs:
                if all(is_valid_ep_subtag(tag, target_ep) for tag in q["axis1"]):
                    old_ep = q["axis2"][0]
                    q["axis2"] = [target_ep]
                    sel_epoch[old_ep] -= 1
                    sel_epoch[target_ep] += 1
                    reassigned += 1
                    break
            if sel_epoch[over_ep] <= TARGET_PER_EPOCH:
                break
    if reassigned:
        print(f"  Reassigned {reassigned} questions between epochs")

    # Type balancing: swap overrepresented types for underrepresented ones
    selected_ids = set(id(q) for q in selected)
    # Swap C→A if C is over and A is under
    if sel_type["C"] > TYPE_TARGETS["C"] + TYPE_TOLERANCE["C"] and sel_type["A"] < TYPE_TARGETS["A"] - TYPE_TOLERANCE["A"]:
        excess_c = sel_type["C"] - (TYPE_TARGETS["C"] + TYPE_TOLERANCE["C"])
        needed_a = (TYPE_TARGETS["A"] - TYPE_TOLERANCE["A"]) - sel_type["A"]
        swap_count = min(excess_c, needed_a, len(remaining))
        if swap_count > 0:
            to_remove = [q for q in selected if q["type"] == "C"][:swap_count]
            replacements = [q for q in remaining if q["type"] == "A"][:swap_count]
            for q in to_remove:
                selected.remove(q)
            for q in replacements:
                selected.append(q)
                remaining.remove(q)
            # Update counts
            sel_type["C"] -= swap_count
            sel_type["A"] += swap_count
            print(f"  Swapped {swap_count} Type C for Type A")

    # Swap E→A if E is over and A is under
    if sel_type["E"] > TYPE_TARGETS["E"] + TYPE_TOLERANCE["E"] and sel_type["A"] < TYPE_TARGETS["A"] - TYPE_TOLERANCE["A"]:
        excess_e = sel_type["E"] - (TYPE_TARGETS["E"] + TYPE_TOLERANCE["E"])
        needed_a = (TYPE_TARGETS["A"] - TYPE_TOLERANCE["A"]) - sel_type["A"]
        swap_count = min(excess_e, needed_a, len([q for q in remaining if q["type"] == "A"]))
        if swap_count > 0:
            to_remove = [q for q in selected if q["type"] == "E"][:swap_count]
            replacements = [q for q in remaining if q["type"] == "A"][:swap_count]
            for q in to_remove:
                selected.remove(q)
            for q in replacements:
                selected.append(q)
                remaining.remove(q)
            sel_type["E"] -= swap_count
            sel_type["A"] += swap_count
            print(f"  Swapped {swap_count} Type E for Type A")

    # If Type D is under, swap from overrepresented type
    if sel_type["D"] < TYPE_TARGETS["D"] - TYPE_TOLERANCE["D"]:
        needed = (TYPE_TARGETS["D"] - TYPE_TOLERANCE["D"]) - sel_type["D"]
        d_candidates = [q for q in remaining if q["type"] == "D"]
        if d_candidates:
            # Remove from most overrepresented type
            for swap_from in ("C", "B", "E"):
                if sel_type.get(swap_from, 0) > TYPE_TARGETS[swap_from] + TYPE_TOLERANCE[swap_from]:
                    to_remove = [q for q in selected if q["type"] == swap_from][:needed]
                    for q in to_remove:
                        selected.remove(q)
                    for q in d_candidates[:len(to_remove)]:
                        selected.append(q)
                        remaining.remove(q)
                    sel_type[swap_from] -= len(to_remove)
                    sel_type["D"] += len(to_remove)
                    print(f"  Swapped {len(to_remove)} Type {swap_from} for Type D")
                    break

    # -----------------------------------------------------------------------
    # Step 6: Final verification
    # -----------------------------------------------------------------------
    print(f"\n--- Step 6: Final verification ---")

    # Trim to exactly TARGET_TOTAL
    if len(selected) > TARGET_TOTAL:
        # Remove from most overrepresented epoch
        final_epoch = Counter()
        for q in selected:
            for ep in q["axis2"]:
                final_epoch[ep] += 1
        over_ep = final_epoch.most_common(1)[0][0]
        excess = len(selected) - TARGET_TOTAL
        to_trim = [q for q in selected if over_ep in q["axis2"]][-excess:]
        selected = [q for q in selected if q not in to_trim]
    elif len(selected) < TARGET_TOTAL:
        # Add from remaining pool
        selected_ids = set(id(q) for q in selected)
        for q in remaining:
            if id(q) not in selected_ids:
                selected.append(q)
                if len(selected) >= TARGET_TOTAL:
                    break

    # Sort by epoch, then category
    epoch_order = {ep: i for i, ep in enumerate(ALL_EPOCHS)}
    cat_order = {c: i for i, c in enumerate("ABCDEFGHIJK")}
    selected.sort(key=lambda q: (
        epoch_order.get(q["axis2"][0], 99),
        cat_order.get(q["axis1"][0][0], 99),
        q["question"]
    ))

    # Assign sequential IDs
    for i, q in enumerate(selected, 1):
        q["id"] = i

    # Final distribution report
    cat_counts = Counter()
    subtag_counts = Counter()
    epoch_counts_final = Counter()
    type_counts_final = Counter()
    intersectional = 0
    tag_pairs = set()

    for q in selected:
        type_counts_final[q["type"]] += 1
        for tag in q["axis1"]:
            cat_counts[tag[0]] += 1
            subtag_counts[tag] += 1
        for ep in q["axis2"]:
            epoch_counts_final[ep] += 1
        if len(q["axis1"]) >= 2:
            intersectional += 1
        for ep in q["axis2"]:
            for tag in q["axis1"]:
                tag_pairs.add((tag, ep))

    total = len(selected)
    print(f"\n{'='*60}")
    print(f"DISTRIBUTION REPORT")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"\nType distribution:")
    for t in sorted(type_counts_final):
        target = TYPE_TARGETS[t]
        tol = TYPE_TOLERANCE[t]
        status = "✓" if target - tol <= type_counts_final[t] <= target + tol else "✗"
        print(f"  {status} {t}: {type_counts_final[t]} (target {target}±{tol})")
    print(f"\nCategory distribution:")
    for c in sorted(cat_counts):
        status = "✓" if cat_counts[c] >= MIN_PER_CAT else "✗"
        print(f"  {status} {c}: {cat_counts[c]} (min {MIN_PER_CAT})")
    print(f"\nEpoch distribution:")
    for ep in ALL_EPOCHS:
        status = "✓" if abs(epoch_counts_final[ep] - TARGET_PER_EPOCH) <= 10 else "✗"
        print(f"  {status} {ep}: {epoch_counts_final[ep]} (target {TARGET_PER_EPOCH}±10)")
    # Tag pair coverage = unique(subtag, epoch) pairs / max possible (60 subtags × 6 epochs)
    max_pairs = len(ALL_SUBTAGS) * len(ALL_EPOCHS)
    tag_pair_coverage = len(tag_pairs) / max_pairs
    print(f"\nDiversity metrics:")
    print(f"  Intersectional density: {intersectional/total*100:.1f}% (target ≥30%)")
    print(f"  Tag pair coverage: {tag_pair_coverage:.3f} (target ≥0.7, max {max_pairs} pairs)")
    print(f"  Unique tag pairs: {len(tag_pairs)}/{max_pairs}")

    under_subtags = [st for st in sorted(ALL_SUBTAGS) if subtag_counts.get(st, 0) < MIN_PER_SUBTAG]
    if under_subtags:
        print(f"\nSubtags below minimum ({MIN_PER_SUBTAG}):")
        for st in under_subtags:
            print(f"  ✗ {st}: {subtag_counts.get(st, 0)}")
    else:
        print(f"\n✓ All subtags meet minimum of {MIN_PER_SUBTAG}")

    # Count failures
    failures = 0
    for t in TYPE_TARGETS:
        if not (TYPE_TARGETS[t] - TYPE_TOLERANCE[t] <= type_counts_final.get(t, 0) <= TYPE_TARGETS[t] + TYPE_TOLERANCE[t]):
            failures += 1
    for c in cat_counts:
        if cat_counts[c] < MIN_PER_CAT:
            failures += 1
    for ep in ALL_EPOCHS:
        if abs(epoch_counts_final.get(ep, 0) - TARGET_PER_EPOCH) > 10:
            failures += 1
    if under_subtags:
        failures += len(under_subtags)

    # Output
    print(f"\nWriting {len(selected)} questions to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Output: {OUTPUT_PATH}")
    if failures:
        print(f"⚠ {failures} targets not met")
        return 1
    else:
        print(f"✓ All targets met!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
