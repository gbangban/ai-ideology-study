import json
from collections import Counter

data = json.load(open('data/raw/questions.json'))

# Current cross-domain distribution
cross_qs = [q for q in data if q["cross_domain"]]
current_cats = Counter()
for q in cross_qs:
    for a in q["axis1"]:
        if a != "cross":
            current_cats[a[0]] += 1

TARGETS = {
    'A': 25, 'B': 30, 'C': 25, 'D': 15, 'E': 10,
    'F': 20, 'G': 15, 'H': 15, 'I': 15, 'J': 15, 'K': 15,
}

print(f"Current: {dict(current_cats)}")
print(f"Target: {TARGETS}")

# Calculate what needs to change
changes = {}
for cat in TARGETS:
    diff = TARGETS[cat] - current_cats.get(cat, 0)
    if diff != 0:
        changes[cat] = diff

print(f"\nChanges needed: {changes}")
print(f"Total to remove: {-sum(v for v in changes.values() if v < 0)}")
print(f"Total to add: {sum(v for v in changes.values() if v > 0)}")

# We need to remove A(86), D(26), E(5) = 117
# We need to add B(0), C(2), F(3), G(1), H(1), I(10), J(8), K(14) = 39
# Wait, the math doesn't add up. Let me recalculate.

# Actually:
# A: 111 -> 25 = remove 86
# B: 30 -> 30 = no change
# C: 23 -> 25 = add 2
# D: 41 -> 15 = remove 26
# E: 15 -> 10 = remove 5
# F: 23 -> 20 = remove 3
# G: 14 -> 15 = add 1
# H: 16 -> 15 = remove 1
# I: 5 -> 15 = add 10
# J: 23 -> 15 = remove 8
# K: 1 -> 15 = add 14

# Total remove: 86 + 26 + 5 + 3 + 1 + 8 = 129
# Total add: 2 + 1 + 10 + 14 = 27
# Net: -102 questions

# We need to maintain 300 cross-domain questions
# So we need to remove 102 and add 0? No, that would give us 198.
# We need to remove 102 and add 102 to maintain 300.

# Let me recalculate:
# Remove: A(86), D(26), E(5), F(3), H(1), J(8) = 129
# Add: C(2), G(1), I(10), K(14) = 27
# Net change: -102
# To maintain 300, we need to add 102 more in categories that need increases

# Actually, let me just calculate what we need:
# A: remove 86
# D: remove 26
# E: remove 5
# F: remove 3
# H: remove 1
# J: remove 8
# Total remove: 129

# C: add 2
# G: add 1
# I: add 10
# K: add 14
# Total add so far: 27
# Need to add 102 more to maintain 300

# Let me distribute the remaining 75 across B, C, F, G, I, K (categories that can accept more)
# B: add 15 (total 45)
# C: add 15 (total 40)
# F: add 10 (total 30)
# G: add 10 (total 25)
# I: add 15 (total 30)
# K: add 15 (total 30)
# Total additional: 80 (close to 75, let me adjust)

# Actually, let me just set reasonable targets that balance the distribution:
NEW_TARGETS = {
    'A': 40,   # Reduce from 111
    'B': 35,   # Increase from 30
    'C': 30,   # Increase from 23
    'D': 20,   # Reduce from 41
    'E': 15,   # Reduce from 15 (same)
    'F': 25,   # Increase from 23
    'G': 20,   # Increase from 14
    'H': 20,   # Increase from 16
    'I': 25,   # Increase from 5
    'J': 20,   # Reduce from 23
    'K': 30,   # Increase from 1
}

print(f"\nNew targets: {NEW_TARGETS}")

changes2 = {}
for cat in NEW_TARGETS:
    diff = NEW_TARGETS[cat] - current_cats.get(cat, 0)
    if diff != 0:
        changes2[cat] = diff

print(f"\nChanges needed for new targets: {changes2}")
print(f"Total to remove: {-sum(v for v in changes2.values() if v < 0)}")
print(f"Total to add: {sum(v for v in changes2.values() if v > 0)}")

# Now let's identify which cross-domain questions to remove
# Priority: remove from overrepresented categories (A, D, J)
# Keep: remove questions with repetitive patterns first

BANNED = [
    "What does the focus on", "What's behind the trend of",
    "What does the narrative of", "What does the emphasis on",
    "What is the gap between", "How would an analysis that",
    "What does the rise of", "What has been the impact",
    "When we examine the explanation", "What is the connection between",
    "Why do women in the", "Why do policies that treat",
    "Isn't it reasonable to expect", "If a person chooses to",
    "Why do workers in the", "In what ways does the",
    "How does the fact that", "What does the intersection of",
    "What is the relationship between", "Why do countries with",
    "What explains the disparity", "How does the relationship between",
    "What explains the concentration", "What's behind the rise of",
    "What does the language of", "What makes it possible for",
    "Is there a way to", "What would it take to",
    "How do these two approaches", "What explains the decline",
    "If a country has", "What's behind the growing",
    "What lies behind the", "Why do countries that",
    "Why do people with", "What would happen if",
    "What explains the boom", "What accounts for the",
    "What does an analysis of", "What alternative explanation for the",
    "What is missing from explanations", "Why do countries that were",
    "How should we address the", "How should we think about",
    "What's driving the increase", "How do we explain the",
    "What explains the trend", "What explains the growing",
    "What explains the gap", "What explains the role",
]

# Find cross-domain questions to remove
cross_indices = [i for i, q_item in enumerate(data) if q_item["cross_domain"]]

# First, find repetitive ones in overrepresented categories
to_remove_repetitive = []
for i in cross_indices:
    q_item = data[i]
    cats = [a[0] for a in q_item["axis1"] if a != "cross"]
    if any(cat in ['A', 'D', 'J'] for cat in cats):
        if any(q_item["question"].strip().startswith(p) for p in BANNED):
            to_remove_repetitive.append(i)

print(f"\nRepetitive cross-domain in A/D/J: {len(to_remove_repetitive)}")

# We need to remove ~129 cross-domain questions
# Remove all repetitive ones first, then fill from A/D/J

to_remove = list(to_remove_repetitive)

# Remove more from A, D, J until we hit target
for cat in ['A', 'D', 'J']:
    current_count = sum(1 for i in cross_indices if cat in [a[0] for a in data[i]["axis1"] if a != "cross"])
    target = NEW_TARGETS[cat]
    to_remove_count = current_count - target
    print(f"Category {cat}: {current_count} -> {target}, remove {to_remove_count}")
    
    if to_remove_count > 0:
        # Find more questions in this category to remove
        for i in cross_indices:
            if i in to_remove:
                continue
            q_item = data[i]
            cats = [a[0] for a in q_item["axis1"] if a != "cross"]
            if cat in cats and len(to_remove) < len(to_remove) + to_remove_count:
                to_remove.append(i)
                if len([x for x in to_remove if cat in [a[0] for a in data[x]["axis1"] if a != "cross"]]) >= to_remove_count:
                    break

print(f"\nTotal to remove: {len(to_remove)}")

# Now add new questions for underrepresented categories
# ... (new questions will be added in next step)

# For now, just remove the questions
for i in sorted(to_remove, reverse=True):
    data.pop(i)

# Renumber IDs
for i, q_item in enumerate(data):
    q_item["id"] = i + 1

# Check current state
cross_qs_new = [q for q in data if q["cross_domain"]]
final_cats = Counter()
for q in cross_qs_new:
    for a in q["axis1"]:
        if a != "cross":
            final_cats[a[0]] += 1

print(f"\nAfter removal: {dict(final_cats)}")
print(f"Total questions: {len(data)}")
print(f"Cross-domain: {len(cross_qs_new)}")

json.dump(data, open('data/raw/questions.json', 'w'), indent=2, ensure_ascii=False)
print("\nWritten to data/raw/questions.json")
