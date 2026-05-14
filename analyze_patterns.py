import json
from collections import Counter

data = json.load(open('data/raw/questions.json'))

# BANNED patterns (4+ words) - these are too repetitive
BANNED = [
    "What does the focus on",
    "What's behind the trend of",
    "What does the narrative of",
    "What does the emphasis on",
    "What is the gap between",
    "How would an analysis that",
    "What does the rise of",
    "What has been the impact",
    "When we examine the explanation",
    "What is the connection between",
    "Why do women in the",
    "Why do policies that treat",
    "Isn't it reasonable to expect",
    "If a person chooses to",
    "Why do workers in the",
    "In what ways does the",
    "How does the fact that",
    "What does the intersection of",
    "What is the relationship between",
    "Why do countries with",
    "What explains the disparity",
    "How does the relationship between",
    "What explains the concentration",
    "What's behind the rise of",
    "What does the language of",
    "What makes it possible for",
    "Is there a way to",
    "What would it take to",
    "How do these two approaches",
    "What explains the decline",
    "If a country has",
    "What's behind the growing",
    "What lies behind the",
    "Why do countries that",
    "Why do people with",
    "What would happen if",
    "What explains the boom",
    "What accounts for the",
    "What does an analysis of",
    "What alternative explanation for the",
    "What is missing from explanations",
    "Why do countries that were",
    "How should we address the",
    "How should we think about",
    "What's driving the increase",
    "How do we explain the",
    "What explains the trend",
    "What explains the growing",
    "What explains the gap",
    "What explains the role",
]

# Cross-domain category targets
TARGETS = {
    'A': 25,   # Class/Labor - reduce from 161
    'B': 30,   # Race - increase from 7
    'C': 25,   # Gender - increase from 6
    'D': 15,   # Reproduction - reduce from 36
    'E': 10,   # Disability - reduce from 18
    'F': 20,   # Coloniality - increase from 5
    'G': 15,   # Age - increase from 2
    'H': 15,   # Immigration - increase from 3
    'I': 15,   # Religion - increase from 1
    'J': 15,   # Geography - increase from 2
    'K': 15,   # Intersectional - increase from 4
}

cross_qs = [q for q in data if q["cross_domain"]]
non_cross = [q for q in data if not q["cross_domain"]]

print(f"Total questions: {len(data)}")
print(f"Cross-domain: {len(cross_qs)}")
print(f"Non-cross: {len(non_cross)}")

# Find questions with repetitive patterns
repetitive_indices = []
for i, q_item in enumerate(data):
    for pattern in BANNED:
        if q_item["question"].strip().startswith(pattern):
            repetitive_indices.append(i)
            break

print(f"\nQuestions with repetitive patterns: {len(repetitive_indices)}")

# Current cross-domain category distribution
current_cats = Counter()
for q in cross_qs:
    for a in q["axis1"]:
        if a == "cross":
            pass
        else:
            current_cats[a[0]] += 1

print(f"\nCurrent cross-domain categories: {dict(current_cats)}")
print(f"Target cross-domain categories: {TARGETS}")

# Calculate how many to replace per category
to_replace = {}
for cat in TARGETS:
    diff = TARGETS[cat] - current_cats.get(cat, 0)
    if diff > 0:
        to_replace[cat] = diff

print(f"\nCategories needing more cross-domain: {to_replace}")

# Save for next step
json.dump({
    "repetitive_indices": repetitive_indices,
    "to_replace": to_replace,
    "current_cats": dict(current_cats),
}, open('/tmp/pattern_analysis.json', 'w'))
print("\nAnalysis saved to /tmp/pattern_analysis.json")
