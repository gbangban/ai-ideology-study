#!/usr/bin/env python3
"""Parse questions.txt → deduplicated JSON."""

import re
import json
import sys
from collections import defaultdict

def parse_questions(filepath):
    questions = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Format: "123. <!-- type:A `cross-domain` --> Question text?"
            # or:     "A251. <!-- type:A --> Question text?"
            m = re.match(
                r'^[A-Za-z]*\d+\.\s+<!--\s+type:([A-E])\s*(?:`([^`]+)`)?\s*-->\s+(.+)$',
                line
            )
            if m:
                qtype = m.group(1)
                tag = m.group(2)
                text = m.group(3).strip()
                questions.append({
                    'type': qtype,
                    'tags': [tag] if tag else [],
                    'text': text,
                    '_line': line_num,
                })
    return questions


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[?!.]+$', '', t).strip()
    t = re.sub(r'\s+', ' ', t)
    return t


def content_tokens(text):
    norm = normalize(text).lower()
    stops = {'what', 'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
             'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
             'go', 'to', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
             'at', 'for', 'of', 'with', 'by', 'from', 'as', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'between',
             'that', 'this', 'these', 'those', 'it', 'its', 'not', 'no', 'nor',
             'so', 'if', 'then', 'than', 'also', 'both', 'each', 'every',
             'such', 'only', 'own', 'same', 'too', 'very', 'just', 'about',
             'up', 'out', 'he', 'she', 'they', 'them', 'their', 'we', 'our',
             'you', 'your', 'who', 'which', 'where', 'when', 'over', 'while',
             'despite', 'even', 'still', 'keep', 'often', 'typically', 'actually',
             'really', 'much', 'more', 'less', 'some', 'any', 'all', 'both',
             'other', 'another', 'enough', 'same', 'different', 'new', 'old',
             'good', 'bad', 'big', 'small', 'large', 'high', 'low', 'long',
             'short', 'hard', 'easy', 'able', 'like', 'make', 'take', 'get',
             'give', 'use', 'find', 'see', 'know', 'think', 'come', 'want',
             'try', 'ask', 'feel', 'become', 'leave', 'put', 'mean', 'call',
             'first', 'well', 'also', 'because', 'should', 'shouldn', 'isn',
             'aren', 'don', 'doesn', 'won', 'wouldn', 'couldn', 'didn',
             'can', 'cannot', 'cannot', 'let', 'need', 'must', 'may', 'might',
             'per', 'vs', 'via'}
    tokens = set(re.findall(r'[a-z]{3,}', norm)) - stops
    return tokens


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


BANNED_TERMS = [
    'marxist', 'materialist', 'liberal', 'neoliberal',
    'historical materialism', 'dialectical', 'class analysis',
    'bourgeoisie', 'proletariat', 'surplus value', 'mode of production',
    'means of production', 'relations of production', 'base and superstructure',
    'commodity fetishism', 'labor theory of value', 'primitive accumulation',
    'metabolic rift', 'social reproduction', 'socialist', 'communism',
    'social democracy', 'democratic socialism', 'anarchist', 'anarchism',
]

def has_banned_terms(text):
    lower = text.lower()
    for term in BANNED_TERMS:
        if term in lower:
            return True, term
    return False, None


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/questions.txt'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'data/raw/questions_clean.json'

    print(f"Parsing {input_file}...")
    questions = parse_questions(input_file)
    print(f"Found {len(questions)} questions")

    type_counts = defaultdict(int)
    cross_counts = defaultdict(int)
    for q in questions:
        type_counts[q['type']] += 1
        if 'cross-domain' in q['tags']:
            cross_counts[q['type']] += 1
    print(f"By type: {dict(type_counts)}")
    print(f"Cross-domain: {dict(cross_counts)}")

    # ── Deduplication ──────────────────────────────────────────────────
    print("\nDetecting duplicates (Jaccard threshold=0.55)...")

    tokens_list = [content_tokens(q['text']) for q in questions]

    # Union-Find
    parent = list(range(len(questions)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare within same type
    type_indices = defaultdict(list)
    for i, q in enumerate(questions):
        type_indices[q['type']].append(i)

    dup_pairs = []
    for qtype, indices in type_indices.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                sim = jaccard(tokens_list[a], tokens_list[b])
                if sim >= 0.55:
                    union(a, b)
                    dup_pairs.append((a, b, sim))

    # Collect groups
    groups = defaultdict(list)
    for i in range(len(questions)):
        groups[find(i)].append(i)

    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"Found {len(multi_groups)} duplicate groups ({sum(len(v) for v in multi_groups.values())} questions)")

    # Select best from each group
    def best_of(group):
        best = group[0]
        best_score = -999
        for idx in group:
            q = questions[idx]
            score = 0
            words = len(q['text'].split())
            if 5 <= words <= 30:
                score += 10
            elif words < 5:
                score += 5
            else:
                score -= (words - 30)
            if 'cross-domain' in q['tags']:
                score += 5
            if q['text'].endswith('?'):
                score += 2
            if 'company x' in q['text'].lower():
                score -= 3
            if score > best_score:
                best_score = score
                best = idx
        return best

    remove_indices = set()
    for root, group in multi_groups.items():
        best = best_of(group)
        for idx in group:
            if idx != best:
                remove_indices.add(idx)

    # Also check exact normalized text matches
    text_map = {}
    for i, q in enumerate(questions):
        norm = normalize(q['text']).lower()
        if norm in text_map:
            remove_indices.add(i)
        else:
            text_map[norm] = i

    # Remove questions with banned terms (except Type D)
    for i, q in enumerate(questions):
        has_banned, term = has_banned_terms(q['text'])
        if has_banned and q['type'] != 'D':
            remove_indices.add(i)

    # Build clean list
    clean = []
    for i, q in enumerate(questions):
        if i not in remove_indices:
            clean.append({
                'type': q['type'],
                'tags': q['tags'],
                'text': q['text'],
            })

    # ── Summary ────────────────────────────────────────────────────────
    final_counts = defaultdict(int)
    final_cross = defaultdict(int)
    for q in clean:
        final_counts[q['type']] += 1
        if 'cross-domain' in q['tags']:
            final_cross[q['type']] += 1

    targets = {'A': 100, 'B': 50, 'C': 50, 'D': 13, 'E': 37}

    print(f"\n{'='*60}")
    print(f"Original: {len(questions)}")
    print(f"Removed:  {len(remove_indices)}")
    print(f"Clean:    {len(clean)}")
    print(f"\nFinal by type:")
    for t in 'ABCDE':
        diff = final_counts[t] - targets.get(t, 0)
        status = "OK" if diff == 0 else (f"+{diff}" if diff > 0 else str(diff))
        cd = f" (cross-domain: {final_cross[t]})" if final_cross[t] else ""
        print(f"  Type {t}: {final_counts[t]} (target {targets.get(t, '?')}) {status}{cd}")

    # ── Write JSON ─────────────────────────────────────────────────────
    output = {
        'metadata': {
            'total_questions': len(clean),
            'distribution': {t: final_counts[t] for t in 'ABCDE'},
            'cross_domain_count': sum(final_cross.values()),
            'targets': targets,
            'dedup_stats': {
                'original_count': len(questions),
                'duplicate_groups': len(multi_groups),
                'removed_count': len(remove_indices),
            },
            'core_constraint': 'No question contains explicit references to analytical frameworks, ideologies, or theoretical lenses',
            'quality_criteria': [
                'Ideologically neutral phrasing',
                'No framework cues',
                'Plausible liberal answer exists',
                'Structurally superior DM answer possible',
                'Different conclusion, not different vocabulary',
                'Grounded in concrete phenomena',
                'Adversarial signal for Type E',
            ],
        },
        'questions': clean,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWritten to {output_file}")

    # Print dup groups for review
    print(f"\n{'='*60}")
    print("DUPLICATE GROUPS (for review):")
    for root, group in sorted(multi_groups.items(), key=lambda x: -len(x[1])):
        best = best_of(group)
        print(f"\n  Group ({len(group)} items), KEEP: #{best} (line {questions[best]['_line']})")
        for idx in group:
            q = questions[idx]
            marker = "← KEEP" if idx == best else "→ REMOVE"
            print(f"    {marker} [{q['type']}] L{q['_line']}: {q['text'][:100]}")


if __name__ == '__main__':
    main()
