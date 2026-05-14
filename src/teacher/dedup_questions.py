"""
Deduplicate questions.jsonl by text similarity and tag overlap.

Usage:
    python -m src.teacher.dedup_questions
"""

import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from src.teacher.tag_questions import suggest_tags
from src.teacher.topics import QuestionBase, coverage_report


def normalize(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def main():
    input_path = "data/raw/questions.jsonl"
    output_json = "data/raw/questions.json"
    output_jsonl = "data/raw/questions.jsonl"

    # Load
    with open(input_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} questions")

    # Auto-tag
    for q in questions:
        q.setdefault("axis1", [])
        q.setdefault("axis2", [])
        if not q["axis1"] or not q["axis2"]:
            a1, a2 = suggest_tags(q["question"])
            q["axis1"] = a1 if a1 else q["axis1"]
            q["axis2"] = a2 if a2 else q["axis2"]

    # Dedup by text similarity (threshold 0.85)
    removed = set()
    text_dupes = []
    for i in range(len(questions)):
        if i in removed:
            continue
        for j in range(i + 1, len(questions)):
            if j in removed:
                continue
            sim = similarity(questions[i]["question"], questions[j]["question"])
            if sim >= 0.85:
                removed.add(j)
                text_dupes.append((i, j, sim))

    print(f"Text dedup: removed {len(text_dupes)} (threshold=0.85)")
    for kept, rm, sim in text_dupes[:5]:
        print(f"  sim={sim:.2f}: kept '{questions[kept]['question'][:60]}...' removed '{questions[rm]['question'][:60]}...'")

    # Dedup by tag overlap + text similarity (threshold 0.6)
    tag_groups = defaultdict(list)
    for i, q in enumerate(questions):
        if i in removed:
            continue
        key = tuple(sorted(q["axis1"])) + tuple(sorted(q["axis2"]))
        tag_groups[key].append(i)

    tag_dupes = []
    for key, indices in tag_groups.items():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if i in removed or j in removed:
                    continue
                sim = similarity(questions[i]["question"], questions[j]["question"])
                if sim >= 0.6:
                    if len(questions[j].get("axis1", [])) > len(questions[i].get("axis1", [])):
                        removed.add(i)
                    else:
                        removed.add(j)
                    tag_dupes.append((i, j, sim, key))

    print(f"Tag dedup: removed {len(tag_dupes)} (threshold=0.6)")

    # Keep
    kept = [q for i, q in enumerate(questions) if i not in removed]
    for i, q in enumerate(kept):
        q["id"] = i + 1

    # Save — JSON (primary), JSONL (secondary/generated)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(kept, f, indent=2)
    with open(output_jsonl, "w") as f:
        for q in kept:
            f.write(json.dumps(q) + "\n")

    print(f"\nFinal: {len(kept)} questions")
    print(f"  Primary: {output_json}")
    print(f"  Secondary: {output_jsonl}")

    # Coverage
    bases = [QuestionBase.from_dict(q) for q in kept]
    print(coverage_report(bases))


if __name__ == "__main__":
    main()