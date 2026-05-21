#!/usr/bin/env python3
"""
Generate rejected responses for DPO training.

Three rejection types to triangulate the target:
1. Liberal Default: mainstream reformist analysis (individual agency, policy design)
2. Jargon Trap: heavy DM terminology without structural rigor (moralistic, rhetorical)
3. Shallow DM: lazy/incomplete DM analysis (superficial, handwavy)

These are template-based generators. For production quality, replace with
LLM-generated responses using liberal-reformist and jargon system prompts.

Usage:
    python3 -m src.teacher.generate_rejected_responses \
        --input data/processed/dpo_dataset.json \
        --output data/processed/rejected_responses.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


LIBERAL_TEMPLATES = [
    "This issue can be addressed through better policy design and institutional reform. "
    "Research shows that targeted interventions, such as {policy}, can produce measurable improvements. "
    "The key is empowering individuals with better opportunities and ensuring markets function efficiently. "
    "Government should focus on creating the right incentives while avoiding overreach that could stifle innovation.",

    "The evidence suggests this is primarily a matter of individual choices and market dynamics. "
    "When people have access to quality information and fair opportunities, they make decisions that benefit themselves and society. "
    "Policy should focus on removing barriers to entry and ensuring competition. "
    "Programs that invest in education and skills development have shown the strongest results in addressing {topic}.",

    "This problem is best understood through the lens of institutional economics and behavioral science. "
    "Studies show that well-designed policy frameworks can align individual incentives with social outcomes. "
    "The solution lies in improving governance, strengthening regulatory oversight, and investing in human capital. "
    "Countries that have implemented {policy} have seen significant improvements in {topic}.",
]

LIBERAL_POLICIES = [
    "progressive taxation", "universal basic income", "education reform",
    "market-based carbon pricing", "antitrust enforcement", "zoning reform",
    "healthcare market competition", "financial literacy programs",
    "public-private partnerships", "regulatory streamlining",
]

JARGON_TEMPLATES = [
    "The bourgeoisie has consistently demonstrated its opposition to progressive change. "
    "We must recognize the evil of the ruling class and their deliberate attempts to suppress the working people. "
    "The capitalists are greedy and selfish, hoarding wealth while the proletariat suffers. "
    "We need to stand in solidarity against the oppressors and fight for a more just society.",

    "This is clearly a manifestation of the inherent evil of the current system. "
    "The ruling elite deliberately exploit the vulnerable for their own enrichment. "
    "We must condemn the moral bankruptcy of those in power and demand immediate action. "
    "The revolution will come when the people rise up against their oppressors.",

    "The problem is that the wealthy have too much power and refuse to share. "
    "Billionaires are the root cause of all social problems, and we must hold them accountable. "
    "The system is rigged by corrupt elites who care only about their own interests. "
    "We need moral leadership to fight against this injustice.",
]

SHALLOW_TEMPLATES = [
    "This reflects the contradictions inherent in the current mode of production. "
    "The relations of production create systemic tensions that cannot be resolved within the existing framework. "
    "The base determines the superstructure, and changes in one will affect the other.",

    "The answer lies in understanding the material conditions of the working class. "
    "Class struggle is the driving force of history, and this situation is no exception. "
    "The contradictions of the system will eventually lead to its transformation.",

    "This can be understood through the lens of historical materialism. "
    "The economic base shapes all social relations, and the current arrangement reflects the interests of the dominant class. "
    "Dialectical analysis reveals the underlying contradictions at play.",
]


def generate_liberal_default(question: str) -> str:
    """Generate a liberal-reformist rejected response."""
    template = random.choice(LIBERAL_TEMPLATES)
    policy = random.choice(LIBERAL_POLICIES)
    topic = question.split("?")[0].split("How")[0].split("Why")[0].split("What")[0].strip()
    if not topic:
        topic = "this issue"
    return template.format(policy=policy, topic=topic)


def generate_jargon_trap(question: str) -> str:
    """Generate a jargon-heavy but structurally shallow rejected response."""
    return random.choice(JARGON_TEMPLATES)


def generate_shallow_dm(question: str) -> str:
    """Generate a lazy/incomplete DM rejected response."""
    return random.choice(SHALLOW_TEMPLATES)


REJECTION_TYPES = ["liberal_default", "jargon_trap", "shallow_dm"]
REJECTION_GENERATORS = {
    "liberal_default": generate_liberal_default,
    "jargon_trap": generate_jargon_trap,
    "shallow_dm": generate_shallow_dm,
}


def generate_all_rejections(question: str) -> list[dict]:
    """Generate all three rejection types for a question."""
    rejections = []
    for rtype, generator in REJECTION_GENERATORS.items():
        response = generator(question)
        rejections.append({
            "type": rtype,
            "content": response,
        })
    return rejections


def main():
    parser = argparse.ArgumentParser(description="Generate rejected responses for DPO training")
    parser.add_argument("--input", default="data/processed/dpo_dataset.json", help="Input DPO dataset JSON")
    parser.add_argument("--output", default="data/processed/rejected_responses.jsonl", help="Output JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            question = record.get("question", "")
            rejections = generate_all_rejections(question)
            output = {
                "id": record.get("id"),
                "question": question,
                "rejections": rejections,
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Generated {3 * len(records)} rejection responses for {len(records)} questions -> {args.output}")


if __name__ == "__main__":
    main()
