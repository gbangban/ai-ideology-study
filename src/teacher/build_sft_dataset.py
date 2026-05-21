#!/usr/bin/env python3
"""
Build trace-aligned SFT dataset from converted JSON.

Constructs ShareGPT-format samples with Qwen3.5 native thinking tokens:
  User: <question>
  Assistant: <thought><reasoning trace></thought><synthesis>

Usage:
    python3 -m src.teacher.build_sft_dataset \
        --input data/processed/sft_dataset.json \
        --output data/processed/sft_dataset.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def extract_synthesis(answer: str) -> str:
    """
    Extract the Synthesis section from the structured answer.

    The teacher answer has format:
      ### Structural Analysis
      **Material Conditions** ...
      ...
      ### Synthesis
      <essay text>

    We want the Synthesis as the final answer after the thought block.
    """
    if "### Synthesis" in answer:
        idx = answer.index("### Synthesis")
        return answer[idx + len("### Synthesis"):].strip()
    # Fallback: return full answer
    return answer.strip()


def build_sft_sample(record: dict) -> dict:
    """
    Build a single SFT sample with trace alignment.

    Format:
      conversations: [
        {role: "user", content: "<question>"},
        {role: "assistant", content: "<thought><trace></thought><synthesis>"}
      ]
    """
    question = record.get("question", "").strip()
    trace = record.get("answer__reasoning_content", "").strip()
    answer = record.get("answer", "").strip()
    synthesis = extract_synthesis(answer)

    assistant_content = ""
    if trace:
        assistant_content = f"<thought>\n{trace}\n</thought>\n{synthesis}"
    else:
        assistant_content = synthesis

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_sft_dataset(input_path: str, output_path: str) -> int:
    """
    Build full SFT dataset from JSON records.

    Returns number of samples written.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    samples = []
    for record in records:
        sample = build_sft_sample(record)
        if sample["conversations"][0]["content"] and sample["conversations"][1]["content"]:
            samples.append(sample)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Build trace-aligned SFT dataset")
    parser.add_argument("--input", default="data/processed/sft_dataset.json", help="Input JSON file")
    parser.add_argument("--output", default="data/processed/sft_dataset.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    count = build_sft_dataset(args.input, args.output)
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
