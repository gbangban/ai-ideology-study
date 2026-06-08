#!/usr/bin/env python3
"""Tagless evaluation for v4.

Evaluates a v4-trained model on the merged dataset WITHOUT tags in the prompt.
This tests whether the model has internalized the reasoning process or is just
learning to produce tags as a formatting exercise.

Usage:
    python3 -m src.student.tagless_eval \
        --model checkpoints/lora_adapters/grpo_v4/checkpoint-1000 \
        --dataset data/processed/grpo_train_merged.jsonl \
        --output evals/tagless_v4_results.jsonl \
        --samples 200
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.reward_outcome import compute_outcome_reward

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("tagless-eval")


def _strip_vision_config(model_path: str):
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    stripped = False
    for key in list(config.keys()):
        if "vision" in key.lower():
            del config[key]
            stripped = True
    if stripped:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def evaluate(model_path, dataset_path, output_path, samples=200, seed=42):
    """Evaluate model without tags in prompt."""
    from unsloth import FastLanguageModel

    random.seed(seed)

    logger.info(f"Loading model from {model_path}...")
    _strip_vision_config(model_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    from src.student import fix_mistral_tokenizer
    fix_mistral_tokenizer(tokenizer)

    # Load and sample dataset
    docs = []
    with open(dataset_path) as f:
        for line in f:
            docs.append(json.loads(line))
    sampled = random.sample(docs, min(samples, len(docs)))
    logger.info(f"Evaluating on {len(sampled)} samples")

    results = []
    correct = 0
    total = 0

    for i, doc in enumerate(sampled):
        # Tagless prompt: just the question, no tag instructions
        messages = [{"role": "user", "content": doc["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_len:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)

        outcome = compute_outcome_reward(doc, completion)

        # Check if model still produces tags without being asked
        has_planning = "<planning>" in completion
        has_commitment = "<commitment>" in completion
        has_reflection = "<reflection>" in completion
        has_monitor = "<monitor>" in completion

        result = {
            "doc": doc,
            "completion": completion,
            "outcome_reward": outcome,
            "has_planning": has_planning,
            "has_commitment": has_commitment,
            "has_reflection": has_reflection,
            "has_monitor": has_monitor,
        }
        results.append(result)

        if outcome > 0.5:
            correct += 1
        total += 1

        if (i + 1) % 25 == 0:
            acc = correct / total
            logger.info(f"Evaluated {i+1}/{len(sampled)} | Accuracy: {acc:.3f}")

    accuracy = correct / total if total > 0 else 0.0

    # Tag usage statistics
    tag_stats = {
        "planning": sum(1 for r in results if r["has_planning"]) / total,
        "commitment": sum(1 for r in results if r["has_commitment"]) / total,
        "reflection": sum(1 for r in results if r["has_reflection"]) / total,
        "monitor": sum(1 for r in results if r["has_monitor"]) / total,
    }

    # By dataset type
    from collections import defaultdict
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        dt = r["doc"].get("dataset_type", "unknown")
        by_type[dt]["total"] += 1
        if r["outcome_reward"] > 0.5:
            by_type[dt]["correct"] += 1

    type_accuracy = {k: v["correct"] / v["total"] for k, v in by_type.items()}

    summary = {
        "accuracy": accuracy,
        "total_samples": total,
        "tag_usage": tag_stats,
        "by_dataset_type": type_accuracy,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Tagless evaluation complete:")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  Tag usage: {tag_stats}")
    logger.info(f"  By type: {type_accuracy}")
    logger.info(f"  Results: {output_path}")
    logger.info(f"  Summary: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Tagless evaluation for v4")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to v4 model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/grpo_train_merged.jsonl",
        help="Path to merged dataset",
    )
    parser.add_argument(
        "--output",
        default="evals/tagless_v4_results.jsonl",
        help="Output path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(args.model, args.dataset, args.output, args.samples, args.seed)


if __name__ == "__main__":
    main()
