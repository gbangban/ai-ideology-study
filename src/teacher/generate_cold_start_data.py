#!/usr/bin/env python3
"""Generate cold-start SFT data with RLVMR tags using the base model.

Samples from the merged dataset and generates tagged demonstrations.
The base model (Qwen3.5-9B) is prompted to produce <planning>, <commitment>,
<reflection>, and <monitor> tags in its output.

Usage:
    python3 -m src.teacher.generate_cold_start_data \
        --dataset data/processed/grpo_train_merged.jsonl \
        --output data/processed/cold_start_sft.jsonl \
        --samples 500
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("cold-start-data")

COLD_START_SYSTEM = """You are an analytical assistant. For each question, structure your response using these tags:

<planning>
First, identify the key variables, treatment, outcome, and context. Briefly outline your analytical approach.
</planning>

<commitment>
State your definitive answer: positive (+), negative (-), null (None), or mixed.
</commitment>

<reflection>
Consider potential weaknesses or alternative interpretations in your analysis.
</reflection>

<monitor>
Note any contextual constraints, assumptions, or limitations in your reasoning.
</monitor>

After the tags, provide a brief synthesis of your reasoning."""


def build_tagged_prompt(doc) -> str:
    """Build a user prompt that encourages tagged output."""
    prompt = doc["prompt"]
    return f"{prompt}\n\nStructure your response using <planning>, <commitment>, <reflection>, and <monitor> tags as described in the system prompt."


def generate_with_base_model(model, tokenizer, docs, output_path):
    """Generate tagged demonstrations using the base model."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    records = []
    for i, doc in enumerate(docs):
        messages = [
            {"role": "system", "content": COLD_START_SYSTEM},
            {"role": "user", "content": build_tagged_prompt(doc)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_len:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)

        record = {
            "messages": messages + [{"role": "assistant", "content": completion}],
            "ground_truth": doc,
        }
        records.append(record)

        if (i + 1) % 25 == 0:
            logger.info(f"Generated {i + 1}/{len(docs)} samples")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Saved {len(records)} samples to {output_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate cold-start SFT data")
    parser.add_argument(
        "--dataset",
        default="data/processed/grpo_train_merged.jsonl",
        help="Path to merged training dataset",
    )
    parser.add_argument(
        "--output",
        default="data/processed/cold_start_sft.jsonl",
        help="Output path for SFT data",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Base model for generation",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load dataset
    docs = []
    with open(args.dataset) as f:
        for line in f:
            docs.append(json.loads(line))
    logger.info(f"Loaded {len(docs)} documents")

    # Sample
    sampled = random.sample(docs, min(args.samples, len(docs)))
    logger.info(f"Sampled {len(sampled)} documents for cold-start generation")

    import torch
    generate_with_base_model(args.model, None, sampled, args.output)


if __name__ == "__main__":
    main()
