#!/usr/bin/env python3
"""Generate qualitative responses from 4 SFT model variants on eval questions.

Runs in evals/.venv/ outside container. Each model is loaded once, generates
all 21 responses, then unloaded before the next model.

Usage:
    cd evals && source .venv/bin/activate
    python3 scripts/generate_eval_responses.py
    python3 scripts/generate_eval_responses.py --model dm  # single model only
    python3 scripts/generate_eval_responses.py --questions 1,2,3  # specific questions
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model definitions: (label, display_name, path)
# All paths are local — no network access required.
MODELS = [
    (
        "baseline",
        "Baseline (Qwen3.5-9B)",
        "/mnt/c/Users/Guy/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ),
    (
        "dm",
        "DM SFT",
        "/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330",
    ),
    (
        "liberal",
        "Liberal SFT",
        "/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781648666/liberal-checkpoint-330",
    ),
    (
        "libertarian",
        "Libertarian SFT",
        "/mnt/c/Users/Guy/.unsloth/studio/exports/Qwen_Qwen3.5-9B_1781703763/libertarian-checkpoint-330",
    ),
]

QUESTIONS_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "eval_questions.json"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "eval_questions_responses.json"

GENERATION_CONFIG = dict(
    max_new_tokens=1024,
    do_sample=False,
    temperature=1.0,
    pad_token_id=None,
)


def load_questions(question_ids=None):
    """Load eval questions, optionally filtering by question ids."""
    with open(QUESTIONS_PATH) as f:
        all_questions = json.load(f)
    if question_ids:
        all_questions = [q for q in all_questions if q["id"] in question_ids]
    return all_questions


def build_prompt(question_text):
    """Build a simple user prompt for open-ended generation.

    Uses a direct question format without system prompt framing.
    """
    return f"Question: {question_text}\n\nAnswer:"


def generate_for_model(model_label, model_name, model_path, questions, results, dry_run=False):
    """Load a model, generate responses for all questions, then unload.

    If dry_run, skips model loading/generation and writes placeholder responses.
    """
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} ({model_path})")
    if dry_run:
        print("[DRY RUN] skipping model load & generation")
    print(f"{'=' * 60}")

    if dry_run:
        for i, q in enumerate(questions, 1):
            qid = str(q["id"])
            if model_label not in results[qid]["responses"]:
                results[qid]["responses"][model_label] = "[dry-run placeholder]"
            print(f"  [{i}/{len(questions)}] Q{qid}: placeholder written")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded. Generating {len(questions)} responses...")

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        qtext = q["question"]
        prompt = build_prompt(qtext)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG,
            )
        elapsed = time.time() - start

        generated = outputs[0][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        response = response.strip()

        results[str(qid)]["responses"][model_label] = response
        print(f"  [{i}/{len(questions)}] Q{qid}: {len(response)} chars, {elapsed:.1f}s")

    # Unload model
    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"Model unloaded. VRAM freed.")


def main():
    parser = argparse.ArgumentParser(description="Generate eval question responses")
    parser.add_argument("--model", choices=[m[0] for m in MODELS], help="Run single model only")
    parser.add_argument("--questions", help="Comma-separated question ids to run")
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths, write placeholders, skip GPU")
    args = parser.parse_args()

    question_ids = None
    if args.questions:
        question_ids = [int(x.strip()) for x in args.questions.split(",")]

    questions = load_questions(question_ids)
    print(f"Loaded {len(questions)} questions")

    if args.model:
        models = [m for m in MODELS if m[0] == args.model]
    else:
        models = MODELS

    output_path = Path(args.output) if args.output else OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to support resuming
    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Loaded existing results from {output_path}")

    # Initialize missing entries
    for q in questions:
        qid = str(q["id"])
        if qid not in results:
            results[qid] = {
                "question": q["question"],
                "type": q.get("type_label", q.get("type", "Unknown")),
                "responses": {},
            }

    # Pre-flight: verify all model paths exist
    for model_label, model_name, model_path in models:
        mp = Path(model_path)
        if not mp.exists():
            print(f"ERROR: Model path not found: {model_path}")
            sys.exit(1)

    for model_label, model_name, model_path in models:
        generate_for_model(model_label, model_name, model_path, questions, results, dry_run=args.dry_run)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
