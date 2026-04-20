"""
Teacher Phase - Synthetic Data Generation

Generates DM-aligned synthetic training samples using llama.cpp
with a GGUF quantized Qwen model.
"""

import json
from pathlib import Path
from typing import List, Optional

from src.teacher.prompts import generate_dm_prompt
from src.teacher.validators import (
    validate_dm_response,
    generate_with_retry,
    is_valid_dm_sample,
)
from src.teacher.sample_utils import create_sample, format_as_jsonl


def _get_llama():
    """Lazy import of llama_cpp to allow testing without the dependency."""
    from llama_cpp import Llama

    return Llama


def generate_single_sample(
    llm,
    question: str,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict:
    """
    Generate a single DM-aligned sample with retry logic.

    Args:
        llm: Loaded llama.cpp model
        question: The question to generate a response for
        max_retries: Maximum retry attempts for invalid responses
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        dict: Validated ShareGPT-format sample
    """
    prompt = generate_dm_prompt(question)

    def generate_response():
        response = llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["Question:", "User:", "\n\n"],
        )
        return response["choices"][0]["text"]

    answer = generate_with_retry(generate_response, max_retries=max_retries)

    return create_sample(question, answer)


def generate_batch(
    llm,
    questions: List[str],
    batch_size: int = 50,
    max_retries: int = 3,
    temperature: float = 0.7,
    checkpoint_path: Optional[str] = None,
) -> List[dict]:
    """
    Generate samples in batches with checkpointing support.

    Args:
        llm: Loaded llama.cpp model
        questions: List of questions to generate responses for
        batch_size: Number of samples to process before checkpoint
        max_retries: Maximum retry attempts per sample
        temperature: Sampling temperature
        checkpoint_path: Optional path to save/load checkpoint

    Returns:
        List[dict]: List of generated samples
    """
    samples = []
    start_idx = 0

    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            samples = checkpoint["samples"]
            start_idx = checkpoint["completed_count"]
            print(f"Resumed from checkpoint: {start_idx}/{len(questions)} samples")

    for i in range(start_idx, len(questions)):
        question = questions[i]
        print(f"Generating sample {i + 1}/{len(questions)}: {question[:50]}...")

        sample = generate_single_sample(
            llm=llm,
            question=question,
            max_retries=max_retries,
            temperature=temperature,
        )

        samples.append(sample)

        if (i + 1) % batch_size == 0 and checkpoint_path:
            save_checkpoint(samples, i + 1, checkpoint_path)
            print(f"Checkpoint saved: {i + 1}/{len(questions)} samples")

    return samples


def save_checkpoint(samples: List[dict], completed_count: int, checkpoint_path: str):
    """
    Save generation checkpoint for resuming interrupted runs.

    Args:
        samples: Generated samples so far
        completed_count: Number of questions processed
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        "samples": samples,
        "completed_count": completed_count,
    }

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


def load_questions(filepath: str) -> List[str]:
    """
    Load questions from a text file (one per line).

    Args:
        filepath: Path to questions file

    Returns:
        List[str]: List of questions
    """
    with open(filepath, "r") as f:
        questions = [line.strip() for line in f if line.strip()]

    return questions


def save_samples(samples: List[dict], output_path: str):
    """
    Save samples to JSONL file.

    Args:
        samples: List of generated samples
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(format_as_jsonl(samples))

    print(f"Saved {len(samples)} samples to {output_path}")


def main(
    model_path: str = "checkpoints/base_model/Qwen3.5-27B-Instruct-Q4_K_M.gguf",
    questions_path: str = "data/raw/questions.txt",
    output_path: str = "data/processed/sft_dataset.jsonl",
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    temperature: float = 0.7,
    max_retries: int = 3,
    batch_size: int = 50,
):
    """
    Main entry point for teacher phase generation.

    Args:
        model_path: Path to GGUF model
        questions_path: Path to input questions file
        output_path: Path for output JSONL file
        n_gpu_layers: Number of GPU layers (-1 for all)
        n_ctx: Context size
        temperature: Sampling temperature
        max_retries: Maximum retries per sample
        batch_size: Checkpoint batch size
    """
    print(f"Loading model from {model_path}...")
    Llama = _get_llama()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=True,
    )

    print(f"Loading questions from {questions_path}...")
    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    checkpoint_path = "data/processed/checkpoint.json"

    print("Generating DM-aligned samples...")
    samples = generate_batch(
        llm=llm,
        questions=questions,
        batch_size=batch_size,
        max_retries=max_retries,
        temperature=temperature,
        checkpoint_path=checkpoint_path,
    )

    save_samples(samples, output_path)

    valid_count = sum(1 for s in samples if is_valid_dm_sample(s))
    print(f"Validation: {valid_count}/{len(samples)} samples are DM-aligned")


if __name__ == "__main__":
    main()
