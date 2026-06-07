#!/usr/bin/env python3
"""Generate cold-start SFT data with RLVMR tags using the base model.

Samples from the merged dataset and generates tagged demonstrations.
The base model (Qwen3.5-9B) is prompted to produce <planning>, <commitment>,
<reflection>, and <monitor> tags in its output.

Uses the 27B teacher model for high-quality demonstrations (not the 9B student).
Supports GGUF (default) or Unsloth 4-bit backends to fit on RTX 5090 32GB.

Memory estimates:
  Qwen3.6-27B Q4_K_XL (default teacher): ~17GB weights + KV cache ~3GB = ~20GB
  Qwen3.5-9B   Q4_K_M   (--use-9b):      ~5.5GB weights + KV cache ~1GB = ~7GB
  Qwen3.5-9B   NF4      (--model):        ~4.7GB weights + compute ~3.5GB = ~8GB

Usage:
    # Default: 27B teacher GGUF (~20GB VRAM, auto-downloads if not cached):
    python3 -m src.teacher.generate_cold_start_data \
        --dataset data/processed/grpo_train_merged.jsonl \
        --output data/processed/cold_start_sft.jsonl \
        --samples 200

    # Smaller 9B fallback (~7GB VRAM):
    python3 -m src.teacher.generate_cold_start_data \
        --use-9b \
        --dataset data/processed/grpo_train_merged.jsonl \
        --output data/processed/cold_start_sft.jsonl \
        --samples 200

    # Unsloth 4-bit backend (~8GB VRAM):
    python3 -m src.teacher.generate_cold_start_data \
        --model Qwen/Qwen3.5-9B \
        --dataset data/processed/grpo_train_merged.jsonl \
        --output data/processed/cold_start_sft.jsonl \
        --samples 200
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("cold-start-data")

# Default GGUF: teacher model (27B) for high-quality cold-start demonstrations.
#
# NOTE: The experimental design specifies Unsloth/Qwen3.5-27B as the teacher.
# However, unsloth/Qwen3.5-27B-GGUF does not exist on HuggingFace.
# We use unsloth/Qwen3.6-27B-GGUF-MTP as the closest available 27B GGUF teacher.
# This is a known deviation from the experimental design — Qwen3.6 is a different
# model family than Qwen3.5. If a Qwen3.5-27B GGUF becomes available, switch back.
# Falls back to 9B if --use-9b flag is passed or VRAM is insufficient.
DEFAULT_GGUF_REPO = "unsloth/Qwen3.6-27B-MTP-GGUF"
DEFAULT_GGUF_FILE = "Qwen3.6-27B-UD-Q4_K_XL.gguf"

# Smaller fallback for constrained VRAM
FALLBACK_GGUF_REPO = "unsloth/Qwen3.5-9B-GGUF"
FALLBACK_GGUF_FILE = "Qwen3.5-9B-Q4_K_M.gguf"


def resolve_gguf_path(gguf_path_str, use_9b=False):
    """Resolve a GGUF path, downloading via huggingface-cli if needed.

    Accepts:
      - Absolute path to .gguf file
      - HF repo ID (e.g. unsloth/Qwen3.6-27B-GGUF-MTP) — downloads default file
      - None — uses DEFAULT_GGUF_REPO + DEFAULT_GGUF_FILE (27B teacher)

    If use_9b=True, defaults to the 9B fallback instead.
    """
    if use_9b:
        repo_id = FALLBACK_GGUF_REPO
        filename = FALLBACK_GGUF_FILE
    elif not gguf_path_str:
        repo_id = DEFAULT_GGUF_REPO
        filename = DEFAULT_GGUF_FILE
    elif gguf_path_str.endswith(".gguf"):
        p = Path(gguf_path_str)
        if p.exists():
            return str(p)
        logger.error(f"GGUF file not found: {p}")
        sys.exit(1)
    else:
        repo_id = gguf_path_str
        filename = DEFAULT_GGUF_FILE

    hf_cache_dir = os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface"),
    )
    repo_dir_name = f"models--{repo_id.replace('/', '--')}"
    # HF cache stores repos under HF_HOME/hub/ when using huggingface_hub,
    # but directly under HF_HOME when using `hf download --local-dir`.
    # Check both locations.
    cache_base = Path(hf_cache_dir) / "hub" / repo_dir_name / "snapshots"
    if not cache_base.exists():
        cache_base = Path(hf_cache_dir) / repo_dir_name / "snapshots"

    if cache_base.exists():
        found = list(cache_base.rglob(filename))
        if found:
            best = max(found, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found cached GGUF: {best}")
            return str(best)

    logger.info(
        f"GGUF not cached. Downloading {repo_id}/{filename}..."
    )
    hf_cli = shutil.which("hf")
    if not hf_cli:
        logger.error(
            "Cannot download GGUF: 'hf' CLI not found. "
            "Install with 'pip install huggingface_hub' or download manually."
        )
        sys.exit(1)

    snapshot_dir = next(cache_base.iterdir(), None) if cache_base.exists() else None
    if not snapshot_dir:
        snapshot_dir = cache_base / "tmp"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [hf_cli, "download", repo_id, filename, "--local-dir", str(snapshot_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Download failed:\n{result.stderr}")
        sys.exit(1)

    found = [snapshot_dir / filename]
    if found[0].exists():
        logger.info(f"Downloaded GGUF to: {found[0]}")
        return str(found[0])

    logger.error(f"Download succeeded but cannot locate {filename}")
    sys.exit(1)


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


def load_model_gguf(gguf_path, max_seq_length=4096):
    """Load model via llama-cpp-python (GGUF backend). ~7GB VRAM for Q4_K_M."""
    from llama_cpp import Llama

    logger.info(f"Loading GGUF model: {gguf_path}")
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=-1,
        n_ctx=max_seq_length,
        verbose=False,
    )
    return llm, "gguf"


def load_model_unsloth(model_name, max_seq_length=2048):
    """Load model via Unsloth 4-bit NF4. ~8GB VRAM."""
    from unsloth import FastLanguageModel

    logger.info(f"Loading Unsloth 4-bit model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    return (model, tokenizer), "unsloth"


def generate_gguf(llm, docs, output_path):
    """Generate using llama-cpp-python GGUF backend."""
    records = []
    for i, doc in enumerate(docs):
        messages = [
            {"role": "system", "content": COLD_START_SYSTEM},
            {"role": "user", "content": build_tagged_prompt(doc)},
        ]

        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
        completion = result["choices"][0]["message"]["content"].strip()

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
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} samples to {output_path}")
    return records


def generate_unsloth(model, tokenizer, docs, output_path):
    """Generate using Unsloth FastLanguageModel 4-bit backend."""
    import torch

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
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
        default=None,
        help="HuggingFace model name/path for Unsloth 4-bit loading (e.g. Qwen/Qwen3.5-9B). ~8GB VRAM. Mutually exclusive with --gguf-path.",
    )
    parser.add_argument(
        "--gguf-path",
        default=DEFAULT_GGUF_REPO,
        help=f"GGUF model for llama-cpp backend. Accepts absolute .gguf path or HF repo ID. ~20GB VRAM for 27B. Default: {DEFAULT_GGUF_REPO} (auto-downloaded if not cached).",
    )
    parser.add_argument(
        "--use-9b",
        action="store_true",
        help=f"Use the smaller 9B fallback GGUF ({FALLBACK_GGUF_REPO}) instead of the 27B teacher. ~7GB VRAM.",
    )
    args = parser.parse_args()

    if args.model and args.gguf_path != DEFAULT_GGUF_REPO:
        parser.error("Specify only one of --model or --gguf-path, not both")
    use_unsloth = args.model is not None

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

    if use_unsloth:
        model, tokenizer = load_model_unsloth(args.model)
        generate_unsloth(model, tokenizer, sampled, args.output)
    else:
        gguf_path = resolve_gguf_path(args.gguf_path, use_9b=args.use_9b)
        llm, _ = load_model_gguf(gguf_path)
        generate_gguf(llm, sampled, args.output)


if __name__ == "__main__":
    main()
