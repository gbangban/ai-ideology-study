#!/usr/bin/env python3
"""
Merge GRPO LoRA adapter into SFT base model.

Strategy: Load SFT base in BF16 on CPU (not GPU), load GRPO LoRA on CPU,
merge LoRA weights into base weights on CPU, save as BF16.
This avoids all VRAM pressure since the model never touches GPU.

Usage:
    python3 merge_grpo_checkpoint.py [options]

Examples:
    python3 merge_grpo_checkpoint.py \\
        --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \\
        --grpo-checkpoint /studio/exports/grpo_checkpoints/checkpoint-250 \\
        --output /studio/exports/grpo_merged/checkpoint-250
"""

import argparse
import json
import logging
import sys
import gc
import time
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(directory: str) -> str:
    """Find the latest checkpoint-N directory in a directory."""
    base = Path(directory)
    if not base.exists():
        logger.error(f"Directory does not exist: {directory}")
        return ""

    checkpoints = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, str(d)))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return ""

    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoints[-1]
    logger.info(f"Found latest checkpoint: step {latest_step} at {latest_path}")
    return latest_path


def strip_vision_config(model_path: str):
    """Remove vision_config from config.json."""
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
        logger.info(f"Stripped vision config from {config_path}")


def merge_grpo_checkpoint(
    base_model_path: str,
    grpo_lora_path: str,
    output_path: str,
    max_shard_size: str = "5GB",
) -> str:
    """
    Merge GRPO LoRA adapter into SFT base model using CPU-only loading.

    Loads everything on CPU to avoid VRAM issues with the 32GB GPU.
    The merge happens in CPU memory, then saves to disk.

    Args:
        base_model_path: Path to SFT merged checkpoint.
        grpo_lora_path: Path to GRPO LoRA adapter directory.
        output_path: Where to save the merged model.
        max_shard_size: Maximum shard size for saved model.

    Returns:
        Path to the merged model directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import PeftModel

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strip vision config
    strip_vision_config(base_model_path)
    strip_vision_config(grpo_lora_path)

    # Parse max_shard_size
    if max_shard_size.endswith("GB"):
        shard_bytes = int(max_shard_size.replace("GB", "")) * 1024 ** 3
    elif max_shard_size.endswith("MB"):
        shard_bytes = int(max_shard_size.replace("MB", "")) * 1024 ** 2
    else:
        shard_bytes = int(max_shard_size)

    # Step 1: Load base model on CPU in BF16
    logger.info(f"[1/4] Loading SFT base model (BF16, CPU) from {base_model_path}...")
    t0 = time.time()

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info(f"  Loaded in {time.time() - t0:.1f}s")

    # Step 2: Load GRPO LoRA adapter on CPU
    logger.info(f"[2/4] Loading GRPO LoRA adapter from {grpo_lora_path}...")
    t1 = time.time()

    model = PeftModel.from_pretrained(
        model,
        grpo_lora_path,
        is_trainable=False,
    )
    logger.info(f"  Loaded in {time.time() - t1:.1f}s")

    # Step 3: Merge LoRA weights into base
    logger.info("[3/4] Merging LoRA weights into base model...")
    t2 = time.time()

    model = model.merge_and_unload()
    logger.info(f"  Merged in {time.time() - t2:.1f}s")

    del t0, t1, t2
    gc.collect()

    # Step 4: Save merged model
    logger.info(f"[4/4] Saving merged BF16 model to {output_path}...")
    t3 = time.time()

    model.save_pretrained(
        output_path,
        max_shard_size=shard_bytes,
        safe_serialization=True,
    )
    logger.info(f"  Model saved in {time.time() - t3:.1f}s")

    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    del model, tokenizer
    gc.collect()

    # Save metadata
    metadata = {
        "base_model": base_model_path,
        "grpo_lora": grpo_lora_path,
        "save_dtype": "bfloat16",
        "merge_tool": "merge_grpo_checkpoint.py",
        "merge_method": "cpu-only-merge",
        "merge_time_seconds": round(time.time() - time.time(), 1),
    }
    with open(output_dir / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    total_size = sum(
        f.stat().st_size
        for f in output_dir.rglob("*.safetensors")
        if f.is_file()
    )
    num_shards = len(list(output_dir.rglob("*.safetensors")))
    logger.info(f"  Total size: {total_size / 1e9:.2f} GB ({num_shards} shards)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge GRPO LoRA adapter into SFT base model (CPU-only)"
    )
    parser.add_argument(
        "--base-model",
        default="/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330",
        help="Path to SFT merged checkpoint",
    )
    parser.add_argument(
        "--grpo-checkpoint",
        default=None,
        help="Path to GRPO LoRA checkpoint",
    )
    parser.add_argument(
        "--grpo-dir",
        default="/app/checkpoints/lora_adapters/grpo_adapter",
        help="Directory containing GRPO checkpoints",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum shard size (default: 5GB)",
    )
    args = parser.parse_args()

    # Resolve GRPO checkpoint path
    grpo_path = args.grpo_checkpoint
    if grpo_path is None:
        grpo_path = find_latest_checkpoint(args.grpo_dir)
        if not grpo_path:
            logger.error(f"No checkpoints found in {args.grpo_dir}")
            sys.exit(1)

    if not Path(grpo_path).exists():
        logger.error(f"GRPO checkpoint not found: {grpo_path}")
        sys.exit(1)

    # Resolve output path
    if args.output is None:
        ckpt_name = Path(grpo_path).name
        output_path = f"/app/checkpoints/merged_grpo/{ckpt_name}"
    else:
        output_path = args.output

    logger.info("=" * 60)
    logger.info("GRPO LoRA Merge (CPU-only)")
    logger.info("=" * 60)
    logger.info(f"Base model:     {args.base_model}")
    logger.info(f"GRPO LoRA:      {grpo_path}")
    logger.info(f"Output:         {output_path}")
    logger.info("=" * 60)

    merge_grpo_checkpoint(
        base_model_path=args.base_model,
        grpo_lora_path=grpo_path,
        output_path=output_path,
        max_shard_size=args.max_shard_size,
    )

    logger.info("=" * 60)
    logger.info("Done. Model ready for evaluation.")
    logger.info(f"Run:")
    logger.info(f"  FINETUNED_MODEL_DIR={output_path}")
    logger.info(f"  ./evals/scripts/run_finetuned_bf16.sh")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
