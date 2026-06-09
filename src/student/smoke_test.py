#!/usr/bin/env python3
"""GRPO Training Smoke Test — runs one training step to validate the full stack.

Executes inside the Docker container with real model, real rewards, real GRPOTrainer.
Subsamples to a small number of prompts for fast execution (~30-60s).

Usage:
    docker exec ml-training python3 -m src.student.smoke_test --track outcome
    docker exec ml-training python3 -m src.student.smoke_test --track process
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

# torch.compile is controlled via --compile flag (disabled by default).
# When enabled, overrides grpo_config.torch_compile = True.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _validate_track(track: str) -> str:
    """Validate that track is one of the supported values."""
    if track not in ("outcome", "process"):
        raise ValueError(f"track must be 'outcome' or 'process', got '{track}'")
    return track


def smoke_test(
    track: str,
    base_model: str,
    dataset_path: str,
    num_prompts: int = 2,
    enable_compile: bool = False,
) -> None:
    """Run a single training step to validate the full stack.

    Args:
        track: 'outcome' for v3, 'process' for v4.
        base_model: Path to merged checkpoint.
        dataset_path: Path to JSONL dataset.
        num_prompts: Number of prompts to subsample (default 2).
    """
    track = _validate_track(track)

    # CRITICAL: Unsloth must be imported before trl/transformers/peft for optimizations
    try:
        import unsloth  # noqa: F401
    except ImportError:
        pass

    from src.student.grpo_config_outcome import (
        DEFAULT_CONFIG as OUTCOME_DEFAULT_CONFIG,
        create_grpo_config as create_grpo_config_outcome,
    )
    from src.student.grpo_config_process import (
        DEFAULT_CONFIG as PROCESS_DEFAULT_CONFIG,
        REWARD_WEIGHTS,
        create_grpo_config as create_grpo_config_process,
    )
    from src.student.reward_outcome import compute_outcome_reward
    from src.student.reward_process import (
        RLVMR_REQUIRED_TAGS,
        compute_process_rewards,
    )
    from src.student.train_grpo_base import (
        build_outcome_dataset,
        build_reward_fn_with_docs,
        patch_unsloth_chunked_log_softmax,
        strip_vision_config,
    )

    # Must patch before loading model / creating trainer
    patch_unsloth_chunked_log_softmax()

    default_config = OUTCOME_DEFAULT_CONFIG if track == "outcome" else PROCESS_DEFAULT_CONFIG

    print(f"=== GRPO Smoke Test: {track} ===")
    print(f"Prompts: {num_prompts}")
    print(f"Generations per prompt: {default_config['grpo_g']}")
    print(f"Model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print()

    # Step 1: Strip vision config
    strip_vision_config(base_model)

    # Step 2: Load model
    from unsloth import FastLanguageModel

    logger.info(f"Loading model from {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
        gpu_memory_utilization=0.95,
    )

    try:
        import torch
        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:
        vram_gb = -1

    print(f"[PASS] Model loaded (VRAM: {vram_gb:.1f} GB)")

    # Step 3: Extract tokenizer and apply fix
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    from src.student import fix_mistral_tokenizer
    fix_mistral_tokenizer(tokenizer)

    # Step 4: Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=default_config["lora_rank"],
        lora_alpha=default_config["lora_alpha"],
        lora_dropout=default_config["lora_dropout"],
        target_modules=default_config["target_modules"],
    )
    model = FastLanguageModel.for_training(model)
    print(
        f"[PASS] LoRA applied (rank={default_config['lora_rank']}, "
        f"alpha={default_config['lora_alpha']})"
    )

    # Step 5: Build dataset
    dataset = build_outcome_dataset(dataset_path, tokenizer)
    indices = list(range(min(num_prompts, len(dataset))))
    dataset = dataset.select(indices)
    print(f"[PASS] Dataset built ({len(dataset)} prompts)")

    # Step 6: Build reward functions
    doc_index = {row["prompt"]: row["doc"] for row in dataset}

    def _combined_process_reward(completion: str, doc: Dict[str, Any]) -> float:
        outcome = compute_outcome_reward(doc, completion)
        process = compute_process_rewards(
            completion,
            outcome,
            required_tags=RLVMR_REQUIRED_TAGS,
            penalty_per_tag=REWARD_WEIGHTS["lambda_format"],
        )
        return sum(process.values())

    if track == "outcome":
        outcome_fn = build_reward_fn_with_docs(
            lambda completions, docs: [
                compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)
            ],
            doc_index,
        )
        reward_funcs = [outcome_fn]
        reward_count = 1
    else:
        outcome_fn = build_reward_fn_with_docs(
            lambda completions, docs: [
                compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)
            ],
            doc_index,
        )
        process_fn = build_reward_fn_with_docs(
            lambda completions, docs: [
                _combined_process_reward(c, doc)
                for c, doc in zip(completions, docs)
            ],
            doc_index,
        )
        reward_funcs = [outcome_fn, process_fn]
        reward_count = 2

    print(f"[PASS] Reward functions created ({reward_count} reward fn)")

    # Step 7: Create config
    if track == "outcome":
        grpo_config = create_grpo_config_outcome(
            output_dir="/tmp/smoke_test_grpo_outcome",
            max_steps=1,
            save_steps=99999,
            logging_steps=1,
            torch_compile=enable_compile,
        )
    else:
        grpo_config = create_grpo_config_process(
            output_dir="/tmp/smoke_test_grpo_process",
            max_steps=1,
            save_steps=99999,
            logging_steps=1,
            torch_compile=enable_compile,
        )

    # Step 8: Instantiate trainer
    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )
    print("[PASS] GRPOTrainer initialized")

    # Step 8.5: Initialize trackio for experiment tracking
    import trackio

    track_config = {
        "training_method": "GRPO" if track == "outcome" else "GRPO-DualAdvantage",
        "track": track,
        "version": "v3" if track == "outcome" else "v4",
        "num_prompts": num_prompts,
        "group_size": default_config["grpo_g"],
        "beta": default_config["beta"],
        "learning_rate": default_config["learning_rate"],
        "lora_rank": default_config["lora_rank"],
        "lora_alpha": default_config["lora_alpha"],
        "max_completion_length": default_config["max_completion_length"],
        "reward_count": reward_count,
        "smoke_test": True,
    }
    if track == "process":
        track_config["alpha"] = default_config.get("alpha", 0.5)
        track_config["lambda_kl"] = default_config.get("lambda_kl", 0.01)
        track_config["clip_epsilon"] = default_config.get("clip_epsilon", 0.2)

    run_name = os.environ.get("TRACKIO_RUN_NAME", f"smoke-test-grpo-{track}")
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=run_name,
        config=track_config,
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
        auto_log_gpu=True,
    )
    trackio_initialized = True

    # Step 9: Run one step
    result = trainer.train()
    metrics = result.metrics

    # Step 10: Validate
    loss = metrics.get("train_loss", None)
    loss_str = f"{loss:.4f}" if loss is not None else "N/A"

    # Log metrics to trackio
    track_metrics = {
        "loss": loss if loss is not None else float("nan"),
        "reward_count": reward_count,
        "num_prompts": num_prompts,
        "track": track,
    }
    for key, val in metrics.items():
        if isinstance(val, (int, float)) and key not in track_metrics:
            track_metrics[key] = val
    try:
        trackio.log(track_metrics)
        print("[PASS] Metrics logged to trackio")
    except Exception as e:
        print(f"[WARN] Failed to log to trackio: {e}")

    if loss is not None and not math.isfinite(loss):
        print(f"[FAIL] Loss is not finite: {loss}")
        if trackio_initialized:
            trackio.finish()
        sys.exit(1)

    print(f"[PASS] Training step completed")
    print(f"  Loss: {loss_str}")

    if trackio_initialized:
        trackio.finish()

    print("[PASS] All validations passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO Training Smoke Test (one step, full stack)"
    )
    parser.add_argument(
        "--track",
        required=True,
        choices=["outcome", "process"],
        help="Training track: 'outcome' (v3) or 'process' (v4)",
    )
    parser.add_argument(
        "--base-model",
        default="checkpoints/merged/cold_start_merged",
        help="Path to SFT merged checkpoint",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/processed/grpo_train_merged.jsonl",
        help="Path to merged training dataset JSONL",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=2,
        help="Number of prompts to subsample (default: 2)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (disabled by default)",
    )
    args = parser.parse_args()

    try:
        smoke_test(
            track=args.track,
            base_model=args.base_model,
            dataset_path=args.dataset_path,
            num_prompts=args.num_prompts,
            enable_compile=args.compile,
        )
    except Exception:
        logger.error("Smoke test failed, flushing VRAM...")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
