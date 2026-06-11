#!/usr/bin/env python3
"""GRPO Training Smoke Test — validates the full stack.

Full mode: runs one training step with real model, real rewards, real GRPOTrainer.
Dry-run mode: runs TrackingManager lifecycle without models or GPU.

Usage:
    # Full smoke test (requires GPU, loads model, runs 1 training step):
    docker exec ml-training python3 -m src.student.smoke_test --track outcome

    # Dry-run smoke test (no GPU, no models, validates tracking pipeline only):
    docker exec ml-training python3 -m src.student.smoke_test --track outcome --dry-run
    docker exec ml-training python3 -m src.student.smoke_test --track process --dry-run
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

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


def dry_run_smoke_test(
    track: str,
    output_dir: str,
) -> None:
    """Run TrackingManager lifecycle dry-run via the training script's --dry-run path.

    Delegates to train_grpo_outcome.dry_run_tracking or train_grpo_process.dry_run_tracking.
    No models loaded, no GPU used. Validates the full tracking pipeline end-to-end.
    """
    track = _validate_track(track)

    if track == "outcome":
        from src.student.train_grpo_outcome import dry_run_tracking
    else:
        from src.student.train_grpo_process import dry_run_tracking

    print(f"=== Dry-Run Smoke Test: {track} ===")
    print(f"Output dir: {output_dir}")
    print(f"Track: {track} (v{'3' if track == 'outcome' else '4'})")
    print()

    dry_run_tracking(output_dir)

    print()
    print("[PASS] Dry-run smoke test completed successfully")


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

    # Step 6: Initialize TrackingManager before building reward functions
    from src.student.train_grpo_base import TrackingManager

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
        "smoke_test": True,
    }
    if track == "process":
        track_config["alpha"] = default_config.get("alpha", 0.5)
        track_config["lambda_kl"] = default_config.get("lambda_kl", 0.01)
        track_config["clip_epsilon"] = default_config.get("clip_epsilon", 0.2)

    tracker = TrackingManager()
    tracker.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=os.environ.get("TRACKIO_RUN_NAME", f"smoke-test-grpo-{track}"),
        config=track_config,
        track=track,
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
    )
    tracking_active = tracker._active
    if tracking_active:
        print("[PASS] TrackingManager initialized")
    else:
        print("[WARN] TrackingManager init failed, tracking will be no-op")

    # Step 7: Build reward functions (wrapped for tracking)
    doc_index = {row["prompt"]: row["doc"] for row in dataset}

    def _get_smoke_reward_specs() -> list:
        def _combined_process_reward(completion: str, doc: Dict[str, Any]) -> float:
            outcome = compute_outcome_reward(doc, completion)
            process = compute_process_rewards(
                completion,
                outcome,
                required_tags=RLVMR_REQUIRED_TAGS,
                penalty_per_tag=REWARD_WEIGHTS["lambda_format"],
            )
            return sum(process.values())

        outcome_spec = (
            "outcome",
            lambda completions, docs: [
                compute_outcome_reward(doc, c) for c, doc in zip(completions, docs)
            ],
        )
        if track == "outcome":
            return [outcome_spec]
        process_spec = (
            "process",
            lambda completions, docs: [
                _combined_process_reward(c, doc) for c, doc in zip(completions, docs)
            ],
        )
        return [outcome_spec, process_spec]

    reward_funcs = tracker.build_reward_functions(_get_smoke_reward_specs(), doc_index)
    reward_count = len(reward_funcs)

    print(f"[PASS] Reward functions created ({reward_count} reward fn)")

    # Step 8: Create config
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

    # Step 9: Instantiate trainer with TrackingCallback
    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )
    if tracking_active:
        tracker.attach_to_trainer(trainer)
    print("[PASS] GRPOTrainer initialized")

    # Step 10: Run one step
    result = trainer.train()
    metrics = result.metrics

    # Step 11: Validate and log final metrics
    loss = metrics.get("train_loss", None)
    loss_str = f"{loss:.4f}" if loss is not None else "N/A"

    if tracking_active:
        # Log all metrics as scalars
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
            tracker.log_rewards(1, track_metrics)
            print("[PASS] Final metrics logged to trackio")
        except Exception as e:
            print(f"[WARN] Failed to log final metrics to trackio: {e}")

        # Log completion sample trace from the last reward data
        try:
            if tracker._reward_table_rows:
                last_row = tracker._reward_table_rows[-1]
                tracker.log_completion_sample(
                    1, last_row.get("prompt", ""), last_row.get("completion", "")
                )
                print("[PASS] Completion sample logged")
            else:
                print("[WARN] No reward table rows for completion sample")
        except Exception as e:
            print(f"[WARN] Failed to log completion sample: {e}")

        # Generate summary report
        try:
            tracker.generate_report_from_trainer(trainer)
            print("[PASS] Summary report generated")
        except Exception as e:
            print(f"[WARN] Failed to generate report: {e}")

    if loss is not None and not math.isfinite(loss):
        print(f"[FAIL] Loss is not finite: {loss}")
        tracker.finish()
        sys.exit(1)

    print(f"[PASS] Training step completed")
    print(f"  Loss: {loss_str}")

    tracker.finish()

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
    parser.add_argument(
        "--output-dir",
        default="checkpoints/lora_adapters/grpo_v3_outcome",
        help="Output directory (used for run naming in dry-run mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run TrackingManager lifecycle without models or GPU (validates tracking pipeline only)",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run_smoke_test(
            track=args.track,
            output_dir=args.output_dir,
        )
        return

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
