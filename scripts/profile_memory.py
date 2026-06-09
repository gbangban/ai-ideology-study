#!/usr/bin/env python3
"""Standalone Memory Profiling Script for GRPO Training.

Runs a few training steps with VRAM memory snapshots at each phase,
then prints a summary report and saves it to the output directory.

Unlike smoke_test.py, this script does NOT disable torch.compile.
Use --compile flag to enable compile for the training run.

Usage:
    docker exec ml-training python3 scripts/profile_memory.py --track outcome --steps 3
    docker exec ml-training python3 scripts/profile_memory.py --track process --compile --steps 3
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def profile_memory(
    track: str,
    base_model: str,
    dataset_path: str,
    steps: int,
    use_compile: bool,
    output_dir: str,
) -> None:
    """Run a few training steps with VRAM memory profiling.

    Args:
        track: 'outcome' for v3, 'process' for v4.
        base_model: Path to merged checkpoint.
        dataset_path: Path to JSONL dataset.
        steps: Number of training steps to run.
        use_compile: Whether to enable torch.compile.
        output_dir: Directory to save the memory report.
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
    from src.utils.memory_profiler import (
        MemorySnapshot,
        TrainingMemoryTracker,
        format_vram,
        force_memory_cleanup,
        get_vram_allocated_gb,
        get_vram_peak_gb,
    )

    import torch

    tracker = TrainingMemoryTracker()
    snapshots: List[MemorySnapshot] = []

    # Must patch before loading model / creating trainer
    patch_unsloth_chunked_log_softmax()

    default_config = OUTCOME_DEFAULT_CONFIG if track == "outcome" else PROCESS_DEFAULT_CONFIG

    compile_status = "ENABLED" if use_compile else "disabled"
    print(f"=== GRPO Memory Profiler: {track} ===")
    print(f"Steps: {steps}")
    print(f"Generations per prompt: {default_config['grpo_g']}")
    print(f"torch.compile: {compile_status}")
    print(f"Model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print()

    # Snapshot 0: Initial state
    snapshots.append(MemorySnapshot.capture("initial"))
    tracker.record(0, "initial")
    print(f"[SNAPSHOT] initial — allocated: {format_vram(snapshots[-1].allocated_bytes)}")

    # Step 1: Strip vision config
    strip_vision_config(base_model)

    # Step 2: Load model
    snapshots.append(MemorySnapshot.capture("before_model_load"))
    tracker.record(0, "before_model_load")
    print(f"[SNAPSHOT] before_model_load — allocated: {format_vram(snapshots[-1].allocated_bytes)}")

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

    snapshots.append(MemorySnapshot.capture("after_model_load"))
    tracker.record(0, "after_model_load")
    delta = (snapshots[-1].allocated_bytes - snapshots[-2].allocated_bytes) / (1024 ** 3)
    print(
        f"[SNAPSHOT] after_model_load — allocated: {format_vram(snapshots[-1].allocated_bytes)} "
        f"(+{delta:.2f} GB)"
    )

    # Step 3: Extract tokenizer and apply fix
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    from src.student import fix_mistral_tokenizer
    fix_mistral_tokenizer(tokenizer)

    # Step 4: Apply LoRA
    snapshots.append(MemorySnapshot.capture("before_lora"))
    tracker.record(0, "before_lora")
    print(f"[SNAPSHOT] before_lora — allocated: {format_vram(snapshots[-1].allocated_bytes)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=default_config["lora_rank"],
        lora_alpha=default_config["lora_alpha"],
        lora_dropout=default_config["lora_dropout"],
        target_modules=default_config["target_modules"],
    )
    model = FastLanguageModel.for_training(model)

    snapshots.append(MemorySnapshot.capture("after_lora"))
    tracker.record(0, "after_lora")
    delta = (snapshots[-1].allocated_bytes - snapshots[-2].allocated_bytes) / (1024 ** 3)
    print(
        f"[SNAPSHOT] after_lora — allocated: {format_vram(snapshots[-1].allocated_bytes)} "
        f"(+{delta:.2f} GB)"
    )

    # Step 5: Build dataset
    dataset = build_outcome_dataset(dataset_path, tokenizer)
    num_prompts = min(max(2, steps), len(dataset))
    indices = list(range(num_prompts))
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
            output_dir=output_dir,
            max_steps=steps,
            save_steps=99999,
            logging_steps=1,
        )
    else:
        grpo_config = create_grpo_config_process(
            output_dir=output_dir,
            max_steps=steps,
            save_steps=99999,
            logging_steps=1,
        )

    # Override torch_compile based on --compile flag
    grpo_config.torch_compile = use_compile
    print(f"[PASS] GRPOConfig created (torch_compile={use_compile})")

    # Snapshot before trainer creation
    snapshots.append(MemorySnapshot.capture("before_trainer"))
    tracker.record(0, "before_trainer")
    print(f"[SNAPSHOT] before_trainer — allocated: {format_vram(snapshots[-1].allocated_bytes)}")

    # Step 8: Instantiate trainer
    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
    )

    snapshots.append(MemorySnapshot.capture("after_trainer"))
    tracker.record(0, "after_trainer")
    delta = (snapshots[-1].allocated_bytes - snapshots[-2].allocated_bytes) / (1024 ** 3)
    print(
        f"[SNAPSHOT] after_trainer — allocated: {format_vram(snapshots[-1].allocated_bytes)} "
        f"(+{delta:.2f} GB)"
    )

    # Step 9: Run training
    print(f"\n=== Running {steps} training step(s) ===\n")
    result = trainer.train()
    metrics = result.metrics

    snapshots.append(MemorySnapshot.capture("after_training"))
    tracker.record(steps, "after_training")
    print(
        f"[SNAPSHOT] after_training — allocated: {format_vram(snapshots[-1].allocated_bytes)}"
    )

    # Validate loss
    loss = metrics.get("train_loss", None)
    if loss is not None and not math.isfinite(loss):
        print(f"[FAIL] Loss is not finite: {loss}")
        sys.exit(1)

    loss_str = f"{loss:.4f}" if loss is not None else "N/A"
    print(f"  Loss: {loss_str}")

    # Step 10: Memory cleanup test (delete references first)
    del trainer
    del model
    del dataset
    del reward_funcs
    del outcome_fn
    if track == "process":
        del process_fn
    del _combined_process_reward
    cleanup = force_memory_cleanup()
    snapshots.append(MemorySnapshot.capture("after_cleanup"))
    print(f"\n[CLEANUP] Freed {cleanup['freed_allocated_gb']:.2f} GB allocated, "
          f"{cleanup['freed_reserved_gb']:.2f} GB reserved")
    print(f"[SNAPSHOT] after_cleanup — allocated: {format_vram(snapshots[-1].allocated_bytes)}")

    # Print VRAM summary
    print("\n" + "=" * 60)
    print("VRAM USAGE REPORT")
    print("=" * 60)
    print(f"  Track          : {track}")
    print(f"  torch.compile  : {use_compile}")
    print(f"  Steps          : {steps}")
    print(f"  Prompts        : {num_prompts}")
    print(f"  Generations/prompt : {default_config['grpo_g']}")
    print(f"  LoRA rank/alpha: {default_config['lora_rank']}/{default_config['lora_alpha']}")
    print("-" * 60)
    print("  Phase                    Allocated       Delta")
    print("-" * 60)

    prev_bytes = 0
    for i, snap in enumerate(snapshots):
        delta_gb = (snap.allocated_bytes - prev_bytes) / (1024 ** 3)
        delta_str = f"+{delta_gb:+.2f} GB" if i > 0 else ""
        print(f"  {snap.label:<24s} {format_vram(snap.allocated_bytes):>14s}  {delta_str}")
        prev_bytes = snap.allocated_bytes

    print("-" * 60)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"  PEAK allocated     : {peak_allocated:.2f} GB (PyTorch tensors only)")
    print(f"  PEAK reserved      : {peak_reserved:.2f} GB (includes caching allocator, what nvidia-smi sees)")
    print(f"  Training loss      : {loss_str}")
    print("=" * 60)

    # Save report to output directory with unique run ID
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"{ts}_{run_id}"

    os.makedirs(output_dir, exist_ok=True)
    report_path = Path(output_dir) / f"memory_profile_{track}_{run_label}.json"

    report = {
        "track": track,
        "torch_compile": use_compile,
        "steps": steps,
        "num_prompts": num_prompts,
        "generations_per_prompt": default_config["grpo_g"],
        "lora_rank": default_config["lora_rank"],
        "lora_alpha": default_config["lora_alpha"],
        "training_loss": loss,
        "peak_vram_gb": peak_reserved,
        "peak_allocated_gb": peak_allocated,
        "peak_reserved_gb": peak_reserved,
        "snapshots": [
            {
                "label": s.label,
                "allocated_gb": round(s.allocated_bytes / (1024 ** 3), 2),
                "reserved_gb": round(s.reserved_bytes / (1024 ** 3), 2),
                "peak_gb": round(s.peak_bytes / (1024 ** 3), 2),
            }
            for s in snapshots
        ],
        "cleanup": {
            k: round(v, 2) for k, v in cleanup.items()
        },
        "run_id": run_id,
        "run_label": run_label,
        "tracker_summary": tracker.summary(),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SAVED] Report written to {report_path}")

    # Log to trackio
    try:
        import trackio
        trackio.init(
            project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
            name=f"memory-profile-{track}-{run_label}",
            config={
                "track": track,
                "torch_compile": use_compile,
                "steps": steps,
                "num_prompts": num_prompts,
                "generations_per_prompt": default_config["grpo_g"],
            },
            server_url=os.environ.get("TRACKIO_SERVER_URL"),
            auto_log_gpu=True,
        )
        trackio.log({
            "peak_allocated_gb": peak_allocated,
            "peak_reserved_gb": peak_reserved,
            "training_loss": loss,
            "steps": steps,
            **{f"snapshot.{s['label']}.allocated_gb": s["allocated_gb"] for s in report["snapshots"]},
            **{f"snapshot.{s['label']}.reserved_gb": s["reserved_gb"] for s in report["snapshots"]},
        })
        trackio.finish()
        print("[PASS] Metrics logged to trackio")
    except Exception as e:
        print(f"[WARN] Failed to log to trackio: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO Memory Profiler — runs training steps with VRAM snapshots"
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
        "--steps",
        type=int,
        default=3,
        help="Number of training steps to run (default: 3)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the training run",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/memory_profile_report",
        help="Directory to save the memory report (default: /tmp/memory_profile_report)",
    )
    args = parser.parse_args()

    try:
        profile_memory(
            track=args.track,
            base_model=args.base_model,
            dataset_path=args.dataset_path,
            steps=args.steps,
            use_compile=args.compile,
            output_dir=args.output_dir,
        )
    except Exception:
        logger.error("Memory profiler failed, flushing VRAM...")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
