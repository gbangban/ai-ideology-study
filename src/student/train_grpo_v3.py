#!/usr/bin/env python3
"""GRPO v3: Outcome rewards on causal dataset.

Same reward structure as v2 (directional_assertion, dm_alignment, mechanism_commitment)
but trained on the synthetic causal dataset instead of SFT questions.

Usage:
    python3 -m src.student.train_grpo_v3
    python3 -m src.student.train_grpo_v3 --max-steps 500 --resume-step 250
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("grpo-v3")

from src.student.grpo_config import GRPO_CONFIG
from src.student.train_grpo import find_latest_checkpoint, train


def main():

    parser = argparse.ArgumentParser(description="GRPO v3: Outcome rewards on causal dataset")
    parser.add_argument(
        "--base-model",
        default=GRPO_CONFIG["base_model"],
        help="Path to SFT merged checkpoint or GRPO checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/lora_adapters/grpo_adapter_v3",
        help="Output directory for GRPO v3 adapter",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/processed/grpo_causal_dataset.jsonl",
        help="Path to causal dataset JSONL",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=GRPO_CONFIG["max_steps"],
        help="Maximum training steps",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="Resume from checkpoint at this step",
    )
    parser.add_argument(
        "--find-checkpoint",
        action="store_true",
        help="List available checkpoints and exit",
    )
    args = parser.parse_args()

    config = GRPO_CONFIG.copy()
    config["questions_path"] = args.dataset_path
    config["max_steps"] = args.max_steps

    # Find checkpoint mode
    if args.find_checkpoint:
        step, path = find_latest_checkpoint(args.output_dir)
        if step > 0:
            print(f"Latest checkpoint: step {step} at {path}")
            for d in sorted(Path(args.output_dir).iterdir()):
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    print(f"  {d.name}")
        else:
            print(f"No checkpoints found in {args.output_dir}")
        return

    # Auto-resume
    resume_step = args.resume_step
    base_model = args.base_model
    if resume_step > 0:
        ckpt_path = f"{args.output_dir}/checkpoint-{resume_step}"
        if Path(ckpt_path).exists():
            base_model = ckpt_path
            logger.info(f"Auto-resuming from {ckpt_path}")
        else:
            logger.warning(f"Checkpoint {ckpt_path} not found, using base-model as-is")

    try:
        train(config, base_model, args.output_dir, resume_step=resume_step)
    except Exception:
        logger.error("Training failed, flushing VRAM...")
        import torch
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
