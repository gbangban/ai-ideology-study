#!/usr/bin/env python3
"""Shared training utilities for checkpointing and resumption."""

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str) -> tuple:
    """Find the latest checkpoint directory in output_dir.

    Returns (step, path) or (0, "") if no checkpoint found.
    Looks for directories matching ``checkpoint-{int}`` or ``step-{int}``.
    """
    if not Path(output_dir).exists():
        return 0, ""

    checkpoints = []
    for d in Path(output_dir).iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("checkpoint-"):
            prefix = "checkpoint-"
        elif name.startswith("step-"):
            prefix = "step-"
        else:
            continue
        try:
            step_num = int(name[len(prefix):])
            checkpoints.append((step_num, str(d)))
        except ValueError:
            continue

    if not checkpoints:
        return 0, ""

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1]


def list_checkpoints(output_dir: str) -> list:
    """Return sorted list of (step, path) for all checkpoints."""
    if not Path(output_dir).exists():
        return []
    results = []
    for d in Path(output_dir).iterdir():
        if not d.is_dir():
            continue
        name = d.name
        prefix = None
        if name.startswith("checkpoint-"):
            prefix = "checkpoint-"
        elif name.startswith("step-"):
            prefix = "step-"
        if prefix is None:
            continue
        try:
            step_num = int(name[len(prefix):])
            results.append((step_num, str(d)))
        except ValueError:
            continue
    results.sort(key=lambda x: x[0])
    return results


def save_training_state(step, epoch, optimizer, scheduler, extra: dict, ckpt_dir: str):
    """Save optimizer, scheduler, and extra state alongside model weights.

    Args:
        step: Global training step.
        epoch: Current epoch number (1-indexed).
        optimizer: Torch optimizer to save state dict from.
        scheduler: LR scheduler to save state dict from.
        extra: Additional key-value pairs to persist (e.g., rewards, loss history).
        ckpt_dir: Directory to save ``training_state.pt`` into.
    """
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        **extra,
    }, f"{ckpt_dir}/training_state.pt")


def load_training_state(ckpt_dir: str, optimizer, scheduler) -> dict:
    """Load training state from a checkpoint directory.

    Returns the full state dict (including ``extra`` fields). Returns ``None``
    if no ``training_state.pt`` exists.
    """
    state_path = Path(ckpt_dir) / "training_state.pt"
    if not state_path.exists():
        return None
    state = torch.load(state_path, weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def save_checkpoint(step, model, tokenizer, optimizer, scheduler, output_dir: str, extra: dict = None):
    """Save a full checkpoint (model + tokenizer + training state).

    Args:
        step: Global step number for checkpoint naming.
        model: Model to save via ``save_pretrained``.
        tokenizer: Tokenizer to save via ``save_pretrained``.
        optimizer: Torch optimizer.
        scheduler: LR scheduler.
        output_dir: Base output directory.
        extra: Optional dict of additional state to persist.
    """
    ckpt_dir = f"{output_dir}/checkpoint-{step}"
    logger.info(f"Saving checkpoint to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    save_training_state(
        step=step,
        epoch=extra.get("epoch", 0) if extra else 0,
        optimizer=optimizer,
        scheduler=scheduler,
        extra=extra or {},
        ckpt_dir=ckpt_dir,
    )


def resume_if_available(output_dir: str, output_prefix: str = "checkpoint") -> tuple:
    """Check for latest checkpoint and return resume info.

    Returns (step, ckpt_path) tuple. Step is 0 if nothing to resume.
    Logs a hint to the user if a checkpoint exists but isn't being auto-resumed.
    Callers should use the returned step > 0 to decide whether to resume.
    """
    latest_step, latest_path = find_latest_checkpoint(output_dir)
    if latest_step > 0:
        logger.info(
            f"Found {output_prefix} at step {latest_step}: {latest_path}\n"
            f"To resume, pass --resume-step {latest_step} --base-model {latest_path}"
        )
    return latest_step, latest_path
