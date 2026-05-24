#!/usr/bin/env python3
"""
Custom GRPO Training Script

Group Relative Policy Optimization implemented from scratch (no TRL GRPOTrainer/vLLM dependency).
Loads SFT merged checkpoint via Unsloth NF4, trains with custom GRPO loop.

Usage:
    python3 -m src.student.train_grpo \
        --base-model /path/to/sft/checkpoint \
        --output-dir checkpoints/lora_adapters/grpo_adapter
"""

import argparse
import itertools
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.grpo_config import GRPO_CONFIG
from src.student.rewards import (
    compute_directional_assertion,
    compute_format_reward,
    compute_length_reward,
    compute_dm_alignment_judge,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GRPODataset(TorchDataset):
    """Simple dataset of prompts for GRPO training."""

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def compute_kl_penalty(
    log_probs_new: torch.Tensor,
    log_probs_ref: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute KL divergence penalty: beta * (log_pi_ref - log_pi_new)."""
    return beta * (log_probs_ref - log_probs_new)


def compute_advantage(
    rewards: List[float],
    group_size: int,
) -> torch.Tensor:
    """Compute group-relative advantage: (r_i - mean) / std within each group."""
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    means = rewards_tensor.view(-1, group_size).mean(dim=1, keepdim=True)
    stds = rewards_tensor.view(-1, group_size).std(dim=1, keepdim=True).clamp(min=1e-8)
    advantages = ((rewards_tensor - means.flatten()) / stds.flatten()).detach()
    return advantages


def get_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Get log probabilities for generated tokens (after prompt)."""
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    labels = input_ids[:, prompt_length:]
    logits_shifted = logits[:, prompt_length - 1: -1, :]
    log_probs = F.log_softmax(logits_shifted, dim=-1)

    label_mask = (labels != -100)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs, label_mask


def generate_completions(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[str]:
    """Generate G completions for a single prompt in a single batched call."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    repeated_inputs = {
        "input_ids": inputs["input_ids"].repeat(group_size, 1),
        "attention_mask": inputs["attention_mask"].repeat(group_size, 1),
    }

    with torch.no_grad():
        output_ids = model.generate(
            **repeated_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    completions = []
    for i in range(group_size):
        generated = output_ids[i][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        completions.append(text)
    return completions


def compute_rewards(
    completions: List[str],
    weights: dict,
    tokenizer,
    judge_model=None,
    judge_tokenizer=None,
) -> List[float]:
    """Compute weighted sum of all reward functions for a batch of completions."""
    n = len(completions)
    total_scores = [0.0] * n

    if "directional_assertion" in weights:
        w = weights["directional_assertion"]
        for i, c in enumerate(completions):
            total_scores[i] += w * compute_directional_assertion(c)

    if "format" in weights:
        w = weights["format"]
        for i, c in enumerate(completions):
            total_scores[i] += w * compute_format_reward(c)

    if "length" in weights:
        w = weights["length"]
        for i, c in enumerate(completions):
            tokens = len(tokenizer.encode(c, add_special_tokens=False))
            total_scores[i] += w * compute_length_reward(tokens)

    if "dm_alignment" in weights and judge_model is not None and judge_tokenizer is not None:
        w = weights["dm_alignment"]
        dm_scores = compute_dm_alignment_judge(completions, judge_model, judge_tokenizer)
        for i, s in enumerate(dm_scores):
            total_scores[i] += w * s

    return total_scores


def find_latest_checkpoint(output_dir: str) -> Tuple[int, str]:
    """Find the latest checkpoint directory in output_dir.

    Returns (step, path) or (0, "") if no checkpoint found.
    """
    if not Path(output_dir).exists():
        return 0, ""

    checkpoints = []
    for d in Path(output_dir).iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step_num = int(d.name.split("-")[1])
                checkpoints.append((step_num, str(d)))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return 0, ""

    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoints[-1]
    return latest_step, latest_path


def save_training_state(
    step: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    total_rewards: list,
    ckpt_dir: str,
):
    """Save optimizer, scheduler, and reward history alongside model weights."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rewards": total_rewards,
    }, f"{ckpt_dir}/training_state.pt")


def _strip_vision_config(model_path: str):
    """Remove vision_config from model config.json to avoid image processor init errors.

    Qwen3.5 checkpoints carry vision_config from the base multimodal model even
    though we only use the text LM head. This causes transformers to try loading
    an image processor which fails.

    Strips from both the given path and any HF cache copies.
    """
    def _strip_at(path: str):
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            return False

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
        return stripped

    # Strip from the provided path
    _strip_at(model_path)

    # Also strip from HF cache if the model was downloaded
    import os
    hf_cache = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
    if Path(hf_cache).exists():
        for cache_dir in Path(hf_cache).iterdir():
            if not cache_dir.is_dir():
                continue
            # Search one level deeper for the actual model dir
            sub_dirs = list(cache_dir.iterdir()) if cache_dir.is_dir() else []
            for sub in sub_dirs:
                if sub.is_dir() and (sub / "config.json").exists():
                    _strip_at(str(sub))


def train(config: dict, base_model_path: str, output_dir: str, resume_step: int = 0):
    """Run GRPO training.

    Args:
        config: Training configuration dict.
        base_model_path: Path to SFT merged checkpoint or latest GRPO checkpoint.
        output_dir: Output directory for GRPO adapter.
        resume_step: If >0, resume from this step (loads checkpoint state).
    """
    from unsloth import FastLanguageModel

    # Check for latest checkpoint if not explicitly resuming
    if resume_step == 0:
        latest_step, latest_path = find_latest_checkpoint(output_dir)
        if latest_step > 0:
            logger.info(
                f"Found checkpoint at step {latest_step}: {latest_path}\n"
                f"To resume, pass --resume-step {latest_step} --base-model {latest_path}"
            )
    else:
        logger.info(f"Resuming from step {resume_step}")

    # Load model with NF4 quantization
    # Qwen3.5 checkpoints carry vision_config from the base model even though
    # we only use the text LM. Strip it to avoid image processor init errors.
    logger.info(f"Loading model from {base_model_path}...")
    _strip_vision_config(base_model_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Qwen3.5 returns a VLProcessor that tries to process images on every call.
    # Extract the underlying text tokenizer so we bypass the image pipeline entirely.
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        logger.info("Extracted text tokenizer from VLProcessor (text-only mode)")

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )
    model = FastLanguageModel.for_training(model)
    logger.info(f"LoRA applied: rank={config['lora_rank']}, alpha={config['lora_alpha']}")

    # Load judge model
    judge_model = None
    judge_tokenizer = None
    if config["reward_weights"].get("dm_alignment", 0) > 0:
        logger.info(f"Loading judge model: {config['judge_model']}...")
        _strip_vision_config(config["judge_model"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        judge_model = AutoModelForCausalLM.from_pretrained(
            config["judge_model"],
            device_map="auto",
            quantization_config=bnb_config,
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(config["judge_model"])
        logger.info(f"Judge model loaded on {judge_model.device}")

    # Load and prepare dataset
    logger.info(f"Loading questions from {config['questions_path']}...")
    with open(config["questions_path"], "r") as f:
        data = json.load(f)
    questions = [q["question"] for q in data]
    logger.info(f"  Loaded {len(questions)} questions")

    prompts = []
    for q in questions:
        chat = [{"role": "user", "content": q}]
        prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)

    dataset = GRPODataset(prompts)
    dataloader = DataLoader(dataset, batch_size=config["per_device_train_batch_size"], shuffle=True)
    dataloader_iter = iter(itertools.cycle(dataloader))

    # Training hyperparameters
    group_size = config["grpo_g"]
    beta = config.get("beta", 0.1)
    lr = config["learning_rate"]
    max_steps = config["max_steps"]
    max_completion_tokens = config["max_completion_length"]
    warmup_steps = config["warmup_steps"]
    save_steps = config["save_steps"]
    logging_steps = config["logging_steps"]
    gradient_accum_steps = config.get("gradient_accumulation_steps", 1)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # Resume training state if requested
    total_rewards = []
    start_step = resume_step
    if resume_step > 0:
        state_path = f"{output_dir}/checkpoint-{resume_step}/training_state.pt"
        if Path(state_path).exists():
            state = torch.load(state_path, weights_only=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            total_rewards = state.get("rewards", [])
            logger.info(
                f"Resumed optimizer + scheduler from step {resume_step}, "
                f"{len(total_rewards)} reward history entries"
            )
        else:
            logger.warning(
                f"No training_state.pt at step {resume_step}, "
                f"starting optimizer/scheduler fresh (model weights loaded from checkpoint)"
            )

    # Training loop
    model.train()
    step = start_step
    start_time = time.time()

    logger.info(f"Starting GRPO training...")
    logger.info(f"  Steps: {max_steps}, G: {group_size}, LR: {lr}, Beta: {beta}")
    logger.info(f"  Starting from step: {start_step}")
    logger.info(f"  Estimated remaining: {((max_steps - start_step) / max_steps * 10):.0f}-{((max_steps - start_step) / max_steps * 12):.0f}h")

    while step < max_steps:
        batch_prompts = next(dataloader_iter)
        batch_start = time.time()

        # Generate completions for each prompt in batch
        all_completions = []
        all_prompt_texts = []
        for prompt in batch_prompts:
            completions = generate_completions(
                model, tokenizer, prompt,
                group_size=group_size,
                max_new_tokens=max_completion_tokens,
            )
            all_completions.extend(completions)
            all_prompt_texts.extend([prompt] * group_size)

        # Compute rewards
        rewards = compute_rewards(
            all_completions,
            config["reward_weights"],
            tokenizer,
            judge_model,
            judge_tokenizer,
        )

        # Compute advantages
        advantages = compute_advantage(rewards, group_size)

        # Policy update
        model.train()

        batch_loss = 0.0
        n_samples = len(all_completions)

        # Snapshot LoRA weights for reference policy
        lora_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_weights[name] = param.data.clone()

        # Process in mini-batches to manage VRAM
        mini_batch_size = 4
        for mb_start in range(0, n_samples, mini_batch_size):
            mb_end = min(mb_start + mini_batch_size, n_samples)
            mb_advantages = advantages[mb_start:mb_end]

            # Tokenize prompt + completion pairs
            mb_texts = []
            mb_prompt_lengths = []
            for i in range(mb_start, mb_end):
                full_text = all_prompt_texts[i] + all_completions[i]
                mb_texts.append(full_text)
                prompt_enc = tokenizer(all_prompt_texts[i], add_special_tokens=False)
                mb_prompt_lengths.append(len(prompt_enc["input_ids"]))

            tokenized = tokenizer(
                mb_texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(model.device)

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # Reference policy log probs (restore original LoRA weights)
            for name, param in model.named_parameters():
                if name in lora_weights:
                    param.data.copy_(lora_weights[name])
            model.eval()
            with torch.no_grad():
                ref_outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            ref_logits = ref_outputs.logits
            model.train()

            # New policy log probs (current LoRA weights, WITH gradients)
            new_outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            new_logits = new_outputs.logits

            # Compute log probs for completion tokens only
            for b_idx in range(len(mb_texts)):
                prompt_len = mb_prompt_lengths[b_idx]
                labels = input_ids[b_idx, prompt_len:]
                new_logit_shifted = new_logits[b_idx, prompt_len - 1: -1, :]
                ref_logit_shifted = ref_logits[b_idx, prompt_len - 1: -1, :]

                label_mask = (labels != -100) & (labels != tokenizer.pad_token_id)
                if label_mask.sum() == 0:
                    continue

                new_log_probs = F.log_softmax(new_logit_shifted, dim=-1)
                ref_log_probs = F.log_softmax(ref_logit_shifted, dim=-1)

                new_token_lp = new_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                ref_token_lp = ref_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

                # Policy ratio
                old_log_probs = ref_token_lp.detach()
                log_ratio = new_token_lp - old_log_probs
                ratio = log_ratio.exp()

                # Clipped PPO-style objective (MIN, not MAX)
                adv = mb_advantages[b_idx]
                pg_loss_unclipped = -(ratio * adv).mean()
                pg_loss_clipped = -(torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv).mean()
                pg_loss = torch.min(pg_loss_unclipped, pg_loss_clipped)

                # KL penalty
                kl = (old_log_probs - new_token_lp).mean()
                total_loss = pg_loss + beta * kl

                total_loss.backward()
                batch_loss += total_loss.item()

        if step % gradient_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

        avg_reward = sum(rewards) / len(rewards)
        total_rewards.append(avg_reward)
        elapsed = time.time() - batch_start
        est_remaining = (elapsed / step) * (max_steps - step) / 3600

        if step % logging_steps == 0 or step == 1:
            logger.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {batch_loss / len(all_completions):.4f} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Time: {elapsed:.1f}s | "
                f"ETA: {est_remaining:.1f}h"
            )

        # Save checkpoint
        if step % save_steps == 0:
            ckpt_dir = f"{output_dir}/checkpoint-{step}"
            logger.info(f"Saving checkpoint to {ckpt_dir}...")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            save_training_state(step, optimizer, scheduler, total_rewards, ckpt_dir)

    # Final save
    logger.info(f"Training complete. Saving final adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Total time: {(time.time() - start_time) / 3600:.1f}h")

    # Save reward history
    with open(f"{output_dir}/reward_history.json", "w") as f:
        json.dump(total_rewards, f)
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for DM Alignment")
    parser.add_argument(
        "--base-model",
        default=GRPO_CONFIG["base_model"],
        help="Path to SFT merged checkpoint or GRPO checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        default=GRPO_CONFIG["output_dir"],
        help="Output directory for GRPO adapter",
    )
    parser.add_argument(
        "--questions-path",
        default=GRPO_CONFIG["questions_path"],
        help="Path to questions.json",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="Resume from checkpoint at this step (sets base-model to checkpoint dir automatically)",
    )
    parser.add_argument(
        "--find-checkpoint",
        action="store_true",
        help="List available checkpoints and exit",
    )
    args = parser.parse_args()

    config = GRPO_CONFIG.copy()
    config["questions_path"] = args.questions_path

    # Find checkpoint mode
    if args.find_checkpoint:
        step, path = find_latest_checkpoint(args.output_dir)
        if step > 0:
            print(f"Latest checkpoint: step {step} at {path}")
            # List all checkpoints
            for d in sorted(Path(args.output_dir).iterdir()):
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    print(f"  {d.name}")
        else:
            print(f"No checkpoints found in {args.output_dir}")
        return

    # Auto-resume: if --resume-step given, find the checkpoint and set base-model
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
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        logger.error(f"VRAM after flush: allocated={allocated/1e9:.1f}GB, reserved={reserved/1e9:.1f}GB")
        raise


if __name__ == "__main__":
    main()
