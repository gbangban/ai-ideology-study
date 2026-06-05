#!/usr/bin/env python3
"""GRPO v4: RLVMR process rewards + outcome rewards on causal dataset.

Adds planning, commitment, monitor, and format_penalty rewards on top of
the three outcome rewards. Requires model to produce <planning>, <commitment>,
<monitor> tags for process rewards to activate.

Usage:
    python3 -m src.student.train_grpo_v4
    python3 -m src.student.train_grpo_v4 --max-steps 500 --resume-step 250
"""

import argparse
import csv
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import wandb
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unsloth import FastLanguageModel

from src.student.grpo_config import GRPO_CONFIG
from src.student.rewards import (
    compute_directional_assertion,
    compute_dm_keyword_alignment,
    compute_mechanism_commitment,
    compute_planning_reward,
    compute_commitment_reward,
    compute_monitor_reward,
    compute_format_penalty,
)
from src.student.sglang_client import SglangClient
from src.student.train_grpo import (
    GRPODataset,
    compute_advantage,
    find_latest_checkpoint,
    generate_completions,
)

logger = logging.getLogger("grpo-v4")


REWARD_WEIGHTS_V4 = {
    "directional_assertion": 0.25,
    "dm_alignment": 0.15,
    "mechanism_commitment": 0.15,
    "planning": 0.15,
    "commitment": 0.15,
    "monitor": 0.10,
    "format_penalty": 0.05,
}


def compute_rewards_v4(
    completions: List[str],
    weights: dict,
    tokenizer=None,
    judge_model=None,
    judge_tokenizer=None,
    sglang_client=None,
) -> Tuple:
    """Compute all 8 reward components for v4."""
    n = len(completions)
    total_scores = [0.0] * n
    dm_scores = [0.0] * n
    dir_scores = [0.0] * n
    mech_scores = [0.0] * n
    planning_scores = [0.0] * n
    commitment_scores = [0.0] * n
    monitor_scores = [0.0] * n
    format_scores = [0.0] * n

    if "directional_assertion" in weights:
        w = weights["directional_assertion"]
        for i, c in enumerate(completions):
            s = compute_directional_assertion(c)
            dir_scores[i] = s
            total_scores[i] += w * s

    if "dm_alignment" in weights:
        w = weights["dm_alignment"]
        for i, c in enumerate(completions):
            s = compute_dm_keyword_alignment(c)
            dm_scores[i] = s
            total_scores[i] += w * s

    if "mechanism_commitment" in weights:
        w = weights["mechanism_commitment"]
        for i, c in enumerate(completions):
            s = compute_mechanism_commitment(c)
            mech_scores[i] = s
            total_scores[i] += w * s

    if "planning" in weights:
        w = weights["planning"]
        for i, c in enumerate(completions):
            s = compute_planning_reward(c)
            planning_scores[i] = s
            total_scores[i] += w * s

    if "commitment" in weights:
        w = weights["commitment"]
        for i, c in enumerate(completions):
            s = compute_commitment_reward(c)
            commitment_scores[i] = s
            total_scores[i] += w * s

    if "monitor" in weights:
        w = weights["monitor"]
        for i, c in enumerate(completions):
            s = compute_monitor_reward(c)
            monitor_scores[i] = s
            total_scores[i] += w * s

    if "format_penalty" in weights:
        w = weights["format_penalty"]
        for i, c in enumerate(completions):
            s = compute_format_penalty(c)
            format_scores[i] = s
            total_scores[i] += w * s

    return (total_scores, dm_scores, dir_scores, mech_scores,
            planning_scores, commitment_scores, monitor_scores, format_scores)


def train_v4(
    config: dict,
    base_model_path: str,
    output_dir: str,
    dataset_path: str,
    resume_step: int = 0,
):
    """GRPO v4 training loop with RLVMR process rewards."""

    # Load model
    logger.info(f"Loading model: {base_model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        logger.info("Extracted text tokenizer from VLProcessor (text-only mode)")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )
    model = FastLanguageModel.for_training(model)
    logger.info(f"LoRA applied: rank={config['lora_rank']}, alpha={config['lora_alpha']}")

    # No judge model for v4 (dm_alignment is rule-based)
    judge_model = None
    judge_tokenizer = None
    sglang_client = None

    # Load causal dataset
    logger.info(f"Loading JSONL dataset from {dataset_path}...")
    raw_prompts = []
    with open(dataset_path) as f:
        for line in f:
            raw_prompts.append(json.loads(line)["prompt"])
    logger.info(f"  Loaded {len(raw_prompts)} prompts")

    prompts = []
    for p in raw_prompts:
        chat = [{"role": "user", "content": p}]
        prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)

    dataset = GRPODataset(prompts)
    dataloader = DataLoader(dataset, batch_size=config["per_device_train_batch_size"], shuffle=True)
    dataloader_iter = iter(itertools.cycle(dataloader))

    # Hyperparameters
    group_size = config["grpo_g"]
    beta = config.get("beta", 0.1)
    lr = config["learning_rate"]
    max_steps = config["max_steps"]
    max_completion_tokens = config["max_completion_length"]
    warmup_steps = config["warmup_steps"]
    save_steps = config["save_steps"]
    logging_steps = config["logging_steps"]
    gradient_accum_steps = config.get("gradient_accumulation_steps", 1)

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

    total_rewards = []
    start_step = resume_step
    if resume_step > 0:
        state_path = f"{output_dir}/checkpoint-{resume_step}/training_state.pt"
        if Path(state_path).exists():
            state = torch.load(state_path, weights_only=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            total_rewards = state.get("rewards", [])
            logger.info(f"Resumed from step {resume_step}, {len(total_rewards)} reward entries")
        else:
            logger.warning(f"No training_state.pt at step {resume_step}, starting fresh")

    model.train()
    step = start_step
    logger.info(f"Starting GRPO v4 training...")
    logger.info(f"  Steps: {max_steps}, G: {group_size}, LR: {lr}, Beta: {beta}")
    logger.info(f"  Rewards: {REWARD_WEIGHTS_V4}")

    # W&B
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    wandb_base_url = os.environ.get("WANDB_BASE_URL")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    if wandb_base_url:
        wandb.base_url = wandb_base_url
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dm-align-grpo"),
        name=os.environ.get("WANDB_RUN_NAME", "grpo-v4-rlvmr"),
        config={**config, "reward_weights": REWARD_WEIGHTS_V4},
        mode=wandb_mode,
        save_code=False,
    )

    # CSV logger
    csv_path = f"{output_dir}/training_log.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["step", "loss", "avg_reward", "dm_reward", "dir_reward", "mech_reward",
                          "planning_reward", "commitment_reward", "monitor_reward", "format_reward",
                          "elapsed_s", "vram_gb"])

    while step < max_steps:
        batch_prompts = next(dataloader_iter)
        batch_start = time.time()

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

        rewards, dm_comp, dir_comp, mech_comp, planning_comp, commitment_comp, monitor_comp, format_comp = compute_rewards_v4(
            all_completions, REWARD_WEIGHTS_V4, tokenizer, judge_model, judge_tokenizer, sglang_client,
        )

        advantages = compute_advantage(rewards, group_size)

        model.train()
        optimizer.zero_grad()

        batch_loss = 0.0
        n_samples = len(all_completions)

        lora_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_weights[name] = param.data.clone()

        all_texts = []
        all_prompt_lengths = []
        for i in range(n_samples):
            all_texts.append(all_prompt_texts[i] + all_completions[i])
            prompt_enc = tokenizer(all_prompt_texts[i], add_special_tokens=False)
            all_prompt_lengths.append(len(prompt_enc["input_ids"]))

        for i in range(n_samples):
            text = all_texts[i]
            tokenized = tokenizer(
                text, truncation=True, max_length=2048, return_tensors="pt",
            ).to(model.device)
            input_ids = tokenized["input_ids"]
            attn_mask = tokenized["attention_mask"]
            prompt_len = all_prompt_lengths[i]

            for name, param in model.named_parameters():
                if name in lora_weights:
                    param.data.copy_(lora_weights[name])
            model.eval()
            with torch.no_grad():
                ref_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
            ref_logits = ref_outputs.logits[0]
            del ref_outputs

            model.train()
            new_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
            new_logits = new_outputs.logits[0]

            completion_slice = input_ids[0, prompt_len:]
            ref_log_probs = torch.log_softmax(ref_logits[prompt_len - 1:], dim=-1)[0, completion_slice]
            new_log_probs = torch.log_softmax(new_logits[prompt_len - 1:], dim=-1)[0, completion_slice]

            log_ratio = new_log_probs - ref_log_probs
            ratio = log_ratio.exp()
            adv = advantages[i]

            loss = 0.0
            for r, a in zip(ratio, adv):
                unclosed = r * a
                clipped = torch.clamp(r, 1 - beta, 1 + beta) * a
                loss += -torch.min(unclosed, clipped)
            loss = loss / len(ratio)

            loss.backward()
            batch_loss += loss.item()

            del new_outputs, new_logits, new_log_probs, ref_log_probs, ratio, log_ratio

        step += 1
        if step % gradient_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_reward = sum(rewards) / len(rewards)
        total_rewards.append(avg_reward)
        elapsed = time.time() - batch_start
        est_remaining = (elapsed / step) * (max_steps - step) / 3600

        if step % logging_steps == 0 or step == 1:
            vram = torch.cuda.memory_allocated(model.device) / 1e9
            logger.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {batch_loss / len(all_completions):.4f} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Time: {elapsed:.1f}s | "
                f"ETA: {est_remaining:.1f}h | "
                f"VRAM: {vram:.1f}GB"
            )

        avg_dm = sum(dm_comp) / len(dm_comp)
        avg_dir = sum(dir_comp) / len(dir_comp)
        avg_mech = sum(mech_comp) / len(mech_comp)
        avg_planning = sum(planning_comp) / len(planning_comp)
        avg_commitment = sum(commitment_comp) / len(commitment_comp)
        avg_monitor = sum(monitor_comp) / len(monitor_comp)
        avg_format = sum(format_comp) / len(format_comp)

        csv_writer.writerow([
            step,
            f"{batch_loss / len(all_completions):.4f}",
            f"{avg_reward:.4f}",
            f"{avg_dm:.4f}",
            f"{avg_dir:.4f}",
            f"{avg_mech:.4f}",
            f"{avg_planning:.4f}",
            f"{avg_commitment:.4f}",
            f"{avg_monitor:.4f}",
            f"{avg_format:.4f}",
            f"{elapsed:.1f}",
            f"{torch.cuda.memory_allocated(model.device) / 1e9:.1f}",
        ])
        csv_f.flush()

        wandb.log({
            "step": step,
            "loss": batch_loss / len(all_completions),
            "avg_reward": avg_reward,
            "dm_reward": avg_dm,
            "dir_reward": avg_dir,
            "mech_reward": avg_mech,
            "planning_reward": avg_planning,
            "commitment_reward": avg_commitment,
            "monitor_reward": avg_monitor,
            "format_reward": avg_format,
            "batch_time_s": elapsed,
            "vram_gb": torch.cuda.memory_allocated(model.device) / 1e9,
            "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr,
        })

        if step % save_steps == 0:
            ckpt_dir = f"{output_dir}/checkpoint-{step}"
            model.save_pretrained(ckpt_dir, safe_serialization=True)
            tokenizer.save_pretrained(ckpt_dir)
            from src.student.train_grpo import save_training_state
            save_training_state(step, optimizer, scheduler, total_rewards, ckpt_dir)
            logger.info(f"Saved checkpoint at step {step}")

    csv_f.close()
    final_dir = f"{output_dir}/checkpoint-{max_steps}"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    from src.student.train_grpo import save_training_state
    save_training_state(max_steps, optimizer, scheduler, total_rewards, final_dir)
    logger.info(f"Final model saved to {final_dir}")
    wandb.finish()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="GRPO v4: RLVMR process rewards on causal dataset")
    parser.add_argument(
        "--base-model",
        default=GRPO_CONFIG["base_model"],
        help="Path to SFT merged checkpoint or GRPO checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/lora_adapters/grpo_adapter_v4",
        help="Output directory for GRPO v4 adapter",
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
    config["max_steps"] = args.max_steps

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
        train_v4(config, base_model, args.output_dir, args.dataset_path, resume_step=resume_step)
    except Exception:
        logger.error("Training failed, flushing VRAM...")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
