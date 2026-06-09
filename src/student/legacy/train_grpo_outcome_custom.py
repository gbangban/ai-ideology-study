#!/usr/bin/env python3
"""GRPO v3: Outcome rewards only, flat advantage, correctness-based.

Uses correctness-based outcome rewards from real benchmark ground truth
(EconCausal, Corr2Cause). No process-level rewards, no tags, no dual advantage.
Serves as the control condition for v4.

Usage:
    python3 -m src.student.legacy.train_grpo_outcome_custom \
        --base-model /path/to/sft/checkpoint \
        --output-dir checkpoints/lora_adapters/grpo_v3_outcome
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
from typing import List

import torch
import torch.nn.functional as F
import trackio
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.student.grpo_config_outcome import DEFAULT_CONFIG as GRPO_CONFIG_V3
from src.student.reward_outcome import compute_outcome_reward

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("grpo-v3")


class GRPODatasetV3(TorchDataset):
    """Dataset of (prompt_text, doc) pairs for v3 training."""

    def __init__(self, prompt_texts: List[str], docs: List[dict]):
        self.prompt_texts = prompt_texts
        self.docs = docs

    def __len__(self):
        return len(self.prompt_texts)

    def __getitem__(self, idx):
        return self.prompt_texts[idx], self.docs[idx]


def grpo_collate_fn(batch):
    """Collate (prompt_text, doc) tuples without converting dicts to tensors."""
    prompts = [item[0] for item in batch]
    docs = [item[1] for item in batch]
    return prompts, docs


def compute_advantage(rewards: List[float], group_size: int) -> torch.Tensor:
    """Group-relative advantage: (r_i - mean) / std within each group."""
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    means = rewards_tensor.view(-1, group_size).mean(dim=1, keepdim=True)
    stds = rewards_tensor.view(-1, group_size).std(dim=1, keepdim=True).clamp(min=1e-8)
    advantages = ((rewards_tensor - means.flatten()) / stds.flatten()).detach()
    return advantages


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


def find_latest_checkpoint(output_dir: str) -> tuple:
    """Find the latest checkpoint directory in output_dir."""
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
    return checkpoints[-1]


def save_training_state(step, optimizer, scheduler, total_rewards, ckpt_dir):
    """Save optimizer, scheduler, and reward history."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rewards": total_rewards,
    }, f"{ckpt_dir}/training_state.pt")


def _strip_vision_config(model_path: str):
    """Remove vision_config from model config.json."""
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


def train(config, base_model_path, output_dir, resume_step=0, enable_profile=False, enable_compile=False):
    """GRPO v3 training with outcome rewards only, flat advantage."""
    from unsloth import FastLanguageModel

    if resume_step == 0:
        latest_step, latest_path = find_latest_checkpoint(output_dir)
        if latest_step > 0:
            logger.info(
                f"Found checkpoint at step {latest_step}: {latest_path}\n"
                f"To resume, pass --resume-step {latest_step} --base-model {latest_path}"
            )
    else:
        logger.info(f"Resuming from step {resume_step}")

    logger.info(f"Loading model from {base_model_path}...")
    _strip_vision_config(base_model_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        logger.info("Extracted text tokenizer from VLProcessor")

    from src.student import fix_mistral_tokenizer
    fix_mistral_tokenizer(tokenizer)
    logger.info("Applied Mistral tokenizer regex fix")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )
    model = FastLanguageModel.for_training(model)
    logger.info(f"LoRA applied: rank={config['lora_rank']}, alpha={config['lora_alpha']}")

    if enable_compile:
        from src.student.train_grpo_base import maybe_compile_model
        model, was_compiled = maybe_compile_model(model, enable=True)
        if was_compiled:
            logger.info("Model successfully compiled with torch.compile")
        else:
            logger.warning("torch.compile requested but fell back to uncompiled model")

    tracker = None
    if enable_profile:
        from src.utils.memory_profiler import TrainingMemoryTracker, force_memory_cleanup
        tracker = TrainingMemoryTracker()
        cleanup = force_memory_cleanup()
        logger.info(f"Pre-training VRAM: {cleanup['after_allocated_gb']:.2f} GB")
        tracker.record(0, "after_model_setup")

    # Load dataset
    logger.info(f"Loading dataset from {config['dataset_path']}...")
    docs = []
    with open(config["dataset_path"], "r") as f:
        for line in f:
            docs.append(json.loads(line))
    logger.info(f"  Loaded {len(docs)} documents")

    prompt_texts = []
    for doc in docs:
        chat = [{"role": "user", "content": doc["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(prompt_text)

    dataset = GRPODatasetV3(prompt_texts, docs)
    dataloader = DataLoader(dataset, batch_size=config["per_device_train_batch_size"], shuffle=True, collate_fn=grpo_collate_fn)
    dataloader_iter = iter(itertools.cycle(dataloader))

    group_size = config["grpo_g"]
    beta = config["beta"]
    lr = config["learning_rate"]
    max_steps = config["max_steps"]
    max_completion_tokens = config["max_completion_length"]
    warmup_steps = config["warmup_steps"]
    save_steps = config["save_steps"]
    logging_steps = config["logging_steps"]
    sample_steps = config.get("sample_steps", 100)
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
            logger.info(f"Resumed optimizer + scheduler from step {resume_step}")
        else:
            logger.warning(f"No training_state.pt at step {resume_step}, starting fresh")

    model.train()
    step = start_step
    start_time = time.time()

    logger.info(f"Starting GRPO v3 training (outcome rewards only, flat advantage)...")
    logger.info(f"  Steps: {max_steps}, G: {group_size}, LR: {lr}, Beta: {beta}")
    logger.info(f"  Logging every {logging_steps} steps, sampling every {sample_steps} steps")
    print("=" * 90)
    print(f"{'Step':>6} {'Loss':>8} {'PG':>8} {'KL':>8} {'Reward':>8} {'MinR':>6} {'MaxR':>6} {'MedR':>6} {'LR':>10} {'Time':>6} {'ETA(h)':>7} {'VRAM':>7}")
    print("=" * 90)

    # Trackio
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=os.environ.get("TRACKIO_RUN_NAME", "grpo-v3-outcome-only"),
        config=config,
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/training_log.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["step", "loss", "avg_reward", "elapsed_s", "vram_gb", "lr"])

    while step < max_steps:
        batch_prompts, batch_docs = next(dataloader_iter)
        batch_start = time.time()

        all_completions = []
        all_prompt_texts = []
        all_docs = []
        if tracker:
            with tracker.phase(step, "generation"):
                for prompt, doc in zip(batch_prompts, batch_docs):
                    completions = generate_completions(
                        model, tokenizer, prompt,
                        group_size=group_size,
                        max_new_tokens=max_completion_tokens,
                    )
                    all_completions.extend(completions)
                    all_prompt_texts.extend([prompt] * group_size)
                    all_docs.extend([doc] * group_size)
        else:
            for prompt, doc in zip(batch_prompts, batch_docs):
                completions = generate_completions(
                    model, tokenizer, prompt,
                    group_size=group_size,
                    max_new_tokens=max_completion_tokens,
                )
                all_completions.extend(completions)
                all_prompt_texts.extend([prompt] * group_size)
                all_docs.extend([doc] * group_size)

        rewards = [compute_outcome_reward(doc, c) for c, doc in zip(all_completions, all_docs)]
        advantages = compute_advantage(rewards, group_size)

        model.train()
        optimizer.zero_grad()

        batch_loss = 0.0
        batch_pg_loss = 0.0
        batch_kl_loss = 0.0
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

            # Reference forward (no gradients)
            for name, param in model.named_parameters():
                if name in lora_weights:
                    param.data.copy_(lora_weights[name])
            model.eval()
            if tracker and i == 0:
                with tracker.phase(step, "forward_ref"):
                    with torch.no_grad():
                        ref_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                    ref_logits = ref_outputs.logits[0]
            else:
                with torch.no_grad():
                    ref_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                ref_logits = ref_outputs.logits[0]
            del ref_outputs

            model.train()
            if tracker and i == 0:
                with tracker.phase(step, "forward_new"):
                    new_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                    new_logits = new_outputs.logits[0]
            else:
                new_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                new_logits = new_outputs.logits[0]

            new_logit_shifted = new_logits[prompt_len - 1:, :]
            ref_logit_shifted = ref_logits[prompt_len - 1:, :]
            labels = input_ids[0, prompt_len:]

            label_mask = (labels != -100) & (labels != tokenizer.pad_token_id)
            if label_mask.sum() == 0:
                del new_outputs, new_logits, ref_logits
                continue

            new_log_probs = F.log_softmax(new_logit_shifted, dim=-1)
            ref_log_probs = F.log_softmax(ref_logit_shifted, dim=-1)

            new_token_lp = new_log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            ref_token_lp = ref_log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)

            old_log_probs = ref_token_lp.detach()
            log_ratio = new_token_lp - old_log_probs
            ratio = log_ratio.exp()

            adv = advantages[i]
            pg_loss_unclipped = -(ratio * adv).mean()
            pg_loss_clipped = -(torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv).mean()
            pg_loss = torch.min(pg_loss_unclipped, pg_loss_clipped)

            kl = (new_token_lp - old_log_probs).mean()
            total_loss = pg_loss - beta * kl

            total_loss.backward()
            batch_loss += total_loss.item()
            batch_pg_loss += pg_loss.item()
            batch_kl_loss += kl.item()

            del new_outputs, new_logits, new_log_probs, ref_log_probs, new_token_lp, ref_token_lp, ref_logits

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

        rewards_sorted = sorted(rewards)
        reward_min = rewards_sorted[0]
        reward_max = rewards_sorted[-1]
        reward_med = rewards_sorted[len(rewards_sorted) // 2]

        vram = torch.cuda.memory_allocated(model.device) / 1e9
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
        avg_pg = batch_pg_loss / len(all_completions)
        avg_kl = batch_kl_loss / len(all_completions)

        print(
            f"{step:>6d} "
            f"{batch_loss / len(all_completions):>8.4f} "
            f"{avg_pg:>8.4f} "
            f"{avg_kl:>8.4f} "
            f"{avg_reward:>8.3f} "
            f"{reward_min:>6.2f} "
            f"{reward_max:>6.2f} "
            f"{reward_med:>6.2f} "
            f"{current_lr:>10.2e} "
            f"{elapsed:>6.1f}s "
            f"{est_remaining:>7.1f}h "
            f"{vram:>6.1f}G",
            flush=True,
        )

        if step % logging_steps == 0 or step == 1:
            logger.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {batch_loss / len(all_completions):.4f} | "
                f"PG: {avg_pg:.4f} | "
                f"KL: {avg_kl:.4f} | "
                f"Avg Reward: {avg_reward:.3f} "
                f"(min={reward_min:.2f}, max={reward_max:.2f}, med={reward_med:.2f}) | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s | "
                f"ETA: {est_remaining:.1f}h | "
                f"VRAM: {vram:.1f}GB"
            )
            if enable_profile:
                from src.utils.memory_profiler import get_vram_peak_gb
                logger.info(f"Peak VRAM: {get_vram_peak_gb():.2f} GB")
                logger.info(tracker.summary())

        if sample_steps > 0 and step % sample_steps == 0:
            model.eval()
            sample_prompt = batch_prompts[0]
            sample_completions = generate_completions(
                model, tokenizer, sample_prompt,
                group_size=min(2, group_size),
                max_new_tokens=min(256, max_completion_tokens),
            )
            model.train()
            sample_rew = [compute_outcome_reward(batch_docs[0], c) for c in sample_completions]
            print(f"\n--- Sample @ step {step} ---")
            doc_prompt = batch_docs[0].get("prompt", "")[:150]
            print(f"Prompt: {doc_prompt}...")
            for i, (comp, rew) in enumerate(zip(sample_completions, sample_rew)):
                print(f"  Completion {i} (reward={rew:.3f}): {comp[:300]}...")
            print(f"--- End sample ---\n")

        csv_writer.writerow([
            step,
            f"{batch_loss / len(all_completions):.4f}",
            f"{avg_reward:.4f}",
            f"{elapsed:.1f}",
            f"{torch.cuda.memory_allocated(model.device) / 1e9:.1f}",
            f"{scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr:.2e}",
        ])
        csv_f.flush()

        trackio.log({
            "step": step,
            "loss": batch_loss / len(all_completions),
            "avg_reward": avg_reward,
            "batch_time_s": elapsed,
            "vram_gb": torch.cuda.memory_allocated(model.device) / 1e9,
            "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr,
        })

        if step % save_steps == 0:
            ckpt_dir = f"{output_dir}/checkpoint-{step}"
            logger.info(f"Saving checkpoint to {ckpt_dir}...")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            save_training_state(step, optimizer, scheduler, total_rewards, ckpt_dir)

    logger.info(f"Training complete. Saving final adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Total time: {(time.time() - start_time) / 3600:.1f}h")

    with open(f"{output_dir}/reward_history.json", "w") as f:
        json.dump(total_rewards, f)
    csv_f.close()
    trackio.finish()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="GRPO v3 Training (outcome rewards only)")
    parser.add_argument(
        "--base-model",
        default=GRPO_CONFIG_V3["base_model"],
        help="Path to SFT merged checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default=GRPO_CONFIG_V3["output_dir"],
        help="Output directory for GRPO adapter",
    )
    parser.add_argument(
        "--dataset-path",
        default=GRPO_CONFIG_V3["dataset_path"],
        help="Path to merged training dataset JSONL",
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed memory profiling per training phase",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for forward passes",
    )
    args = parser.parse_args()

    config = GRPO_CONFIG_V3.copy()
    config["dataset_path"] = args.dataset_path

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
        train(
            config, base_model, args.output_dir,
            resume_step=resume_step,
            enable_profile=args.profile,
            enable_compile=args.compile,
        )
    except Exception:
        logger.error("Training failed, flushing VRAM...")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
