#!/usr/bin/env python3
"""Cold-start SFT training for v3/v4.

Trains the base model on tagged demonstrations for 5 epochs to teach
the model the RLVMR output format before RLVMR training.

Both v3 and v4 use the same cold-start SFT checkpoint as base.
This isolates process rewards as the only variable between v3 and v4.

Usage:
    python3 -m src.student.train_cold_start_sft \
        --data data/processed/cold_start_sft.jsonl \
        --output checkpoints/lora_adapters/cold_start_sft
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import trackio
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("cold-start-sft")


class SFTDataset(TorchDataset):
    """Simple dataset of (input_ids, labels) for SFT."""

    def __init__(self, records, tokenizer, max_length=4096):
        self.samples = []
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for rec in records:
            messages = rec.get("messages", [])
            if not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokenized = tokenizer(
                text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt"
            )
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            labels = input_ids.clone()

            # Find the assistant message start to mask prompt tokens
            prompt_messages = []
            for msg in messages:
                if msg["role"] == "assistant":
                    break
                prompt_messages.append(msg)

            if prompt_messages:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
                user_end = len(prompt_tokens["input_ids"])
                labels[:user_end] = -100

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
        logger.info(f"Prepared {len(self.samples)} SFT samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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


def train(data_path, base_model, output_dir, epochs=5, batch_size=1, lr=1e-5, resume_step=0):
    """Run cold-start SFT training."""
    from unsloth import FastLanguageModel
    import torch.nn.functional as F

    logger.info(f"Loading model from {base_model}...")
    _strip_vision_config(base_model)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
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
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = FastLanguageModel.for_training(model, use_gradient_checkpointing="unsloth")
    logger.info("LoRA applied for cold-start SFT")

    # Load data
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records")

    dataset = SFTDataset(records, tokenizer, max_length=4096)
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = len(dataloader) * epochs
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # Resume from checkpoint if requested
    from src.student.training_utils import find_latest_checkpoint, load_training_state, save_checkpoint, resume_if_available
    if resume_step > 0:
        ckpt_path = f"{output_dir}/checkpoint-{resume_step}"
        state = load_training_state(ckpt_path, optimizer, scheduler)
        if state:
            logger.info(
                f"Resumed optimizer + scheduler from step {resume_step}, "
                f"epoch {state.get('epoch', '?')}, avg_loss {state.get('avg_loss', '?'):.4f}"
            )
        else:
            logger.warning(
                f"No training_state.pt at step {resume_step}, "
                f"starting optimizer/scheduler fresh (model weights loaded from checkpoint)"
            )
    else:
        latest_step, latest_path = resume_if_available(output_dir, "checkpoint")

    # Trackio
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=os.environ.get("TRACKIO_RUN_NAME", "cold-start-sft"),
        config={"epochs": epochs, "batch_size": batch_size, "lr": lr},
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
    )

    model.train()
    start_time = time.time()
    global_step = 0
    batch_times = []
    save_interval = max(50, len(dataloader) // 2)  # save at least once per epoch

    print("=" * 80)
    print(f"{'Epoch':>5} {'Step':>6} {'Loss':>8} {'BatchLoss':>10} {'LR':>10} {'BatchTime':>10} {'VRAM':>7}")
    print("=" * 80)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            global_step += 1
            if global_step % save_interval == 0:
                save_checkpoint(
                    global_step, model, tokenizer, optimizer, scheduler,
                    output_dir, extra={"epoch": epoch + 1, "avg_loss": epoch_loss / (batch_idx + 1)},
                )
            batch_elapsed = time.time() - batch_start
            batch_times.append(batch_elapsed)

            if global_step % 5 == 0 or global_step == 1:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
                vram = torch.cuda.memory_allocated(model.device) / 1e9
                avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])
                print(
                    f"{epoch+1:>5d} "
                    f"{global_step:>6d} "
                    f"{epoch_loss / (batch_idx+1):>8.4f} "
                    f"{loss.item():>10.4f} "
                    f"{current_lr:>10.2e} "
                    f"{avg_batch_time:>10.2f}s "
                    f"{vram:>6.1f}G",
                    flush=True,
                )

        avg_loss = epoch_loss / len(dataloader)
        epoch_elapsed = time.time() - epoch_start
        elapsed = time.time() - start_time
        vram = torch.cuda.memory_allocated(model.device) / 1e9
        logger.info(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Step: {global_step} | Epoch Time: {epoch_elapsed/60:.1f}m | "
            f"Total Time: {elapsed/60:.1f}m | "
            f"VRAM: {vram:.1f}GB"
        )

 

        trackio.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "step": global_step,
            "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr,
        })

    # Save
    logger.info(f"Saving cold-start SFT adapter to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Total time: {(time.time() - start_time) / 60:.1f}m")

    trackio.finish()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Cold-start SFT for v3/v4")
    parser.add_argument(
        "--data",
        default="data/processed/cold_start_sft.jsonl",
        help="Path to cold-start SFT data",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-9B",
        help="Base model path",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/lora_adapters/cold_start_sft",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
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

    from src.student.training_utils import list_checkpoints
    if args.find_checkpoint:
        checkpoints = list_checkpoints(args.output)
        if checkpoints:
            print(f"Latest checkpoint: step {checkpoints[-1][0]} at {checkpoints[-1][1]}")
            for step, path in checkpoints:
                print(f"  checkpoint-{step}")
        else:
            print(f"No checkpoints found in {args.output}")
        return

    resume_step = args.resume_step
    base_model = args.base_model
    if resume_step > 0:
        ckpt_path = f"{args.output}/checkpoint-{resume_step}"
        if Path(ckpt_path).exists():
            base_model = ckpt_path
            logger.info(f"Auto-resuming from {ckpt_path}")
        else:
            logger.warning(f"Checkpoint {ckpt_path} not found, using base-model as-is")

    train(args.data, base_model, args.output, args.epochs, args.batch_size, args.lr, resume_step=resume_step)


if __name__ == "__main__":
    main()
