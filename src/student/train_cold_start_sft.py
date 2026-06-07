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
import wandb
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

    def __init__(self, records, tokenizer, max_length=1024):
        self.samples = []
        for rec in records:
            messages = rec.get("messages", [])
            if not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokenized = tokenizer(
                text, truncation=True, max_length=max_length, return_tensors="pt"
            )
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            labels = input_ids.clone()

            # Find the assistant message start to mask prompt tokens
            # Accumulate messages up to (but not including) the assistant response
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


def train(data_path, base_model, output_dir, epochs=5, batch_size=1, lr=1e-5):
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
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"].squeeze() for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"].squeeze() for b in batch], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [b["labels"].squeeze() for b in batch], batch_first=True, padding_value=-100
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
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
        name=os.environ.get("WANDB_RUN_NAME", "cold-start-sft"),
        config={"epochs": epochs, "batch_size": batch_size, "lr": lr},
        mode=wandb_mode,
        save_code=False,
    )

    model.train()
    start_time = time.time()
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
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

            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Step: {global_step} | Time: {elapsed/60:.1f}m | "
            f"VRAM: {torch.cuda.memory_allocated(model.device)/1e9:.1f}GB"
        )

        wandb.log({
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

    wandb.finish()
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
    args = parser.parse_args()

    train(args.data, args.base_model, args.output, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
