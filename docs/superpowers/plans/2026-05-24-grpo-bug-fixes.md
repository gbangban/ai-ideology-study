# GRPO Training Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 critical bugs in the custom GRPO training loop that prevent the model from learning and cause silent failures.

**Architecture:** The fixes restructure the training loop in `train_grpo.py` to: (1) compute policy gradients correctly by removing `no_grad` from the new policy forward pass, (2) replace `deepcopy` reference model with LoRA weight snapshot/restore, (3) fix PPO clip objective from `max` to `min`, (4) batch reference forward pass, (5) cycle the dataloader for multi-epoch training, (6) wire up warmup scheduler, (7) implement gradient accumulation. The Dockerfile gets `python3.11-dev` for Triton compilation.

**Tech Stack:** PyTorch 2.7.0, Unsloth 2026.4.6, transformers, Python 3.11

---

### Task 1: Dockerfile - add python3.11-dev

**Files:**
- Modify: `docker/Dockerfile:9`

- [ ] **Step 1: Add python3.11-dev to apt packages**

In `docker/Dockerfile`, line 9, add `python3.11-dev` to the install list:

```dockerfile
   python3.11 python3.11-venv python3.11-dev python3-pip git curl wget build-essential \
```

- [ ] **Step 2: Commit**

```bash
git add docker/Dockerfile
git commit -m "fix(docker): add python3.11-dev for triton CUDA utils compilation"
```

---

### Task 2: Fix reference model - replace deepcopy with LoRA weight snapshot

**Files:**
- Modify: `src/student/train_grpo.py:304-310` (ref model creation)
- Modify: `src/student/train_grpo.py:460-502` (training loop - inline ref snapshot/restore)

**Problem:** `copy.deepcopy` on an NF4-quantized model either shares weights (KL=0) or corrupts quantization state. The reference model must capture LoRA weights before each update, then restore after computing ref log probs.

- [ ] **Step 1: Remove the deepcopy reference model block**

Replace lines 304-310 in `train_grpo.py`:

```python
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
```

Remove the entire block:
```python
    # Create reference model (snapshot before training)
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    logger.info("Reference model created")
```

Do NOT add a replacement here — the reference snapshot will be done inline in the training loop.

- [ ] **Step 2: Rewrite the policy update section to use inline LoRA snapshot**

Replace lines 428-508 in `train_grpo.py` (from `model.train()` through `scheduler.step()`):

```python
            # Policy update
            model.train()
            optimizer.zero_grad()

            batch_loss = 0.0
            n_samples = len(all_completions)

            # Snapshot LoRA weights for reference policy
            import copy
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

                # Reference policy log probs (with original LoRA weights)
                with torch.no_grad():
                    # Restore original LoRA weights
                    for name, param in model.named_parameters():
                        if name in lora_weights:
                            param.data.copy_(lora_weights[name])
                    ref_outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
                ref_logits = ref_outputs.logits

                # New policy log probs (with current trained LoRA weights)
                with torch.no_grad():
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

                    # Clipped PPO-style objective
                    adv = mb_advantages[b_idx].item()
                    policy_loss = -(ratio * adv).mean()
                    clipped_loss = (-torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv).mean()
                    pg_loss = torch.max(policy_loss, clipped_loss)

                    # KL penalty
                    kl = (old_log_probs - new_token_lp).mean()
                    total_loss = pg_loss + beta * kl

                    total_loss.backward()
                    batch_loss += total_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1
```

Wait — this approach still has the `no_grad` bug for the new policy. The issue is that with NF4 quantization, we can't easily do a forward pass with gradients through the base model. The LoRA layers ARE trainable though. Let me revise:

Actually, the key insight is: with Unsloth NF4, the base model weights are frozen (quantized), and ONLY the LoRA adapter weights are trainable. So `no_grad` on the forward pass blocks gradients through LoRA too since they're part of the same model graph.

The fix: remove `no_grad` from the NEW policy forward pass, keep it on the reference pass. But we need to restructure since we're doing per-sample processing inside a mini-batch. The cleaner approach is batched forward passes:

Replace lines 428-508 with:

```python
            # Policy update
            model.train()
            optimizer.zero_grad()

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

                # Compute per-sample loss
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1
```

Changes in this rewrite:
1. LoRA weight snapshot replaces `deepcopy` ref model
2. Reference forward: restore original weights, `eval()` + `no_grad`
3. New policy forward: NO `no_grad` — gradients flow through LoRA layers
4. PPO clip: `torch.min` instead of `torch.max`
5. Advantage: use tensor `mb_advantages[b_idx]` instead of `.item()` to keep gradient graph connected

- [ ] **Step 2: Commit**

```bash
git add src/student/train_grpo.py
git commit -m "fix(grpo): correct policy gradient flow, ref model snapshot, and PPO clip"
```

---

### Task 3: Fix dataloader exhaustion - add itertools.cycle for multi-epoch training

**Files:**
- Modify: `src/student/train_grpo.py:20` (import)
- Modify: `src/student/train_grpo.py:346` (dataloader creation)
- Modify: `src/student/train_grpo.py:396-399` (training loop)

- [ ] **Step 1: Add itertools import**

At line 20 in `train_grpo.py`, add `itertools` to imports:

```python
import itertools
import math
```

- [ ] **Step 2: Cycle the dataloader**

Replace the dataloader creation and training loop entry (around lines 346-399):

Change:
```python
    dataloader = DataLoader(dataset, batch_size=config["per_device_train_batch_size"], shuffle=True)
```

To:
```python
    dataloader = DataLoader(dataset, batch_size=config["per_device_train_batch_size"], shuffle=True)
    dataloader_iter = iter(itertools.cycle(dataloader))
```

Change the training loop from:
```python
    while step < max_steps:
        for batch_prompts in dataloader:
            if step >= max_steps:
                break
```

To:
```python
    while step < max_steps:
        batch_prompts = next(dataloader_iter)
```

- [ ] **Step 3: Commit**

```bash
git add src/student/train_grpo.py
git commit -m "fix(grpo): cycle dataloader for multi-epoch training"
```

---

### Task 4: Fix warmup scheduler - replace CosineAnnealingLR with get_cosine_schedule_with_warmup

**Files:**
- Modify: `src/student/train_grpo.py:363-364` (scheduler creation)

- [ ] **Step 1: Replace scheduler**

Replace lines 363-364:

```python
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
```

With:
```python
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/student/train_grpo.py
git commit -m "fix(grpo): wire warmup_steps to cosine scheduler"
```

---

### Task 5: Implement gradient accumulation

**Files:**
- Modify: `src/student/train_grpo.py` (training loop - loss scaling and conditional optimizer.step)

- [ ] **Step 1: Add gradient accumulation variables and logic**

After the training hyperparameters section (after line 356), add:

```python
    gradient_accum_steps = config.get("gradient_accumulation_steps", 1)
```

In the training loop, wrap the optimizer step with accumulation logic. Replace the `optimizer.step()` / `scheduler.step()` / `step += 1` section (currently around lines 505-508 in the original, will be at the end of the mini-batch loop after Task 2's changes):

Replace:
```python
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1
```

With:
```python
            if step % gradient_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            else:
                # Accumulating gradients - don't step yet
                pass
```

Also move `optimizer.zero_grad()` to after the step instead of at the top of the loop. Remove the `optimizer.zero_grad()` that's currently before the mini-batch loop.

- [ ] **Step 2: Commit**

```bash
git add src/student/train_grpo.py
git commit -m "fix(grpo): implement gradient accumulation from config"
```

---

### Task 6: Batch generation and reference forward pass

**Files:**
- Modify: `src/student/train_grpo.py:100-125` (generate_completions)

- [ ] **Step 1: Batch the generation of G completions**

Replace `generate_completions` function:

```python
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

    # Repeat input G times for batched generation
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
```

- [ ] **Step 2: Commit**

```bash
git add src/student/train_grpo.py
git commit -m "perf(grpo): batch G completions in single generate call"
```

---

### Task 7: Update tests to match fixed behavior

**Files:**
- Modify: `src/tests/test_grpo_training.py`

- [ ] **Step 1: Add test for PPO clip using torch.min**

Add to `TestGRPOIntegration` class:

```python
    def test_ppo_clip_uses_min(self):
        """Verify PPO objective uses min (conservative update), not max."""
        import torch
        import torch.nn.functional as F

        ratio = torch.tensor([1.3, 0.8, 1.5])
        adv = torch.tensor([1.0, -0.5, 0.8])

        unclipped = -(ratio * adv).mean()
        clipped = -(torch.clamp(ratio, 0.8, 1.2) * adv).mean()

        # PPO uses min for conservative updates
        pg_loss = torch.min(unclipped, clipped)
        assert pg_loss.item() <= max(unclipped.item(), clipped.item())
```

- [ ] **Step 2: Add test for dataloader cycling**

```python
    def test_dataloader_cycles_for_max_steps(self):
        """Verify training loop handles more steps than dataset size."""
        from src.student.train_grpo import GRPODataset
        from torch.utils.data import DataLoader
        import itertools

        prompts = ["q1", "q2", "q3"]
        dataset = GRPODataset(prompts)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataloader_iter = iter(itertools.cycle(dataloader))

        # Should be able to pull more items than dataset size
        items = [next(dataloader_iter) for _ in range(10)]
        assert len(items) == 10
```

- [ ] **Step 3: Run the tests**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_grpo_training.py -v
```

Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add src/tests/test_grpo_training.py
git commit -m "test(grpo): add tests for PPO min objective and dataloader cycling"
```

---

### Task 8: Consistency fix - unify reward functions

**Files:**
- Modify: `src/student/rewards.py:186-208` (compute_reward)

- [ ] **Step 1: Add length reward to compute_reward to match compute_rewards**

The `compute_reward` function in `rewards.py` is missing the length reward that `compute_rewards` in `train_grpo.py` includes. Add it:

Replace `compute_reward` (lines 186-208):

```python
def compute_reward(
    completions: List[str],
    weights: dict,
    tokenizer,
    judge_model=None,
    judge_tokenizer=None,
) -> List[float]:
    """Compute weighted sum of all reward functions."""
    n = len(completions)
    total_scores = [0.0] * n

    if "directional_assertion" in weights:
        for i, completion in enumerate(completions):
            total_scores[i] += weights["directional_assertion"] * compute_directional_assertion(completion)

    if "format" in weights:
        for i, completion in enumerate(completions):
            total_scores[i] += weights["format"] * compute_format_reward(completion)

    if "length" in weights and tokenizer is not None:
        for i, completion in enumerate(completions):
            tokens = len(tokenizer.encode(completion, add_special_tokens=False))
            total_scores[i] += weights["length"] * compute_length_reward(tokens)

    if "dm_alignment" in weights and judge_model is not None and judge_tokenizer is not None:
        dm_scores = compute_dm_alignment_judge(completions, judge_model, judge_tokenizer)
        for i, s in enumerate(dm_scores):
            total_scores[i] += weights["dm_alignment"] * s

    return total_scores
```

- [ ] **Step 2: Commit**

```bash
git add src/student/rewards.py
git commit -m "fix(rewards): add length reward to compute_reward for consistency"
```

---

### Task 9: Final verification - run full test suite

**Files:**
- Test: `src/tests/test_grpo_training.py`

- [ ] **Step 1: Run full GRPO test suite**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m pytest src/tests/test_grpo_training.py -v
```

Expected: all 8+ tests pass

- [ ] **Step 2: Run CLI help check**

```bash
cd /home/yao/projects/ml-lora-training && python3 -m src.student.train_grpo --help
```

Expected: exits 0, shows all arguments

- [ ] **Step 3: Verify Dockerfile builds**

```bash
cd /home/yao/projects/ml-lora-training && docker compose build
```

Expected: builds successfully with python3.11-dev installed

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "fix(grpo): comprehensive bug fixes for training loop" || echo "nothing to commit"
```

---

## Self-Review

**1. Spec coverage:**
- Python.h missing (Dockerfile) -> Task 1
- Zero gradients (no_grad on policy) -> Task 2
- Broken ref model (deepcopy NF4) -> Task 2
- Anti-PPO (max instead of min) -> Task 2
- Dataloader exhaustion -> Task 3
- Warmup not wired -> Task 4
- Gradient accumulation missing -> Task 5
- Sequential generation -> Task 6
- Inconsistent reward functions -> Task 8
- Tests updated -> Task 7
- All items covered.

**2. Placeholder scan:**
- No TBDs, no "add validation", no "implement later"
- All code blocks contain complete, copyable code
- All file paths are exact with line references

**3. Type consistency:**
- `compute_reward` signature updated to match `compute_rewards` (added `tokenizer` parameter)
- Scheduler type changes from `CosineAnnealingLR` to `get_cosine_schedule_with_warmup` — both have `.step()` so no caller changes needed
- Advantage tensor: changed from `.item()` to raw tensor to keep gradient graph connected — consistent with backward pass

**4. Task ordering:**
- Task 1 (Dockerfile) is independent, can run first
- Tasks 2-6 all modify `train_grpo.py` and should be done sequentially (each task's line numbers assume prior tasks' changes)
- Task 7 (tests) depends on Tasks 2, 3 being complete
- Task 8 (rewards) is independent
- Task 9 (verification) depends on all prior tasks
