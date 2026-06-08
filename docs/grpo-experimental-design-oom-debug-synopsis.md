# GRPO Experimental Design + OOM Debugging Synopsis

**Generated:** 2026-06-08
**Purpose:** Enable an external party to debug why the latest GRPO v3/v4 runs OOM while prior successful runs (v1 at 500 steps, v2 at 500 steps) completed without issue.

---

## 1. Hardware and Environment

- **GPU:** RTX 5090, 32GB VRAM
- **Container:** Docker Desktop on Windows, WSL2 bridge via `scripts/ddk`
- **Python:** 3.11, PyTorch 2.7.0, Unsloth 2026.4.6
- **Quantization:** NF4 via BitsAndBytes (Unsloth FastLanguageModel)
- **Student model:** Qwen/Qwen3.5-9B Instruct, architecture `Qwen3_5ForConditionalGeneration`
- **Base checkpoint for v3/v4:** `checkpoints/merged/cold_start_merged` (BF16, 4 shards, ~17.9GB on disk)
  - This is the SFT checkpoint merged with cold-start SFT LoRA adapter (5 epochs, 200 tagged demos)

---

## 2. Training Loop Architecture (All Versions Share Same Pattern)

The custom GRPO loop in `train_grpo_v3.py` and `train_grpo_v4.py` follows this per-step sequence:

```
1. Generate G completions (batched, single model.generate() call)
2. Compute rewards (CPU-side, regex-based)
3. Compute advantages (group-relative normalization)
4. Snapshot LoRA weights for reference policy
5. For each of G samples (SEQUENTIAL loop):
   a. Tokenize prompt + completion (individual tokenizer call)
   b. Restore ref LoRA weights, eval() + no_grad forward pass
   c. Restore train LoRA weights, train() forward pass WITH gradients
   d. Compute PPO clipped loss + KL penalty per sample
   e. Backward pass, accumulate gradients
   f. Delete all intermediate tensors
6. Gradient clip + optimizer step (every gradient_accumulation_steps)
```

**Critical VRAM note:** Step 5 is a SEQUENTIAL loop over G samples, each doing TWO full forward passes through the 9B model. This is the VRAM bottleneck.

---

## 3. Successful Runs (Baseline for Comparison)

### GRPO v1 (`grpo_adapter` directory, 500 steps completed)

- **Config:** `grpo_g=8`, `max_completion_length=512`, `per_device_train_batch_size=1`
- **Dataset:** `data/raw/questions.json` (1,500 DM-oriented questions)
- **Rewards:** 3 keyword-based (dm_alignment 0.45, directional_assertion 0.30, mechanism_commitment 0.25)
- **Judge:** Disabled (`judge_backend="disabled"`)
- **Result:** Completed 500 steps. VRAM stable at ~12GB. No OOM.
- **Evidence:** `checkpoints/lora_adapters/grpo_adapter/training_log.csv` has 500 rows, VRAM column shows 12.0-12.2GB throughout.
- **Training outcome:** No measurable improvement on benchmarks. DM keyword reward saturated, directional reward stalled at 0.0, mechanism commitment at -0.5.

### GRPO v2 (`grpo_adapter_v2` directory, 500 steps completed)

- **Config:** `grpo_g=8`, `max_completion_length=512`, `per_device_train_batch_size=1`
- **Dataset:** Same 1,500 questions
- **Rewards:** Same 3 functions, but ASYMMETRIC directional assertion (hedging is costly, not neutral)
- **Judge:** Disabled
- **Result:** Completed 500 steps. VRAM stable at ~9GB. No OOM.
- **Evidence:** `checkpoints/lora_adapters/grpo_adapter_v2/training_log.csv` has 500 rows, VRAM column shows 8.9-9.1GB throughout.
- **Training outcome:** DM reward saturated at 0.94 by end. Directional assertion still near 0.0. Mechanism commitment oscillated around -0.25. Same reward hacking as v1 but with slightly different dynamics.

**Key observation:** Both successful runs used `grpo_g=8` with 512-token completions and never exceeded 12GB VRAM. The per-sample sequential loop worked fine with these parameters.

---

## 4. v3/v4 Configuration Changes (OOM Runs)

### What changed from v2 to v3/v4:

| Parameter | v1/v2 (successful) | v3/v4 (OOM) | Impact |
|---|---|---|---|
| `grpo_g` | 8 | **2** | Fewer completions per step, should REDUCE VRAM |
| `max_steps` | 500 | **1,000** | No per-step VRAM impact |
| `max_completion_length` | 512 | 512 | Same |
| `base_model` | SFT checkpoint (NF4 load) | **Merged cold-start BF16 checkpoint** | CRITICAL DIFFERENCE |
| Dataset | 1,500 questions (short prompts) | ~8,300 prompts (EconCausal + Corr2Cause, LONGER prompts) | Longer context = more VRAM per forward pass |
| Rewards | 3 keyword functions (CPU) | Correctness-based + process rewards (CPU) | No VRAM impact |

### The OOM Problem

- **v3 directory:** `checkpoints/lora_adapters/grpo_v3/` exists but `training_log.csv` is **0 bytes** — training failed before step 1 completed
- **v4 directory:** `checkpoints/lora_adapters/grpo_v4/` does not exist — training never started
- No Docker logs, no W&B traces, no run logs captured the error

### Most Likely Root Cause: Base Model Loading Difference

The successful v1/v2 runs loaded the SFT checkpoint via Unsloth NF4:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

v3/v4 load the **merged cold-start checkpoint** which is a full BF16 model:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="checkpoints/merged/cold_start_merged",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

The merged checkpoint is a full BF16 safetensors model (~17.9GB on disk). When Unsloth loads this with `load_in_4bit=True`, it should quantize to NF4 at load time. However, there are two potential issues:

1. **Unsloth may not re-quantize a merged BF16 checkpoint the same way it quantizes the original.** The original SFT checkpoint is sharded safetensors from Studio's NF4 export. The merged checkpoint is full BF16 produced by `merge_grpo_checkpoint.py`. If Unsloth loads BF16 first then quantizes, there could be a transient memory spike exceeding 32GB.

2. **Longer prompts from EconCausal/Corr2Cause.** The original 1,500 DM questions are relatively short. EconCausal prompts include context passages that could be 500+ tokens. With `max_seq_length=2048`, the attention kv-cache for G=2 completions of 512 tokens each on 2048-length sequences is larger per forward pass.

3. **Per-sample sequential loop without mini-batching.** The v3/v4 code processes samples one at a time in a for-loop, doing two forward passes per sample. With G=2 this is manageable, but if the base model loading uses more memory, the headroom disappears.

---

## 5. VRAM Budget Analysis

### Successful v2 run VRAM (from training_log.csv):
- Base model (NF4): ~6-7GB
- LoRA adapter: ~0.1GB
- KV cache (G=8, seq=2048, completion=512): ~2-3GB
- Gradients (per-sample): ~1-2GB peak
- **Total peak:** ~12GB (measured)

### Expected v3/v4 VRAM:
- Base model (NF4 from BF16 merge): **UNKNOWN** — could be 6-7GB if quantization works, or 18GB if it loads BF16 first
- LoRA adapter: ~0.1GB
- KV cache (G=2, longer prompts): ~1-2GB
- Gradients (per-sample): ~1-2GB peak
- **If NF4 loads correctly:** ~10GB (should be fine)
- **If BF16 loads first:** ~22GB + gradients = potential OOM at ~25-28GB

### The merge script (`scripts/merge_grpo_checkpoint.py`):
- Saves as `bfloat16` (confirmed in `merge_metadata.json`)
- Produces 4 safetensors shards totaling ~17.9GB
- This is a standard HF-compatible BF16 checkpoint

---

## 6. Debugging Steps for External Party

### Step 1: Verify the OOM is real
Run a minimal test to confirm the error:
```bash
python3 -c "
from unsloth import FastLanguageModel
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='checkpoints/merged/cold_start_merged',
    max_seq_length=2048,
    load_in_4bit=True,
)
print('Model loaded, VRAM:', torch.cuda.memory_allocated() / 1e9, 'GB')
print('Model device:', model.device)
# Check if actually NF4
for name, param in list(model.named_parameters())[:3]:
    print(f'  {name}: dtype={param.dtype}')
"
```

If model loads at ~6-7GB with NF4 dtype, the OOM is elsewhere. If it loads at ~18GB with bf16, the quantization is not applying.

### Step 2: Check if the BF16 merge is the problem
Compare loading the original SFT checkpoint vs the merged checkpoint:
```bash
python3 -c "
from unsloth import FastLanguageModel
import torch

# Original SFT checkpoint (known to work)
m1, _ = FastLanguageModel.from_pretrained(
    model_name='/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330',
    max_seq_length=2048, load_in_4bit=True)
print('SFT checkpoint VRAM:', torch.cuda.memory_allocated() / 1e9, 'GB')
del m1; torch.cuda.empty_cache()

# Merged cold-start checkpoint (suspected OOM source)
m2, _ = FastLanguageModel.from_pretrained(
    model_name='checkpoints/merged/cold_start_merged',
    max_seq_length=2048, load_in_4bit=True)
print('Merged checkpoint VRAM:', torch.cuda.memory_allocated() / 1e9, 'GB')
"
```

### Step 3: If merge is fine, check per-step VRAM with long prompts
```bash
python3 -c "
from unsloth import FastLanguageModel
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='checkpoints/merged/cold_start_merged',
    max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
model = FastLanguageModel.for_training(model)

# Simulate a long EconCausal prompt
prompt = 'Answer this economic causal question: ' + 'X causes Y. ' * 200
chat = [{'role': 'user', 'content': prompt}]
text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to(model.device)
print('Prompt tokens:', inputs['input_ids'].shape[1])

# Two forward passes (ref + new)
model.eval()
with torch.no_grad():
    out1 = model(**inputs, use_cache=False)
print('After ref forward:', torch.cuda.memory_allocated() / 1e9, 'GB')
del out1

model.train()
out2 = model(**inputs, use_cache=False)
print('After new forward:', torch.cuda.memory_allocated() / 1e9, 'GB')
del out2
"
```

### Step 4: Check actual error from container logs
```bash
ddk logs ml-training --tail=200 2>/dev/null
# or if ddk not available:
docker logs ml-training --tail=200
```

---

## 7. Known Code Issues That Could Contribute to OOM

### Issue A: No tensor cleanup between ref and new forward passes
In `train_grpo_v3.py:324-341` and `train_grpo_v4.py:394-410`:
```python
# Reference forward
for name, param in model.named_parameters():
    if name in lora_weights:
        param.data.copy_(lora_weights[name])
model.eval()
with torch.no_grad():
    ref_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
ref_logits = ref_outputs.logits[0]
del ref_outputs          # <-- deletes outputs but ref_logits stays in memory

model.train()
new_outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
new_logits = new_outputs.logits[0]
```

At this point, BOTH `ref_logits` and `new_logits` are in GPU memory simultaneously. For a 9B model with 2048 sequence length and vocabulary ~152K, each logits tensor is ~2048 x 152K x 4 bytes = ~1.2GB. Two of them = ~2.4GB peak. This is manageable but adds up.

### Issue B: LoRA weight snapshot could be large
```python
lora_weights = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        lora_weights[name] = param.data.clone()
```

With r=16 and 7 target modules on a 9B model, the LoRA weights are small (~0.1GB). Not a concern.

### Issue C: Per-sample tokenization holds all texts in memory
```python
all_texts = []
all_prompt_lengths = []
for i in range(n_samples):
    all_texts.append(all_prompt_texts[i] + all_completions[i])
    prompt_enc = tokenizer(all_prompt_texts[i], add_special_tokens=False)
    all_prompt_lengths.append(len(prompt_enc["input_ids"]))
```

With G=2 and 512-token completions, this is negligible. Even with G=8 it's minimal (Python strings, not tensors).

### Issue D: The `_strip_vision_config` function modifies config.json in-place
This runs before model load. If the merged checkpoint's config.json already has vision config stripped, this is a no-op. If it doesn't, the function modifies the file on disk, which could cause issues on subsequent runs if the file is corrupted.

---

## 8. Experimental Design Summary (for Context)

### Problem being addressed
SFT+DPO on DM-aligned data caused a hedging regression on EconCausal (-4 to -13pp) while improving Corr2Cause (+38pp). The model replaces correct directional answers (`+`) with `mixed` hedging.

### v3 design (control condition)
- Outcome rewards only, flat advantage, free-form output
- Correctness-based rewards from real benchmark ground truth (EconCausal `answer`, Corr2Cause `relation`)
- Dataset: EconCausal (2,943 prompts) + Corr2Cause (5,000 sampled) + synthetic (360 prompts) = ~8,300
- `grpo_g=2`, 1,000 steps, `beta=0.1`

### v4 design (treatment condition)
- Dual advantage: `A_traj` (outcome) + `A_MR` (process), combined with `alpha=0.5`
- Process rewards: planning (success-conditional), commitment (anti-hedging), reflection (success-conditional), monitor, format penalty
- Tagged output: `<planning>`, `<commitment>`, `<reflection>`, `<monitor>`
- `lambda_kl=0.01` (weaker than v3's `beta=0.1`)
- `clip_epsilon=0.2` (standard PPO)
- Same dataset, same base checkpoint

### Cold-start SFT (prerequisite for both)
- 200 prompts, teacher (Qwen3.5-27B) generates 3 tagged completions each = 600 samples
- 5-epoch SFT to teach tag format
- Merged into base model via `merge_grpo_checkpoint.py`
- Cold-start completed successfully (checkpoints at steps 50, 100, 150, 200)
- Merge completed successfully (BF16, 4 shards, 145.8 seconds)

---

## 9. Hypothesis for OOM

**Primary hypothesis:** The merged BF16 checkpoint loads differently through Unsloth than the original Studio NF4 checkpoint. Unsloth's `FastLanguageModel.from_pretrained` with `load_in_4bit=True` may not re-quantize a merged BF16 safetensors model the same way it quantizes the original. This could cause the base model to consume ~18GB instead of ~7GB, leaving insufficient headroom for the dual forward passes and gradients.

**Secondary hypothesis:** Longer EconCausal prompts (context passages of 500+ tokens) increase the attention KV-cache and intermediate activation memory per forward pass, pushing a tight VRAM budget over the edge even with NF4 quantization.

**Tertiary hypothesis:** A transient memory spike during LoRA application (`FastLanguageModel.get_peft_model`) on top of a BF16-loaded model exceeds 32GB before garbage collection.

---

## 10. Recommended Fixes

1. **Verify NF4 loading of merged checkpoint** (Step 1 above). If it loads as BF16, the fix is to re-export the merged checkpoint in a format Unsloth can quantize, or load the merge differently.

2. **If NF4 loads fine, reduce `max_seq_length`** from 2048 to 1024. EconCausal prompts + 512-token completions should fit in 1024.

3. **Add `torch.cuda.empty_cache()` calls** between ref forward and new forward passes in the per-sample loop.

4. **Delete `ref_logits` before the new forward pass:**
   ```python
   ref_logits = ref_outputs.logits[0]
   del ref_outputs
   # ... later, before new forward:
   del ref_logits  # move deletion earlier
   ```
   Actually, ref_logits is needed for the loss computation, so this requires restructuring to compute per-sample loss immediately rather than storing all logits.

5. **Reduce `grpo_g` to 1** temporarily to verify it's a VRAM issue (if G=1 works, it confirms the per-sample loop is the bottleneck).

6. **Use `gradient_accumulation_steps` more effectively.** Currently set to 4, but with `per_device_train_batch_size=1` and `grpo_g=2`, each step processes only 2 samples. The gradient accumulation is over STEPS, not samples, so it accumulates 2-sample gradients 4 times before stepping. This is correct but means the optimizer state is larger relative to the effective batch.

7. **Add VRAM monitoring at each phase** of the training loop to identify exactly where memory peaks:
   ```python
   print(f'After generation: {torch.cuda.memory_allocated()/1e9:.1f}GB')
   print(f'After ref forward: {torch.cuda.memory_allocated()/1e9:.1f}GB')
   print(f'After new forward: {torch.cuda.memory_allocated()/1e9:.1f}GB')
   ```

---

## 11. File Map

| File | Role |
|---|---|
| `src/student/train_grpo.py` | v2 training (successful, 500 steps, g=8) |
| `src/student/train_grpo_v3.py` | v3 training (OOM, g=2, correctness rewards) |
| `src/student/train_grpo_v4.py` | v4 training (OOM, g=2, dual advantage) |
| `src/student/grpo_config.py` | v2 config (g=8, 500 steps, keyword rewards) |
| `src/student/grpo_config_v4.py` | v3/v4 configs (g=2, 1,000 steps) |
| `src/student/rewards.py` | v2 reward functions (keyword-based) |
| `src/student/rewards_v3v4.py` | v3/v4 reward functions (correctness-based + process) |
| `scripts/merge_grpo_checkpoint.py` | CPU-only BF16 merge script |
| `checkpoints/lora_adapters/grpo_adapter/training_log.csv` | v1 training metrics (500 steps, 12GB VRAM) |
| `checkpoints/lora_adapters/grpo_adapter_v2/training_log.csv` | v2 training metrics (500 steps, 9GB VRAM) |
| `checkpoints/lora_adapters/grpo_v3/training_log.csv` | v3 training metrics (0 bytes - OOM) |
| `checkpoints/merged/cold_start_merged/` | Merged cold-start BF16 checkpoint (17.9GB) |
| `checkpoints/lora_adapters/cold_start_sft/` | Cold-start SFT LoRA adapter (5 epochs) |
