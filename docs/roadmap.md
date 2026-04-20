This updated plan incorporates the hardware specificities of the RTX 5090 (32GB) and the software optimizations required for Unsloth to ensure a 100% completion rate without OOM (Out of Memory) errors.
------------------------------
## 🏗️ Updated Project Plan: DM-Align Qwen 27B## 1. Model Strategy

* Base Model: Qwen 2.5/3.5 27B (Instruct version).
* Teacher Mode: Q4_K_M GGUF via llama-cpp-python.
* Student Mode: 4-bit QLoRA via Unsloth (Auto-quantized from BF16).
* Objective: Distill Dialectical Materialist (DM) reasoning chains into the LoRA adapter.

## 2. Phase 1: Synthetic Data Generation (The Teacher)

* Script: src/teacher_generate.py
* Method: Zero-shot CoT (Chain of Thought).
* Dataset Size: 1,000–1,500 samples.
* Structure:
* Input: Generic socio-economic question.
   * Thought: Internal DM analysis (Material conditions -> Contradiction -> Superstructure).
   * Output: Final synthesized DM response.
* Storage: data/distillation_dataset.jsonl in ShareGPT or ChatML format.

## 3. Phase 2: Optimized Training (The Student)

* Script: src/student_train.py
* Framework: Unsloth + PyTorch.
* VRAM Management (Critical for 32GB):
* max_seq_length = 4096 (Safest limit for 27B on 32GB).
   * load_in_4bit = True (Using Unsloth's 4-bit loading).
   * gradient_checkpointing = "unsloth".
* LoRA Hyperparameters:
* r = 32, lora_alpha = 32.
   * target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"].
* Optimization: Use adamw_8bit or paged_adamw_8bit.

## 4. Phase 3: Validation & Export

* Manual Spot-Check: Test on 20 "Liberal-Trap" questions (e.g., "Is universal basic income the ultimate solution to poverty?").
* Export: Save as LoRA Adapter (.safetensors).
* Merge: Use Unsloth’s model.save_pretrained_gguf to export a merged GGUF for local deployment (requires ~64GB System RAM).

------------------------------
## 📊 Technical Requirements Matrix

| Resource | Requirement | Note |
|---|---|---|
| GPU | NVIDIA RTX 5090 (32GB) | Essential for 27B QLoRA headroom. |
| System RAM | 64GB+ | Necessary for the GGUF merge/export phase. |
| Storage | 100GB SSD | Weights (~18GB) + Dataset + Checkpoints. |
| Library | unsloth[colab-new] | Even for local, use the latest Unsloth build. |

------------------------------
## ⚡ Execution Roadmap## Step 1: Environment Hardening

* Install CUDA 12.x and PyTorch 2.4+.
* Install Unsloth from source to ensure 27B support.
* Verify llama-cpp-python has CUBLAS enabled for Teacher inference.

## Step 2: Teacher Loop (Estimated: 4–6 hours)

* Run generation in batches.
* Implement a Retry Logic in the script: if the Teacher fails to include "Material Conditions" in the response, discard and regenerate.

## Step 3: Student Training (Estimated: 2–3 hours)

* Execute training with gradient_accumulation_steps=4.
* Monitor VRAM via nvidia-smi. Target: 29GB/32GB peak usage.

------------------------------
## 🚀 Ready to begin?
I can now provide the specific code for either:

   1. The teacher_generate.py script with the DM validation logic.
   2. The student_train.py Unsloth configuration for the RTX 5090.

Which script should I generate first?

