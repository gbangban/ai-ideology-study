# Design Brief: Fixing DM Alignment Training After GRPO Failure

**Date**: 2026-06-04
**Status**: Seeking feedback before implementation
**Author**: AI agent + project owner

---

## Problem Statement

After three attempts to improve a DM (Dialectical Materialism) aligned language model beyond SFT, GRPO training has failed to produce usable results:

- **GRPO Run 1** (step 250): No measurable improvement. An initial eval reported 0.0% HumanEval due to a broken eval pipeline; the corrected run shows 71.3% (within noise of 71.9% baseline). Code generation was preserved throughout.
- **GRPO Run 2** (step 500, current): DM keyword reward saturated at 0.94, but directional commitment reward stalled at 0.0 and mechanism commitment at -0.5. The model learned to insert DM vocabulary without changing its fundamental hedging behavior.

The SFT baseline is already functional: +38pp on Corr2Cause (formal causal inference), neutral on HumanEval/IFEval/GPQA/MMLU, but -4 to -13pp on EconCausal (applied economic causal reasoning). The dominant failure mode is `+ -> mixed` hedging the model learns to be skeptical of straightforward positive causal effects.

**Goal**: Improve DM reasoning quality without breaking existing capabilities (code generation, instruction following, causal sign prediction).

---

## Current Architecture

```
Pipeline: Questions (1,500 AI-generated)
    -> Teacher answers via Unsloth Studio (Qwen3.5-27B, DM system prompts)
    -> Studio SFT with QLoRA (Qwen3.5-9B student, NF4)
    -> Export LoRA adapter -> merge to BF16
    -> GRPO training (custom implementation, NOT Studio)
    -> Evaluation + GGUF export
```

**Hardware**: RTX 5090 (32GB), Docker Desktop on Windows with WSL2 bridge.

**Models**:
- Teacher: Unsloth/Qwen3.5-27B (data generation only)
- Student: Qwen/Qwen3.5-9B Instruct, SFT-merged BF16 checkpoint
- Judge: Qwen/Qwen3.5-4B via SG-Lang HTTP server (separate container)

---

## Why GRPO Failed (Root Cause Analysis)

### Reward hacking
The DM keyword alignment reward requires only 2/3 pattern categories to score 1.0. The model learned to sprinkle keywords like "structural" and "accumulation" into responses without any change in reasoning quality. This is easy pattern matching, not genuine alignment.

### Flat gradient zone
The directional assertion reward uses `(positive_count - hedging_count)` scoring. When the model generates equal amounts of committed and hedging language, the score is exactly 0.0 with no gradient signal. The model converged to this flat zone and stopped learning.

### Reward weight imbalance
DM keyword reward at 0.45 weight saturated quickly (0.94), consuming most of the learning signal. Directional assertion at 0.30 weight with 0.0 output contributed nothing to the advantage computation. Mechanism commitment at 0.25 weight actively dragged total reward down at -0.5.

### No code collapse (eval bug)
An initial eval incorrectly reported 0.0% HumanEval. The corrected run at `grpo/bf16/.../results_2026-05-28T12-38-27.794259.json` shows 71.3% — code generation was preserved. The real problem is GRPO failed to improve DM reasoning quality while preserving all existing capabilities.

### Training set mismatch
GRPO trains on `data/raw/questions.json` — 1,500 DM-oriented structural analysis questions. The hedging problem manifests on EconCausal-style causal direction questions ("does X increase Y?"), which are absent from the training set. GRPO cannot learn "don't hedge on causal direction" from questions that don't ask about causal direction. This is a fundamental data mismatch, not just a reward engineering problem.

---

## Proposed Approaches

### Approach 1: Fix GRPO Rewards (Incremental)

Rewrite the reward functions and re-run GRPO:

**Reward changes:**
- DM reward: Penalize responses containing both DM keywords AND hedging patterns. Forces the model to either commit directionally with DM framing or receive no reward.
- Directional reward: Replace ratio scoring with binary gate: `1.0 if hedging_count == 0 AND positive_count > 0, else -0.5`. Eliminates the 0.0 flat zone.
- Mechanism reward: Remove temporarily until directional learning stabilizes.
- SG-Lang judge: Add as 40% weight for DM quality assessment. Judge evaluates genuine structural analysis vs. keyword stuffing.
- Code preservation reward: Detect coding tasks and reward code output to prevent collapse.

**New weights:**
```
directional_assertion: 0.35  (binary, no flat zone)
dm_alignment:          0.30  (rule-based + hedging penalty)
dm_judge:              0.25  (SG-Lang LLM judge)
code_preservation:     0.10  (binary code detection)
```

**Trade-offs:**
- Pro: Fastest to implement, builds on existing training infrastructure
- Pro: Can validate with 50-100 step test runs before committing to 500
- Con: Still GRPO, which has failed twice with different reward configs
- Con: Reward functions remain inherently gameable
- Con: Adding LLM judge increases batch time from ~60s to ~120s

### Approach 2: Rejection Sampling + SFT (Different Paradigm)

Replace GRPO with iterative rejection sampling:

**Process:**
1. Use current SFT model to generate 8-16 responses per question
2. Score each with SG-Lang judge for DM quality AND directional commitment
3. Keep only responses scoring above threshold on BOTH criteria
4. Retrain SFT on filtered dataset
5. Evaluate after each round to catch regressions
6. Repeat for 2-3 rounds

**Why this works better:**
- SFT objective is stable and doesn't collapse code generation when trained on complete high-quality examples
- No reward function to game the model learns from explicit examples of good behavior
- Each round produces an evaluable model, enabling early detection of regressions
- VRAM-friendly: no reference model snapshots needed

**Trade-offs:**
- Pro: No reward engineering, no gradient flat zones, proven to work in literature (Online DPO, Rejection Fine-Tuning)
- Pro: Compatible with 32GB GPU constraint
- Pro: Each iteration produces a concrete, evaluable model
- Con: Requires generating and storing filtered datasets between rounds
- Con: Loses online learning benefit of GRPO
- Con: 2-3 rounds needed, each requiring full SFT training

### Approach 3: Teacher-Student DPO with Preference Pairs (Recommended)

Use the 27B teacher model to generate high-quality DM-aligned answers, then train the 9B student via DPO using preference pairs. This leverages existing DPO infrastructure and avoids reward engineering entirely.

**Process:**

**Phase 1: Generate preference pairs**
1. For each of the 1,500 questions, generate a "chosen" response using the 27B teacher with DM system prompts
2. Generate a "rejected" response using the current SFT student model (which exhibits hedging)
3. Quality-filter: verify the teacher's response shows both DM reasoning AND directional commitment
4. Verify the pair is actually a preference (teacher response is meaningfully different from student response)

**Phase 2: DPO training**
1. Train DPO on preference pairs using existing `train_dpo.py` pipeline
2. Use conservative hyperparameters: beta=0.1, LR=5e-7, max_steps=300
3. Evaluate after every 50 steps on held-out questions to catch regressions early

**Phase 3: Capability preservation**
1. Mix in "neutral" preference pairs where chosen = rejected (identity pairs) for non-DM questions to preserve baseline behavior
2. Include coding task pairs where the chosen response includes code to prevent HumanEval collapse
3. Evaluate on full benchmark suite after training

**Preference pair generation:**
```
For each question:
  chosen = teacher.generate(question, DM_system_prompt)  # 27B with DM prompts
  rejected = student.generate(question)                   # current SFT model (hedges)

  If judge(chosen) >= 0.7 AND has_directional_commitment(chosen):
      If similarity(chosen, rejected) < 0.8:  # meaningfully different
          save pair
```

**DPO config:**
```
base_model: SFT-merged BF16 checkpoint (current best model)
beta: 0.1
learning_rate: 5e-7
max_steps: 300  (reduced from 500 for safety)
warmup_steps: 50
lr_scheduler: cosine
batch_size: 1, gradient_accumulation: 4
```

**Trade-offs:**
- Pro: DPO is mathematically more stable than GRPO (no advantage estimation, no reward model)
- Pro: Preference signal is explicit and human-interpretable: "teacher answer > student answer"
- Pro: Leverages existing DPO pipeline (`train_dpo.py` already works)
- Pro: No reward function to game
- Pro: Teacher model (27B) is more capable than any reward model for quality assessment
- Con: Requires teacher model generation (27B is slower, but data generation is one-time cost)
- Con: Need to verify teacher actually produces directional commitment (may need prompt engineering)
- Con: Current DPO pair generator uses "stub" rejected responses, not real model output needs rewrite

---

## Critical Open Questions

### 1. Does the teacher model produce directional commitment?
**AUDITED (2026-06-04)**: 250 teacher answers from `data/processed/batch_00000.json` audited using hedging/commitment patterns from `src/student/rewards.py`. Results: 29.6% committed, 4.0% hedging, 66.4% neutral (no patterns matched).

**Findings**: The patterns are unreliable for structural analysis answers. "it depends" in "capitalism depends on X" is not hedging. "increases" in "cost increases" is not causal commitment. More critically, the SFT dataset questions ask for structural analysis ("why is income inequality increasing?"), not causal direction ("does minimum wage increase unemployment?"). The teacher's answers on these questions are irrelevant to the hedging problem because the questions don't require committing to a causal direction.

**Implication**: To fix EconCausal hedging via DPO, we need preference pairs on EconCausal-style questions, not DM analysis questions. The SFT dataset answers cannot serve as chosen responses for anti-hedging pairs.

### 2. What's the right training data mix?
The 1,500 questions are all DM-oriented. Training exclusively on these didn't break code generation (corrected eval shows 71.3% HumanEval), but also didn't improve DM reasoning. Additionally, the SFT dataset questions don't address the hedging failure mode because they're not causal direction questions. Options:
- A) Include non-DM questions with identity pairs (chosen = rejected) to preserve baseline behavior
- B) Include coding tasks with code-containing chosen responses
- C) Use a smaller subset of DM questions (500-700) and accept narrower alignment
- D) Train only on DM questions but add a KL penalty to the SFT model to prevent drift
- E) **[NEW]** Include EconCausal-style causal direction questions with teacher-chosen/student-rejected pairs to directly target hedging

### 3. How many DPO steps before catastrophic drift?
DPO is more stable than GRPO but can still cause catastrophic forgetting. Recommendation: start with 300 steps and evaluate at 50-step intervals. Stop early if HumanEval drops below 60%.

### 4. Should we use the SG-Lang judge for pair quality filtering?
The judge (Qwen3.5-4B via SG-Lang) could filter teacher responses before creating pairs. This adds quality control but introduces judge bias. Alternative: use rule-based filters for directional commitment (no hedging patterns) which are faster but less nuanced.

---

## Implementation Plan (If Approach 3 Selected)

### New code to write:
1. `src/teacher/generate_teacher_responses.py` - Generate chosen responses using 27B teacher via Studio API or direct inference
2. `src/teacher/generate_student_rejections.py` - Generate rejected responses using current SFT model
3. `src/teacher/filter_preference_pairs.py` - Quality filter pairs using judge + rule-based checks
4. Modified `src/teacher/generate_dpo_pairs.py` - Replace stub rejections with real student output

### Existing code to modify:
1. `src/student/train_dpo.py` - Add evaluation checkpoint every 50 steps, add W&B logging
2. `src/student/dpo_config.py` - Update for new training run

### Evaluation gates:
- Before DPO: audit 50 teacher responses for directional commitment
- During DPO: evaluate at steps 50, 100, 150, 200, 250, 300 on Corr2Cause + HumanEval
- After DPO: full benchmark suite (HumanEval, EconCausal, Corr2Cause, IFEval, GPQA, MMLU)

### Stop conditions:
- HumanEval drops below 60% at any checkpoint
- Corr2Cause drops below 60% (regression from SFT's 74.6%)
- Loss doesn't decrease after 100 steps

---

## Alternative: Hybrid Approach 3 + 2

If DPO alone doesn't achieve sufficient improvement, combine with one round of rejection sampling:

1. DPO training (Approach 3) to establish preference signal
2. Generate responses with DPO-trained model
3. Filter top responses using judge
4. One round of SFT on filtered data to reinforce quality

This gives the stability of DPO plus the explicit quality filtering of rejection sampling, at the cost of an extra training round.

---

## Resource Requirements

| Phase | GPU Time | VRAM | Notes |
|-------|----------|------|-------|
| Teacher response generation | ~4h | 32GB | 1,500 questions at 27B, batch generation via Studio |
| Student rejection generation | ~1h | 8GB | SFT model is small, fast inference |
| DPO training (300 steps) | ~5h | 16GB | NF4 quantization, batch size 1x4 accum |
| Evaluation (full suite) | ~2h | 12GB | BF16 merged model |
| **Total** | **~12h** | | Single GPU, sequential |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Teacher hedges like student | High | Critical - DPO learns hedging | AUDITED: patterns unreliable for structural analysis. For EconCausal questions, need new teacher generation with directional commitment prompt |
| DPO causes capability drift | Medium | High | Include identity pairs for non-DM questions; evaluate at 50-step intervals |
| DPO doesn't improve Corr2Cause | Medium | Medium | Fall back to rejection sampling |
| Insufficient pair quality | Low | Medium | Use judge for filtering; increase pair count |
| VRAM OOM during training | Low | Low | NF4 quantization already handles this |

---

## Decision Points for Feedback

1. **Approach selection**: Approach 3 (DPO with teacher preferences) vs. Approach 2 (rejection sampling) vs. hybrid?
2. **Data mix**: Include non-DM identity pairs for capability preservation, or accept narrow DM-only training?
3. **Teacher audit**: Should we audit teacher responses before building pairs, or build pairs and filter aggressively?
4. **Step budget**: 300 steps with early stopping, or 500 steps with more frequent evaluation?
5. **Judge usage**: Use SG-Lang judge for pair filtering, or rely on rule-based filters only?
