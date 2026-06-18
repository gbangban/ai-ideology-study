# Design Brief: Post-SFT Alignment Strategy

**Date**: 2026-06-04
**Status**: Seeking feedback before implementation
**References**: `docs/Experimental Design.md` v3.0

---

## 1. Context

Per the experimental design (§1 Goal), the project aims to shift the Qwen3.5-9B student model's **default analytical frame** for social, economic, and political phenomena to Dialectical Materialism. The model should spontaneously identify material interests, trace structural constraints, and reach different conclusions from liberal-reformist analysis on neutral prompts.

SFT training is complete. The model was trained on 1,500 AI-generated, quality-filtered questions with teacher (Qwen3.5-27B) DM-aligned answers. The SFT model shows:

| Metric | Baseline | SFT | Delta | Interpretation |
|--------|----------|-----|-------|---------------|
| Corr2Cause | 36.3% | 74.6% | **+38.3pp** | Structural reasoning transfers to formal causal inference |
| HumanEval | 70.73% | 71.9% | ~0pp | Coding preserved (per §5.6 regression tests) |
| EconCausal Task1 Econ | 60.3% | 47.9% | **-12.4pp** | DM epistemic skepticism corrupts applied economics |
| EconCausal Task1 Finance | 56.5% | 43.0% | **-13.5pp** | Same hedging failure mode |
| EconCausal Task2 | 69.7% | 65.9% | **-3.9pp** | Context-dependent tasks less affected |
| EconCausal Task3 | 22.2% | 11.4% | **-10.8pp** | Misinformation-robust tasks worst hit |

The dominant SFT failure mode is `+ -> mixed` hedging: the model converts correct positive causal predictions into ambiguous "mixed" answers. This affects 52-64% of EconCausal regressions across tasks. The model internalized DM skepticism of simple causality and applies it to empirical economics where definitive directional effects are the norm.

**Per §13.7 Design Implication #7**: "DPO must address the `+ -> mixed` hedging bias. The DPO training phase needs explicit chosen/rejected pairs that reward definitive causal predictions when warranted, counteracting the hedging tendency learned during SFT."

---

## 2. Problem: GRPO Failed to Improve DM Reasoning

The experimental design (§9.3) called for GRPO as the post-SFT alignment step. Two GRPO runs have been executed:

### GRPO Run 1 (step 250): No behavioral change, code preserved

An initial eval reported 0.0% HumanEval, but this was a broken eval pipeline — the corrected run shows **71.3%** HumanEval (vs 71.9% baseline), within noise. Code generation is fully preserved. The model showed no measurable improvement on DM reasoning tasks either. The reward function provided insufficient gradient signal for meaningful alignment.

### GRPO Run 2 (step 500, current): Reward hacking without behavioral change

Final step metrics:
- `dm_reward = 0.94` (weight 0.45): Model learned to insert DM keywords
- `dir_reward = 0.0` (weight 0.30): No directional commitment learned
- `mech_reward = -0.5` (weight 0.25): Mechanisms named but paired with hedging
- `avg_reward = 0.297`

The model learned DM vocabulary without changing reasoning. It generates "structural factors shape outcomes, but the relationship is context-dependent" - DM keywords with hedging intact. The directional assertion reward's scoring created a flat gradient zone at 0.0 where the model converged and stopped learning.

### Why GRPO doesn't fit this problem

The experimental design's core constraint (§3) requires the model to **choose** DM analysis over liberal defaults on neutral prompts. GRPO's reward functions can't express this preference directly:

1. **Rule-based rewards are gameable**: Keyword matching rewards vocabulary insertion, not reasoning change (§11 Risk: "DPO collapses to keyword insertion")
2. **LLM judge rewards are slow and noisy**: Adding SG-Lang judge (Qwen3.5-4B) for DM quality assessment adds ~60s per batch and introduces a weaker model's judgment as the training signal
3. **Reward functions can't encode "different conclusion, not different vocabulary"**: This is the experimental design's success criterion (§10), but it requires comparing two responses' reasoning, not scoring one response against patterns
4. **Training set mismatch**: GRPO trains on `data/raw/questions.json` — 1,500 DM-oriented structural analysis questions. The hedging problem manifests on EconCausal-style causal direction questions ("does X increase Y?"), which are absent from the training set. GRPO cannot learn "don't hedge on causal direction" from questions that don't ask about causal direction.

DPO, by contrast, encodes preferences directly: "teacher's committed DM answer > student's hedged answer." The preference signal is human-interpretable and aligns with the experimental design's evaluation strategy (§5.1 Baseline Divergence Test).

### Teacher answer audit findings (2026-06-04)

Audited 250 teacher answers from `data/processed/batch_00000.json` using hedging and commitment patterns from `src/student/rewards.py`:

| Classification | Count | Percentage |
|---|---|---|
| Committed (more commitment than hedging patterns) | 74 | 29.6% |
| Hedging (more hedging than commitment patterns) | 10 | 4.0% |
| Neutral (no patterns matched) | 166 | 66.4% |

**Root cause findings:**

1. **Hedging patterns are too narrow for structural analysis answers**: `_HEDGING_PATTERNS` catches "it depends" (11x) and "uncertain" (6x), but these appear in structural contradiction descriptions ("capitalism depends on X yet Y"), not as hedging about causal direction. The patterns don't capture the actual `+ -> mixed` hedging that caused EconCausal failure.

2. **Commitment patterns are too broad**: 109 matches for `increases/reduces/strengthens/weakens` are false positives — these are common descriptive words, not causal directional claims. "Cost increases" is not the same as "X increases Y."

3. **SFT dataset questions don't ask about causal direction**: The 1,500 DM questions ask for structural analysis ("why is income inequality increasing?"), not causal direction ("does minimum wage increase unemployment?"). The teacher's answers on these questions are irrelevant to the hedging problem because the questions don't require committing to a causal direction.

**Implication for DPO**: To fix the EconCausal hedging problem, DPO needs preference pairs on EconCausal-style questions where the teacher commits to a causal direction and the student hedges. Reusing SFT dataset answers as DPO chosen responses won't address the hedging problem because the questions are structurally different. The DPO dataset must include new question types that match the failure mode.

---

## 3. Proposed Approach: Teacher-Student DPO

Replace GRPO with DPO using preference pairs derived from the teacher (27B) and student (9B) models. This approach:

- Uses the existing DPO pipeline (`src/student/train_dpo.py`, already functional)
- Avoids reward engineering entirely
- Leverages the teacher model's superior reasoning as the quality signal
- Aligns with §4.2 DPO Pair Construction: "Chosen: DM-aligned response (teacher), Rejected: Liberal/default response"
- Addresses §13.7 Design Implication #7 explicitly: chosen responses include definitive causal predictions, rejected responses include over-hedged alternatives

### 3.1 Preference Pair Generation

**Chosen responses** (teacher, 27B):
- For DM structural analysis questions: reuse SFT dataset answers from `data/processed/batch_00000.json` — these are structurally committed and don't require causal direction commitment
- **For anti-hedging pairs**: generate NEW teacher responses on EconCausal-style causal direction questions. The SFT dataset answers are irrelevant here because the questions don't ask about causal direction. Teacher must be prompted to commit to a directional prediction: "When evidence supports a clear directional effect, state it directly."
- Generate using the DM system prompt from `src/teacher/prompts.py` (`DM_SYSTEM_PROMPT`) with the directional commitment modification

**Rejected responses** (student, 9B SFT model):
- Generate using the current SFT-merged model (which exhibits the `+ -> mixed` hedging behavior)
- For anti-hedging pairs: feed the student EconCausal-style questions to elicit hedged responses
- These are the model's actual hedged outputs, not stub templates
- Per §4.2: "The rejected response must be a *plausible* liberal answer, not a trivial placeholder"

**Pair quality filter**:
- `judge(chosen) >= 0.7` using SG-Lang judge for DM quality
- `chosen` contains no hedging patterns (rule-based check)
- `similarity(chosen, rejected) < 0.8` - the pair must be meaningfully different
- If chosen and rejected are too similar, the DPO signal is too weak to drive change

**Pattern limitations** (from audit):
- Current `_HEDGING_PATTERNS` and `POSITIVE_PATTERNS` from `src/student/rewards.py` are unreliable for structural analysis answers. "it depends" in "capitalism depends on X" is not hedging. "increases" in "cost increases" is not causal commitment.
- For anti-hedging pairs on EconCausal questions, use the SG-Lang judge (JUDGE_SYSTEM_PROMPT) for quality assessment rather than rule-based pattern matching. The judge's 4-axis evaluation (structural analysis, contradiction tracing, frame critique, conclusion divergence) is more reliable than keyword patterns.

### 3.2 Capability Preservation Data

Per §4.3 Negative Data for Preserving General Reasoning:

Include identity pairs for non-DM domains to signal that general reasoning should not change:

- **Coding tasks**: 200 HumanEval-style prompts where chosen = rejected = correct code. Standard DPO capability preservation.
- **Factual QA**: 200 MMLU-style questions where chosen = rejected = correct answer. Preserves knowledge.
- **Science/math**: 100 GPQA-style questions where chosen = rejected. Preserves reasoning.

Total preservation pairs: ~500, mixed with ~1,500 DM preference pairs for a 3:1 DM-to-preservation ratio.

### 3.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | SFT merged BF16 checkpoint | Current best model |
| Beta | 0.1 | Per existing DPO config, conservative |
| Learning rate | 5e-7 | Per existing DPO config |
| Max steps | 300 | Conservative starting point; GRPO showed no improvement at 250 or 500 |
| Warmup steps | 50 | Per existing config |
| Scheduler | Cosine | Per existing config |
| Batch size | 1, gradient accum | 4 | VRAM constraints (32GB GPU, NF4) |
| LoRA rank/alpha | 16/16 | Same as SFT to reduce drift risk |

**Evaluation checkpoints**: Every 50 steps, evaluate on:
- HumanEval (code preservation)
- Corr2Cause (DM capability preservation)
- 20-sample held-out DM questions (alignment progress)

**Stop conditions**:
- HumanEval drops below 60% (from 72% baseline)
- Corr2Cause drops below 60% (from 74.6% SFT)
- Loss plateaus for 100 steps

### 3.4 Addressing the Hedging Bias Specifically

Per §13.7 Design Implication #7, construct explicit anti-hedging pairs:

For EconCausal-style questions where the ground truth is `+`:
- **Chosen**: Definitive directional response ("X increases Y because...")
- **Rejected**: Hedged response ("The relationship between X and Y is mixed and context-dependent...")

This directly targets the `+ -> mixed` failure mode that accounts for 52-64% of EconCausal regressions. Include 200-300 such pairs drawn from EconCausal task formats.

---

## 4. Alternative Approaches Considered

### Approach A: Fix GRPO Rewards (Rejected)

Rewrite reward functions with binary directional gate, DM+hedging penalty, code preservation reward, and SG-Lang judge integration.

**Rejected because**: GRPO has failed twice with different reward configurations to produce meaningful DM reasoning improvement. The fundamental problem is that reward functions can't express the experimental design's core requirement: "different conclusion, not different vocabulary" (§3 Question Quality Criteria #6). Reward functions score individual responses; the experimental design requires comparing responses to detect reasoning divergence. DPO's preference pairs encode this comparison directly.

### Approach B: Rejection Sampling + SFT (Reserved as fallback)

Generate 8-16 responses per question with SFT model, filter with judge, retrain SFT on filtered data, repeat 2-3 rounds.

**Reserved because**: Simpler than DPO and avoids reward engineering. Each round produces an evaluable model. But it requires the student model to be able to generate high-quality DM responses some of the time, which may not be true given the hedging problem. If DPO fails to improve Corr2Cause beyond SFT levels, fall back to this approach.

---

## 5. Implementation Plan

### New Code

1. **`src/teacher/generate_teacher_chosen.py`** - Generate chosen responses using 27B teacher
   - If Studio-generated SFT data is sufficient quality, reuse it as chosen responses
   - Otherwise, regenerate with directional commitment prompt modification
   - Output: `data/processed/teacher_chosen.jsonl`

2. **`src/teacher/generate_student_rejected.py`** - Generate rejected responses using SFT student
   - Load SFT merged BF16 model, generate responses for each question
   - Output: `data/processed/student_rejected.jsonl`

3. **`src/teacher/build_preference_pairs.py`** - Quality-filtered pair construction
   - Join chosen + rejected by question ID
   - Apply quality filters (judge score, hedging check, similarity threshold)
   - Add capability preservation pairs (identity pairs for code/factual/science)
   - Add anti-hedging pairs for EconCausal-style questions
   - Output: `data/processed/dpo_pairs_v2.jsonl`

### Modified Code

4. **`src/student/train_dpo.py`** - Add evaluation checkpoints and W&B logging
   - Evaluate at every 50 steps on HumanEval + Corr2Cause + held-out DM questions
   - Stop training if stop conditions met
   - Log to W&B for tracking

5. **`src/student/dpo_config.py`** - Update for new run
   - `max_steps: 300`
   - `save_steps: 50` (more frequent saves for checkpoint evaluation)
   - `output_dir: checkpoints/lora_adapters/dpo_adapter_v2`

### Data Flow

```
data/raw/questions.json (1,500 questions)
    -> generate_teacher_chosen.py -> data/processed/teacher_chosen.jsonl
    -> generate_student_rejected.py -> data/processed/student_rejected.jsonl
    -> build_preference_pairs.py -> data/processed/dpo_pairs_v2.jsonl
        (1,500 DM pairs + 500 preservation pairs + 300 anti-hedging pairs = 2,300 total)
    -> train_dpo.py (300 steps, eval every 50)
        -> checkpoints/lora_adapters/dpo_adapter_v2/
    -> Full evaluation suite
```

---

## 6. Resource Requirements

| Phase | GPU Time | VRAM | Notes |
|-------|----------|------|-------|
| Teacher response audit (50 samples) | ~10 min | 12GB | Manual inspection |
| Teacher regeneration (if needed, ~500 questions) | ~2h | 16GB | 27B via Studio or direct |
| Student rejection generation (1,500 questions) | ~1h | 8GB | SFT model, fast inference |
| Judge-based pair filtering | ~30 min | 4GB | SG-Lang HTTP judge |
| DPO training (300 steps) | ~4h | 16GB | NF4, batch 1x4 accum |
| Evaluation at 6 checkpoints | ~3h | 12GB | BF16 merged, 3 tasks x 6 checkpoints |
| Final full evaluation | ~1.5h | 12GB | All benchmarks |
| **Total** | **~12h** | | Single GPU, sequential |

---

## 7. Critical Open Questions

1. **DPO on SFT questions won't fix hedging** (CONFIRMED by audit): The SFT dataset's 1,500 questions ask for structural analysis, not causal direction. DPO pairs built from these questions will teach the student to match the teacher's structural analysis style, which it already does (Corr2Cause 74.6%). This won't address the `+ -> mixed` EconCausal failure mode. Anti-hedging pairs must use new EconCausal-style questions.

2. **DPO on SFT questions — is it still worthwhile?** If the pairs won't fix hedging, what would they teach? Possibly: deeper DM reasoning on structural questions, or nothing (model already matches teacher on these). Need to decide whether to include SFT-question pairs at all, or focus exclusively on anti-hedging pairs for EconCausal-style questions.

3. **What's the right DM-to-preservation ratio?** The experimental design (§4.3) calls for negative data to preserve general reasoning. 3:1 (DM:preservation) is proposed, but 2:1 may be safer as a standard DPO caution against overfitting. The preservation pairs act as an anchor preventing drift.

4. **Should anti-hedging pairs use EconCausal ground truth?** We know the ground truth for EconCausal tasks. Using this to construct pairs where chosen = correct directional prediction and rejected = hedged "mixed" would directly target the measured failure mode. However, this risks overfitting to the EconCausal format rather than generalizing the anti-hedging behavior.

5. **300 steps sufficient?** DPO is more stable than GRPO, but 300 steps with 2,300 pairs means ~5 epochs over the data. May need more steps for meaningful alignment, or fewer to avoid drift. The 50-step evaluation checkpoints will guide this.

---

## 8. Success Criteria (From Experimental Design §10)

| Metric | SFT Current | DPO Target | Measurement |
|--------|-------------|------------|-------------|
| Corr2Cause | 74.6% | >= 75% | Preserve SFT gain |
| HumanEval | 71.9% | >= 60% | Code preservation |
| EconCausal Task1 Econ | 47.9% | >= 50% | Reduce hedging bias |
| Baseline divergence (conclusion) | unknown | >= 40% | Per §5.1 |
| Baseline divergence (reasoning) | unknown | >= 60% | Per §5.1 |
| Vocabulary-only change | unknown | <= 20% | Per §5.1 |

---

## 9. Decision Points for Feedback

1. **Approach**: Teacher-student DPO vs. rejection sampling SFT vs. hybrid?
2. **Teacher audit**: Audit 50 teacher responses first, or build pairs and filter aggressively?
3. **Data mix**: 3:1 DM-to-preservation ratio, or more conservative 2:1?
4. **Anti-hedging pairs**: Use EconCausal ground truth for explicit pairs, or rely on general teacher-vs-student preference signal?
5. **Step budget**: 300 steps with early stopping, or different budget?
6. **Evaluation frequency**: Every 50 steps, or more/less frequent?
