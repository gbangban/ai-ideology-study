# Paper Synthesis Report: Hedging Analysis & RLVMR Integration

**Date:** 2026-06-04
**Context:** GRPO v2 reward redesign, EconCausal regression diagnosis, new training data approaches

---

## 1. The Hedging Problem — Root Cause Confirmed

The papers converge on one diagnosis: **the model's hedging isn't a reward bug, it's a data + prior problem.**

- **2604.21334** (ideological bias): LLMs have an intervention-oriented bias — they over-predict complex multi-causal outcomes. DM training amplified this: the model learned "everything depends on structural conditions" and applies it uniformly. That's why `+` → `mixed` is the dominant failure mode.
- **2602.20571** (causal reasoning benchmark): The bottleneck in applied causal reasoning is identification details, not formal logic. The +38pp Corr2Cause improvement proves the model's formal causal logic is fine. The -12pp EconCausal regression is about misidentifying when a simple causal relationship holds vs. when context matters.
- **2505.18931** (ReCITE): Even Claude Opus 4.5 only scores F1=0.535 on causal graph extraction. LLM causal reasoning is fundamentally weak, and explicitness is the single biggest factor. The SFT data was explicit DM reasoning — the model never learned to handle implicit causality.

---

## 2. Revisions.md Approaches — Assessment

### Approach 1: Synthetic Counterfactual Context Flipping — **VALID, directly addresses hedging**

Strongest idea. Forces the model to give two distinct directional answers for two different contexts, penalizes hedging in both. Directly trains the skill EconCausal tests — context-dependent directional commitment. Non-leaking because synthetic prompts, not benchmark data.

**Paper alignment:** 2604.21334 confirms the model needs to learn when to commit vs. when to be skeptical. Context flipping forces this discrimination.

### Approach 2: Multi-Variable Abstract Graphing — **VALID, but secondary**

Corr2Cause already jumped +38pp. This would push it higher but doesn't address the hedging problem. Maintenance improvement, not a fix.

**Paper alignment:** 2602.20571 shows that improving formal causal logic without fixing identification details won't help EconCausal.

### Approach 3: Orthogonality Prompts (Null-Effect Training) — **VALID, critical for Task 3**

Task 3 regression (-10.80pp) is the "everything is related" fallacy. Training the model to recognize genuinely null causal relationships is exactly what's needed. Complement to approach 1.

**Paper alignment:** 2505.18931 shows models generate spurious causal relationships (high precision, low recall). This directly counters that failure mode.

### Missing Approach: Commitment-without-hedging Data

Need a fourth approach: straightforward causal relationships where the correct answer IS a clear direction. The model needs examples where `+` is correct and it should commit. This is what 2604.21334 identifies as the gap — the model learned multi-causal skepticism and applies it even when the empirical truth is simple.

---

## 3. Paper Alignment Summary

| Paper | Aligns With | Application |
|-------|-------------|-------------|
| **2604.21334** (ideological bias) | All 3 approaches | Explains WHY hedging happens; validates context-flipping as the fix |
| **2602.20571** (causal benchmark) | Approach 1 | Shows identification details are the bottleneck, not logic |
| **2505.18931** (ReCITE) | Approach 3 | Shows spurious causality is the dominant error |
| **2410.01810** (propaganda) | None directly | Confirms SFT > DPO for ideological shift; suggests GRPO data matters more than reward design |
| **2507.22844** (RLVMR) | New paradigm (see below) | Process-level rewards, independent of dataset question |

---

## 4. RLVMR Integration — Regardless of New Approaches

RLVMR is the most actionable paper for the training paradigm.

### Immediate Integration (Fits Current Architecture)

**1. Tagged reasoning output format**

Instead of free-form completions, require the model to output with meta-reasoning tags:
- `<planning>` — decompose the causal question, identify variables and context
- `<reasoning>` — trace material conditions and structural mechanisms
- `<commitment>` — explicit directional claim (not hedging)
- `<monitor>` — self-check: does the answer match the context given?

**2. Process-level rule-based rewards** (add to existing reward functions)

- `compute_planning_reward(text)`: +0.5 if `<planning>` present and identifies >=2 variables
- `compute_commitment_reward(text)`: +1.0 if `<commitment>` contains a definitive direction, -0.5 if hedging
- `compute_monitor_reward(text)`: +0.5 if `<monitor>` references the prompt's specific context
- Format penalty: -0.1 per missing required tag

**3. Advantage grouping by tag**

RLVMR's key insight: compare exploration steps against other exploration steps, not all steps. For GRPO, group rewards by reasoning phase and normalize advantages within each group. This prevents the model from gaming one reward component.

**4. Cold-start SFT**

RLVMR needs 200 tagged trajectories before RL. Generate these with the 27B teacher producing tagged DM reasoning, then SFT on the tagged format. Prerequisite for stable RL training with tagged output.

### Why This Works with the New Approaches

The tagged format makes context-flipping (approach 1) more effective because `<planning>` forces the model to explicitly acknowledge the context before committing. It makes orthogonality training (approach 3) more effective because `<reasoning>` exposes whether the model is genuinely finding no mechanism or hallucinating connections.

### Combined Reward Structure

```
Total = 0.30 × directional_assertion    (anti-hedging, existing)
      + 0.20 × dm_alignment             (DM keywords, existing)
      + 0.15 × mechanism_commitment     (existing)
      + 0.15 × planning                 (RLVMR: explicit decomposition)
      + 0.10 × commitment               (RLVMR: definitive claim)
      + 0.10 × monitor                  (RLVMR: context self-check)
```

The RLVMR components don't replace the existing rewards — they add process-level supervision that makes the outcome rewards more effective. Without process rewards, the model finds shortcuts (hedging). With process rewards, it has to produce structurally sound reasoning to get any reward.

---

## 5. Revision: If Corr2Cause Improvement Is Also Hedging

**Alternative interpretation:** The +38pp Corr2Cause gain isn't genuine causal reasoning improvement — it's the model's hedging/skepticism prior producing more "False" answers, which happen to match the benchmark's answer distribution better than the SFT model's answers. The model isn't reasoning better; it's refusing to affirm causal claims, and "False" is the hedging default in a binary format.

**How this changes the analysis:**

1. **No intact causal reasoning.** The previous "formal good, applied bad" split collapses. The model has no demonstrated strength in causal reasoning — hedging is universal across task formats.

2. **Approach 2 (abstract graphing) loses justification.** It was predicated on Corr2Cause showing the model can handle causal graphs. Without that evidence, the approach is speculative.

3. **Fix must be universal, not domain-specific.** Can't patch EconCausal in isolation. Counterfactual context-flipping (approach 1) must include abstract causal questions alongside economic ones. The skepticism prior from SFT needs to be overridden across all causal reasoning.

4. **RLVMR becomes more critical.** If there's no domain where the model reasons well, process-level rewards are the only mechanism to break the hedging prior. The `<commitment>` tag forcing a definitive claim is the structural intervention needed to override the skepticism prior from SFT.
