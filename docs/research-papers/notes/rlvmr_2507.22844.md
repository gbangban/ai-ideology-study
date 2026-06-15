# RLVMR Notes

**arXiv:** 2507.22844 | **Authors:** Zhang, Chen, Li, Tu, Li (Tencent Hunyuan AI Digital Human)
**Title:** Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents

## Problem Statement

Standard RL for LLM agents (GRPO, PPO) optimizes only for final task success. This creates **inefficient exploration** — agents find solutions but through flawed, redundant, illogical reasoning paths. They get rewarded for success even when the path to success is broken.

Consequences:
- High repetitive action rates (31.2% for GRPO 7B on unseen tasks)
- Poor generalization to novel task categories
- Agents mimic training data action distributions rather than developing genuine reasoning

## Core Insight

Reward the **reasoning process**, not just the outcome. Draw from metacognitive theory — effective problem-solving requires "thinking about thinking": planning, monitoring, exploration, reflection.

## Method: RLVMR

### Two-Phase Training

**Phase 1 — Cold-Start SFT**
- 200 expert trajectories annotated with meta-reasoning tags by a stronger model (GPT-4)
- 5 epochs, LR 1e-5, batch size 16/GPU
- Purpose: teach the model to produce syntactically correct tagged output, NOT to solve tasks

**Phase 2 — RL with GRPO-MR**
- Critic-free policy gradient (no separate value model)
- 100 epochs RL, 16 environments × 8 rollouts each
- Max 30 steps per episode

### Meta-Reasoning Tags

Four XML-style tags the agent uses to label its cognitive state at each step:

| Tag | Purpose | When to Use |
|-----|---------|-------------|
| `<planning>` | Decompose task into high-level steps | Task start or replanning needed |
| `<explore>` | Generate hypotheses to navigate uncertainty | Facing obstacles, unexpected results |
| `<reflection>` | Analyze errors, formulate corrections | After consecutive failures |
| `<monitor>` | Track progress against plan | Routine execution, on-track |

### Reward Structure

**Outcome Reward R(tau):** Binary — reward rs for task success, 0 otherwise.

**Process Rewards (per-step):**
- **Planning reward:** Granted if `<planning>` step and trajectory ultimately succeeds
- **Exploration reward:** Granted if action targets a new object/location (discourages repetition)
- **Reflection reward:** Granted if `<reflection>` is followed by corrective action after failures
- **Format penalty:** -0.1 if output doesn't match `<tag>...<action>...</action>` structure

All process rewards are **programmatic and rule-based** — no learned reward model needed.

### GRPO-MR Algorithm

Advantage combines two signals:

1. **Trajectory-level relative advantage:** Normalized outcome reward across batch
   A_traj = (R(tau_k) - mu_R) / sigma_R

2. **Meta-reasoning relative advantage:** Rewards normalized within each tag group
   A_MR = (r_MR - mu_tag) / sigma_tag

3. **Combined:** A_t = alpha * A_traj + (1 - alpha) * A_MR  (alpha=0.5)

Optimized with clipped surrogate + KL regularization against reference policy (lambda_KL=0.01).

Key design: grouping by meta-reasoning tag means exploration steps are compared against other exploration steps, not against planning steps. This provides context-aware credit assignment.

## Results

### ALFWorld (Qwen2.5-7B)

| Method | L0 (seen) | L1 (unseen variants) | L2 (unseen categories) |
|--------|-----------|---------------------|----------------------|
| ReAct | 23.1% | 28.5% | 27.0% |
| SFT | 63.3% | 57.0% | 37.5% |
| GRPO | 79.3% | 77.3% | 52.3% |
| GiGPO | 89.5% | 90.2% | 67.2% |
| **RLVMR** | **91.4%** | **91.8%** | **83.6%** |

RLVMR beats GiGPO by 16.4pp on the hardest split (L2).

### ScienceWorld (Qwen2.5-7B)

| Method | L0 | L1 | L2 |
|--------|-----|-----|-----|
| GRPO | 49.1% | 30.1% | 26.6% |
| GiGPO | 53.4% | 35.2% | 25.8% |
| **RLVMR** | **67.2%** | **43.0%** | **32.2%** |

### Small Model Outperforming Large

Qwen2.5-1.5B + RLVMR achieves 87.9% on ALFWorld L1, beating GPT-4o (66.0%) and DeepSeek-R1 (70.2%).

### Efficiency Improvements

7B model repetitive action rate on L2:
- GRPO: 31.2%
- RLVMR: 11.7%

Average actions per task (ALFWorld L2):
- GRPO: 21.7 steps
- RLVMR: 15.4 steps (28.1% reduction)

### Ablation (1.5B, ALFWorld L2)

| Variant | Success Rate |
|---------|-------------|
| Full RLVMR | 56.3% |
| w/o outcome reward (AT) | 12.5% (catastrophic) |
| w/o meta-reasoning reward (AMC) | 45.3% (-11pp) |
| w/o cold-start SFT | 40.6% (-15.7pp) |

All three components are essential. Removing outcome reward collapses performance — process rewards alone can't orient the agent toward the goal. Removing cold-start causes format instability, especially for small models.

## Key Takeaways for Our Project

1. **Process-level rewards are cheap and effective.** RLVMR's rewards are rule-based, not learned — no judge model needed. This is directly applicable to our GRPO pipeline where we currently use a separate judge backend.

2. **Tagging reasoning steps works.** The four-tag system (planning, explore, reflection, monitor) is simple, parseable, and effective. We could adapt this for our DM alignment training — tag reasoning steps and reward structurally sound reasoning.

3. **Cold-start SFT is critical for small models.** 200 trajectories is minimal. Without it, RL training is unstable because the model can't produce parseable outputs.

4. **GRPO benefits from dense rewards.** Our current GRPO uses outcome-only rewards. Adding process-level rewards (even simple rule-based ones) should reduce repetitive patterns and improve generalization.

5. **The advantage grouping by tag type is clever.** Comparing exploration steps against other exploration steps, not against all steps, gives better credit assignment. This is a simple modification to standard GRPO.

6. **Format penalties are important.** A small -0.1 penalty for malformed output keeps the model producing structured responses throughout training.

## Relevance to DM-Align Project

Our GRPO training currently rewards based on a judge model evaluating final responses. RLVMR suggests we should also reward intermediate reasoning behaviors:

- Reward when the model explicitly plans its reasoning approach
- Reward exploration of alternative interpretations
- Reward self-correction after failed reasoning attempts
- Reward monitoring of alignment with the task

These could be implemented as lightweight regex-based rewards on tagged output, eliminating the need for the judge model on every step and reducing compute costs.
