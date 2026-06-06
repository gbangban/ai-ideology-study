# GRPO v3/v4 Proposal: Summary

## The Problem

After supervised fine-tuning on Dialectical Materialism-aligned data, our student model (Qwen3.5-9B) shows a paradoxical pattern on causal reasoning benchmarks. Formal causal inference improved dramatically — Corr2Cause accuracy jumped from 36.3% to 74.6%, a +38 percentage point gain. But applied economic causal reasoning collapsed. On EconCausal, the model lost 4 to 13 percentage points across four subtasks, with the worst regressions on Task 1 Economics (-12.36pp) and Task 1 Finance (-13.49pp).

The dominant failure mode is hedging. The model replaces correct directional answers — "X causes Y" or "X has a positive effect on Y" — with ambiguous responses like "the effect depends on structural conditions" or "the relationship is mixed." The model has internalized that everything depends on context, and applies this rule uniformly, even when the prompt defines a specific context and empirical evidence supports a clear directional effect. It's overfit to skepticism.

## The Intervention

We propose using reinforcement learning with process-level rewards to break the hedging equilibrium. The approach adapts RLVMR (Reinforcement Learning with Verifiable Meta-Reasoning Rewards), a method from Zhang et al. (2025) that improved task-completion agent performance by rewarding tagged reasoning steps rather than only final outcomes.

The core insight is that hedging is a structural problem, not a factual one. The model knows the right answer but won't commit. Outcome rewards alone — rewarding correct answers — don't address this because the model can game them by hedging and receiving partial credit. Process rewards force the model to produce structured reasoning with a committed conclusion, removing hedging as a reward-maximizing strategy.

We compare four conditions to isolate what works:

- **SFT baseline**: The current model, no further training
- **GRPO v2**: Outcome-only rewards on the original SFT questions (free-form output)
- **GRPO v3**: Outcome-only rewards on synthetic causal data (free-form output)
- **GRPO v4**: Outcome + process rewards on the same synthetic causal data (tagged output)

Comparing v3 vs v4 on identical data isolates whether process-level rewards improve causal reasoning beyond outcome rewards alone. Comparing v3 vs v2 isolates whether the synthetic causal data matters.

## The Data

We generated a synthetic dataset of 620 prompts across five categories. The dataset was designed to train conditional commitment — the ability to commit to a directional answer within a defined context while remaining flexible across contexts.

**Causal graph queries** (160 prompts, currently excluded): Programmatic DAGs with d-separation questions. These have verifiable ground truth — the answer is True or False, determined by algorithm. They're currently excluded because the outcome rewards are keyword-based and score near zero on True/False answers. Reactivating them requires a graph correctness reward that checks the model's answer against the programmatic ground truth.

**Context-flip pairs** (280 prompts): The same causal relationship presented under two different institutional contexts. For example, "Does minimum wage affect employment?" under "monopoly labor market" vs "competitive labor market." The model is rewarded for producing distinct directional answers under different contexts. This teaches conditional commitment — committing within a context while remaining flexible across contexts.

**Null-effect pairs** (100 prompts): Orthogonal variable pairs where the correct answer is null effect. These have verifiable ground truth (answer is always null). A null correctness reward checks whether the model committed to null, zero, or no effect. This trains the model to resist forcing connections between unrelated variables.

**Contradiction pairs** (80 prompts): Opposing claims on the same topic. The model is rewarded for quality reasoning about both sides. These are inherently subjective — there's no single correct answer — so the outcome rewards measure reasoning quality, not factual accuracy.

Of the 460 active prompts (excluding DAG), only 100 (null-effect) have formally verifiable answers. The remaining 360 rely on keyword-based quality proxies. This is a known limitation. The outcome rewards for context-flip and contradiction prompts measure keyword density and structural quality, not factual correctness.

## The Reward Structure

This is where the proposal diverges from a naive implementation. After auditing our initial v4 code against the RLVMR paper, we identified 13 gaps. The revised design follows the paper's dual advantage computation rather than a flat weighted sum.

**Outcome rewards** are correctness. Using EconCausal and Corr2Cause as training sets with ground truth answers, the outcome reward checks whether the model's commitment matches the correct label (`+`, `-`, `mixed`, or `null`). This eliminates the hedging equilibrium: if the correct answer is `+` and the model says `mixed`, it gets 0.0 outcome reward regardless of process reward. The model can't game `mixed` because it's marked wrong when the answer is directional. When the correct answer IS `mixed`, the commitment reward correctly awards it — the reward treats any single committed label equally.

**Process rewards** measure reasoning structure. The model produces tagged output with four tags: planning, reasoning, commitment, and reflection. Each tag has a reward function. Planning rewards variable identification. Commitment rewards a definitive answer and penalizes hedging. Reflection rewards self-critique. A format penalty applies for missing tags.

Critically, the commitment reward was revised to reward commitment to any single label — including mixed — rather than only directional answers. The original implementation rewarded only positive or negative commitments, which would penalize correct mixed answers. The revised version treats the absence of a committed label as hedging, regardless of which label is chosen.

**Dual advantage computation** is the core algorithmic change. In standard GRPO, all rewards are summed into a single advantage estimate. RLVMR separates outcome and process rewards into two advantage estimates, normalizes each independently within prompt groups, and combines them with a mixing parameter alpha (set to 0.5). This prevents high process rewards from masking poor outcomes and vice versa. If a completion has great reasoning but a wrong answer, it gets positive process advantage but negative outcome advantage, resulting in near-zero net update. If it has poor reasoning but a correct answer, it gets the opposite — rewarding the outcome but not reinforcing bad reasoning patterns.

**Success-conditional planning** mirrors the paper's design: the planning reward is only awarded if the outcome reward exceeds a threshold. This prevents the model from producing performative planning on wrong answers — planning that looks good structurally but leads to incorrect conclusions.

## Tag Adaptations

The tag set was adapted from RLVMR for single-turn causal reasoning, not copied verbatim.

RLVMR's `<explore>` tag penalizes repetition between turns — it's an anti-repetition mechanism for multi-turn agents. In single-turn causal reasoning, repetition is irrelevant. The problem is hedging. Our `<commitment>` tag is the domain equivalent: it rewards a definitive answer and penalizes hedging language.

RLVMR's `<reflection>` tag rewards self-critique. We reintroduced this tag (it was absent in the initial v4 implementation) and made it outcome-conditional. Reflection is only rewarded if the outcome reward exceeds a threshold, preventing performative reflection on wrong answers.

RLVMR's `<monitor>` tag checks context constraints. We dropped it as a required tag because its function overlaps with reflection and it adds format burden without a clear reward signal.

## Cold-Start SFT

RLVMR's ablation study shows a 15.7 percentage point drop when cold-start SFT is omitted. The model needs examples of the tagged format before RL training begins. Without cold-start SFT, the model may never produce tags, and the process reward signal remains near zero throughout training.

We generate tagged demonstrations using the teacher model (Qwen3.5-27B) with system prompts that instruct tagged output. A 5-epoch SFT on these demonstrations teaches the format. The merged model then serves as the starting point for GRPO v4 training.

## Expected Results

We hypothesize that v4 will produce fewer hedged responses than v3 on the same data. The commitment tag structure removes hedging as a reward-maximizing strategy. The dual advantage computation ensures that process rewards don't mask poor outcomes. We expect improvement on EconCausal Task 1 subtasks, where hedging is the dominant failure mode.

A success criterion is statistically significant improvement on at least one Task 1 subtask at p < 0.05. We also track directional assertion rate — the fraction of answers that are directional rather than mixed — as a secondary metric.

## Key Risks

**Correctness reward resolves the hedging game.** Using EconCausal/Corr2Cause with ground truth means the outcome reward is factual, not structural. The model can't game `mixed` because it's marked wrong when the answer is directional. The dual advantage ensures process rewards don't rescue wrong answers. The remaining risk is that the model learns format compliance without genuine reasoning improvement — correct answer by chance, good tags, but no transfer to held-out questions.

**Tag format transfer.** Evaluation benchmarks expect free-form answers. If the model reasons well only within the tagged format, the skill may not transfer to untagged evaluation.

**Small dataset.** 460 prompts with 8 completions over 500 steps produces limited diversity. Overfitting to synthetic data would show as high training reward scores without benchmark improvement.

**Single-turn limitation.** RLVMR's multi-turn loop has no analog in single-turn causal reasoning. The tag adaptations are faithful to the paper's principles but not a reproduction of the multi-turn mechanism.

## Implementation Status

The data generation pipeline is complete and tested. The initial v4 implementation exists but requires a rewrite to match the paper's dual advantage computation, success-conditional planning, generalized commitment reward, reflection reward, and KL regularization. The cold-start SFT pipeline is implemented but not yet executed. The filtered dataset (460 prompts, DAG excluded) is ready for training.

The execution plan follows six phases: cold-start SFT, reward function rewrites, dual advantage implementation, KL regularization and clipping fixes, training runs for all four conditions, and evaluation with comparison. DAG questions are parked pending a graph correctness reward implementation.

## Why This Matters

The SFT+DPO pipeline improved the model's formal reasoning but degraded its applied reasoning. This is a measurable, quantifiable regression that a targeted RL intervention can address. The proposal isolates the contribution of process rewards through clean comparisons, uses verifiable rewards where possible, and acknowledges limitations where rewards are structural proxies rather than factual signals. The approach is conservative: it adapts an established method (RLVMR) to a new domain (causal reasoning) rather than proposing novel algorithmic changes.
