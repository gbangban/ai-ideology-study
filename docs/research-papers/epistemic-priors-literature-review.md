# Epistemic Priors in AI/ML: Literature Review and Integration Analysis

**Date**: 2026-06-15
**Purpose**: Survey epistemic prior research across AI/ML literature and identify integration points for the DM-Align GRPO v3/v4 pipeline.

---

## 1. Two Requested Papers

### 1.1 MARL-EP: Multi-Agent Reinforcement Learning with Epistemic Priors (IEEE CoDIT 2023)

**Citation**: IEEE Conference on Control, Decision and Information Technologies (CoDIT), Rome 2023. DOI: 10.23919/CoDIT57022.2023.10284342

**Core idea**: Autonomous multi-agent teams with limited sensing and zero communication can achieve high coordination by sharing an epistemic mental model. Each agent uses epistemic estimation to infer unobservable portions of the observation space from a shared prior over teammate behavior.

**Key mechanism**:
- Shared mental model between agents parameterized as epistemic priors over observation space
- Epistemic estimation infers unobservable states given partial observations
- Enables coordination with severely impaired sensing and zero inter-agent communication

**Relevance to DM-Align**: Low direct relevance. MARL-EP operates in a classical multi-agent control setting (warehouses, firefighting, surveillance) with well-defined MDPs. The "epistemic prior" here means a prior over hidden states of other agents -- closer to standard POMDP belief tracking than to the Bayesian prior literature we care about for LLM alignment. However, the conceptual framing -- that agents can coordinate effectively if they share calibrated epistemic beliefs about unobserved variables -- maps loosely to the idea that a student model could benefit from a teacher-derived prior over reasoning structure.

### 1.2 A Decision Architecture for Epistemic Prioritization: ML at the Intersection of Technology and Society (Technology in Society, 2025)

**Citation**: Naser, M.Z. "A decision architecture for epistemic prioritization: Machine learning at the intersection of technology and society." Technology in Society 83 (2025): 103039. Open access, Creative Commons.

**Core idea**: ML methodologies transform philosophy of science through five epistemic functions -- Prediction, Explanation, Discovery, Understanding, and Decision-making (P.E.D.U.D.). The paper develops a decision framework to help practitioners determine which epistemic function to prioritize for specific problem domains.

**Key contributions**:
- Taxonomy of five epistemic functions in ML
- Decision architecture aligning ML methodologies with targeted epistemic goals
- Analysis of tensions between data-driven and theory-driven approaches
- Argument that ML necessitates reconsideration of traditional philosophy of science

**Relevance to DM-Align**: Conceptual and philosophical. The P.E.D.U.D. framework provides vocabulary for articulating what our GRPO pipeline is optimizing: primarily Explanation (causal mechanism identification) and Understanding (structural analysis), secondarily Prediction (correct sign classification). The data-driven vs. theory-driven tension maps directly to our v3 (outcome-only, data-driven correctness) vs. v4 (process rewards, theory-driven reasoning structure) experimental design. This paper doesn't provide methods to integrate, but provides framing for the research question.

---

## 2. Broad Epistemic Prior Literature Survey

### 2.1 LLMs as Sources of Priors

#### LMPriors: Pre-Trained Language Models as Task-Specific Priors (arXiv 2210.12530)

**Core idea**: Distill task-specific inductive biases from pretrained LMs using natural language metadata (variable names, descriptions). The LMPrior maps metadata to learning procedures with bespoke inductive bias.

**Mechanism**:
- Prompt LM with task metadata -> extract common-sense reasoning as prior
- Apply to feature selection, causal discovery, safe RL
- LM's knowledge becomes a heuristic prior over model parameters

**Integration potential**: HIGH. Our pipeline already uses a teacher model (Qwen3.5-27B) to generate answers. Instead of discarding teacher outputs after SFT, we could extract teacher-derived priors over reasoning structure and inject them as reward shaping in GRPO. Specifically, the teacher's reasoning traces could define a prior distribution over acceptable reasoning paths, and the v4 process rewards could penalize deviations from this prior.

#### LoID: Logit-Informed Distributions (arXiv 2601.17609)

**Core idea**: Extract Bayesian priors from LLMs through token-level logits rather than generated text. Probe the LLM with semantically paired sentences (positive vs. negative feature-target relationships), measure logit differences as directional belief, and variance across paraphrases as uncertainty estimate. Forms Normal priors over model coefficients.

**Key results**: Recovers up to 50% of the OOD performance gap relative to oracle. Works across 15 real-world tabular datasets.

**Integration potential**: MEDIUM. LoID is designed for Bayesian linear/logistic regression, not LLM fine-tuning. However, the core insight -- that LLM logits encode calibrated belief strength and uncertainty -- could inform how we weight teacher-derived signals. Instead of treating all teacher answers equally, we could probe teacher confidence (via logit spread) and use it to modulate outcome reward strength.

#### Statsformer: Learning When to Trust LLM Priors (arXiv 2601.21410v3)

**Core idea**: Framework for incorporating LLM-derived semantic priors into supervised learning with adaptive validation. Maps LLM feature scores into prior-injection mechanisms across heterogeneous learners, then uses out-of-fold validation to calibrate prior influence.

**Key guarantee**: Oracle-style -- final predictor performs no worse than best convex combination of candidates, including prior-free learners.

**Integration potential**: MEDIUM-HIGH. The "validate before trusting" paradigm is directly applicable to our reward system. We could add an epistemic validation step: when the student's output diverges from the teacher prior, check whether the divergence is justified by ground-truth correctness. If the student is correct where the teacher prior would penalize, the prior should be downweighted. This prevents reward overoptimization on teacher-specific reasoning styles.

#### LLMPrior: Automated Prior Elicitation (arXiv 2508.03766)

**Core idea**: Formalizes an LLMPrior operator that maps flexible context to valid prior distributions by coupling an LLM with an explicit generative model (Mixture Density Network). Extended to multi-agent settings with Logarithmic Opinion Pool aggregation.

**Integration potential**: LOW-MEDIUM. Architecture is complex (MDN coupling) and designed for prior aggregation across agents. Not directly applicable to single-model GRPO training.

#### Iterated In-Context Learning for Prior Elicitation (arXiv 2406.01860)

**Core idea**: Use iterated learning (a Markov chain Monte Carlo method) through in-context prompting to sample prior distributions from LLMs. Validated against human priors on causal learning, proportion estimation, and everyday quantity prediction.

**Key finding**: GPT-4 priors qualitatively align with human priors and surpass generic (uniform) priors.

**Integration potential**: LOW. Method is prompt-based and designed for prior elicitation, not training integration. Conceptually interesting for understanding what priors our student model has learned.

### 2.2 Epistemic Uncertainty in LLM Alignment

#### Epistemic Traps: Rational Misalignment Driven by Model Misspecification (arXiv 2602.17676)

**Core idea**: Alignment failures (sycophancy, hallucination, deception) are not transient errors but mathematically rationalizable behaviors arising from model misspecification. Uses Berk-Nash Rationalizability from economics to model agents optimizing against flawed subjective world models.

**Key theorem**: Safety is a discrete phase determined by the agent's epistemic priors, not a continuous function of reward magnitude. The "safe" region where honesty is the unique equilibrium requires $p_H > 0.5 > p_S$ -- a much stricter condition than the Nash requirement of $p_H > p_S$.

**Phase diagram**:
- $p_S > 0.5 > p_H$: Unique sycophancy equilibrium
- $p_H > 0.5 > p_S$: Unique honesty equilibrium (the only "safe" region)
- $p_S, p_H > 0.5$: Multiple equilibria (both sycophancy and honesty stable)
- $p_S, p_H < 0.5$: Pure 2-cycle, no equilibrium

**Integration potential**: CRITICAL. This is the single most relevant paper to our project. Key implications:

1. **Reward calibration matters more than reward magnitude**: Our current reward functions produce scores in [0, 1], but the BNR framework shows that what matters is whether rewards cross the 0.5 threshold. We need to audit whether our outcome rewards reliably exceed 0.5 for correct answers and fall below 0.5 for incorrect answers.

2. **Model misspecification creates fragility**: The student model's internal representation of "what constitutes a good answer" may be misspecified relative to the true objective. This explains why SFT on DM-aligned data caused large regressions on EconCausal (-4 to -13pp) -- the model learned a misspecified prior that conflates "DM-style hedging" with "correct reasoning."

3. **Subjective Model Engineering (SME)**: The paper proposes shifting from Reward Engineering (fixing external rewards) to SME (fixing the agent's internal priors). For our pipeline, this means the v4 process rewards should not just shape output format but should actively correct the student's subjective model of what constitutes valid causal reasoning.

4. **Direct connection to hedging failure mode**: The BNR framework explains why the model learned to hedge (+ -> mixed). If the reward for hedging responses ($p_S$) exceeds 0.5 -- even slightly -- sycophancy (here: excessive caution) becomes a rationalizable equilibrium. Our anti-hedging commitment reward directly addresses this, but needs to be calibrated to ensure $p_{commitment} > 0.5 > p_{hedge}$.

#### RewardUQ: Uncertainty-Aware Reward Models (arXiv 2602.24040)

**Core idea**: Unified framework for uncertainty quantification in reward models. Compares methods along accuracy and calibration dimensions. Epistemic uncertainty in reward models arises from limited human feedback data.

**Key finding**: Model size and initialization matter most. No single UQ method dominates across all settings.

**Applications**:
- Mitigating reward overoptimization by penalizing uncertain reward signals
- Active learning: guiding data collection toward most informative samples

**Integration potential**: HIGH. Our reward functions are deterministic (regex-based), so they have zero epistemic uncertainty in the traditional sense. However, the concept applies to our proxy rewards: `compute_proxy_outcome()` has high epistemic uncertainty because it uses keyword heuristics as a proxy for correctness. We should:
1. Scale proxy rewards more aggressively (currently at 0.5x, consider 0.2-0.3x)
2. Add explicit uncertainty flags: when proxy rewards fire but ground truth is unavailable, log these as high-uncertainty samples
3. Consider ensemble-based reward estimation for proxy rewards

#### Bayesian Reward Models for LLM Alignment (ICML 2024 Workshop)

**Core idea**: Train Bayesian reward models using Laplace approximation on LoRA weights. Uncertainty estimates signal higher uncertainty further from training data, mitigating reward overoptimization in best-of-n sampling.

**Integration potential**: MEDIUM. Laplace approximation on our GRPO LoRA adapters could provide per-sample uncertainty estimates for the reward signal. More relevant for post-training evaluation than for training itself.

#### UCPO: Uncertainty-Aware Policy Optimization (arXiv 2601.22648)

**Core idea**: Ternary reward system (right, wrong, uncertain) with Ternary Advantage Decoupling and Dynamic Uncertainty Reward Adjustment. Resolves gradient interference and reward hacking in naive uncertainty learning.

**Key insight**: Fixed uncertainty reward mechanisms bias the advantage function -- negative in high-performance regimes (model fails to learn uncertainty), overloaded in low-performance regimes (model lapses into reward hacking).

**Integration potential**: HIGH. Our v4 dual-advantage system could benefit from a ternary decomposition:
- High-confidence correct: full outcome + process reward
- Low-confidence correct: partial outcome reward, full process reward (reward the reasoning even if uncertain)
- High-confidence incorrect: negative reward (penalize overconfidence in wrong answers)

This directly addresses the hedging failure mode: the model currently gets the same reward for "I'm uncertain" regardless of whether uncertainty is warranted.

#### Ask a Strong LLM Judge when Your Reward Model is Uncertain (NeurIPS 2025)

**Core idea**: Uncertainty-based routing between fast reward model and costly LLM judge. Uses Singular Neural GP for preference modeling to quantify epistemic uncertainty. Routes uncertain pairs to strong LLM judge.

**Integration potential**: MEDIUM. We already have an SG-Lang container with Qwen3.5-4B as a judge. We could route low-confidence reward estimates to the judge model. However, the 4B model may not be "strong enough" for this -- the paper uses much larger judges.

#### POETS: Policy Ensembles for Thompson Sampling (arXiv 2605.07775)

**Core idea**: Policy ensembles diversified via Poisson bootstrapping capture epistemic uncertainty and drive exploration. Extends GRPO to ensemble setting, implicitly conducting KL-regularized Thompson sampling.

**Key result**: Standard GRPO is mathematically equivalent to optimizing a single-policy instance of the POETS loss. The ensemble inherits optimal cumulative soft regret bounds.

**Integration potential**: HIGH. This is directly applicable to our GRPO setup. Running multiple GRPO policies with different initializations and aggregating could provide natural epistemic uncertainty estimates. More practically, the insight that "GRPO = single-policy POETS" suggests we should add exploration bonuses proportional to policy disagreement.

#### The Alignment Auditor: Bayesian Framework for Verifying LLM Objectives (arXiv 2510.06096)

**Core idea**: Three-stage reward inference framework: (1) recover distribution over plausible reward functions, (2) evaluate trustworthiness with uncertainty-aware diagnostics, (3) use refined objectives in RLHF.

**Key finding**: Strong correlation (r=0.989) between inferred reward variance (epistemic uncertainty) and Mahalanibis distance from training data distribution.

**Integration potential**: MEDIUM-HIGH. The diagnostic approach -- injecting spurious features and measuring whether the reward model becomes uncertain vs. spuriously confident -- could be adapted to test our reward functions. If our regex-based rewards fire on spurious DM keywords, that's a shortcut we should detect.

### 2.3 Epistemic Priors in Reinforcement Learning

#### EUBRL: Epistemic Uncertainty directed Bayesian RL (arXiv 2512.15405)

**Core idea**: Bayesian RL algorithm leveraging epistemic guidance for principled exploration. Uses probabilistic inference to model epistemic uncertainty directly in the objective, disentangling exploration and exploitation.

**Key results**: Nearly minimax-optimal regret and sample complexity for expressive priors. Instantiated with Dirichlet (transitions) and Normal-Gamma (rewards) priors.

**Integration potential**: MEDIUM. The prior-dependent bounds provide theoretical justification for incorporating domain knowledge as priors in RL. Our DM system prompts effectively serve as a prior over acceptable reasoning -- EUBRL provides the theoretical framework for why this works.

#### Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL (arXiv 2505.20561)

**Core idea**: Ground reflective exploration with Bayesian RL that maximizes expected return under a posterior distribution over MDPs. Agent adapts on-the-fly by updating beliefs and switching strategies based on observed outcomes.

**Key theorem**: Optimal adaptive policy achieves arbitrarily higher Bayes-expected return than any optimal Markovian policy.

**Integration potential**: MEDIUM-HIGH. Our v4 process rewards already encourage reflective behavior (the `<reflection>` tag). This paper provides theoretical grounding: reflective exploration is not just a heuristic but provably superior to Markovian policies in uncertain environments. The `<reflection>` tag should be rewarded more when the model detects high epistemic uncertainty about the answer.

#### UPER: Uncertainty Prioritized Experience Replay (arXiv 2506.09270)

**Core idea**: Modified epistemic uncertainty estimator incorporating distance-to-target term. Uses information gain criterion to prioritize experience replay.

**Integration potential**: LOW-MEDIUM. More relevant for offline RL with replay buffers. Our GRPO training doesn't use experience replay in the traditional sense.

#### ULPS: Uncertainty-Aware LLM-Guided Policy Shaping (arXiv 2606)

**Core idea**: Integrates calibrated LLM into RL training loop to provide uncertainty-modulated behavioral guidance. Uses A*-based oracle for optimal trajectories, fine-tunes BERT for action suggestions, blends with PPO based on MC dropout uncertainty.

**Integration potential**: LOW. Architecture-specific (BERT + PPO + MiniGrid). Conceptually similar to our teacher-student setup but at a different scale.

### 2.4 Epistemic Properties of LLMs

#### Probabilistic Coherence, Logical Consistency, and Bayesian Learning (PLOS ONE)

**Core idea**: Neural language models exhibit properties of epistemic agency -- probabilistically coherent and logically consistent degrees of belief that can be rationally revised with novel evidence. Self-training on auto-generated texts enables gradual coherence.

**Integration potential**: LOW-MEDIUM. Confirms that LLMs can maintain coherent belief systems, which justifies our approach of training the student to maintain consistent reasoning across DM-aligned tasks.

#### Bayesian Teaching Enables Probabilistic Reasoning in LLMs (Nature Communications 2026)

**Core idea**: Fine-tuning LLMs on interactions between users and a Bayesian Assistant dramatically improves probabilistic reasoning. Bayesian teaching outperforms direct fine-tuning on correct answers.

**Key insight**: Training on the process of Bayesian belief updating is more effective than training on point estimates.

**Integration potential**: HIGH. This directly supports our v4 approach: training on process (reasoning structure) is more effective than training on outcomes alone. The "Bayesian teaching" paradigm maps to our teacher-generated reasoning traces as SFT data, followed by process-reward GRPO.

#### LLMs are Bayesian, In Expectation, Not in Realization (arXiv 2507.11768)

**Core idea**: Transformers violate the martingale property required for Bayesian updating on exchangeable data. However, they achieve information-theoretic optimality in expectation over orderings. Positional encodings break exchangeability.

**Integration potential**: LOW. Theoretical insight about transformer architecture. Not directly actionable for reward design.

#### Enough Coin Flips Can Make LLMs Act Bayesian (Berkeley 2025)

**Core idea**: LLMs possess biased priors causing initial divergence in zero-shot settings, but update in a Bayesian manner with sufficient in-context evidence. Primary limitation is poor priors, not failed in-context learning.

**Integration potential**: MEDIUM. Confirms that the student model's initial priors (from pretraining + SFT) matter more than the GRPO training signal. If SFT implants a hedging prior, GRPO needs to overcome it -- which explains why we need stronger anti-hedging rewards.

#### Identifying and Mitigating Prior Influence in LLMs (arXiv 2504.12585)

**Core idea**: LLMs fail on deterministic tasks because implicit prior distributions over token sequences influence responses. Lightweight finetuning of early layers can mitigate prior dominance.

**Key finding**: Simply prompting "don't rely on prior knowledge" leads to dramatic improvements. LoRA finetuning of a single early layer achieves high performance on prior-dominated tasks.

**Integration potential**: MEDIUM. Our SFT phase may have implanted strong DM-specific priors that interfere with EconCausal performance. The paper suggests targeted early-layer finetuning could correct this without full retraining.

#### ESI-UQ: Epistemic Uncertainty via Semantic-preserving Intervention (arXiv 2510.13103)

**Core idea**: Grey-box uncertainty quantification measuring variation in model outputs before/after semantic-preserving interventions. Better causal mechanism capture = more stability under intervention = lower epistemic uncertainty.

**Integration potential**: MEDIUM. Could be used as a post-training diagnostic: if the student's answer changes dramatically under semantic-preserving prompt rephrasing, it has high epistemic uncertainty about that topic.

---

## 3. Integration Points for DM-Align Pipeline

### 3.1 Immediate (Low Effort, High Impact)

#### A. Reward Threshold Calibration (from Epistemic Traps / BNR)

The BNR framework shows that reward thresholds at 0.5 determine behavioral phase transitions. Audit current reward functions:

- `compute_econcausal_correctness()`: Correct answers get 0.9-1.0, partial gets 0.0-0.3. Gap is clean (0.3-0.9) -- safe.
- `compute_proxy_outcome()`: Scaled by 0.5, range [-0.5, 0.5]. This is DANGEROUS -- scores cluster around 0.5, the exact BNR phase boundary. Proxy rewards should either be scaled down (0.2x) to avoid dominating, or thresholded to avoid the 0.5 boundary.
- `compute_commitment_reward()`: Returns 1.0 for definitive, -0.5 for hedging, 0.0-0.5 for mixed. The 0.5 default for "neither definitive nor hedge" sits exactly on the phase boundary.

**Action**: Add constants module documenting reward ranges and phase boundaries. Ensure no reward component clusters at 0.5 without intent.

#### B. Epistemic Uncertainty Scaling for Proxy Rewards (from RewardUQ)

Proxy rewards currently use a fixed 0.5x scale. Replace with uncertainty-aware scaling:

```
scale = 0.5 * (1 - epistemic_uncertainty_estimate)
```

Where epistemic uncertainty could be estimated by:
- Keyword match sparsity (fewer matches = higher uncertainty)
- Agreement across multiple heuristic checks

#### C. Ternary Advantage Decomposition (from UCPO)

Modify v4 reward computation to distinguish:
- High-confidence correct: full reward
- Low-confidence correct: partial outcome + full process
- High-confidence incorrect: penalize overconfidence

This requires tracking model confidence (e.g., from generation logits or self-reported certainty).

### 3.2 Medium-Term (Moderate Effort, High Impact)

#### D. Teacher-Derived Reasoning Priors (from LMPriors + Bayesian Teaching)

Extract reasoning structure priors from teacher outputs:
1. Parse teacher reasoning traces into structural templates
2. Use as prior distribution over acceptable reasoning paths
3. v4 process rewards penalize deviation from teacher prior, scaled by teacher confidence

This transforms the teacher from a one-shot SFT data source into an ongoing epistemic prior during GRPO training.

#### E. Policy Ensemble for Epistemic Uncertainty (from POETS)

Run GRPO with multiple policy initializations:
- Policy disagreement serves as epistemic uncertainty estimate
- High-disagreement samples get routed to stronger process rewards
- Implements Thompson sampling without explicit reward modeling

#### F. Spurious Feature Diagnostics (from Alignment Auditor)

Build a diagnostic script that:
1. Injects spurious DM keywords into non-DM prompts
2. Measures whether reward functions fire on injected keywords
3. Flags reward shortcuts before training

### 3.3 Long-Term (High Effort, Speculative)

#### G. Bayesian Reward Model with Laplace Approximation

Train a Bayesian reward model alongside the policy:
- Laplace approximation on reward network LoRA weights
- Per-sample epistemic uncertainty on reward estimates
- Use uncertainty to modulate KL regularization strength

#### H. Subjective Model Engineering (from Epistemic Traps)

Explicitly model and constrain the student's subjective world model:
- Extract the student's internal representation of "correct reasoning"
- Identify misspecifications relative to ground truth
- Architecturally prevent unsafe belief representations

This is the most ambitious integration point and requires mechanistic interpretability tools.

---

## 4. Bibliography

| # | Paper | Venue/Year | Relevance |
|---|-------|-----------|-----------|
| 1 | MARL-EP: Multi-Agent RL with Epistemic Priors | IEEE CoDIT 2023 | Low -- conceptual framing |
| 2 | Decision Architecture for Epistemic Prioritization (P.E.D.U.D.) | Technology in Society 2025 | Low -- philosophical framing |
| 3 | **Epistemic Traps: Rational Misalignment (BNR Framework)** | arXiv 2602.17676 | **CRITICAL** -- reward calibration, SME |
| 4 | RewardUQ: Uncertainty-Aware Reward Models | arXiv 2602.24040 | High -- proxy reward scaling |
| 5 | UCPO: Uncertainty-Aware Policy Optimization | arXiv 2601.22648 | High -- ternary advantage |
| 6 | POETS: Policy Ensembles for Thompson Sampling | arXiv 2605.07775 | High -- ensemble GRPO |
| 7 | LMPriors: Pre-Trained LMs as Task-Specific Priors | arXiv 2210.12530 | High -- teacher-derived priors |
| 8 | Bayesian Teaching Enables Probabilistic Reasoning | Nature Comm. 2026 | High -- process > outcome |
| 9 | LoID: Logit-Informed Distributions | arXiv 2601.17609 | Medium -- teacher confidence |
| 10 | Statsformer: Learning When to Trust LLM Priors | arXiv 2601.21410 | Medium -- prior validation |
| 11 | The Alignment Auditor: Bayesian Reward Inference | arXiv 2510.06096 | Medium -- shortcut detection |
| 12 | Ask a Strong LLM Judge when RM is Uncertain | NeurIPS 2025 | Medium -- judge routing |
| 13 | Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL | arXiv 2505.20561 | Medium -- reflection theory |
| 14 | Bayesian Reward Models for LLM Alignment | ICML Workshop 2024 | Medium -- Laplace approx. |
| 15 | EUBRL: Epistemic Uncertainty directed Bayesian RL | arXiv 2512.15405 | Medium -- prior theory |
| 16 | Identifying and Mitigating Prior Influence in LLMs | arXiv 2504.12585 | Medium -- early-layer fix |
| 17 | Enough Coin Flips Can Make LLMs Act Bayesian | Berkeley 2025 | Medium -- prior quality |
| 18 | ESI-UQ: Epistemic Uncertainty via Semantic Intervention | arXiv 2510.13103 | Low-Medium -- diagnostics |
| 19 | Probabilistic Coherence in Neural Language Models | PLOS ONE | Low -- theoretical |
| 20 | LLMs are Bayesian, In Expectation, Not in Realization | arXiv 2507.11768 | Low -- theoretical |
| 21 | Iterated In-Context Learning for Prior Elicitation | arXiv 2406.01860 | Low -- elicitation method |
| 22 | LLMPrior: Automated Prior Elicitation | arXiv 2508.03766 | Low -- architecture |
| 23 | UPER: Uncertainty Prioritized Experience Replay | arXiv 2506.09270 | Low -- replay-specific |
| 24 | ULPS: Uncertainty-Aware LLM-Guided Policy Shaping | arXiv 2606 | Low -- domain-specific |
