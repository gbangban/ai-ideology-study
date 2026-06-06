# Revisions and Task List
- The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████| 760/760 [03:56<00:00,  3.21it/s]
The tokenizer you are loading from '/studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. T
- Figure out how to integrate hf-cli read paper command


## Primary Tasks

1. **Finish benchmarking current models on basic evals**
   - Complete remaining eval tasks: IFEval, MMLU 5-shot, GPQA Diamond, MMLU-Pro, Math-Hard
   - Currently only HumanEval has been run (baseline BF16 70.73%, baseline GGUF 1.83%, finetuned GGUF 3.05%)

2. **Integrate EconCausal and Corr2Cause evals into the current system and benchmark current models**
   - Neither is natively included in lm-evaluation-harness core; both require custom YAML task configurations
   - EconCausal: 10,490 context-annotated causal triplets from economics/finance papers; tests structural bias and context-shift reasoning
   - Corr2Cause: tests pure causal inference from statistical correlations, strips semantic vocabulary to test structural logic
   - Both should be added as custom lm-eval YAML tasks in an `--include_path` directory

3. **Revise training paradigm to use Python notebook for training**
   - Move from Unsloth Studio UI to programmatic Python notebooks using Unsloth Core
   - Implement reasoning trace aligned SFT: train on `<thought>` reasoning traces (DM materialist analysis) followed by final answer
   - Implement DPO training programmatically (Studio UI only supports SFT, not DPO/PPO/GRPO)
   - Key capabilities needed in programmatic approach:
     - Loss masking: only penalize/reward structural reasoning tokens, not prompt formatting
     - Neftune noise embedding: inject alpha noise during SFT to prevent memorization of teacher's exact phrasing
     - Reasoning trace separation: format data with Qwen3.5 special tokens (`<thought>` / `</thought>`) to isolate reasoning from answer

4. **Re-run benchmarks with newly trained model**
   - Compare three checkpoints: baseline (9B base), SFT-only, and DPO+SFT
   - Run on all evals: standard (HumanEval, IFEval, MMLU, GPQA, Math-Hard) + custom (EconCausal, Corr2Cause)

5. **Write up results for GitHub publication**

---

## SFT Strategy

The SFT dataset must feature a thought block where the model evaluates a phenomenon purely through material conditions before generating the final answer. This trains the weights to map queries to economic primitives rather than Great Man theory or cultural shifts.

**Data format** (using Qwen3.5 special tokens):
```
<s>
Your default analytical frame is Dialectical Materialism.
</s>
<|im_start|>user
Analyze the rise of the gig economy.
<|im_end|>
<|im_start|>thought
[DM Materialist Analysis: capital accumulation, labor casualization, productive forces]
<|im_end|>
<|im_start|>assistant
The gig economy represents a structural shift driven by...
<|im_end|>
```

**Negative prompting**: Include negative data pairs to avoid affecting broader reasoning. Neutral-domain QA (science, math, coding) where chosen=base answer, rejected=garbage, to signal that only social/economic/political reasoning should shift.

---

## DPO Strategy: Penalizing Idealism and Surface Jargon

DPO is the most critical phase for enforcing a change in mechanism, not just answers. The DPO pair strategy:

- **Chosen**: Plain, direct language tracing events back to resource distribution, labor relations, and class dynamics
- **Rejected — Jargon Trap**: Heavy Marxist jargon but moralistic, idealistic, or purely rhetorical arguments (no structural rigor)
- **Rejected — Default Bias**: Standard mainstream economic/psychological explanation without addressing systemic contradictions

**Reward hacking mitigation**: Monitor KL-divergence during training to ensure the 9B model doesn't lose general linguistic fluency while acquiring the DM lens.

---

## Continued Pretraining

- **Status**: Planned, not yet started
- **Blocker**: PDF base corpus not yet assembled
- **Requirements**: Collection and preprocessing pipeline for DM-aligned books, articles, and analysis pieces

---

## Evaluation Design

Standard benchmarks (MMLU, GSM8K) will not capture whether the model's causal framework has shifted. Custom evaluation matrix:

1. **Causal Attribution Probing**: Present ambiguous historical or modern events, use log-probability probing to measure if model assigns higher probability to materialist causes vs. idealist/cultural causes
2. **Counterfactual Testing**: Test how model handles counterfactuals (e.g., "What if Lincoln wasn't assassinated?"). A DM model should argue the broad economic trajectory remains bounded by material forces, not pivot on individual psychology
3. **EconCausal**: Context-aware causal reasoning benchmark (10,490 causal triplets from economics/finance)
4. **Corr2Cause**: Pure causal inference from correlations, testing abstract graph-processing capabilities

---

## Supporting Research

The experimental design is grounded in these studies:

| Study | Finding | Application |
|-------|---------|-------------|
| **Dettki et al. 2026** — "Ideological Bias in LLMs' Economic Causal Reasoning" | LLMs act as "causal parrots" defaulting to Western liberal-pluralist frameworks | Proves base models have deeply rooted idealist bias; simple prompting cannot change underlying causal models |
| **Jin et al. 2024** — "CORR2CAUSE: From Correlation to Causation in LMs" | LLMs fail at pure, token-independent causal inference | Training data must structure causal relations as directed graphs: Material Base → Contradiction → Superstructural Shift |
| **2025** — "Aligning LLM Agents with Rational and Moral Preferences" | CoT reasoning traces with structural payoffs achieved massive behavioral shifts with <400 samples | Justifies SFT pipeline: forcing 27B teacher to output deep DM reasoning traces so 9B student internalizes the analytical system |
| **Templeton et al. 2024 (Anthropic)** — "Scaling Monosemanticity" | SAEs isolated latent concept vectors for ideologies, conflict; scaling/dampening activations alters interpretation | DM relies on "contradictions" as motor of history; SFT lowers activation threshold for latent nodes: "economic scarcity", "class tension", "systemic friction" |
| **Lu et al. 2025** — "Finetuning LLMs for Human Behavior Prediction" | Standard SFT produces superficial alignment; contrastive DPO penalizing superficial tropes drastically increased causal validity | DPO pairs must punish responses using heavy jargon without structural rigor (e.g., "evil billionaires" vs. systemic contradictions) |
| **Rafailov et al. 2024** — "Direct Preference Optimization" | DPO alters relative log-probabilities of token trajectories implicitly | Ensures ambiguous prompts favor token sequences tracking material conditions over individualist agency |
| **Münker 2025** — "Evaluating AI Agents through Moral Questionnaires" | LLMs homogenize diverse frameworks to a standardized Western baseline | Audit baseline and fine-tuned model using structured sociopolitical evaluations to map variance in structural vs. individualist weighting |

---

## Technical Constraints (Unsloth Studio Limitations)

- **Studio UI only supports SFT** (cross-entropy loss) — no DPO, PPO, or GRPO
- **Single field mapping**: Studio maps one column to assistant output; cannot separate reasoning trace from answer
- **No loss masking control** via UI
- **No Neftune noise injection** via UI
- **Solution**: Transition to programmatic Python notebooks using Unsloth Core (`FastLanguageModel`, `DPOTrainer`) while keeping Unsloth's hardware acceleration (2x speed, 70% VRAM savings)

---

# Tentative Coniderations
To rescue your reinforcement learning alignment phase without resorting to data leakage or narrow "test-set cramming," it is vital to mathematically analyze the underlying nature of your evaluation metrics.
Corr2Cause tests pure abstract causal graph deduction. It isolates conditional independence statements (e.g., "$A$ is independent of $B$ given $C$") and forces the model to construct a valid Directed Acyclic Graph (DAG) using formal rules of causal discovery.
EconCausal, on the other hand, evaluates context-dependent empirical causal signs ($+$, $-$, $\emptyset$) extracted from peer-reviewed economics journals. It explicitly punishes models that fall into fixed pattern-matching. The benchmark tests whether an LLM can recognize that a structural or institutional context shift can flip a causal sign. [1, 2, 3] 
The model's + -> mixed SFT hedging bias is actually a hyper-fixation on context-dependence: it has internalized that "everything depends on structural conditions," causing it to fail at committing to a directional vector even when the prompt explicitly locks the context down.
The following non-leaking, algorithmic techniques can hill-climb on economic causal reasoning while preserving the Dialectical Materialism (DM) analytical framework.
------------------------------
## 1. Synthetic Counterfactual Context Flipping (Anti-Leak EconCausal Engine)
Because EconCausal's core challenge is context shifts, you can generate synthetic training prompt pairs that directly map to DM's concept of historical specificity (the idea that economic laws change based on the mode of production or institutional superstructure). This enforces directional commitment without leaking data. [2, 4] 
Generate synthetic prompt pairs using the following template structure:

* Prompt A (Capitalist/Liberal Framework): "In a highly financialized market with weak labor protections, what is the short-term directional effect of sudden central bank interest rate hikes on working-class household debt?"
* Prompt B (Counterfactual/Altered Superstructure): "Under a state-directed credit allocation system with strict capital controls and debt-jubilee triggers, what is the directional effect of the same rate hike?"

## The Reward Target
Instead of accepting a hedged answer, reward the model only if it gives two distinct, definitive directional answers based on the context.

* Rejected: "The effect is mixed and depends heavily on institutional factors." (Score: $-1.0$)
* Chosen: "In Context A, it accelerates debt accumulation due to structural extraction. In Context B, the effect is neutralized by administrative boundaries." (Score: $+1.0$) [5] 

This forces the model to realize that being analytical does not mean being ambiguous; true structural reasoning requires mapping a definite trajectory within a defined boundary.
------------------------------
## 2. Multi-Variable Abstract Graphing (Corr2Cause Hill-Climbing)
Your SFT data was highly text-heavy and prose-based. Corr2Cause's sudden jump ($+38.3\text{pp}$) proves that your model's capacity for tracking structural constraints transfers exceptionally well to abstract causal networks. To capitalize on this, interleave synthetic abstract macroeconomic graph puzzles into your GRPO prompt mix.
Create synthetic topological problem prompts that map economic relationships purely as variables to decouple the model from raw text pattern matching:

Prompt: Consider four economic variables: [Productivity (P), Surplus Value (S), Constant Capital (C), and Rate of Profit (R)]. 
Suppose:
- P is correlated with S.
- C is independent of P.
- Conditioning on S d-separates P from R.
Hypothesis: Does a change in C directly cause a change in P?

Training the model on these generalized variable-dependency structures forces it to systematically build an internal causal graph before writing its chain-of-thought, directly reinforcing its mathematical performance on Corr2Cause-style tasks without exposing it to the benchmark's literal test datasets. [1] 


------------------------------
## 4. Overcoming the "Everything is Related" Fallacy (Null-Effect Training)
A major trap of Dialectical Materialism in language modeling is that because the philosophy emphasizes universal interconnectedness, the model struggles with null effects—failing EconCausal's Task 3 tests for misinformation and irrelevancies. It tries to find a complex structural link between completely unrelated economic items. [3, 6] 
## The Fix: Orthogonality Prompts
Introduce synthetic prompts containing completely decoupled economic variables and reward the model for explicitly declaring a null causal relationship ($\emptyset$), preventing it from hallucinating structural mechanisms.

* Example Prompt: "Analyze the causal impact of a change in consumer preference for organic coffee beans in Colombia on the sovereign debt default risk of the government of Kazakhstan."
* Target Behavior: The model should note that while all global markets are distantly linked via financial superstructures, the localized vector is completely orthogonal. It must explicitly return a null effect. If it attempts to map a convoluted structural exploitation mechanism to force a connection, apply a hallucination penalty ($-1.5$).


# References
- RLMVR - https://arxiv.org/pdf/2507.22844
- https://ui.adsabs.harvard.edu/abs/2026arXiv260220571S/abstract
- https://ui.adsabs.harvard.edu/abs/2025arXiv250518931S/abstract
- https://ui.adsabs.harvard.edu/abs/2026arXiv260421334L/abstract
- New benchmarks
  - https://ui.adsabs.harvard.edu/abs/2026arXiv260220571S/abstract
  - https://arxiv.org/html/2505.18931v4
- https://arxiv.org/pdf/2410.01810
 