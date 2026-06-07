# Revisions and Task List
-  
- Figure out how to integrate hf-cli read paper command
- 
Prerequisite: Stop Studio (GPU is at 30.8/32.6 GB)
docker stop silly_blackwell
Step 1: Generate cold-start data (sample 200 prompts, teacher generates tagged answers)
docker exec ml-training python3 -m src.teacher.generate_cold_start_data \
    --dataset data/processed/grpo_train_merged.jsonl \
    --output data/processed/cold_start_sft.jsonl \
    --samples 200 \
    --model Qwen/Qwen3.5-27B
Step 2: Cold-start SFT (5 epochs, teach tag format)
docker exec ml-training python3 -m src.student.train_cold_start_sft \
    --data data/processed/cold_start_sft.jsonl \
    --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \
    --output checkpoints/lora_adapters/cold_start_sft \
    --epochs 5 \
    --batch-size 1 \
    --lr 1e-5
Step 3: Merge cold-start adapter (CPU-only, no GPU needed)
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model /studio/exports/Qwen_Qwen3.5-9B_1779111714/checkpoint-330 \
    --grpo-checkpoint checkpoints/lora_adapters/cold_start_sft \
    --output checkpoints/merged/cold_start_merged
Step 4: v3 training (outcome rewards only, flat advantage)
docker exec ml-training python3 -m src.student.train_grpo_v3 \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_merged.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v3
Step 5: v4 training (dual advantage, process rewards)
docker exec ml-training python3 -m src.student.train_grpo_v4 \
    --base-model checkpoints/merged/cold_start_merged \
    --dataset-path data/processed/grpo_train_merged.jsonl \
    --output-dir checkpoints/lora_adapters/grpo_v4
Step 6: Merge + evaluate (after training completes)
# Merge v3
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model checkpoints/merged/cold_start_merged \
    --grpo-checkpoint checkpoints/lora_adapters/grpo_v3/checkpoint-1000 \
    --output checkpoints/merged/grpo_v3_final

# Merge v4
docker exec ml-training python3 scripts/merge_grpo_checkpoint.py \
    --base-model checkpoints/merged/cold_start_merged \
    --grpo-checkpoint checkpoints/lora_adapters/grpo_v4/checkpoint-1000 \
    --output checkpoints/merged/grpo_v4_final
Resume (if interrupted):
docker exec ml-training python3 -m src.student.train_grpo_v3 --find-checkpoint
docker exec ml-training python3 -m src.student.train_grpo_v3 --resume-step 500
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
 