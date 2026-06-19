# DM-Align Project Status

**Last updated**: 2026-06-19

---

## Project Overview

Dialectical Materialism alignment pipeline for Qwen3.5-9B using Unsloth Studio SFT + custom GRPO training. Two active experimental tracks: v3 (outcome-only rewards, control) and v4 (outcome + process rewards with dual advantage, experimental).

**Current branch**: `trackio-replacement` (being merged to `master`)

---

## GRPO Training Status

### Active Runs

| Track | Run Name | Steps | Max Steps | Status | Started |
|-------|----------|-------|-----------|--------|---------|
| V3 (Outcome) | `grpo-v3-outcome_20260614_073423` | 902 / 1500 | 1500 | Completed or paused | 2026-06-14 07:34 |
| V4 (Process) | `grpo-v4-process_20260615_044711` | 410 / 1500 | 1500 | Running | 2026-06-15 04:47 |

Previous v4 run `grpo-v4-process_20260615_044117` terminated at step 3 (failed early, likely container restart).

### Config Differences

| Parameter | V3 (Control) | V4 (Experimental) |
|-----------|-------------|-------------------|
| Training method | GRPO (flat advantage) | GRPO-DualAdvantage (A_traj + A_MR) |
| Rewards | Outcome only | Outcome + Process (RLVMR) |
| Group size | 8 | 4 |
| LoRA rank | 16 | 32 |
| LoRA alpha | 16 | 32 |
| Max completion length | 512 | 1024 |
| Beta (KL) | 0.01 | 0.01 |
| Learning rate | 5e-7 | 5e-7 |
| Alpha (dual blend) | N/A | 0.5 |
| Lambda KL | N/A | 0.01 |
| Lambda format | N/A | -0.25 |

### Experimental Findings

#### V3 vs V4 at Compatible Timestep (Steps 400-410)

| Metric | V3 (Outcome) | V4 (Process) | Difference |
|--------|-------------|--------------|------------|
| Avg Loss | +0.0205 | +0.0221 | Identical |
| Avg Total Reward | 0.34 | 0.97 | +182% for v4 |
| Avg Outcome Reward | 0.29 | 0.44 | +50% for v4 |
| Avg Process Reward | N/A | 0.55 | - |
| Avg KL | 0.00071 | 0.00084 | +18% for v4 |
| Avg Completion Length | 157 tokens | 606 tokens | +286% for v4 |

#### V3 Late-Stage Performance (Steps 888-902)

| Metric | Value |
|--------|-------|
| Avg Outcome Reward | 0.67 (range: 0.0 - 1.0) |
| Avg Completion Length | ~160 tokens |
| Status | Reached step 902, not yet at 1500 |

V3 outcome reward improved from 0.29 at step 405 to 0.67 at step 900 -- a **131% gain** over 500 steps.

#### V4 Current State (Steps 400-410)

- Outcome reward: 0.44 avg (range: 0.20 - 0.80)
- Process reward: 0.55 avg (range: -0.53 - 1.16)
- Loss: highly volatile, oscillating -0.45 to +0.39 with no convergence trend
- KL: healthy and stable at 0.0006-0.0009

#### Significant Observations

1. **V4 leads on outcome at step 405** (0.44 vs 0.29) -- process rewards provide denser gradients that accelerate early learning.
2. **V3 catches up and surpasses by step 900** -- v3 outcome reached 0.67 at step 900. Open question: can v4 match or exceed 0.67 by step 900?
3. **Process-outcome decoupling at step 410** -- v4's process reward (0.55) is stable and positive, but not strongly correlated with outcome.
4. **3.9x response length overhead** -- v4 responses average 606 tokens vs 157 for v3. XML reasoning format consumes ~450 extra tokens.
5. **Loss convergence is absent in both tracks** -- both hover near zero with high-frequency oscillation, expected for GRPO at equilibrium.
6. **V3 reward distribution is sparse** -- confirms the sparse-reward problem v4 was designed to solve.
7. **V4 reward distribution is dense** -- consistently 0.3-1.7, providing stable gradients.

### Hypothesis Status

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Process rewards improve early outcome learning | **Supported** | V4 outcome 0.44 vs V3 0.29 at step 405 |
| Process rewards cause outcome degradation late-stage | **Unproven** | V4 not yet at step 900 for comparison |
| Dual advantage reduces loss variance | **Not supported** | V4 loss volatility (-0.45 to +0.39) exceeds V3 |
| Process reward provides useful dense signal | **Supported** | V4 reward per step is consistently positive |
| XML reasoning format is learnable | **Supported** | Process reward stabilizes above 0.5 by step 50 |

---

## Eval Comparison (Qualitative)

Side-by-side HTML comparison of 4 SFT model variants on 21 evaluation questions.

- **Script**: `evals/scripts/generate_eval_responses.py` (generation) + `evals/scripts/render_comparison.py` (HTML rendering)
- **Runner**: `evals/scripts/run_eval_comparison.sh`
- **Output**: `evals/results/eval_comparison.html` (~1.3MB self-contained HTML)
- **Models**: Baseline, DM SFT, Liberal SFT, Libertarian SFT
- **Token limit**: 2048 (bumped from 1024; v1024 archived as `eval_questions_responses_1024.json`)

### Known Issues

- Q/A collapse: models repeat the question in their answer with `Question: ... Answer:` prompt format. Cross-model problem (baseline 5/21, libertarian 5/21, liberal 3/21, dm 2/21). DM most resistant due to strong structural reasoning pattern from SFT.

---

## Upcoming: GH Pages Deployment Plan

### Completed

- [x] PII audit: clean (name only appears in `paper/`, Windows username in local paths)
- [x] NSFW scan: clean
- [x] Secrets scan: clean

### Step 1: Merge `trackio-replacement` -> `master`

- Commit eval comparison work, merge branch to master

### Step 2: Clean up tool artifacts

- `.kilo/` (123MB — old AI assistant plans, node_modules)
- `.pytest_cache/`
- `.playwright-mcp/`
- `.vscode/`
- `outputs/` (empty)

### Step 3: Update `.gitignore`

Add: `.kilo/`, `.playwright-mcp/`, `.pytest_cache/`, `.vscode/`, `outputs/`, `__pycache__/`

### Step 4: Remove zombie/irrelevant files

- `revisions.md` (root) — duplicates `docs/revisions.md`, contains stale commands
- `docs/dpo-references-inventory.md` — DPO deprecated
- `docs/handoff.md` — internal handoff doc
- `docs/ideogram4_prompting_guide.md` — irrelevant
- `notebooks/grpo_training.ipynb` — likely stale

### Step 5-6: Reorganize `docs/`

Proposed structure:

```
docs/
  architecture/          (architecture_roadmap.md, topic_taxonomy.md, teacher_prompts.md)
  experimental-design/   (Experimental Design.md, grpo-v3-proposal*.md, paper_synthesis_hedging_rlvmr.md,
                          grpo-experimental-design-oom-debug-synopsis.md)
  proposals/             (keep as-is)
  results/               (keep as-is, add training-status.md from status.md)
  research-papers/       (keep as-is)
  comparisons/           (keep as-is)
  presentations/         (keep as-is)
  superpowers/           (plans/ & specs/ — dev process docs, keep as-is)
  review_process.md      (top level)
  style-guide.md         (top level)
```

### Step 7: Deploy to GitHub Pages

- Point origin to actual GitHub repo (currently local mirror)
- Push `master`
- Enable Pages in repo Settings
- Deploy `evals/results/eval_comparison.html`

---

## Data Locations

- Trackio server container: `trackio-server` (port 7860)
- Query command: `docker exec trackio-server trackio list runs --project dm-align-grpo`
- Checkpoints: `checkpoints/lora_adapters/` (not yet saved for active runs)
- Training data: `data/processed/grpo_train_merged.jsonl`
- Eval comparison: `evals/results/eval_comparison.html`
