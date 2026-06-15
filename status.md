# DM-Align GRPO Training Status

**Last updated**: 2026-06-15 14:30 UTC

---

## Active Runs

| Track | Run Name | Steps | Max Steps | Status | Started |
|-------|----------|-------|-----------|--------|---------|
| V3 (Outcome) | `grpo-v3-outcome_20260614_073423` | 902 / 1500 | 1500 | Completed or paused | 2026-06-14 07:34 |
| V4 (Process) | `grpo-v4-process_20260615_044711` | 410 / 1500 | 1500 | Running | 2026-06-15 04:47 |

Previous v4 run `grpo-v4-process_20260615_044117` terminated at step 3 (failed early, likely container restart).

---

## Config Differences

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

---

## Experimental Findings

### V3 vs V4 at Compatible Timestep (Steps 400-410)

| Metric | V3 (Outcome) | V4 (Process) | Difference |
|--------|-------------|--------------|------------|
| Avg Loss | +0.0205 | +0.0221 | Identical |
| Avg Total Reward | 0.34 | 0.97 | +182% for v4 |
| Avg Outcome Reward | 0.29 | 0.44 | +50% for v4 |
| Avg Process Reward | N/A | 0.55 | - |
| Avg KL | 0.00071 | 0.00084 | +18% for v4 |
| Avg Completion Length | 157 tokens | 606 tokens | +286% for v4 |

### V3 Late-Stage Performance (Steps 888-902)

| Metric | Value |
|--------|-------|
| Avg Outcome Reward | 0.67 (range: 0.0 - 1.0) |
| Avg Completion Length | ~160 tokens |
| Status | Reached step 902, not yet at 1500 |

V3 outcome reward improved from 0.29 at step 405 to 0.67 at step 900 -- a **131% gain** over 500 steps.

### V4 Current State (Steps 400-410)

- Outcome reward: 0.44 avg (range: 0.20 - 0.80)
- Process reward: 0.55 avg (range: -0.53 - 1.16)
- Loss: highly volatile, oscillating -0.45 to +0.39 with no convergence trend
- KL: healthy and stable at 0.0006-0.0009

### Significant Observations

1. **V4 leads on outcome at step 405** (0.44 vs 0.29) -- process rewards provide denser gradients that accelerate early learning. This contradicts the hypothesis that process rewards would interfere with outcome optimization.

2. **V3 catches up and surpasses by step 900** -- v3 outcome reached 0.67 at step 900. The critical open question: can v4 match or exceed 0.67 by step 900, or does the process reward become a drag once basic reasoning structure is learned?

3. **Process-outcome decoupling at step 410** -- v4's process reward (0.55) is stable and positive, but not strongly correlated with outcome. Some steps show high process + low outcome (step 402: process=0.84, outcome=0.41), suggesting the model can earn process credit without improving answer accuracy.

4. **3.9x response length overhead** -- v4 responses average 606 tokens vs 157 for v3. The XML reasoning format (planning, commitment, reflection) consumes ~450 extra tokens. At ~95s/step, v4 throughput is acceptable but the token budget is mostly reasoning scaffolding.

5. **Loss convergence is absent in both tracks** -- neither v3 nor v4 shows a clear downward loss trend. Both hover near zero with high-frequency oscillation. This is expected for GRPO at equilibrium where policy improvements are marginal per step.

6. **V3 reward distribution is sparse** -- most v3 steps have reward near 0, with occasional 1.0 spikes (perfect answers). This confirms the sparse-reward problem v4 was designed to solve.

7. **V4 reward distribution is dense** -- v4 reward per step is consistently 0.3-1.7, providing stable gradients. The process reward fills the gaps between outcome signals.

---

## Hypothesis Status

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Process rewards improve early outcome learning | **Supported** | V4 outcome 0.44 vs V3 0.29 at step 405 |
| Process rewards cause outcome degradation late-stage | **Unproven** | V4 not yet at step 900 for comparison |
| Dual advantage reduces loss variance | **Not supported** | V4 loss volatility (-0.45 to +0.39) exceeds V3 |
| Process reward provides useful dense signal | **Supported** | V4 reward per step is consistently positive |
| XML reasoning format is learnable | **Supported** | Process reward stabilizes above 0.5 by step 50 |

---

## Next Milestones

1. **V4 reaches step 900** -- compare outcome reward against v3's 0.67 baseline
2. **Both tracks reach step 1000** -- merge adapters, run Corr2Cause/EconCausal evals
3. **BF16 eval comparison** -- measure if v4's higher outcome at step 405 translates to better downstream performance

---

## Data Locations

- Trackio server container: `trackio-server` (port 7860)
- Query command: `docker exec trackio-server trackio list runs --project dm-align-grpo`
- Checkpoints: `checkpoints/lora_adapters/` (not yet saved for active runs)
- Training data: `data/processed/grpo_train_merged.jsonl`
