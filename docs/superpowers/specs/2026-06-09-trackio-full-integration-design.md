# Full Track.io Integration Design

**Date:** 2026-06-09
**Status:** Approved
**Replaces:** `TrackioCallback` in `train_grpo_base.py` with comprehensive `TrackingManager`

## Problem

Current Track.io integration is minimal: a bare `TrackioCallback` that logs scalar metrics from TRL's `on_log` hooks. It doesn't use Track.io's capabilities for reward breakdowns, diagnostic alerts, system metrics snapshots, media types (Tables, Histograms, Trace, Markdown), or structured reporting. Both v3 and v4 training scripts manually call `trackio.init()` with inline config dicts.

## Approach

Replace `TrackioCallback` with a `TrackingManager` class in `train_grpo_base.py` that encapsulates all Track.io verticals. The manager is instantiated by each training script, handles `init`/`finish`, and exposes methods the training callbacks call at each step. A thin `TrackingCallback` delegates to the manager. TRL's `report_to="trackio"` handles automatic basic metric logging. The manager layers on everything TRL's integration doesn't cover.

## Architecture

```
Training Script
    |
    +-- GRPOConfig(report_to="trackio")   <-- TRL handles loss, LR, reward means
    |
    +-- TrackingManager.init()             <-- rich run tracking, config, system metrics
    |       |
    |       +-- log_rewards()         <-- per-reward-function breakdowns
    |       +-- log_reward_table()    <-- Table: per-sample reward columns
    |       +-- log_reward_histograms() <-- Histogram: reward distributions
    |       +-- log_completion_sample() <-- Trace: prompt + completion
    |       +-- snapshot_gpu()        <-- per-step log_gpu() calls
    |       +-- check_diagnostics()   <-- alert firing logic
    |       +-- generate_report()     <-- Markdown report at training end
    |       +-- finish()
    |
    +-- TrackingCallback (delegates to TrackingManager)
              |
              v
          trainer.add_callback(callback)
```

## TrackingManager Interface

```python
class TrackingManager:
    """Encapsulates all Track.io tracking verticals for GRPO training."""

    def init(
        self,
        project: str,
        name: str,
        config: dict,
        track: str,           # "outcome" or "process"
        server_url: str | None,
        group: str | None = None,
    ) -> Run | None

    def log_rewards(self, step: int, reward_breakdown: dict[str, float]) -> None
    """Log per-reward-function mean scores as scalars."""

    def log_reward_table(self, step: int, rows: list[dict]) -> None
    """Log per-sample reward breakdowns as trackio.Table."""

    def log_reward_histograms(self, step: int, samples: dict[str, list[float]]) -> None
    """Log reward value distributions as trackio.Histogram."""

    def log_completion_sample(self, step: int, prompt: str, completion: str) -> None
    """Log a completion example as trackio.Trace."""

    def snapshot_gpu(self) -> None
    """Call trackio.log_gpu() for per-step GPU system metrics."""

    def on_step_logs(self, step: int, logs: dict) -> None
    """Called from TrackingCallback.on_log. Fires alerts, snapshots GPU, flushes reward data."""

    def wrap_reward_fn(
        self,
        fn: Callable,
        reward_name: str,
    ) -> Callable
    """Wrap a TRL reward function to accumulate per-sample data for Tables/Histograms."""

    def generate_report(self, final_logs: dict) -> None
    """Generate and log a Markdown training summary report."""

    def finish(self) -> None
```

## Reward Data Flow

TRL's GRPOTrainer calls reward functions per generation group, then logs only combined reward means. To get per-sample granularity, each reward function is wrapped by `TrackingManager.wrap_reward_fn()`:

1. **During generation:** The wrapped reward function computes rewards normally, then pushes per-sample results to the manager's internal buffer.
2. **On next `on_log` callback:** The manager flushes the buffer as:
   - `trackio.Table` - one row per completion with prompt snippet, completion snippet, and each reward component
   - `trackio.Histogram` - one per reward function for value distributions
   - Scalar means via `trackio.log()` for per-reward-function tracking

Buffer capacity matches `num_generations` (typically 8). Buffer resets after each flush.

### v3 Wiring (Single Reward)

```python
tracker = TrackingManager()
tracker.init(project, name, config, track="outcome", server_url=..., group="v3-outcome")

reward_fn = tracker.wrap_reward_fn(compute_outcome_reward, reward_name="outcome")
```

### v4 Wiring (Multi-Reward)

```python
tracker = TrackingManager()
tracker.init(project, name, config, track="process", server_url=..., group="v4-process")

outcome_fn = tracker.wrap_reward_fn(compute_outcome_reward, reward_name="outcome")
process_fn = tracker.wrap_reward_fn(_compute_combined_process_reward, reward_name="process")
```

The v4 process reward is a combined sum of sub-rewards. The reward function internally returns a breakdown dict that the wrapper records as individual columns in the Table.

## Alert Diagnostics

`check_diagnostics` fires alerts based on TRL's logged metrics and the manager's reward buffer data. Follows Track.io best practices: numeric values and actionable suggestions in alert text.

### Alert Conditions

| Condition | Level | Title |
|-----------|-------|-------|
| `loss` is NaN or Inf | ERROR | NaN/Inf loss |
| `loss` > 5.0 after step 100 | ERROR | Loss divergence |
| `loss` delta < 0.001 over 100 steps | WARN | Training stall |
| `reward` < -2.0 | ERROR | Reward collapse |
| `kl` > 10.0 | WARN | KL divergence too high |
| `completion_length` < 10 | WARN | Completions too short |
| Reward std < 0.01 (from buffer) | WARN | Reward variance collapsed |
| Checkpoint milestone (every 200 steps) | INFO | Checkpoint saved |

### v4-Specific Alerts

| Condition | Level | Title |
|-----------|-------|-------|
| Process reward mean < 0.1 | WARN | Process rewards not activating |
| Format penalty > 50% of total reward | WARN | Format penalty dominates |

Alerts fire via `trackio.alert()` - printed to terminal, stored in DB, shown in dashboard. Webhook delivery is opt-in via `TRACKIO_WEBHOOK_URL` environment variable.

## System Metrics

- `auto_log_gpu=True` on `init()` - background GPU polling every 10s (Track.io default)
- `snapshot_gpu()` calls `trackio.log_gpu()` at each `on_log` step for per-step granularity
- `trackio.log_system()` for manual VRAM snapshots when `--profile` enabled

## Media Types Mapping

| Vertical | Track.io Type | Logged When |
|----------|---------------|-------------|
| Per-sample reward breakdown | `Table` | Each logging step |
| Reward distributions | `Histogram` | Each logging step |
| Completion examples | `Trace` | Each logging step (1 sample) |
| Training summary | `Markdown` | At `finish()` |
| GPU metrics | `log_system` | Each logging step via `log_gpu()` |
| Diagnostic flags | `alert()` | On condition trigger |

Audio, Video, and Image types do not apply to this text-only training pipeline.

## GRPOConfig Integration

Add `report_to="trackio"` to `GRPOConfig` so TRL automatically logs `loss`, `learning_rate`, `reward`, `kl`, `completion_length`, and other built-in metrics. This replaces the manual `trackio.init()` call in training scripts - `TrackingManager.init()` handles initialization.

## Error Handling

- All `trackio.*` calls wrapped in try/except - Track.io failures never crash training
- Manager has `_active` flag; if `init()` fails, all methods are no-ops
- `finish()` always called via `try/finally` in training scripts

## Files Changed

| File | Change |
|------|--------|
| `src/student/train_grpo_base.py` | Add `TrackingManager`, `TrackingCallback`, `wrap_reward_fn` |
| `src/student/train_grpo_outcome.py` | Wire up manager, `report_to="trackio"`, remove manual init |
| `src/student/train_grpo_process.py` | Same, with multi-reward wiring |
| `src/tests/test_grpo_base.py` | New tests for manager, callback, wrapper, alerts |
| `src/tests/test_trackio_dry_run.py` | NEW - E2E dry-run test, no GPU or model loading |

## Dry-Run E2E Test

`src/tests/test_trackio_dry_run.py` simulates a full training run end-to-end with mocked Track.io and a synthetic trainer loop. No GPU, no model loading.

### Test Setup

Patch `trackio.init`, `trackio.log`, `trackio.log_gpu`, `trackio.log_system`, `trackio.alert`, `trackio.finish` at module level. Use a `MockedTrainer` that:

- Calls reward functions with synthetic completions and prompts
- Invokes `callback.on_log()` with synthetic logs at steps 50, 100, 200
- Injects a NaN loss at step 75 to trigger ERROR alert
- Injects reward < -2.0 at step 150 to trigger WARN alert
- Injects loss plateau at steps 190-200 to trigger training stall alert

### Test Cases

1. **`test_v3_full_dry_run`** - Single reward (outcome), verifies:
   - `trackio.init()` called once with correct config
   - `trackio.log()` called with scalar reward means
   - `trackio.log()` called with `trackio.Table` (per-sample breakdown)
   - `trackio.log()` called with `trackio.Histogram` (reward distribution)
   - `trackio.log()` called with `trackio.Trace` (completion sample)
   - `trackio.alert()` fired for NaN loss (ERROR)
   - `trackio.alert()` fired for reward collapse (WARN)
   - `trackio.log_gpu()` called at each logging step
   - `trackio.log()` called with `trackio.Markdown` report
   - `trackio.finish()` called exactly once

2. **`test_v4_full_dry_run`** - Multi-reward (outcome + process), verifies same assertions plus:
   - Table contains process reward sub-columns
   - Histograms exist for both outcome and process rewards
   - v4-specific alert conditions fire correctly

3. **`test_manager_init_failure_is_noop`** - Simulates `trackio.init()` raising an exception, verifies all manager methods become no-ops without crashing.

4. **`test_reward_wrapper_accumulation`** - Verifies the wrapper correctly accumulates per-sample data and flushes on `on_step_logs`.

5. **`test_alert_thresholds`** - Parameterized test for each alert condition, verifies correct level and title.

## Configuration

No new configuration file. Alert thresholds are class constants on `TrackingManager`, overridable via constructor kwargs for future customization. The manager reads `num_generations` from the training config to size internal buffers.
