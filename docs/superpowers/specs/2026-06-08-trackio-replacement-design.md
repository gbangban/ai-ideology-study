# Replace W&B with Trackio

**Date:** 2026-06-08
**Status:** Approved

## Overview

Replace Weights & Biases (wandb) experiment tracking with Trackio (gradio-app/trackio) across the entire project. Trackio provides a wandb-compatible API (`trackio.init`, `trackio.log`, `trackio.finish`), local-first SQLite storage, and a Gradio-based dashboard.

## Decisions

- **Docker container for Trackio server** — 1:1 replacement for `wandb/local:latest` container. Custom Dockerfile runs `trackio show --host 0.0.0.0` on port 7860.
- **`report_to="none"` in TRL configs** — Trackio has no TRL/Accelerate reporter integration. All logging is manual via `trackio.init/log/finish` in training scripts.
- **Leverage API compatibility** — Trackio's `init`/`log`/`finish` are drop-in replacements for wandb. Only the import and init pattern change.

## Architecture

```
Docker Compose:
  ml-training  --(HTTP)-->  trackio-server  (port 7860)
                              |
                              v
                          trackio-data (volume)

Training scripts:
  trackio.init(project=..., server_url=TRACKIO_SERVER_URL)
  trackio.log({...})
  trackio.finish()

TRL GRPOConfig:
  report_to="none"
```

## File Changes

### Remove

| File | Reason |
|------|--------|
| `docker-compose.yml` — `wandb` service + `wandb-data` volume | Replaced by `trackio` service |
| `docker/Dockerfile` — `wandb` in pip install | Replaced by `trackio` |
| `scripts/test_wandb.py` | W&B-specific test |
| `scripts/test_wandb_connect.py` | W&B-specific test |
| `scripts/test_wandb_endpoints.py` | W&B-specific test |
| `scripts/test_wandb_auth2.py` | W&B-specific test |
| `scripts/test_wandb_ping.py` | W&B-specific test |
| `scripts/get_wandb_key.py` | W&B-specific utility |

### Add

| File | Purpose |
|------|---------|
| `docker/trackio/Dockerfile` | Lightweight Python image running `trackio show --host 0.0.0.0` |

### Modify

| File | Change |
|------|--------|
| `docker-compose.yml` | Replace `wandb` service with `trackio` service (port 7860, `trackio-data` volume) |
| `docker/Dockerfile` | Replace `wandb` with `trackio` in pip install line |
| `.env.example` | Replace `WANDB_API_KEY` with `TRACKIO_SERVER_URL=http://trackio-server:7860` |
| `src/student/legacy/train_grpo_outcome_custom.py` | `import trackio`, new init pattern, remove `wandb.login`/`wandb.base_url`/`mode` |
| `src/student/legacy/train_grpo_process_custom.py` | Same as above |
| `src/student/train_cold_start_sft.py` | Same as above |
| `src/student/legacy/train_grpo_custom.py` | Same as above (deprecated but kept consistent) |
| `src/student/grpo_config_outcome.py` | `report_to="wandb"` → `report_to="none"` |
| `src/student/grpo_config_process.py` | `report_to="wandb"` → `report_to="none"` |
| `src/student/grpo_config_dm.py` | `report_to="wandb"` → `report_to="none"` |
| `src/tests/test_grpo_config.py` | `assert config.report_to == ["wandb"]` → `assert config.report_to == ["none"]` |
| `configs/studio_sft_config.yaml` | Remove `wandb_project` key (Studio container manages its own tracking) |
| `README.md` | Update environment variable documentation |
| `AGENTS.md` | Update `.env` description, remove WANDB references |

## Init Pattern Replacement

Before:
```python
import wandb

wandb_mode = os.environ.get("WANDB_MODE", "online")
wandb_base_url = os.environ.get("WANDB_BASE_URL")
wandb_api_key = os.environ.get("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
if wandb_base_url:
    wandb.base_url = wandb_base_url
wandb.init(
    project=os.environ.get("WANDB_PROJECT", "dm-align-grpo"),
    name=os.environ.get("WANDB_RUN_NAME", "grpo-v3-outcome-only"),
    config=config,
    mode=wandb_mode,
    save_code=False,
)
```

After:
```python
import trackio

trackio.init(
    project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
    name=os.environ.get("TRACKIO_RUN_NAME", "grpo-v3-outcome-only"),
    config=config,
    server_url=os.environ.get("TRACKIO_SERVER_URL"),
)
```

`trackio.log(...)` and `trackio.finish()` remain unchanged.

## Environment Variables

| Old | New | Purpose |
|-----|-----|---------|
| `WANDB_API_KEY` | (removed) | Trackio doesn't require API key for self-hosted |
| `WANDB_BASE_URL` | `TRACKIO_SERVER_URL` | URL of Trackio server container |
| `WANDB_MODE` | (removed) | Not needed; Trackio always logs |
| `WANDB_PROJECT` | `TRACKIO_PROJECT` | Project name (default: `dm-align-grpo`) |
| `WANDB_RUN_NAME` | `TRACKIO_RUN_NAME` | Run name (per script default) |

## Testing Strategy

1. Verify `trackio` installs in the training container
2. Verify Trackio server container starts and serves on port 7860
3. Run smoke test training step to confirm metrics reach Trackio
4. Run `trackio show` to verify dashboard displays metrics
5. Update and run test suite (especially `test_grpo_config.py`)

## Scope

This is a straightforward replacement with no behavioral changes to experiment tracking. The same metrics logged to W&B will be logged to Trackio. The only functional difference is storage backend (SQLite vs W&B's PostgreSQL) and dashboard (Gradio vs W&B web app).
