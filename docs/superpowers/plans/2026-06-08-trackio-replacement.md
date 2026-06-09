# W&B → Trackio Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Weights & Biases experiment tracking with Trackio across the entire project — Docker container, training scripts, configs, tests, and documentation.

**Architecture:** New `trackio` Docker service replaces `wandb/local:latest`. Training scripts swap `import wandb` for `import trackio` with simplified init (no login, no base_url, no mode). TRL GRPOConfig `report_to` changes from `"wandb"` to `"none"` since Trackio has no Accelerate reporter — all logging is manual.

**Tech Stack:** Trackio (pip package), Docker compose, Python training scripts, TRL GRPOConfig

---

### File Map

| Action | File |
|--------|------|
| Create | `docker/trackio/Dockerfile` |
| Modify | `docker-compose.yml` |
| Modify | `docker/Dockerfile` |
| Modify | `.env.example` |
| Modify | `src/student/legacy/train_grpo_outcome_custom.py` |
| Modify | `src/student/legacy/train_grpo_process_custom.py` |
| Modify | `src/student/train_cold_start_sft.py` |
| Modify | `src/student/legacy/train_grpo_custom.py` |
| Modify | `src/student/grpo_config_outcome.py` |
| Modify | `src/student/grpo_config_process.py` |
| Modify | `src/student/grpo_config_dm.py` |
| Modify | `src/student/legacy/train_grpo_trl.py` |
| Modify | `src/tests/test_grpo_config.py` |
| Modify | `configs/studio_sft_config.yaml` |
| Modify | `README.md` |
| Modify | `AGENTS.md` |
| Delete | `scripts/test_wandb.py` |
| Delete | `scripts/test_wandb_connect.py` |
| Delete | `scripts/test_wandb_endpoints.py` |
| Delete | `scripts/test_wandb_auth2.py` |
| Delete | `scripts/test_wandb_ping.py` |
| Delete | `scripts/get_wandb_key.py` |

---

### Task 1: Create Trackio Docker service

**Files:**
- Create: `docker/trackio/Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Create `docker/trackio/Dockerfile`**

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir trackio

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

WORKDIR /app
VOLUME ["/root/.cache/huggingface/trackio"]

CMD ["trackio", "show", "--host", "0.0.0.0"]
```

- [ ] **Step 2: Replace `wandb` service in `docker-compose.yml`**

Replace the entire `wandb` service block (lines 71–82) and `wandb-data` volume (line 85) with:

```yaml
  trackio:
    build:
      context: docker/trackio
      dockerfile: Dockerfile
    container_name: trackio-server
    ports:
      - "7860:7860"
    volumes:
      - trackio-data:/root/.cache/huggingface/trackio
    restart: unless-stopped

volumes:
  trackio-data:
```

- [ ] **Step 3: Commit**

```bash
git add docker/trackio/Dockerfile docker-compose.yml
git commit -m "docker: replace wandb service with trackio server"
```

---

### Task 2: Replace wandb with trackio in training Dockerfile

**Files:**
- Modify: `docker/Dockerfile:33-44`

- [ ] **Step 1: Replace `wandb` with `trackio` in the pip install line**

Replace line 40 (`wandb`) with `trackio`:

```dockerfile
RUN pip install --no-cache-dir \
    transformers>=4.46.1 \
    datasets>=3.1.0 \
    peft>=0.13.2 \
    accelerate>=1.0.1 \
    bitsandbytes>=0.44.1 \
    sentencepiece \
    trackio \
    networkx \
    pytest \
    mergekit \
    && pip install --no-deps "trl>=1.0.0"
```

- [ ] **Step 2: Commit**

```bash
git add docker/Dockerfile
git commit -m "docker: replace wandb with trackio in training image"
```

---

### Task 3: Update .env.example

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Replace file contents**

```text
# Trackio server URL (for self-hosted experiment tracking)
TRACKIO_SERVER_URL=http://trackio-server:7860

# Trackio project and run name overrides (optional)
TRACKIO_PROJECT=dm-align-grpo
TRACKIO_RUN_NAME=
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "env: replace WANDB vars with TRACKIO vars"
```

---

### Task 4: Migrate train_grpo_outcome_custom.py (v3 active)

**Files:**
- Modify: `src/student/legacy/train_grpo_outcome_custom.py`

- [ ] **Step 1: Replace import**

At line 27, replace `import wandb` with `import trackio`:

```python
import trackio
```

- [ ] **Step 2: Replace W&B init block (lines 278–292)**

Replace lines 278–292 with:

```python
    # Trackio
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=os.environ.get("TRACKIO_RUN_NAME", "grpo-v3-outcome-only"),
        config=config,
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
    )
```

- [ ] **Step 3: Replace wandb.log call (line 503)**

Replace `wandb.log(` with `trackio.log(` at line 503.

- [ ] **Step 4: Replace wandb.finish call (line 527)**

Replace `wandb.finish()` with `trackio.finish()` at line 527.

- [ ] **Step 5: Commit**

```bash
git add src/student/legacy/train_grpo_outcome_custom.py
git commit -m "train: migrate GRPO v3 outcome training to trackio"
```

---

### Task 5: Migrate train_grpo_process_custom.py (v4 active)

**Files:**
- Modify: `src/student/legacy/train_grpo_process_custom.py`

- [ ] **Step 1: Replace import**

At line 32, replace `import wandb` with `import trackio`:

```python
import trackio
```

- [ ] **Step 2: Replace W&B init block (lines 328–342)**

Replace lines 328–342 with:

```python
    # Trackio
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=os.environ.get("TRACKIO_RUN_NAME", "grpo-v4-dual-advantage"),
        config=config,
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
    )
```

- [ ] **Step 3: Replace wandb.log call (line 582)**

Replace `wandb.log(` with `trackio.log(` at line 582.

- [ ] **Step 4: Replace wandb.finish call (line 610)**

Replace `wandb.finish()` with `trackio.finish()` at line 610.

- [ ] **Step 5: Commit**

```bash
git add src/student/legacy/train_grpo_process_custom.py
git commit -m "train: migrate GRPO v4 process training to trackio"
```

---

### Task 6: Migrate train_cold_start_sft.py

**Files:**
- Modify: `src/student/train_cold_start_sft.py`

- [ ] **Step 1: Replace import**

At line 25, replace `import wandb` with `import trackio`:

```python
import trackio
```

- [ ] **Step 2: Replace W&B init block (lines 185–199)**

Replace lines 185–199 with:

```python
    # Trackio
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=os.environ.get("TRACKIO_RUN_NAME", "cold-start-sft"),
        config={"epochs": epochs, "batch_size": batch_size, "lr": lr},
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
    )
```

- [ ] **Step 3: Replace wandb.log call (line 273)**

Replace `wandb.log(` with `trackio.log(` at line 273.

- [ ] **Step 4: Replace wandb.finish call (line 287)**

Replace `wandb.finish()` with `trackio.finish()` at line 287.

- [ ] **Step 5: Commit**

```bash
git add src/student/train_cold_start_sft.py
git commit -m "train: migrate cold-start SFT to trackio"
```

---

### Task 7: Migrate legacy train_grpo_custom.py

**Files:**
- Modify: `src/student/legacy/train_grpo_custom.py`

- [ ] **Step 1: Replace import**

At line 26, replace `import wandb` with `import trackio`:

```python
import trackio
```

- [ ] **Step 2: Replace W&B init block (lines 390–408)**

Replace lines 390–408 with:

```python
    # Initialize Trackio logging
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "dm-align-grpo"),
        name=os.environ.get("TRACKIO_RUN_NAME", "grpo-dm-alignment"),
        config=config,
        server_url=os.environ.get("TRACKIO_SERVER_URL"),
    )
```

- [ ] **Step 3: Replace wandb.log call (line 651)**

Replace `wandb.log(` with `trackio.log(` at line 651.

- [ ] **Step 4: Replace wandb.finish call (line 681)**

Replace `wandb.finish()` with `trackio.finish()` at line 681.

- [ ] **Step 5: Commit**

```bash
git add src/student/legacy/train_grpo_custom.py
git commit -m "train: migrate legacy GRPO training to trackio"
```

---

### Task 8: Update GRPO configs — report_to="none"

**Files:**
- Modify: `src/student/grpo_config_outcome.py:79`
- Modify: `src/student/grpo_config_process.py:112`
- Modify: `src/student/grpo_config_dm.py:80`
- Modify: `src/student/legacy/train_grpo_trl.py:140`

- [ ] **Step 1: Change `report_to` in all four files**

In `src/student/grpo_config_outcome.py` line 79:
```python
        report_to="none",
```

In `src/student/grpo_config_process.py` line 112:
```python
        report_to="none",
```

In `src/student/grpo_config_dm.py` line 80:
```python
        report_to="none",
```

In `src/student/legacy/train_grpo_trl.py` line 140:
```python
        report_to="none",
```

- [ ] **Step 2: Commit**

```bash
git add src/student/grpo_config_outcome.py src/student/grpo_config_process.py src/student/grpo_config_dm.py src/student/legacy/train_grpo_trl.py
git commit -m "config: set report_to=none in all GRPO configs (trackio handles logging manually)"
```

---

### Task 9: Update test assertion

**Files:**
- Modify: `src/tests/test_grpo_config.py:22`

- [ ] **Step 1: Change assertion**

Replace line 22:
```python
    assert config.report_to == ["none"]
```

- [ ] **Step 2: Run the test to verify**

```bash
python -m pytest src/tests/test_grpo_config.py -v
```

Expected: all 7 tests pass

- [ ] **Step 3: Commit**

```bash
git add src/tests/test_grpo_config.py
git commit -m "test: update report_to assertion from wandb to none"
```

---

### Task 10: Update Studio SFT config

**Files:**
- Modify: `configs/studio_sft_config.yaml`

- [ ] **Step 1: Remove `wandb_project` line**

Remove line 37 (`wandb_project: dm-align-sft`) from the logging section. The resulting logging block should be:

```yaml
logging:
  report_to: none
  tensorboard_dir: runs
  log_frequency: 10
```

- [ ] **Step 2: Commit**

```bash
git add configs/studio_sft_config.yaml
git commit -m "config: remove wandb_project from Studio SFT config"
```

---

### Task 11: Delete W&B-specific test scripts

**Files:**
- Delete: `scripts/test_wandb.py`
- Delete: `scripts/test_wandb_connect.py`
- Delete: `scripts/test_wandb_endpoints.py`
- Delete: `scripts/test_wandb_auth2.py`
- Delete: `scripts/test_wandb_ping.py`
- Delete: `scripts/get_wandb_key.py`

- [ ] **Step 1: Remove files**

```bash
git rm scripts/test_wandb.py scripts/test_wandb_connect.py scripts/test_wandb_endpoints.py scripts/test_wandb_auth2.py scripts/test_wandb_ping.py scripts/get_wandb_key.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "scripts: remove W&B-specific test and utility scripts"
```

---

### Task 12: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update setup section (lines 33–38)**

Replace lines 33–38:
```bash
# Edit .env with your keys:
#   TRACKIO_SERVER_URL — http://trackio-server:7860 (Docker) or http://localhost:7860 (host)
#   TRACKIO_PROJECT    — dm-align-grpo (default)
#   HF_TOKEN           — from https://huggingface.co/settings/tokens
```

- [ ] **Step 2: Update Docker services table (lines 44–48)**

Replace the `wandb` row:
```
| `trackio` | `trackio-server` | 7860 | Local Trackio experiment tracking |
```

- [ ] **Step 3: Update docker compose start command (line 58)**

Replace:
```
# Start local Trackio server
docker compose up -d trackio
```

- [ ] **Step 4: Replace W&B Logging section (lines 97–107)**

Replace lines 97–107:
```markdown
### Experiment Tracking

Training logs to Trackio automatically. Configure via environment:

```bash
# .env
TRACKIO_SERVER_URL=http://trackio-server:7860
TRACKIO_PROJECT=dm-align-grpo
TRACKIO_RUN_NAME=grpo-v3-outcome-only
```

View dashboard: `trackio show --project dm-align-grpo` (on host) or visit `http://localhost:7860` (Docker).
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README for trackio experiment tracking"
```

---

### Task 13: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update `.env` reference (line 129)**

Replace:
```
- `.env` contains `WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_MODE` (gitignored, never commit)
```
With:
```
- `.env` contains `TRACKIO_SERVER_URL`, `TRACKIO_PROJECT` (gitignored, never commit)
```

- [ ] **Step 2: Update W&B service reference**

Replace any remaining references to `wandb-server` container with `trackio-server`.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md for trackio"
```

---

### Task 14: Run full test suite

**Files:**
- Test: entire test suite

- [ ] **Step 1: Run tests**

```bash
./scripts/run_e2e_tests.sh
```

Expected: all non-GPU tests pass. Any failures related to W&B imports should be fixed.

- [ ] **Step 2: Fix any test failures**

If any test imports `wandb` or references `WANDB_*` environment variables, update accordingly.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "test: verify full test suite passes with trackio replacement"
```

---

## Self-Review

**Spec coverage:**
- Docker container for Trackio server — Task 1
- Dockerfile dependency swap — Task 2
- .env.example update — Task 3
- All 4 training scripts migrated — Tasks 4–7
- All 3 GRPO configs + 1 TRL script report_to updated — Task 8
- Test assertion updated — Task 9
- Studio SFT config cleaned — Task 10
- W&B test scripts deleted — Task 11
- README updated — Task 12
- AGENTS.md updated — Task 13
- Full test verification — Task 14

**Placeholder scan:** No TBDs, no "add appropriate error handling", no "similar to Task N". Every step has exact line numbers and code.

**Type consistency:** All `trackio.init()` calls use same signature: `project`, `name`, `config`, `server_url`. All `trackio.log()` and `trackio.finish()` calls are direct replacements with no argument changes. Environment variable names are consistent: `TRACKIO_SERVER_URL`, `TRACKIO_PROJECT`, `TRACKIO_RUN_NAME`.
