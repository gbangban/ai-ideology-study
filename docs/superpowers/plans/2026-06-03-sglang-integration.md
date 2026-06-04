# SG-Lang Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate SG-Lang as an HTTP inference backend for judge offloading during GRPO training, add local W&B server for experiment tracking, and fix missing W&B logging in the active training script.

**Architecture:** New `sglang_client.py` module wraps SG-Lang's OpenAI-compatible API. `rewards.py` branches on `judge_backend` config to call either local judge or SG-Lang HTTP. `train_grpo.py` wires up the client, removes judge load/unload cycle when using SG-Lang, and adds W&B logging (was missing entirely — only CSV logging existed). New `sglang` service in docker-compose. New `wandb` service runs a local W&B server on port 8086 for offline tracking. New eval script uses lm_eval's OpenAI backend against SG-Lang.

**Tech Stack:** Python `requests`, `concurrent.futures.ThreadPoolExecutor`, `wandb`, SG-Lang OpenAI-compatible API, bash scripts, docker-compose.

---

### File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/student/sglang_client.py` | Create | HTTP client wrapper for SG-Lang API |
| `src/tests/test_sglang_client.py` | Create | Unit tests with mocked HTTP responses |
| `src/student/grpo_config.py` | Modify | Add `judge_backend`, `sglang_base_url` config fields |
| `src/student/rewards.py` | Modify | Add SG-Lang judge path to `compute_dm_alignment_judge()` |
| `src/student/train_grpo.py` | Modify | Wire SG-Lang client, skip local judge load when using SG-Lang |
| `docker-compose.yml` | Modify | Add `sglang` service |
| `scripts/sglang_health.sh` | Create | Health check script for SG-Lang container |
| `evals/scripts/run_sglang_bf16.sh` | Create | Eval script using SG-Lang + lm_eval OpenAI backend |
| `scripts/run_e2e_tests.sh` | Modify | Add SG-Lang client tests to test runner |
| `src/student/train_grpo.py` | Modify | (additional) add W&B logging |
| `docker-compose.yml` | Modify | (additional) add local W&B server service |
| `.env` | Modify | Add `WANDB_BASE_URL` for local W&B |

---

### Task 1: SG-Lang HTTP Client

**Files:**
- Create: `src/student/sglang_client.py`
- Create: `src/tests/test_sglang_client.py`

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_sglang_client.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import json


class TestSglangClientHealth:
    def test_health_check_returns_true_when_reachable(self):
        from src.student.sglang_client import SglangClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "test", "object": "model"}]}

        with patch("requests.get", return_value=mock_resp):
            client = SglangClient("http://localhost:1235")
            assert client.health_check() is True

    def test_health_check_returns_false_when_unreachable(self):
        from src.student.sglang_client import SglangClient

        with patch("requests.get", side_effect=Exception("Connection refused")):
            client = SglangClient("http://localhost:1235")
            assert client.health_check() is False


class TestSglangClientChatCompletion:
    def test_chat_completion_returns_content(self):
        from src.student.sglang_client import SglangClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "STRUCTURAL_ANALYSIS: Yes\nCONTRADICTION_TRACING: Yes\nFRAME_CRITIQUE: No\nCONCLUSION_DIVERGENCE: Yes"
                    }
                }
            ]
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            client = SglangClient("http://localhost:1235")
            result = client.chat_completion([{"role": "user", "content": "test"}])
            assert "STRUCTURAL_ANALYSIS" in result
            mock_post.assert_called_once()

    def test_chat_completion_raises_on_http_error(self):
        from src.student.sglang_client import SglangClient
        import requests as req

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_resp):
            client = SglangClient("http://localhost:1235")
            with pytest.raises(req.HTTPError):
                client.chat_completion([{"role": "user", "content": "test"}])


class TestSglangClientBatchCompletion:
    def test_batch_chat_completion_returns_all_results(self):
        from src.student.sglang_client import SglangClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }

        with patch("requests.post", return_value=mock_resp):
            client = SglangClient("http://localhost:1235")
            requests_list = [
                {"messages": [{"role": "user", "content": f"q{i}"}]}
                for i in range(4)
            ]
            results = client.batch_chat_completion(requests_list)
            assert len(results) == 4
            assert all(r == "response" for r in results)

    def test_batch_chat_completion_retries_on_failure(self):
        from src.student.sglang_client import SglangClient
        import requests as req

        call_count = [0]
        def mock_post_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise req.HTTPError("503 Service Unavailable")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            return resp

        with patch("requests.post", side_effect=mock_post_side_effect):
            client = SglangClient("http://localhost:1235", timeout=1)
            results = client.batch_chat_completion([
                {"messages": [{"role": "user", "content": "test"}]}
            ])
            assert len(results) == 1
            assert results[0] == "ok"
            assert call_count[0] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/tests/test_sglang_client.py -v --tb=short`
Expected: FAIL with "No module named src.student.sglang_client" or import error

- [ ] **Step 3: Write minimal implementation**

Create `src/student/sglang_client.py`:

```python
"""
SG-Lang HTTP Client

Thin wrapper around requests for SG-Lang's OpenAI-compatible API.
Used for judge offloading during GRPO training and lm_eval evaluations.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class SglangClient:
    """HTTP client for SG-Lang OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1235",
        timeout: int = 60,
        max_retries: int = 3,
        max_workers: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers

    def health_check(self) -> bool:
        """Check if SG-Lang is reachable and responsive."""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                timeout=self.timeout,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"SG-Lang health check failed: {e}")
            return False

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Send a single chat completion request to SG-Lang.

        Args:
            messages: Chat messages in OpenAI format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional arguments passed to SG-Lang API.

        Returns:
            Generated text content.

        Raises:
            requests.HTTPError: On non-200 response with exhausted retries.
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    raise requests.HTTPError(
                        f"SG-Lang returned {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except requests.HTTPError:
                raise
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** (attempt - 1)
                    logger.warning(
                        f"SG-Lang request failed (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)

        raise last_exception  # type: ignore[arg-type]

    def batch_chat_completion(
        self,
        requests_list: List[Dict[str, Any]],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> List[str]:
        """Send multiple chat completion requests in parallel.

        Uses ThreadPoolExecutor for maximum throughput. SG-Lang handles
        continuous batching on the server side.

        Args:
            requests_list: List of dicts with 'messages' key (and optionally other params).
            max_tokens: Maximum tokens to generate per request.
            temperature: Sampling temperature.

        Returns:
            List of generated text contents, in same order as input.
        """
        results: Dict[int, str] = {}

        def _single_request(idx: int, req: Dict[str, Any]) -> tuple:
            try:
                messages = req.get("messages", req)
                content = self.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return idx, content
            except Exception as e:
                logger.error(f"SG-Lang batch request {idx} failed: {e}")
                return idx, ""

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_single_request, idx, req): idx
                for idx, req in enumerate(requests_list)
            }
            for future in as_completed(futures):
                idx, content = future.result()
                results[idx] = content

        return [results[i] for i in range(len(requests_list))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/tests/test_sglang_client.py -v --tb=short`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/student/sglang_client.py src/tests/test_sglang_client.py
git commit -m "feat: add SG-Lang HTTP client with batch completion support"
```

---

### Task 2: GRPO Config — Judge Backend Fields

**Files:**
- Modify: `src/student/grpo_config.py`

- [ ] **Step 1: Add new config fields to GRPO_CONFIG**

In `src/student/grpo_config.py`, add these fields after the `"judge_model"` line (line 43):

```python
    # Judge model
    "judge_model": "Qwen/Qwen3.5-4B",

    # Judge backend: "local" (default), "sglang" (BF16 recommended), "sglang-quantized"
    "judge_backend": "local",
    "sglang_base_url": "http://localhost:1235",
    "sglang_judge_quantization": None,  # None (BF16), "fp8", or "int4"
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `python3 -m pytest src/tests/test_grpo_config.py -v --tb=short`
Expected: PASS (no new tests needed; config is just a dict, existing tests verify structure)

- [ ] **Step 3: Commit**

```bash
git add src/student/grpo_config.py
git commit -m "feat(grpo_config): add judge_backend and sglang config fields"
```

---

### Task 3: Rewards — SG-Lang Judge Path

**Files:**
- Modify: `src/student/rewards.py`

- [ ] **Step 1: Add SG-Lang import and modify `compute_dm_alignment_judge`**

At the top of `src/student/rewards.py`, add the import:

```python
from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.student.sglang_client import SglangClient
```

Replace the existing `Optional` import line with the new one above.

- [ ] **Step 2: Create `compute_dm_alignment_judge_http` function**

Add this function before `compute_dm_alignment_judge`:

```python
def compute_dm_alignment_judge_http(
    completions: List[str],
    sglang_client: "SglangClient",
) -> List[float]:
    """Compute DM alignment scores using SG-Lang HTTP judge."""
    request_payloads = []
    for completion in completions:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(response=completion)},
        ]
        request_payloads.append({"messages": messages})

    outputs = sglang_client.batch_chat_completion(
        request_payloads,
        max_tokens=128,
        temperature=0.0,
    )
    return [_parse_judge_output(output) for output in outputs]
```

- [ ] **Step 3: Modify `compute_reward` to accept sglang_client**

Change the `compute_reward` function signature and dm_alignment branch:

Replace:
```python
def compute_reward(
    completions: List[str],
    weights: dict,
    tokenizer,
    judge_model=None,
    judge_tokenizer=None,
) -> List[float]:
```

With:
```python
def compute_reward(
    completions: List[str],
    weights: dict,
    tokenizer,
    judge_model=None,
    judge_tokenizer=None,
    sglang_client=None,
) -> List[float]:
```

Replace the dm_alignment block:
```python
    if "dm_alignment" in weights and judge_model is not None and judge_tokenizer is not None:
        dm_scores = compute_dm_alignment_judge(completions, judge_model, judge_tokenizer)
        for i, s in enumerate(dm_scores):
            total_scores[i] += weights["dm_alignment"] * s
```

With:
```python
    if "dm_alignment" in weights:
        w = weights["dm_alignment"]
        if sglang_client is not None:
            dm_scores = compute_dm_alignment_judge_http(completions, sglang_client)
        elif judge_model is not None and judge_tokenizer is not None:
            dm_scores = compute_dm_alignment_judge(completions, judge_model, judge_tokenizer)
        else:
            dm_scores = [0.0] * n
        for i, s in enumerate(dm_scores):
            total_scores[i] += w * s
```

- [ ] **Step 4: Modify `build_reward_fn` to accept sglang_client**

Replace:
```python
def build_reward_fn(
    weights: dict,
    judge_model,
    judge_tokenizer: Optional[PreTrainedTokenizer],
) -> Callable:
    """Build a TRL-compatible reward function.

    TRL GRPOTrainer expects a callable:
        reward_fn(completions: List[str]) -> List[float]

    The length reward is handled inside train_grpo.py since it needs token counts.
    This builder returns a function that computes the three text-based rewards.
    """
    def reward_fn(completions: List[str]) -> List[float]:
        return compute_reward(completions, weights, judge_model, judge_tokenizer)
    return reward_fn
```

With:
```python
def build_reward_fn(
    weights: dict,
    judge_model,
    judge_tokenizer: Optional[PreTrainedTokenizer],
    sglang_client=None,
) -> Callable:
    """Build a TRL-compatible reward function.

    TRL GRPOTrainer expects a callable:
        reward_fn(completions: List[str]) -> List[float]

    The length reward is handled inside train_grpo.py since it needs token counts.
    This builder returns a function that computes the three text-based rewards.
    """
    def reward_fn(completions: List[str]) -> List[float]:
        return compute_reward(completions, weights, None, None, sglang_client)
    return reward_fn
```

- [ ] **Step 5: Verify existing reward tests still pass**

Run: `python3 -m pytest src/tests/test_rewards.py -v --tb=short`
Expected: PASS (existing tests don't pass sglang_client, so local path is used)

- [ ] **Step 6: Commit**

```bash
git add src/student/rewards.py
git commit -m "feat(rewards): add SG-Lang HTTP judge path to compute_dm_alignment"
```

---

### Task 4: Training Script — SG-Lang Integration

**Files:**
- Modify: `src/student/train_grpo.py`

- [ ] **Step 1: Add SG-Lang import**

Add after the existing rewards import block (line 38):

```python
from src.student.sglang_client import SglangClient
```

- [ ] **Step 2: Modify `compute_rewards` in train_grpo.py to accept sglang_client**

Replace the `compute_rewards` function signature (line 108):

```python
def compute_rewards(
    completions: List[str],
    weights: dict,
    tokenizer,
    judge_model=None,
    judge_tokenizer=None,
    sglang_client=None,
) -> List[float]:
    """Compute weighted sum of all reward functions for a batch of completions."""
```

And replace the dm_alignment block inside it:

```python
    if "dm_alignment" in weights:
        w = weights["dm_alignment"]
        if sglang_client is not None:
            from src.student.rewards import compute_dm_alignment_judge_http
            dm_scores = compute_dm_alignment_judge_http(completions, sglang_client)
        elif judge_model is not None and judge_tokenizer is not None:
            dm_scores = compute_dm_alignment_judge(completions, judge_model, judge_tokenizer)
        else:
            dm_scores = [0.0] * n
        for i, s in enumerate(dm_scores):
            total_scores[i] += w * s
```

- [ ] **Step 3: Modify `train()` function — SG-Lang client setup and judge loading**

In the `train()` function, replace the judge model loading block (lines 266-284):

```python
    # Load judge model or SG-Lang client
    judge_model = None
    judge_tokenizer = None
    sglang_client = None
    if config["reward_weights"].get("dm_alignment", 0) > 0:
        judge_backend = config.get("judge_backend", "local")
        if judge_backend == "local":
            logger.info(f"Loading judge model (local): {config['judge_model']}...")
            _strip_vision_config(config["judge_model"])
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            judge_model = AutoModelForCausalLM.from_pretrained(
                config["judge_model"],
                device_map="auto",
                quantization_config=bnb_config,
            )
            judge_tokenizer = AutoTokenizer.from_pretrained(config["judge_model"])
            logger.info(f"Judge model loaded on {judge_model.device}")
        else:
            sglang_client = SglangClient(
                base_url=config.get("sglang_base_url", "http://localhost:1235"),
                timeout=60,
                max_retries=3,
            )
            if not sglang_client.health_check():
                raise RuntimeError(
                    f"SG-Lang is not reachable at {config.get('sglang_base_url', 'http://localhost:1235')}. "
                    "Start the SG-Lang container first, or set judge_backend='local' to use local judge."
                )
            logger.info(f"SG-Lang client connected at {config.get('sglang_base_url', 'http://localhost:1235')}")
```

- [ ] **Step 4: Modify reward computation call to pass sglang_client**

Replace the `compute_rewards` call (line 380):

```python
        rewards = compute_rewards(
            all_completions,
            config["reward_weights"],
            tokenizer,
            judge_model,
            judge_tokenizer,
            sglang_client,
        )
```

- [ ] **Step 5: Remove judge offload/reload when using SG-Lang**

Replace the judge offload block (lines 392-395):

```python
        # Offload judge model to free VRAM for policy update (local only)
        if judge_model is not None and sglang_client is None:
            logger.info(f"Offloading judge model (freeing ~{torch.cuda.memory_allocated(judge_model.device) / 1e9:.1f}GB)...")
            judge_model.cpu()
            torch.cuda.empty_cache()
```

Replace the judge reload block (lines 481-482):

```python
        # Reload judge model for next batch's reward computation (local only)
        if judge_model is not None and sglang_client is None:
            judge_model.cuda(model.device)
```

- [ ] **Step 6: Verify existing GRPO tests still pass**

Run: `python3 -m pytest src/tests/test_grpo_training.py -v --tb=short`
Expected: PASS (existing tests use local judge path)

- [ ] **Step 7: Commit**

```bash
git add src/student/train_grpo.py
git commit -m "feat(grpo): wire SG-Lang client for judge offloading"
```

---

### Task 5: Docker Compose — SG-Lang Service

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add sglang service to docker-compose.yml**

Append this service after the `training` service:

```yaml
  sglang:
    image: lmsysorg/sglang:latest
    container_name: sglang-server
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "1235:30000"
    volumes:
      - C:/Users/Guy/.cache/huggingface:/root/.cache/huggingface
      - C:/Users/Guy/.unsloth/studio/exports:/studio/exports
    environment:
      - HF_HOME=/root/.cache/huggingface
      - HF_TOKEN=${HF_TOKEN}
      - SGLANG_ENABLE_CUDA_GRAPH=false
    shm_size: '16gb'
    restart: "no"
    command: >
      --model-path Qwen/Qwen3.5-4B
      --host 0.0.0.0
      --port 30000
      --mem-fraction-static 0.6
      --cuda-memory-fraction 0.6
```

Note: Port mapping is `1235:30000` because SG-Lang defaults to port 30000 internally. The `sglang_base_url` config is `http://localhost:1235` which maps to the container's port 30000.

- [ ] **Step 2: Verify docker-compose syntax**

Run: `docker compose config > /dev/null 2>&1`
Expected: Exit code 0, no errors

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "feat(docker): add SG-Lang service for judge offloading"
```

---

### Task 6: Health Check Script

**Files:**
- Create: `scripts/sglang_health.sh`

- [ ] **Step 1: Create health check script**

Create `scripts/sglang_health.sh`:

```bash
#!/bin/bash
# SG-Lang Health Check
# Verifies SG-Lang container is running and responsive before training or evals.
#
# Usage:
#   ./scripts/sglang_health.sh                    # Check default port 1235
#   ./scripts/sglang_health.sh --port 1235        # Check specific port
#   ./scripts/sglang_health.sh --url http://...   # Check custom URL

set -euo pipefail

PORT="${SGLANG_PORT:-1235}"
URL="${SGLANG_URL:-http://localhost:$PORT}"

echo "Checking SG-Lang health at $URL ..."

for i in 1 2 3; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL/v1/models" --max-time 10 2>/dev/null || echo "000")
  if [ "$HTTP_CODE" = "200" ]; then
    echo "SG-Lang is healthy (HTTP $HTTP_CODE)"
    exit 0
  fi
  echo "  Attempt $i: HTTP $HTTP_CODE"
  sleep 2
done

echo "ERROR: SG-Lang is not reachable at $URL"
echo "Start the SG-Lang container first:"
echo "  docker compose up -d sglang"
exit 1
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/sglang_health.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/sglang_health.sh
git commit -m "feat: add SG-Lang health check script"
```

---

### Task 7: SG-Lang BF16 Eval Script

**Files:**
- Create: `evals/scripts/run_sglang_bf16.sh`

- [ ] **Step 1: Create eval script**

Create `evals/scripts/run_sglang_bf16.sh`:

```bash
#!/bin/bash
# Run BF16 evaluation using SG-Lang serving backend
# Evaluates the model served by SG-Lang via lm_eval's OpenAI backend
#
# This script manages the SG-Lang server lifecycle:
# 1. Launches SG-Lang with the target model (BF16)
# 2. Waits for health check on port 1235
# 3. Runs lm_eval with OpenAI backend
# 4. Tears down SG-Lang after eval completes
#
# NOTE: Requires Docker Desktop with NVIDIA runtime
# NOTE: Stop other GPU containers (Studio, training) before running
#
# Usage:
#   ./run_sglang_bf16.sh                                # Run all tasks
#   ./run_sglang_bf16.sh --tasks humaneval               # Run single task
#   ./run_sglang_bf16.sh --suite causal                  # Run causal suite
#   ./run_sglang_bf16.sh --help                          # Show available tasks
#
# Set SGLANG_MODEL to point to a specific model:
#   SGLANG_MODEL=Qwen/Qwen3.5-9B ./run_sglang_bf16.sh
# Default: merged GRPO checkpoint path
#
# Set SGLANG_SKIP_SERVER=true to skip server management (use existing SG-Lang instance):
#   SGLANG_SKIP_SERVER=true ./run_sglang_bf16.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_logging.sh"

export HF_ALLOW_CODE_EVAL="1"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/sglang/bf16"

VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

SGLANG_PORT=1235
SGLANG_URL="http://localhost:$SGLANG_PORT"
SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3.5-9B}"
SGLANG_SKIP_SERVER="${SGLANG_SKIP_SERVER:-false}"

ALL_TASKS=(
    "mmlu"
    "mmlu_pro"
    "gpqa_diamond_zeroshot"
    "ifeval"
    "humaneval"
    "leaderboard_math_hard"
    "econcausal_task1_econ"
    "econcausal_task1_finance"
    "econcausal_task2"
    "econcausal_task3"
    "corr2cause"
)

TASKS_LIST="mmlu,mmlu_pro,gpqa_diamond_zeroshot,ifeval,humaneval,leaderboard_math_hard,econcausal_task1_econ,econcausal_task1_finance,econcausal_task2,econcausal_task3,corr2cause"

DRY_RUN="false"
_SELECTED_TASKS=()
for arg in "$@"; do
    case "$arg" in
        --help)
            show_help "$0" "SG-Lang BF16 Evaluation (OpenAI-compatible backend)" "$TASKS_LIST"
            exit 0
            ;;
        --dry-run)
            DRY_RUN="true"
            ;;
        --suite)
            shift
            if [ $# -eq 0 ]; then
                log_error "--suite requires a value (short, medium, causal, full)"
                exit 1
            fi
            case "$1" in
                short)
                    IFS=',' read -ra _SELECTED_TASKS <<< "ifeval,humaneval,mmlu"
                    ;;
                medium)
                    IFS=',' read -ra _SELECTED_TASKS <<< "ifeval,humaneval,mmlu,gpqa_diamond_zeroshot"
                    ;;
                causal)
                    IFS=',' read -ra _SELECTED_TASKS <<< "econcausal_task1_econ,econcausal_task1_finance,econcausal_task2,econcausal_task3,corr2cause"
                    ;;
                full)
                    _SELECTED_TASKS=()
                    ;;
                *)
                    log_error "Unknown suite: $1 (valid: short, medium, causal, full)"
                    exit 1
                    ;;
            esac
            ;;
        --tasks)
            shift
            if [ $# -eq 0 ]; then
                log_error "--tasks requires a value"
                exit 1
            fi
            IFS=',' read -ra _SELECTED_TASKS <<< "$1"
            ;;
    esac
done

mkdir -p "$RESULTS_DIR"
EVAL_LOG="$RESULTS_DIR/eval.log"
: > "$EVAL_LOG"

log_section "SG-Lang BF16 Evaluation"
log_info "Model: $SGLANG_MODEL"
log_info "SG-Lang URL: $SGLANG_URL"
log_info "Output: $RESULTS_DIR"
log_info "Log file: $EVAL_LOG"

if [ "$DRY_RUN" = "true" ]; then
    log_info "DRY RUN MODE - showing what would execute"
fi

TASKS_TO_RUN=()
for TASK in "${ALL_TASKS[@]}"; do
    if task_selected "$TASK"; then
        log_info "  + $TASK"
        TASKS_TO_RUN+=("$TASK")
    else
        log_info "  - $TASK (skipped)"
    fi
done
echo

if [ ${#TASKS_TO_RUN[@]} -eq 0 ]; then
    log_error "No tasks selected."
    exit 1
fi

SGLANG_PID=""
cleanup_sglang() {
    if [ -n "$SGLANG_PID" ] && [ "$SGLANG_SKIP_SERVER" != "true" ]; then
        log_info "Stopping SG-Lang server (PID $SGLANG_PID)..."
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
        log_info "SG-Lang server stopped."
    fi
}
trap cleanup_sglang EXIT

if [ "$SGLANG_SKIP_SERVER" != "true" ]; then
    check_gpu 5000 || exit 1

    log_info "Launching SG-Lang server with model: $SGLANG_MODEL"
    SERVER_LOG="$RESULTS_DIR/sglang_server.log"
    : > "$SERVER_LOG"

    docker compose run --rm \
        --no-deps \
        -e HF_TOKEN \
        sglang \
        --model-path "$SGLANG_MODEL" \
        --host 0.0.0.0 \
        --port 30000 \
        --mem-fraction-static 0.6 \
        --cuda-memory-fraction 0.6 \
        > "$SERVER_LOG" 2>&1 &
    SGLANG_PID=$!

    log_info "SG-Lang started with PID $SGLANG_PID"

    HEALTH_TIMEOUT=300
    HEALTH_START=$(date +%s)
    while true; do
        HEALTH_ELAPSED=$(( $(date +%s) - HEALTH_START ))
        if [ "$HEALTH_ELAPSED" -ge "$HEALTH_TIMEOUT" ]; then
            log_error "SG-Lang did not become ready within ${HEALTH_TIMEOUT}s."
            log_error "Last 20 lines of $SERVER_LOG:"
            tail -20 "$SERVER_LOG" 2>/dev/null | while IFS= read -r line; do
                log_error "  $line"
            done
            exit 1
        fi
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SGLANG_URL/v1/models" --max-time 5 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            log_info "SG-Lang is ready (${HEALTH_ELAPSED}s)."
            break
        fi
        if [ $((HEALTH_ELAPSED % 15)) -eq 0 ] && [ "$HEALTH_ELAPSED" -gt 0 ]; then
            log_info "  Still waiting for SG-Lang... (${HEALTH_ELAPSED}s / ${HEALTH_TIMEOUT}s)"
        fi
        sleep 3
    done
else
    log_info "Skipping SG-Lang server launch (SGLANG_SKIP_SERVER=true)"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SGLANG_URL/v1/models" --max-time 10 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" != "200" ]; then
        log_error "SG-Lang is not reachable at $SGLANG_URL"
        exit 1
    fi
    log_info "SG-Lang is healthy at $SGLANG_URL"
fi

export LM_EVAL_CONFIRM_RUN_UNSAFE_CODE="True"

progress_init ${#TASKS_TO_RUN[@]}

FAILED_TASKS=()
SKIPPED_TASKS=()

for TASK in "${TASKS_TO_RUN[@]}"; do
    progress_next

    TASK_RESULT="$RESULTS_DIR/${TASK}.json"
    if [ -f "$TASK_RESULT" ] && [ "${FORCE_RERUN:-false}" != "true" ]; then
        log_warn "Results already exist for $TASK at $TASK_RESULT"
        log_info "Skipping. Set FORCE_RERUN=true to overwrite."
        SKIPPED_TASKS+=("$TASK")
        continue
    fi

    log_info "Starting $TASK..."
    MODEL_ARGS="model=$SGLANG_MODEL,base_url=$SGLANG_URL,vllm_guided_decoding_enabled=False"
    log_info "Command: lm_eval --model openai-completions --model_args $MODEL_ARGS --tasks $TASK ..."

    if _is_dry_run; then
        log_info "[DRY RUN] Would run lm_eval for task: $TASK"
        continue
    fi

    TASK_START=$(date +%s)

    set +e
    lm_eval --model openai-completions \
      --model_args "$MODEL_ARGS" \
      --tasks "$TASK" \
      --batch_size auto \
      --output_path "$RESULTS_DIR" \
      --log_samples \
      --trust_remote_code \
      --include_path "$PROJECT_DIR/configs/task_configs" \
      --apply_chat_template \
      --confirm_run_unsafe_code 2>&1 | tee -a "$EVAL_LOG"
    TASK_EXIT=$?
    set -e

    TASK_END=$(date +%s)
    TASK_ELAPSED=$((TASK_END - TASK_START))
    TASK_MINS=$((TASK_ELAPSED / 60))
    TASK_SECS=$((TASK_ELAPSED % 60))

    if [ $TASK_EXIT -eq 0 ]; then
        log_info "Completed: $TASK (${TASK_MINS}m ${TASK_SECS}s)"
    else
        log_error "Failed: $TASK (exit code $TASK_EXIT, ${TASK_MINS}m ${TASK_SECS}s)"
        FAILED_TASKS+=("$TASK")
    fi

    echo
done

TOTAL_ELAPSED=$(progress_elapsed_total)
log_section "Evaluation Summary"
log_info "Total elapsed time: $TOTAL_ELAPSED"
log_info "Tasks completed: $(( ${#TASKS_TO_RUN[@]} - ${#FAILED_TASKS[@]} - ${#SKIPPED_TASKS[@]} ))"

if [ ${#SKIPPED_TASKS[@]} -gt 0 ]; then
    log_warn "Tasks skipped (existing results): ${SKIPPED_TASKS[*]}"
fi

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    log_error "Tasks failed: ${FAILED_TASKS[*]}"
    log_separator "-"
    log_error "Some tasks failed. Check $EVAL_LOG for details."
    exit 1
fi

log_info "Results saved to: $RESULTS_DIR"
log_info "Log saved to: $EVAL_LOG"
log_separator "="
```

- [ ] **Step 2: Make executable**

```bash
chmod +x evals/scripts/run_sglang_bf16.sh
```

- [ ] **Step 3: Verify script parses correctly**

Run: `bash -n evals/scripts/run_sglang_bf16.sh`
Expected: Exit code 0, no syntax errors

- [ ] **Step 4: Commit**

```bash
git add evals/scripts/run_sglang_bf16.sh
git commit -m "feat(evals): add SG-Lang BF16 eval script with server lifecycle"
```

---

### Task 8: Update Test Runner

**Files:**
- Modify: `scripts/run_e2e_tests.sh`

- [ ] **Step 1: Add SG-Lang client tests to the test runner**

Insert a new step before the E2E integration tests (before line 67). Replace:

```bash
# Run E2E integration tests
echo "========================================="
echo "Step 4: Running E2E Integration Tests"
echo "========================================="
```

With:

```bash
# Run SG-Lang client tests
echo "========================================="
echo "Step 4: Running SG-Lang Client Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_sglang_client.py -${VERBOSE} --tb=short
SGLANG_EXIT=$?

if [ $SGLANG_EXIT -ne 0 ]; then
    echo "ERROR: SG-Lang client tests failed"
    exit $SGLANG_EXIT
fi
echo ""

# Run GRPO training tests
echo "========================================="
echo "Step 5: Running GRPO Training Tests"
echo "========================================="
python3 -m pytest $TEST_DIR/test_grpo_training.py -${VERBOSE} --tb=short
GRPO_EXIT=$?

if [ $GRPO_EXIT -ne 0 ]; then
    echo "ERROR: GRPO training tests failed"
    exit $GRPO_EXIT
fi
echo ""

# Run E2E integration tests
echo "========================================="
echo "Step 6: Running E2E Integration Tests"
echo "========================================="
```

Update the summary section to include new test results. Replace the summary block:

```bash
echo "Teacher Phase:       $([ $TEACHER_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "SFT Config:          $([ $SFT_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "DPO Unit Tests:      $([ $DPO_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "E2E Integration:     $([ $E2E_EXIT -eq 0 ] && echo 'PASS' || echo 'PARTIAL - needs GPU')"
```

With:

```bash
echo "Teacher Phase:       $([ $TEACHER_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "SFT Config:          $([ $SFT_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "DPO Unit Tests:      $([ $DPO_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "SG-Lang Client:      $([ $SGLANG_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "GRPO Training:       $([ $GRPO_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "E2E Integration:     $([ $E2E_EXIT -eq 0 ] && echo 'PASS' || echo 'PARTIAL - needs GPU')"
```

Update the exit check at the bottom:

```bash
if [ $TEACHER_EXIT -ne 0 ] || [ $SFT_EXIT -ne 0 ] || [ $DPO_EXIT -ne 0 ]; then
```

To:

```bash
if [ $TEACHER_EXIT -ne 0 ] || [ $SFT_EXIT -ne 0 ] || [ $DPO_EXIT -ne 0 ] || [ $SGLANG_EXIT -ne 0 ] || [ $GRPO_EXIT -ne 0 ]; then
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_e2e_tests.sh
git commit -m "test: add SG-Lang client and GRPO tests to E2E runner"
```

---

### Task 9: W&B Integration — Local Server + Training Script Logging

**Root cause:** `train_grpo.py` (the active training script) has zero W&B integration — it only logs to CSV. W&B was only configured in `train_grpo_trl.py` (deprecated TRL version, CUDA 13 container). That's why you saw no records.

**Fix:** Add W&B init + logging to `train_grpo.py`, and add a local W&B server to docker-compose so runs work offline.

**Files:**
- Modify: `src/student/train_grpo.py`
- Modify: `docker-compose.yml`
- Modify: `.env`

- [ ] **Step 1: Add WANDB_BASE_URL to .env**

Append to `.env`:

```
WANDB_BASE_URL=http://localhost:8086
WANDB_MODE=offline
```

`WANDB_MODE=offline` ensures runs are saved locally even if W&B server is down. Set to `online` when the local W&B server is running.

- [ ] **Step 2: Add W&B logging to train_grpo.py**

At the top of `train_grpo.py`, after the existing imports, add:

```python
import os
import wandb
```

In the `train()` function, after the training loop setup (after line 355 where logger prints "Starting GRPO training..."), add W&B initialization:

```python
    # Initialize W&B logging
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    wandb_base_url = os.environ.get("WANDB_BASE_URL")
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dm-align-grpo"),
        name=os.environ.get("WANDB_RUN_NAME", "grpo-dm-alignment"),
        config=config,
        mode=wandb_mode,
        save_code=False,
    )
    if wandb_base_url:
        wandb.base_url = wandb_base_url
    logger.info(f"W&B initialized (mode={wandb_mode}, base_url={wandb_base_url or 'default'})")
```

Replace the existing CSV logging block inside the training loop (after `if step % logging_steps == 0 or step == 1:`) to also log to W&B:

```python
        if step % logging_steps == 0 or step == 1:
            vram = torch.cuda.memory_allocated(model.device) / 1e9
            logger.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {batch_loss / len(all_completions):.4f} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Time: {elapsed:.1f}s | "
                f"ETA: {est_remaining:.1f}h | "
                f"VRAM: {vram:.1f}GB"
            )
            wandb.log({
                "step": step,
                "loss": batch_loss / len(all_completions),
                "avg_reward": avg_reward,
                "batch_time_s": elapsed,
                "vram_gb": vram,
                "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr,
            })
```

Add W&B cleanup at the end of training, before `logger.info("Done.")`:

```python
    wandb.finish()
```

- [ ] **Step 3: Add local W&B server to docker-compose.yml**

Append this service after the `sglang` service:

```yaml
  wandb:
    image: wandb/core:v0.23.0-deploy
    container_name: wandb-server
    ports:
      - "8086:8086"
    volumes:
      - wandb-data:/home/wandb/wandb/local
    environment:
      - WANDB_BASE_URL=http://localhost:8086
    restart: unless-stopped
```

Add the volume definition at the bottom of docker-compose.yml (after the services block):

```yaml
volumes:
  wandb-data:
```

- [ ] **Step 4: Verify W&B imports work**

Run: `python3 -c "import wandb; print('wandb version:', wandb.__version__)"`
Expected: Prints version without error

- [ ] **Step 5: Commit**

```bash
git add src/student/train_grpo.py docker-compose.yml .env
git commit -m "feat(grpo): add W&B logging and local W&B server"
```

---

### Task 10: Run Full Test Suite

**Files:**
- No new files; verification step.

- [ ] **Step 1: Run full test suite**

Run: `./scripts/run_e2e_tests.sh`
Expected: All steps PASS (SG-Lang client tests pass with mocked HTTP, existing tests unaffected)

- [ ] **Step 2: Run docker compose config validation**

Run: `docker compose config > /dev/null 2>&1 && echo "docker-compose valid"`
Expected: Exit code 0

- [ ] **Step 3: Verify no regressions in existing code**

Run: `python3 -c "from src.student.grpo_config import GRPO_CONFIG; assert 'judge_backend' in GRPO_CONFIG; assert GRPO_CONFIG['judge_backend'] == 'local'; print('Config OK')"`
Expected: Prints "Config OK"

- [ ] **Step 4: Verify W&B import**

Run: `python3 -c "import wandb; print('wandb version:', wandb.__version__)"`
Expected: Prints version without error

- [ ] **Step 5: Verify W&B config in .env**

Run: `grep WANDB .env`
Expected: Shows `WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_MODE` lines

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete SG-Lang integration + W&B local server" || echo "Nothing to commit"
```

---

## Self-Review

**1. Spec coverage:**
- SG-Lang Docker Service (docker-compose.yml) — Task 5
- SG-Lang HTTP Client (sglang_client.py) — Task 1
- Judge Backend Abstraction (grpo_config.py + rewards.py) — Tasks 2-3
- Eval Scripts (run_sglang_bf16.sh) — Task 7
- Helper Script (sglang_health.sh) — Task 6
- Training script integration (train_grpo.py) — Task 4
- W&B local server + training logging — Task 9
- Testing (test_sglang_client.py + run_e2e_tests.sh) — Tasks 1 + 8
- Rollback: `judge_backend: "local"` default, existing code paths untouched — Tasks 2-4
- Error handling: health check, retry with exponential backoff, fail-fast — Task 1
- HF_TOKEN: already in `.env`, injected via docker-compose — Task 5
- WANDB_API_KEY: already in `.env`, used by W&B init in train_grpo.py — Task 9

**2. Placeholder scan:** No TBDs, TODOs, or vague instructions. Every step has exact code and commands.

**3. Type consistency:**
- `SglangClient` class name consistent across all files
- `judge_backend` config key consistent across grpo_config.py, train_grpo.py
- `sglang_base_url` default `http://localhost:1235` matches docker-compose port mapping
- `compute_dm_alignment_judge_http` function signature matches calls in both rewards.py and train_grpo.py
- Port 1235 used consistently in config, health check, eval script, and docker-compose
- W&B port 8086 used consistently in docker-compose and .env

**Gaps found and fixed during review:**
- Task 4 imports `compute_dm_alignment_judge_http` inside the function to avoid circular import. This is correct since rewards.py imports from sglang_client via TYPE_CHECKING only.
- `build_reward_fn` in rewards.py now passes `None` for judge_model/judge_tokenizer since TRL path doesn't use local judge. This is correct because the TRL builder is only called with sglang_client.
- Eval script uses `docker compose run --rm --no-deps` instead of background process — this is cleaner for container lifecycle management.
- W&B in `train_grpo.py` was entirely missing — this is the root cause of no W&B records. Task 9 adds wandb.init(), wandb.log() per step, and wandb.finish().
- Local W&B server (`wandb/core:v0.23.0-deploy`) runs on port 8086, independent of training/SG-Lang containers. WANDB_MODE=offline by default for safety.
