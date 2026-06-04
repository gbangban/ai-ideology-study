# SG-Lang Integration — Judge Offloading + Evals

Date: 2026-06-03
Status: Approved

## Overview

Integrate SG-Lang (high-performance LLM serving framework) as an external HTTP inference backend for two purposes:
1. **Judge offloading** — move the Qwen3.5-4B DM alignment judge from the training container to SG-Lang. Eliminates per-batch judge load/unload churn and leverages SG-Lang's continuous batching for concurrent scoring of all completions in a batch.
2. **Evals** — serve BF16 student model via SG-Lang for lm_eval benchmarks. SG-Lang's inference engine (continuous batching, radix attention prefix caching) outperforms lm_eval's native HF backend.

The student model (Qwen3.5-9B) remains local in the training container for both generation and gradient updates. No model sync between containers.

**Primary goal: speed via concurrency.** Quantization is a secondary knob for VRAM when needed.

## Architecture

```
[Training Container - ml-training]              [SG-Lang Container - localhost:1235]
┌─────────────────────────────┐                ┌─────────────────────────────┐
│  NF4 Student (9B) + LoRA    │                │  BF16 Judge (4B)            │
│  ~12-15 GB VRAM             │  HTTP POST     │  ~8 GB VRAM                 │
│                             │ ──────────────▶│  (continuous batching)      │
│  generate_completions()     │  (judge calls) │                             │
│  compute_rewards()  ───────┤                │                             │
│  policy_update()            │ ◀─────────────┤                             │
│                             │                │  (evals: BF16 Student 9B)   │
└─────────────────────────────┘                │  ~18 GB VRAM               │
                                               └─────────────────────────────┘
```

Both containers share the same RTX 5090 (32GB) via Docker Desktop/NVIDIA runtime.

### System-Wide VRAM Budget

| Scenario | Training Container | SG-Lang Container | Total System |
|----------|-------------------|-------------------|-------------|
| Current (no SG-Lang, training) | NF4 student ~6GB + NF4 judge ~2GB + KV cache ~6GB | N/A | **~14GB** |
| With SG-Lang (training + judge) | NF4 student ~6GB + KV cache ~6GB | BF16 judge ~8GB + runtime ~2GB | **~22GB** |
| With SG-Lang (evals only) | Stopped | BF16 student ~18GB + KV cache ~4GB | **~22GB** |
| With SG-Lang (evals, quantized judge) | Stopped | INT4 judge ~2GB + runtime ~1GB | **~3GB** |

Judge offloading uses more total VRAM (BF16 judge > NF4 judge) but eliminates per-batch load/unload overhead and enables continuous batching of all completions.

## Components

### 1. SG-Lang Docker Service

New service in `docker-compose.yml`:

- Image: `lmsysorg/sglang:latest` (CUDA 13.0, v0.5.12.post1)
- Port: 1235 (avoiding 1234 conflict)
- GPU: all, NVIDIA runtime
- Volumes: HF cache mount (`C:/Users/Guy/.cache/huggingface`), Studio exports
- Environment: `HF_HOME`, `HF_TOKEN` (injected from host `.env`)
- `HF_TOKEN` is required for downloading gated models (Qwen3.5-9B, Qwen3.5-4B)

The docker-compose service defines the SG-Lang container but does NOT auto-start a model. Model is launched via server command line args:

- **Training (judge):** server started with `--model-path Qwen/Qwen3.5-4B`
- **Evals (student):** server started with `--model-path <student-checkpoint>` and optionally `--quantization fp8`

### 2. SG-Lang HTTP Client

New module: `src/student/sglang_client.py`

Thin wrapper around `requests` for SG-Lang's OpenAI-compatible API:

- `chat_completion(messages, **kwargs)` — wraps `/v1/chat/completions`
- `batch_chat_completion(requests_list)` — sends multiple requests in parallel using `concurrent.futures.ThreadPoolExecutor` for maximum throughput
- `health_check()` — wraps `/v1/models` to verify SG-Lang is reachable
- Connection timeout, retry logic, and error handling

No new dependencies — uses only `requests` (already transitively available via transformers/datasets).

### 3. Judge Backend Abstraction

Modified `rewards.py`:

- `compute_dm_alignment_judge()` accepts either:
  - `judge_model` + `judge_tokenizer` (local, existing behavior)
  - `sglang_client` (HTTP, new behavior)
- Branching logic controlled by `grpo_config.py` `"judge_backend"` field

Modified `grpo_config.py`:

```python
"judge_backend": "local",       # "local", "sglang", or "sglang-quantized"
"sglang_base_url": "http://localhost:1235",
"sglang_judge_quantization": None,  # None (BF16), "fp8", or "int4"
```

Three modes:
- `"local"` — existing behavior, judge loaded/unloaded per batch in training container (default, rollback-safe)
- `"sglang"` — judge served as BF16 in SG-Lang (default recommendation, fastest)
- `"sglang-quantized"` — judge served as FP8/INT4 in SG-Lang (lower VRAM, configurable via `sglang_judge_quantization`)

### 4. Eval Scripts

New scripts in `evals/scripts/`:

- `run_sglang_bf16.sh` — launches SG-Lang with BF16 student model, runs lm_eval suite (primary eval path)

Both follow the existing eval script pattern:
- Same argument parsing (`--tasks`, `--suite`, `--dry-run`)
- Same logging via `eval_logging.sh`
- Results saved to `evals/results/sglang/bf16/`
- Uses lm_eval's OpenAI backend: `--model openai-completions --model_args base_url=http://localhost:1235,vllm_guided_decoding_enabled=False`

The script manages the SG-Lang server lifecycle:
1. Launches SG-Lang server with target model (BF16)
2. Waits for health check on port 1235
3. Runs lm_eval with OpenAI backend
4. SG-Lang handles continuous batching and prefix caching across eval samples
5. Results written to `evals/results/sglang/bf16/`
6. Server is torn down after eval completes
7. `compare_results.py` handles cross-format comparison

### 5. Helper Script

`scripts/sglang_health.sh` — verifies SG-Lang container is running and responsive before training or evals begin.

## Data Flow

### During GRPO Training (judge offloading)

1. Training container generates G=8 completions per prompt locally (unchanged)
2. For DM alignment reward, batches ALL completions (e.g., 32 = 4 prompts × 8 groups) and sends parallel HTTP requests to SG-Lang via `batch_chat_completion()`
3. SG-Lang processes all requests with continuous batching — no per-batch model load/unload
4. Training container parses judge output locally (existing `_parse_judge_output`)
5. Policy update proceeds without VRAM churn from judge model

Speed improvement sources:
- Eliminates judge CPU→GPU transfer per batch (~1-2s saved per batch)
- SG-Lang's continuous batching scores all completions in one GPU pass
- Training container's VRAM is stable (no judge load/unload cycle)

### During Evaluation

1. Eval script launches SG-Lang server with BF16 student model
2. lm_eval sends completion requests via OpenAI-compatible API
3. SG-Lang's continuous batching and radix attention prefix cache process eval samples faster than lm_eval's native HF backend
4. Results written to `evals/results/sglang/bf16/`

## Rollback Strategy

- Existing `train_grpo.py` and `rewards.py` code paths remain functional
- `judge_backend: "local"` is the default — SG-Lang integration is opt-in
- If SG-Lang is unreachable, the training script fails fast with a clear error message
- Existing eval scripts (`run_baseline_bf16.sh`, `run_finetuned_bf16.sh`, `run_grpo_bf16.sh`) are untouched

## Error Handling

- SG-Lang client wraps HTTP errors with descriptive messages
- Training script checks SG-Lang health before starting (when `judge_backend` is not `"local"`)
- Timeout: 60s per judge request batch (32 completions × 128 max tokens is fast)
- Retry: 3 attempts with exponential backoff on transient failures
- If SG-Lang is down during training, the script aborts (don't silently fall back to local)

## Testing

- Unit test for SG-Lang client with mocked HTTP responses
- Integration test: start SG-Lang container, send test completion, verify response format
- Modified test suite to include SG-Lang judge path (skipped if SG-Lang not available)

## Files to Create/Modify

**New files:**
- `src/student/sglang_client.py`
- `evals/scripts/run_sglang_bf16.sh`
- `scripts/sglang_health.sh`

**Modified files:**
- `docker-compose.yml` — add `sglang` service with `HF_TOKEN` env injection
- `.env` — add `HF_TOKEN` variable (if not already present)
- `src/student/grpo_config.py` — add `judge_backend`, `sglang_base_url`, `sglang_judge_quantization` fields
- `src/student/rewards.py` — add SG-Lang judge path to `compute_dm_alignment_judge()`
- `src/student/train_grpo.py` — wire up SG-Lang client when `judge_backend` is not `"local"`
