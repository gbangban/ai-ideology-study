"""
SG-Lang HTTP Client

Thin wrapper around requests for SG-Lang's OpenAI-compatible API.
Used for judge offloading during GRPO training and lm_eval evaluations.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

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
