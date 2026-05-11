"""
Silver Pipeline — Judge Client

Wraps the OpenAI-compatible vLLM chat/completions endpoint for calling
the 72B-class judge model (Qwen 2.5 72B Instruct primary, Llama-3-70B
fallback).

Uses the `openai` Python client (already installed in the project conda
env and used throughout the existing pipeline).

Design notes:
  - No prefill trick (unlike the candidate generator which uses <final>).
    The judge outputs a complete JSON response from scratch.
  - Temperature defaults to 0.3 (baseline), stepping to 0.1 on retries
    (per §9.2).
  - max_tokens is 512 (enough for exact_quote + reasoning + label).
  - Retries at the HTTP level (connection/timeout errors) are separate from
    the Tier 2 retries in the recovery pipeline. HTTP retries are handled
    here; Tier 2 is handled by the caller (judge_runner.py, Phase 3 cont.).

Usage:
    client = JudgeClient(
        base_url="http://127.0.0.1:8000/v1",
        model_id="qwen2.5-72b-instruct",
    )
    raw_output = client.generate(messages, temperature=0.3)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JudgeClient:
    """
    Thin wrapper around the OpenAI chat/completions API for judge calls.

    The base_url should point to the vLLM server endpoint, e.g.:
      http://127.0.0.1:8000/v1
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    model_id: str = ""     # auto-detected from server if empty
    max_tokens: int = 512
    timeout: int = 180     # seconds (72B models are slower than 7B)
    http_retries: int = 3
    http_retry_sleep: float = 5.0

    # Set after connect()
    _client: object = field(default=None, repr=False)
    _connected: bool = field(default=False, repr=False)

    def connect(self) -> str:
        """
        Initialize the OpenAI client and auto-detect the model ID.

        Returns the model ID string. Raises if the server is unreachable.
        """
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.base_url,
            api_key="EMPTY",        # vLLM doesn't require a real key
            timeout=self.timeout,
        )

        # Auto-detect model if not specified
        if not self.model_id:
            models = self._client.models.list()
            if not models.data:
                raise ConnectionError(
                    f"No models available at {self.base_url}. "
                    f"Is the vLLM server running?"
                )
            self.model_id = models.data[0].id

        self._connected = True
        return self.model_id

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Make a single judge call and return the raw text output.

        Args:
            messages: list of {"role": ..., "content": ...} dicts
                      (from prompt_v1.format_judge_messages).
            temperature: sampling temperature (0.3 baseline, 0.1 for retries).
            max_tokens: override default max_tokens if needed.

        Returns:
            Raw text from the model's response.

        Raises:
            ConnectionError if not connected.
            RuntimeError after exhausting HTTP retries.
        """
        if not self._connected:
            self.connect()

        mt = max_tokens if max_tokens is not None else self.max_tokens
        last_error: Optional[Exception] = None

        for attempt in range(self.http_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=mt,
                )
                content = response.choices[0].message.content or ""
                return content.strip()

            except Exception as e:
                last_error = e
                if attempt < self.http_retries:
                    sleep = self.http_retry_sleep * (attempt + 1)
                    print(
                        f"  [JudgeClient] HTTP error (attempt {attempt + 1}/"
                        f"{self.http_retries + 1}): {e}. "
                        f"Retrying in {sleep:.0f}s..."
                    )
                    time.sleep(sleep)

        raise RuntimeError(
            f"JudgeClient: all {self.http_retries + 1} attempts failed. "
            f"Last error: {last_error}"
        )

    def is_connected(self) -> bool:
        return self._connected

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return (
            f"JudgeClient(base_url={self.base_url!r}, "
            f"model_id={self.model_id!r}, status={status})"
        )