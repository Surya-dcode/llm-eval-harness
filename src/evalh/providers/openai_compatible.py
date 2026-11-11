"""Adapter for any server that speaks the OpenAI chat completions API.

This covers OpenAI, Groq, Together, OpenRouter, and local servers such as
Ollama or vLLM. Point ``base_url`` at the server and ``api_key_env`` at the
environment variable that holds the key (leave it out for local servers).
"""

from __future__ import annotations

import os

import httpx

from evalh.config import ModelSpec
from evalh.providers.base import (
    Completion,
    Provider,
    ProviderError,
    ProviderRequest,
    is_retryable_status,
    parse_retry_after,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"

    def __init__(
        self,
        spec: ModelSpec,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(spec, timeout_s)
        headers = {"Content-Type": "application/json"}
        if spec.api_key_env:
            key = os.environ.get(spec.api_key_env)
            if not key:
                raise ProviderError(
                    f"model '{spec.id}': environment variable {spec.api_key_env} is not set"
                )
            headers["Authorization"] = f"Bearer {key}"
        self.url = (spec.base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
        self.client = httpx.AsyncClient(headers=headers, timeout=timeout_s, transport=transport)

    async def complete(self, request: ProviderRequest) -> Completion:
        body = {"model": self.spec.model, "messages": request.messages, **request.params}
        try:
            response = await self.client.post(self.url, json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"timeout: {exc}", retryable=True) from None
        except httpx.TransportError as exc:
            raise ProviderError(f"connection error: {exc}", retryable=True) from None

        if response.status_code >= 400:
            snippet = response.text[:300]
            raise ProviderError(
                f"HTTP {response.status_code}: {snippet}",
                status=response.status_code,
                retryable=is_retryable_status(response.status_code),
                retry_after=parse_retry_after(response.headers.get("retry-after")),
            )
        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError(f"unexpected response shape: {response.text[:300]}") from None

        text = message.get("content") or ""
        usage = payload.get("usage") or {}
        return Completion(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def aclose(self) -> None:
        await self.client.aclose()
