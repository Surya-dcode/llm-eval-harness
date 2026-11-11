"""Adapter for the Anthropic Messages API (install with the ``anthropic`` extra)."""

from __future__ import annotations

import os
from typing import Any

from evalh.config import ModelSpec
from evalh.providers.base import (
    Completion,
    Provider,
    ProviderError,
    ProviderRequest,
    is_retryable_status,
)

DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, spec: ModelSpec, timeout_s: float = 60.0, client: Any = None):
        super().__init__(spec, timeout_s)
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise ProviderError(
                    "the anthropic provider needs the optional SDK: "
                    "pip install 'llm-eval-harness[anthropic]'"
                ) from None
            env_name = spec.api_key_env or "ANTHROPIC_API_KEY"
            key = os.environ.get(env_name)
            if not key:
                raise ProviderError(
                    f"model '{spec.id}': environment variable {env_name} is not set"
                )
            kwargs: dict[str, Any] = {"api_key": key, "max_retries": 0, "timeout": timeout_s}
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
            client = anthropic.AsyncAnthropic(**kwargs)
        self.client = client

    async def complete(self, request: ProviderRequest) -> Completion:
        system_parts = [m["content"] for m in request.messages if m["role"] == "system"]
        messages = [m for m in request.messages if m["role"] != "system"]
        params = dict(request.params)
        params.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        kwargs: dict[str, Any] = {"model": self.spec.model, "messages": messages, **params}
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        try:
            response = await self.client.messages.create(**kwargs)
        except Exception as exc:
            raise _map_error(exc) from None

        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def _map_error(exc: Exception) -> ProviderError:
    """Map SDK exceptions to ProviderError without importing the SDK."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                retry_after = float(headers.get("retry-after"))
            except (TypeError, ValueError):
                retry_after = None
        return ProviderError(
            f"HTTP {status}: {exc}",
            status=status,
            retryable=is_retryable_status(status),
            retry_after=retry_after,
        )
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name:
        return ProviderError(f"{name}: {exc}", retryable=True)
    return ProviderError(f"{name}: {exc}")
