"""Provider registry."""

from __future__ import annotations

from evalh.config import ModelSpec
from evalh.providers.base import Completion, Provider, ProviderError, ProviderRequest


def make_provider(spec: ModelSpec, timeout_s: float) -> Provider:
    """Create the adapter for one model spec."""
    if spec.provider == "fake":
        from evalh.providers.fake import FakeProvider

        return FakeProvider(spec, timeout_s)
    if spec.provider == "openai_compatible":
        from evalh.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(spec, timeout_s)
    if spec.provider == "anthropic":
        from evalh.providers.anthropic import AnthropicProvider

        return AnthropicProvider(spec, timeout_s)
    raise ProviderError(f"unknown provider '{spec.provider}'")


__all__ = ["Completion", "Provider", "ProviderError", "ProviderRequest", "make_provider"]
