"""Provider interface shared by every adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evalh.config import ModelSpec


@dataclass
class ProviderRequest:
    messages: list[dict[str, str]]
    params: dict[str, Any] = field(default_factory=dict)
    # Context about the case. Real providers ignore it; the fake provider uses
    # it to produce deterministic answers.
    case_id: str | None = None
    expected: Any = None
    repeat: int = 0
    purpose: str = "candidate"  # or "judge"


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderError(Exception):
    """A failed call. ``retryable`` marks rate limits, server errors and timeouts."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


def is_retryable_status(status: int) -> bool:
    return status == 429 or status == 408 or 500 <= status < 600


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class Provider(ABC):
    name: str = "base"

    def __init__(self, spec: ModelSpec, timeout_s: float = 60.0):
        self.spec = spec
        self.timeout_s = timeout_s

    @abstractmethod
    async def complete(self, request: ProviderRequest) -> Completion: ...

    async def aclose(self) -> None:
        return None
