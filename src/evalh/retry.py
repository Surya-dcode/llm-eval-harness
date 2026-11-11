"""Retry with exponential backoff for rate limits, server errors and timeouts."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from evalh.providers.base import ProviderError

T = TypeVar("T")


def backoff_delay(attempt: int, base_s: float, max_s: float, retry_after: float | None) -> float:
    """Delay before retry number ``attempt`` (0-based). Honors Retry-After when given."""
    if retry_after is not None:
        return min(retry_after, max_s) if max_s else retry_after
    delay = base_s * (2**attempt)
    delay = min(delay, max_s) if max_s else delay
    # Full jitter keeps parallel workers from retrying in lockstep.
    return random.uniform(delay / 2, delay) if delay else 0.0


async def with_retries(
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    timeout_s: float,
    base_s: float,
    max_s: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[T, int]:
    """Run ``call`` with a timeout, retrying retryable failures.

    Returns (result, attempts). Raises the last ProviderError when retries run
    out or the error is not retryable.
    """
    attempt = 0
    while True:
        try:
            result = await asyncio.wait_for(call(), timeout=timeout_s)
            return result, attempt + 1
        except TimeoutError:
            error = ProviderError(f"timed out after {timeout_s:g}s", retryable=True)
        except ProviderError as exc:
            error = exc
        if not error.retryable or attempt >= max_retries:
            error.attempts = attempt + 1  # type: ignore[attr-defined]
            raise error
        await sleep(backoff_delay(attempt, base_s, max_s, error.retry_after))
        attempt += 1
