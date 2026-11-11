"""Token counting fallback and cost math."""

from __future__ import annotations

import math
from collections.abc import Iterable

# Rough rule of thumb for English text with BPE tokenizers. It is only used
# when a provider does not return usage, and every such row is flagged.
CHARS_PER_TOKEN = 4
PER_MESSAGE_OVERHEAD = 4


def approx_tokens(text: str) -> int:
    """Approximate token count for a piece of text (ceil(chars / 4))."""
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def approx_message_tokens(messages: Iterable[dict]) -> int:
    """Approximate prompt tokens for a chat message list."""
    total = 0
    for message in messages:
        total += PER_MESSAGE_OVERHEAD + approx_tokens(str(message.get("content", "")))
    return total


def call_cost(
    input_tokens: int,
    output_tokens: int,
    price_in_per_mtok: float,
    price_out_per_mtok: float,
) -> float:
    """Cost in USD of one call, given prices per one million tokens."""
    return (input_tokens * price_in_per_mtok + output_tokens * price_out_per_mtok) / 1_000_000
