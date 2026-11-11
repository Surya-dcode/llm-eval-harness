"""Helpers for pulling JSON out of model output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


class JSONExtractError(ValueError):
    pass


def parse_json(text: str, extract: bool = True) -> Any:
    """Parse JSON from model output.

    With ``extract=False`` the whole (stripped) text must be valid JSON.
    With ``extract=True`` a fenced code block is unwrapped first, and if the
    text still does not parse, the first JSON object or array found in it is
    decoded. Raises JSONExtractError with a short reason when nothing parses.
    """
    stripped = text.strip()
    if not extract:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise JSONExtractError(f"invalid JSON: {exc.msg} (position {exc.pos})") from None

    fence = _FENCE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char in "{[":
                try:
                    value, _ = decoder.raw_decode(stripped, index)
                    return value
                except json.JSONDecodeError:
                    continue
        raise JSONExtractError(
            f"no valid JSON found: {first_error.msg} (position {first_error.pos})"
        ) from None


def get_path(data: Any, dotted: str) -> tuple[bool, Any]:
    """Look up a dotted path (``a.b.0.c``) in parsed JSON. Returns (found, value)."""
    current = data
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current
