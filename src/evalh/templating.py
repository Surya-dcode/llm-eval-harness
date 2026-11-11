"""Prompt template rendering.

Two syntaxes are supported:

* ``double_brace`` (default): Jinja-style ``{{ name }}`` placeholders. Only
  variable substitution is supported (with dotted access such as
  ``{{ customer.name }}``), no loops or filters.
* ``format``: Python ``str.format`` with ``{name}`` placeholders.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\}\}")


class TemplateError(ValueError):
    """Raised when a template cannot be rendered with the given variables."""


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _lookup(variables: dict[str, Any], dotted: str) -> Any:
    current: Any = variables
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise TemplateError(f"missing template variable '{dotted}'")
    return current


def placeholders(template: str, syntax: str = "double_brace") -> list[str]:
    """Return the top-level variable names a template refers to."""
    if syntax == "double_brace":
        return sorted({m.group(1).split(".")[0] for m in _PLACEHOLDER.finditer(template)})
    names = set()
    for _, field, _, _ in string.Formatter().parse(template):
        if field:
            names.add(re.split(r"[.\[]", field, maxsplit=1)[0])
    return sorted(names)


def render(template: str, variables: dict[str, Any], syntax: str = "double_brace") -> str:
    """Render ``template`` with ``variables``. Raises TemplateError on a missing name."""
    if syntax == "double_brace":
        return _PLACEHOLDER.sub(lambda m: _to_text(_lookup(variables, m.group(1))), template)
    if syntax == "format":
        try:
            return template.format(**variables)
        except KeyError as exc:
            raise TemplateError(f"missing template variable {exc}") from None
        except (IndexError, ValueError) as exc:
            raise TemplateError(f"bad format template: {exc}") from None
    raise TemplateError(f"unknown template syntax '{syntax}'")
