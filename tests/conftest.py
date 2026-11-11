from __future__ import annotations

import copy
from typing import Any

import pytest

from evalh.config import parse_task

BASE_TASK: dict[str, Any] = {
    "name": "unit",
    "system_prompt": "Reply with JSON.",
    "prompt_template": "Extract from: {{ text }}",
    "cases": [
        {"id": "c1", "vars": {"text": "Acme, 10 USD"}, "expected": {"vendor": "Acme", "total": 10}},
        {"id": "c2", "vars": {"text": "Beta, 20 EUR"}, "expected": {"vendor": "Beta", "total": 20}},
    ],
    "scorers": [{"type": "field_match"}],
    "models": [
        {
            "id": "fake-a",
            "provider": "fake",
            "model": "fake-a",
            "price_in_per_mtok": 1.0,
            "price_out_per_mtok": 2.0,
        }
    ],
    "execution": {"backoff_base_s": 0, "max_retries": 2},
}


def task_dict(**overrides: Any) -> dict[str, Any]:
    data = copy.deepcopy(BASE_TASK)
    data.update(copy.deepcopy(overrides))
    return data


@pytest.fixture
def make_task():
    def _make(**overrides: Any):
        return parse_task(task_dict(**overrides))

    return _make
