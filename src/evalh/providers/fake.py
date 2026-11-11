"""A deterministic, seedable fake provider for tests and demos.

It answers from each case's ``expected`` value, so a correct answer is easy
to produce, and it can be told to get some cases wrong. No network calls.

Params (all optional, under ``params`` in the model spec):

* ``seed`` (int): changes which answers are wrong when ``accuracy`` < 1.
* ``accuracy`` (0..1): chance that each (case, repeat) is answered correctly.
* ``wrong_cases``: case ids that are always answered wrong.
* ``malformed_cases``: case ids that get broken JSON back.
* ``error_cases``: case ids that fail with a non-retryable HTTP 400.
* ``fail_first`` (int): every call first fails this many times with HTTP 429.
* ``latency_ms``: a number, or ``[min, max]`` for a seeded random delay.
* ``prose_wrap`` (bool): wrap JSON answers in a sentence and a code fence.
* ``report_usage`` (bool, default true): when false, no token usage is
  returned, so the harness falls back to an approximate count.
* ``fixed_response`` (str): always return this text.
* ``judge_min_score`` (1..5, default 3): lowest score the fake judge gives.

Generation params such as ``temperature`` or ``max_tokens`` are accepted and
ignored.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalh.config import ModelSpec
from evalh.providers.base import Completion, Provider, ProviderError, ProviderRequest
from evalh.tokens import approx_message_tokens, approx_tokens

_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_CURRENCIES = ["USD", "EUR", "GBP", "CAD", "INR", "JPY"]


class FakeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    accuracy: float = Field(default=1.0, ge=0, le=1)
    wrong_cases: list[str] = Field(default_factory=list)
    malformed_cases: list[str] = Field(default_factory=list)
    error_cases: list[str] = Field(default_factory=list)
    fail_first: int = Field(default=0, ge=0)
    latency_ms: float | list[float] = 0
    prose_wrap: bool = False
    report_usage: bool = True
    fixed_response: str | None = None
    judge_min_score: int = Field(default=3, ge=1, le=5)
    # Accepted and ignored so a fake model can mirror a real model's params.
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


def _rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _perturb_value(value: Any, rng: random.Random) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return round(value * 1.1 + 1, 2)
    if isinstance(value, str):
        date = _DATE.match(value)
        if date:
            year, month, day = (int(g) for g in date.groups())
            return (
                f"{year:04d}-{day:02d}-{month:02d}"
                if day <= 12 and day != month
                else (f"{year:04d}-{month:02d}-{(day % 28) + 1:02d}")
            )
        if value.upper() in _CURRENCIES:
            choices = [c for c in _CURRENCIES if c != value.upper()]
            return rng.choice(choices)
        words = value.split()
        if len(words) > 1:
            return " ".join(words[:-1])
        return value + " Ltd"
    return None


def _wrong_answer(expected: Any, rng: random.Random) -> Any:
    if isinstance(expected, dict) and expected:
        answer = dict(expected)
        key = rng.choice(sorted(answer))
        answer[key] = _perturb_value(answer[key], rng)
        return answer
    if isinstance(expected, list) and expected:
        return expected[:-1]
    if isinstance(expected, str):
        return "I am not sure."
    return None


def _render(answer: Any, prose_wrap: bool) -> str:
    if isinstance(answer, (dict, list)):
        text = json.dumps(answer, ensure_ascii=False)
        if prose_wrap:
            return f"Here is the result:\n```json\n{text}\n```"
        return text
    if isinstance(answer, str):
        return answer
    return "" if answer is None else str(answer)


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, spec: ModelSpec, timeout_s: float = 60.0):
        super().__init__(spec, timeout_s)
        self.options = FakeParams.model_validate(spec.params)
        self._attempts: dict[str, int] = {}
        self.calls = 0

    async def complete(self, request: ProviderRequest) -> Completion:
        self.calls += 1
        opts = self.options
        key = hashlib.sha256(
            json.dumps(
                [request.case_id, request.repeat, request.purpose, request.messages],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        rng = _rng(
            opts.seed, self.spec.model, request.purpose, request.case_id, request.repeat, key
        )

        delay = opts.latency_ms
        if isinstance(delay, list):
            low, high = (delay[0], delay[-1]) if delay else (0, 0)
            delay = rng.uniform(low, high)
        if delay:
            await asyncio.sleep(delay / 1000)

        attempt = self._attempts.get(key, 0)
        self._attempts[key] = attempt + 1
        if attempt < opts.fail_first:
            raise ProviderError("HTTP 429: fake rate limit", status=429, retryable=True)
        if request.case_id in opts.error_cases:
            raise ProviderError("HTTP 400: fake bad request", status=400, retryable=False)

        text = self._answer(request, rng)
        if not opts.report_usage:
            return Completion(text=text)
        return Completion(
            text=text,
            input_tokens=approx_message_tokens(request.messages),
            output_tokens=approx_tokens(text),
        )

    def _answer(self, request: ProviderRequest, rng: random.Random) -> str:
        opts = self.options
        if opts.fixed_response is not None:
            return opts.fixed_response
        if request.purpose == "judge":
            score = rng.randint(opts.judge_min_score, 5)
            return json.dumps({"score": score, "reason": "fake judge verdict"})
        if request.case_id in opts.malformed_cases:
            return _render(request.expected, False)[:-3] or "{"
        correct = request.case_id not in opts.wrong_cases and rng.random() < opts.accuracy
        if request.expected is None:
            return f"fake response for {request.case_id}"
        answer = request.expected if correct else _wrong_answer(request.expected, rng)
        return _render(answer, opts.prose_wrap)
