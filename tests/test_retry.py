import asyncio

import httpx
import pytest

from evalh.config import ModelSpec
from evalh.providers.base import ProviderError, ProviderRequest
from evalh.providers.openai_compatible import OpenAICompatibleProvider
from evalh.retry import backoff_delay, with_retries
from evalh.runner import Runner


class Sleeps:
    def __init__(self):
        self.delays = []

    async def __call__(self, delay):
        self.delays.append(delay)


def spec(**overrides):
    data = {
        "id": "oa",
        "provider": "openai_compatible",
        "model": "test-model",
        "base_url": "http://testserver/v1",
        "price_in_per_mtok": 1.0,
        "price_out_per_mtok": 2.0,
    }
    data.update(overrides)
    return ModelSpec.model_validate(data)


def mock_server(responses):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen), len(responses)) - 1]

    return httpx.MockTransport(handler), seen


OK = httpx.Response(
    200,
    json={
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    },
)


def test_retry_on_fake_429_then_success(make_task):
    task = make_task()
    task.models[0].params = {"fail_first": 2}
    sleeps = Sleeps()
    rows = asyncio.run(Runner(task, sleep=sleeps).run())
    assert all(r.error is None for r in rows)
    assert [r.attempts for r in rows] == [3, 3]
    assert len(sleeps.delays) == 4


def test_retries_exhausted_becomes_failed_row(make_task):
    task = make_task()
    task.models[0].params = {"fail_first": 10}
    rows = asyncio.run(Runner(task, sleep=Sleeps()).run())
    assert all(r.error and "429" in r.error for r in rows)
    assert all(r.attempts == 3 for r in rows)  # max_retries 2 in the base task
    assert all(r.score == 0 and not r.passed and r.cost_usd == 0 for r in rows)


def test_non_retryable_error_is_not_retried(make_task):
    task = make_task()
    task.models[0].params = {"error_cases": ["c2"]}
    sleeps = Sleeps()
    rows = asyncio.run(Runner(task, sleep=sleeps).run())
    assert rows[0].error is None
    assert rows[1].error.startswith("HTTP 400") and rows[1].attempts == 1
    assert sleeps.delays == []


def test_openai_compatible_retries_429_and_honors_retry_after():
    transport, seen = mock_server(
        [httpx.Response(429, headers={"retry-after": "2"}, json={"error": "slow down"}), OK]
    )
    provider = OpenAICompatibleProvider(spec(), transport=transport)
    request = ProviderRequest(
        messages=[{"role": "user", "content": "hi"}], params={"temperature": 0}
    )
    sleeps = Sleeps()

    async def go():
        try:
            return await with_retries(
                lambda: provider.complete(request),
                max_retries=3,
                timeout_s=5,
                base_s=1,
                max_s=30,
                sleep=sleeps,
            )
        finally:
            await provider.aclose()

    completion, attempts = asyncio.run(go())
    assert (completion.text, completion.input_tokens, completion.output_tokens) == ("hello", 12, 3)
    assert attempts == 2
    assert sleeps.delays == [2.0]
    assert len(seen) == 2
    body = seen[0].read()
    assert b'"model":"test-model"' in body.replace(b" ", b"") and b"temperature" in body


@pytest.mark.parametrize(
    "status, retryable", [(500, True), (503, True), (400, False), (401, False)]
)
def test_openai_compatible_status_mapping(status, retryable):
    transport, _ = mock_server([httpx.Response(status, text="nope")])
    provider = OpenAICompatibleProvider(spec(), transport=transport)

    async def go():
        try:
            await provider.complete(ProviderRequest(messages=[{"role": "user", "content": "hi"}]))
        finally:
            await provider.aclose()

    with pytest.raises(ProviderError) as info:
        asyncio.run(go())
    assert info.value.status == status
    assert info.value.retryable is retryable


def test_openai_compatible_sends_bearer_key(monkeypatch):
    monkeypatch.setenv("TEST_EVALH_KEY", "sk-test")
    transport, seen = mock_server([OK])
    provider = OpenAICompatibleProvider(spec(api_key_env="TEST_EVALH_KEY"), transport=transport)

    async def go():
        try:
            await provider.complete(ProviderRequest(messages=[{"role": "user", "content": "hi"}]))
        finally:
            await provider.aclose()

    asyncio.run(go())
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert str(seen[0].url) == "http://testserver/v1/chat/completions"


def test_openai_compatible_missing_key_env(monkeypatch):
    monkeypatch.delenv("TEST_EVALH_MISSING", raising=False)
    with pytest.raises(ProviderError, match="TEST_EVALH_MISSING is not set"):
        OpenAICompatibleProvider(spec(api_key_env="TEST_EVALH_MISSING"))


def test_missing_usage_falls_back_to_approximate_tokens(make_task):
    task = make_task()
    task.models[0].params = {"report_usage": False}
    rows = asyncio.run(Runner(task).run())
    assert all(r.tokens_approximate for r in rows)
    assert all(r.input_tokens > 0 and r.output_tokens > 0 and r.cost_usd > 0 for r in rows)


def test_timeout_is_retried():
    attempts = 0

    async def slow():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(1)
        return "done"

    result, count = asyncio.run(
        with_retries(slow, max_retries=1, timeout_s=0.05, base_s=0, max_s=0, sleep=Sleeps())
    )
    assert (result, count, attempts) == ("done", 2, 2)


def test_backoff_delay_grows_and_caps():
    assert backoff_delay(0, 1, 30, None) <= 1
    assert 4 <= backoff_delay(3, 1, 30, None) <= 8
    assert backoff_delay(10, 1, 30, None) <= 30
    assert backoff_delay(0, 1, 30, 45) == 30
    assert backoff_delay(0, 0, 0, None) == 0
