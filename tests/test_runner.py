import asyncio
import json
from types import SimpleNamespace

import pytest

from evalh.config import ModelSpec
from evalh.providers.anthropic import AnthropicProvider
from evalh.providers.base import Completion, Provider, ProviderError, ProviderRequest
from evalh.providers.fake import FakeProvider
from evalh.runner import Runner
from tests.conftest import task_dict

JUDGE_MODEL = {
    "id": "judge",
    "provider": "fake",
    "model": "fake-judge",
    "price_in_per_mtok": 10.0,
    "price_out_per_mtok": 10.0,
}


def judge_task(make_task, judge_params):
    return make_task(
        judge_models=[{**JUDGE_MODEL, "params": judge_params}],
        scorers=[{"type": "llm_judge", "judge": "judge", "rubric": "Correct fields?"}],
    )


def test_judge_parse_failure_is_scored_not_raised(make_task):
    task = judge_task(make_task, {"fixed_response": "Looks good to me!"})
    rows = asyncio.run(Runner(task).run())
    assert len(rows) == 2
    for row in rows:
        assert row.error is None
        assert row.score == 0 and not row.passed
        (score,) = row.scores
        assert score.details["error"] == "judge_parse_error"
        assert score.details["raw"] == "Looks good to me!"
        assert row.judge_cost_usd > 0


def test_judge_valid_reply_scores(make_task):
    task = judge_task(make_task, {"fixed_response": json.dumps({"score": 5, "reason": "exact"})})
    rows = asyncio.run(Runner(task).run())
    assert all(r.score == 1.0 and r.passed for r in rows)
    assert rows[0].scores[0].details["reason"] == "exact"


def test_judge_call_failure_is_scored(make_task):
    task = judge_task(make_task, {"fail_first": 99})
    task.execution.max_retries = 0
    rows = asyncio.run(Runner(task).run())
    assert all(r.scores[0].details["error"] == "judge_call_failed" for r in rows)


def test_per_case_scorers_override_task_scorers(make_task):
    data = task_dict()
    data["cases"][1]["scorers"] = [{"type": "regex", "pattern": "Beta"}]
    from evalh.config import parse_task

    task = parse_task(data)
    rows = asyncio.run(Runner(task).run())
    assert [s.name for s in rows[0].scores] == ["field_match"]
    assert [s.name for s in rows[1].scores] == ["regex"]


def test_duplicate_scorer_names_get_suffixes(make_task):
    task = make_task(scorers=[{"type": "regex", "pattern": "A"}, {"type": "regex", "pattern": "B"}])
    rows = asyncio.run(Runner(task).run())
    assert [s.name for s in rows[0].scores] == ["regex", "regex_2"]


def test_model_filter_and_repeats(make_task):
    data_models = [
        {**JUDGE_MODEL, "id": "m1", "model": "m1"},
        {**JUDGE_MODEL, "id": "m2", "model": "m2"},
    ]
    task = make_task(models=data_models)
    rows = asyncio.run(Runner(task).run(model_ids=["m2"], repeats=3))
    assert {r.model_id for r in rows} == {"m2"}
    assert sorted({r.repeat for r in rows}) == [0, 1, 2]
    with pytest.raises(ValueError, match="unknown model id"):
        asyncio.run(Runner(task).run(model_ids=["m3"]))


def test_fake_provider_is_deterministic_and_seedable(make_task):
    task = make_task(repeats=5)
    task.models[0].params = {"accuracy": 0.5, "seed": 1}
    first = [r.output for r in asyncio.run(Runner(task).run())]
    second = [r.output for r in asyncio.run(Runner(task).run())]
    assert first == second
    task.models[0].params = {"accuracy": 0.5, "seed": 2}
    third = [r.output for r in asyncio.run(Runner(task).run())]
    assert third != first


def test_fake_wrong_and_malformed_cases(make_task):
    task = make_task()
    task.models[0].params = {"wrong_cases": ["c1"], "malformed_cases": ["c2"]}
    rows = asyncio.run(Runner(task).run())
    assert 0 < rows[0].score < 1  # one field perturbed
    assert rows[1].scores[0].details["error"].startswith("no valid JSON found")


class GateProvider(Provider):
    """Counts how many calls are in flight at once."""

    active = 0
    peak = 0

    async def complete(self, request: ProviderRequest) -> Completion:
        GateProvider.active += 1
        GateProvider.peak = max(GateProvider.peak, GateProvider.active)
        await asyncio.sleep(0.01)
        GateProvider.active -= 1
        return Completion(text='{"vendor": "Acme", "total": 10}', input_tokens=1, output_tokens=1)


def test_per_provider_concurrency_limit(make_task):
    task = make_task(repeats=10, execution={"concurrency": 3, "backoff_base_s": 0})
    GateProvider.active = GateProvider.peak = 0
    rows = asyncio.run(Runner(task, provider_factory=lambda s, t: GateProvider(s, t)).run())
    assert len(rows) == 20
    assert GateProvider.peak == 3

    task.execution.provider_concurrency = {"fake": 1}
    GateProvider.peak = 0
    asyncio.run(Runner(task, provider_factory=lambda s, t: GateProvider(s, t)).run())
    assert GateProvider.peak == 1


def test_missing_sdk_or_key_fails_before_any_call(make_task, monkeypatch):
    task = make_task(
        models=[
            {**JUDGE_MODEL, "id": "ok"},
            {
                "id": "oa",
                "provider": "openai_compatible",
                "model": "x",
                "api_key_env": "TEST_EVALH_UNSET",
                "price_in_per_mtok": 0,
                "price_out_per_mtok": 0,
            },
        ]
    )
    monkeypatch.delenv("TEST_EVALH_UNSET", raising=False)
    created = []

    def factory(spec, timeout):
        from evalh.providers import make_provider

        provider = make_provider(spec, timeout)
        created.append(provider)
        return provider

    with pytest.raises(ProviderError, match="TEST_EVALH_UNSET"):
        asyncio.run(Runner(task, provider_factory=factory).run())
    assert all(isinstance(p, FakeProvider) and p.calls == 0 for p in created)


# Anthropic adapter, tested with a stand-in client (no SDK, no network).


class StubMessages:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def anthropic_spec():
    return ModelSpec(
        id="a", provider="anthropic", model="test-model", price_in_per_mtok=1, price_out_per_mtok=1
    )


def test_anthropic_adapter_maps_request_and_usage():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Hi "),
            SimpleNamespace(type="text", text="there"),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=2),
    )
    messages = StubMessages(response)
    provider = AnthropicProvider(anthropic_spec(), client=SimpleNamespace(messages=messages))
    request = ProviderRequest(
        messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}],
        params={"temperature": 0},
    )
    completion = asyncio.run(provider.complete(request))
    assert (completion.text, completion.input_tokens, completion.output_tokens) == (
        "Hi there",
        11,
        2,
    )
    assert messages.kwargs["system"] == "Be brief."
    assert messages.kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert messages.kwargs["max_tokens"] == 1024


class FakeStatusError(Exception):
    def __init__(self, status):
        super().__init__("rate limited")
        self.status_code = status
        self.response = SimpleNamespace(headers={"retry-after": "3"})


def test_anthropic_adapter_maps_errors():
    provider = AnthropicProvider(
        anthropic_spec(), client=SimpleNamespace(messages=StubMessages(FakeStatusError(429)))
    )
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.complete(ProviderRequest(messages=[{"role": "user", "content": "x"}])))
    assert (info.value.status, info.value.retryable, info.value.retry_after) == (429, True, 3.0)
