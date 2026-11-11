import asyncio

import pytest

from evalh.cache import DiskCache, cache_key
from evalh.providers.fake import FakeProvider
from evalh.runner import Runner


def test_cache_key_changes_with_inputs():
    messages = [{"role": "user", "content": "hi"}]
    base = cache_key("fake", "m", {"temperature": 0}, messages)
    assert base == cache_key("fake", "m", {"temperature": 0}, [dict(messages[0])])
    assert base != cache_key("fake", "m", {"temperature": 1}, messages)
    assert base != cache_key("fake", "other", {"temperature": 0}, messages)
    assert base != cache_key("openai_compatible", "m", {"temperature": 0}, messages)
    assert base != cache_key("fake", "m", {"temperature": 0}, [{"role": "user", "content": "hi!"}])
    assert base != cache_key("fake", "m", {"temperature": 0}, messages, repeat=1)
    assert base != cache_key("fake", "m", {"temperature": 0}, messages, base_url="http://x")


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("ab" * 32) is None
    cache.set("ab" * 32, {"text": "hello"})
    assert cache.get("ab" * 32) == {"text": "hello"}
    assert (cache.hits, cache.misses) == (1, 1)


def test_disabled_cache_never_stores(tmp_path):
    cache = DiskCache(tmp_path, enabled=False)
    cache.set("cd" * 32, {"text": "x"})
    assert cache.get("cd" * 32) is None
    assert not any(tmp_path.iterdir())


class CountingFactory:
    def __init__(self):
        self.providers = []

    def __call__(self, spec, timeout_s):
        provider = FakeProvider(spec, timeout_s)
        self.providers.append(provider)
        return provider

    @property
    def calls(self):
        return sum(p.calls for p in self.providers)


def test_second_run_is_served_from_cache(make_task, tmp_path):
    task = make_task(repeats=2)
    first_factory = CountingFactory()
    first = asyncio.run(Runner(task, DiskCache(tmp_path), first_factory).run())
    assert first_factory.calls == 4
    assert not any(r.cached for r in first)

    second_factory = CountingFactory()
    cache = DiskCache(tmp_path)
    second = asyncio.run(Runner(task, cache, second_factory).run())
    assert second_factory.calls == 0
    assert (cache.hits, cache.misses) == (4, 0)
    assert all(r.cached for r in second)
    assert [r.output for r in second] == [r.output for r in first]
    assert [r.cost_usd for r in second] == [r.cost_usd for r in first]


def test_no_cache_always_calls(make_task, tmp_path):
    task = make_task()
    asyncio.run(Runner(task, DiskCache(tmp_path), CountingFactory()).run())
    factory = CountingFactory()
    rows = asyncio.run(Runner(task, DiskCache(tmp_path, enabled=False), factory).run())
    assert factory.calls == 2
    assert not any(r.cached for r in rows)


def test_errors_are_not_cached(make_task, tmp_path):
    task = make_task()
    task.models[0].params = {"error_cases": ["c1"]}
    asyncio.run(Runner(task, DiskCache(tmp_path), CountingFactory()).run())
    factory = CountingFactory()
    cache = DiskCache(tmp_path)
    asyncio.run(Runner(task, cache, factory).run())
    assert factory.calls == 1  # only the failed case is retried on the next run
    assert cache.hits == 1


def test_cached_rows_use_current_prices(make_task, tmp_path):
    task = make_task()
    first = asyncio.run(Runner(task, DiskCache(tmp_path), CountingFactory()).run())
    task.models[0].price_in_per_mtok *= 2
    task.models[0].price_out_per_mtok *= 2
    factory = CountingFactory()
    second = asyncio.run(Runner(task, DiskCache(tmp_path), factory).run())
    assert factory.calls == 0
    for a, b in zip(first, second, strict=True):
        assert b.cost_usd == pytest.approx(a.cost_usd * 2)
