"""Runs every case against every model, scores the outputs and returns rows."""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from evalh.cache import DiskCache, cache_key
from evalh.config import Case, ModelSpec, TaskConfig
from evalh.providers import Provider, ProviderError, ProviderRequest, make_provider
from evalh.results import ResultRow
from evalh.retry import with_retries
from evalh.scorers import JudgeReply, LLMJudge, ScoreContext, ScoreResult, combine
from evalh.tokens import approx_message_tokens, approx_tokens, call_cost

ProviderFactory = Callable[[ModelSpec, float], Provider]


@dataclass
class CallOutcome:
    text: str | None
    error: str | None
    latency_ms: float | None
    attempts: int
    cached: bool
    input_tokens: int
    output_tokens: int
    tokens_approximate: bool
    cost_usd: float


class Runner:
    def __init__(
        self,
        task: TaskConfig,
        cache: DiskCache | None = None,
        provider_factory: ProviderFactory = make_provider,
        sleep: Callable[[float], Any] = asyncio.sleep,
        progress: bool = False,
    ):
        self.task = task
        self.cache = cache or DiskCache(enabled=False)
        self.provider_factory = provider_factory
        self.sleep = sleep
        self.progress = progress
        self._providers: dict[str, Provider] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._done = 0
        self._total = 0

    # ----------------------------------------------------------- plumbing

    def _provider(self, spec: ModelSpec) -> Provider:
        if spec.id not in self._providers:
            timeout = spec.timeout_s or self.task.execution.timeout_s
            self._providers[spec.id] = self.provider_factory(spec, timeout)
        return self._providers[spec.id]

    def _judge_specs(self) -> list[ModelSpec]:
        specs = [self.task.find_model(i) for i in sorted(_judge_ids(self.task))]
        return [s for s in specs if s is not None]

    def _semaphore(self, spec: ModelSpec) -> asyncio.Semaphore:
        group = spec.group
        if group not in self._semaphores:
            self._semaphores[group] = asyncio.Semaphore(self.task.execution.limit_for(group))
        return self._semaphores[group]

    async def call_model(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        *,
        case_id: str,
        expected: Any,
        repeat: int,
        purpose: str = "candidate",
    ) -> CallOutcome:
        key = cache_key(spec.provider, spec.model, spec.params, messages, repeat, spec.base_url)
        hit = self.cache.get(key)
        if hit is not None:
            return self._outcome(spec, hit, attempts=0, cached=True)

        provider = self._provider(spec)
        execution = self.task.execution
        request = ProviderRequest(
            messages=messages,
            params=spec.params,
            case_id=case_id,
            expected=expected,
            repeat=repeat,
            purpose=purpose,
        )

        async def attempt() -> tuple[Any, float]:
            start = time.perf_counter()
            completion = await provider.complete(request)
            return completion, (time.perf_counter() - start) * 1000

        async with self._semaphore(spec):
            try:
                (completion, latency_ms), attempts = await with_retries(
                    attempt,
                    max_retries=execution.max_retries,
                    timeout_s=spec.timeout_s or execution.timeout_s,
                    base_s=execution.backoff_base_s,
                    max_s=execution.backoff_max_s,
                    sleep=self.sleep,
                )
            except ProviderError as exc:
                return CallOutcome(
                    text=None,
                    error=str(exc),
                    latency_ms=None,
                    attempts=getattr(exc, "attempts", 1),
                    cached=False,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_approximate=False,
                    cost_usd=0.0,
                )

        approximate = completion.input_tokens is None or completion.output_tokens is None
        record = {
            "text": completion.text,
            "input_tokens": completion.input_tokens
            if completion.input_tokens is not None
            else approx_message_tokens(messages),
            "output_tokens": completion.output_tokens
            if completion.output_tokens is not None
            else approx_tokens(completion.text),
            "tokens_approximate": approximate,
            "latency_ms": latency_ms,
        }
        self.cache.set(key, record)
        return self._outcome(spec, record, attempts=attempts, cached=False)

    @staticmethod
    def _outcome(
        spec: ModelSpec, record: dict[str, Any], attempts: int, cached: bool
    ) -> CallOutcome:
        input_tokens = int(record["input_tokens"])
        output_tokens = int(record["output_tokens"])
        return CallOutcome(
            text=record["text"],
            error=None,
            latency_ms=float(record["latency_ms"]),
            attempts=attempts,
            cached=cached,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_approximate=bool(record.get("tokens_approximate", False)),
            cost_usd=call_cost(
                input_tokens, output_tokens, spec.price_in_per_mtok, spec.price_out_per_mtok
            ),
        )

    # ---------------------------------------------------------- one call

    async def _run_one(self, run_id: str, spec: ModelSpec, case: Case, repeat: int) -> ResultRow:
        messages = self.task.build_messages(case)
        outcome = await self.call_model(
            spec, messages, case_id=case.id, expected=case.expected, repeat=repeat
        )
        row = ResultRow(
            run_id=run_id,
            task=self.task.name,
            model_id=spec.id,
            provider=spec.provider,
            model=spec.model,
            case_id=case.id,
            repeat=repeat,
            output=outcome.text,
            error=outcome.error,
            latency_ms=outcome.latency_ms,
            attempts=outcome.attempts,
            cached=outcome.cached,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            tokens_approximate=outcome.tokens_approximate,
            price_in_per_mtok=spec.price_in_per_mtok,
            price_out_per_mtok=spec.price_out_per_mtok,
            cost_usd=outcome.cost_usd,
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        if outcome.text is not None:
            await self._score(row, case, messages[-1]["content"])
        self._tick()
        return row

    async def _score(self, row: ResultRow, case: Case, prompt: str) -> None:
        judge_cost = 0.0

        async def judge(model_id: str, messages: list[dict[str, str]]) -> JudgeReply:
            nonlocal judge_cost
            spec = self.task.find_model(model_id)
            if spec is None:
                return JudgeReply(text=None, error=f"unknown judge model '{model_id}'")
            outcome = await self.call_model(
                spec,
                messages,
                case_id=case.id,
                expected=None,
                repeat=row.repeat,
                purpose="judge",
            )
            judge_cost += outcome.cost_usd
            return JudgeReply(text=outcome.text, error=outcome.error, cost_usd=outcome.cost_usd)

        ctx = ScoreContext(case_id=case.id, expected=case.expected, prompt=prompt, judge=judge)
        scorers = self.task.scorers_for(case)
        results: list[ScoreResult] = []
        used: dict[str, int] = {}
        for scorer in scorers:
            try:
                result = await scorer.score(row.output or "", ctx)
            except Exception as exc:  # a scorer bug should not kill the whole run
                result = ScoreResult(
                    name=scorer.display_name,
                    score=0.0,
                    passed=False,
                    details={"error": f"scorer crashed: {type(exc).__name__}: {exc}"},
                )
            used[result.name] = used.get(result.name, 0) + 1
            if used[result.name] > 1:
                result.name = f"{result.name}_{used[result.name]}"
            results.append(result)
        row.scores = results
        row.score, row.passed = combine(results, [s.weight for s in scorers])
        row.judge_cost_usd = judge_cost

    def _tick(self) -> None:
        self._done += 1
        if self.progress:
            end = "\n" if self._done == self._total else ""
            print(f"\r  {self._done}/{self._total} calls", end=end, file=sys.stderr, flush=True)

    # -------------------------------------------------------------- run

    async def run(
        self,
        model_ids: list[str] | None = None,
        repeats: int | None = None,
        run_id: str | None = None,
    ) -> list[ResultRow]:
        models = select_models(self.task, model_ids)
        repeats = repeats or self.task.repeats
        cases = self.task.all_cases()
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        jobs = [(spec, case, r) for spec in models for case in cases for r in range(repeats)]
        self._total = len(jobs)
        self._done = 0
        # Create every adapter up front so a missing key or SDK fails before
        # any paid call is made.
        try:
            for spec in [*models, *self._judge_specs()]:
                self._provider(spec)
            rows = await asyncio.gather(*(self._run_one(run_id, *job) for job in jobs))
        finally:
            for provider in self._providers.values():
                await provider.aclose()
            self._providers.clear()
        return list(rows)


def _judge_ids(task: TaskConfig) -> set[str]:
    ids = set()
    for case in task.all_cases():
        for scorer in task.scorers_for(case):
            if isinstance(scorer, LLMJudge):
                ids.add(scorer.judge)
    return ids


def select_models(task: TaskConfig, model_ids: list[str] | None) -> list[ModelSpec]:
    models = task.candidate_models()
    if not model_ids:
        return models
    known = {m.id: m for m in models}
    unknown = [m for m in model_ids if m not in known]
    if unknown:
        raise ValueError(f"unknown model id(s) {unknown}; available: {sorted(known)}")
    return [known[m] for m in model_ids]
