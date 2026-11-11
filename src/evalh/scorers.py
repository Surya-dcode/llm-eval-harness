"""Scorers turn one model output into a 0 to 1 score plus details.

Each scorer is a pydantic model, so its options are validated when the task
file is loaded. The ``type`` field selects the scorer.
"""

from __future__ import annotations

import math
import re
import string
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from evalh.jsonutil import JSONExtractError, get_path, parse_json


class ScoreResult(BaseModel):
    name: str
    score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class JudgeReply:
    """What the runner hands back to the llm_judge scorer for one judge call."""

    text: str | None
    error: str | None = None
    cost_usd: float = 0.0


JudgeFn = Callable[[str, list[dict[str, str]]], Awaitable[JudgeReply]]


@dataclass
class ScoreContext:
    case_id: str
    expected: Any
    prompt: str
    judge: JudgeFn | None = None


class _ScorerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    weight: float = Field(default=1.0, gt=0)

    @property
    def display_name(self) -> str:
        return self.name or self.type  # type: ignore[attr-defined]

    def check_case(self, expected: Any) -> str | None:
        """Return an error message if this scorer cannot score a case."""
        return None

    async def score(self, output: str, ctx: ScoreContext) -> ScoreResult:  # pragma: no cover
        raise NotImplementedError

    def _result(self, score: float, passed: bool, **details: Any) -> ScoreResult:
        return ScoreResult(name=self.display_name, score=score, passed=passed, details=details)


# ---------------------------------------------------------------- exact_match


def normalize_text(
    text: str,
    *,
    case_sensitive: bool = False,
    strip_whitespace: bool = True,
    collapse_whitespace: bool = True,
    strip_punctuation: bool = False,
) -> str:
    if strip_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    if strip_whitespace:
        text = text.strip()
    if not case_sensitive:
        text = text.casefold()
    return text


class ExactMatch(_ScorerBase):
    """Output equals the expected string (or any string in a list) after normalization."""

    type: Literal["exact_match"]
    expected: str | list[str] | None = None
    case_sensitive: bool = False
    strip_whitespace: bool = True
    collapse_whitespace: bool = True
    strip_punctuation: bool = False

    def _targets(self, expected: Any) -> list[str] | None:
        value = self.expected if self.expected is not None else expected
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return value
        return None

    def check_case(self, expected: Any) -> str | None:
        if self._targets(expected) is None:
            return (
                "exact_match needs a string (or list of strings) in the case 'expected' "
                "field or the scorer 'expected' option"
            )
        return None

    def _norm(self, text: str) -> str:
        return normalize_text(
            text,
            case_sensitive=self.case_sensitive,
            strip_whitespace=self.strip_whitespace,
            collapse_whitespace=self.collapse_whitespace,
            strip_punctuation=self.strip_punctuation,
        )

    async def score(self, output: str, ctx: ScoreContext) -> ScoreResult:
        targets = self._targets(ctx.expected) or []
        got = self._norm(output)
        for target in targets:
            if got == self._norm(target):
                return self._result(1.0, True, matched=target)
        return self._result(0.0, False, got=got[:200], expected=targets)


# ---------------------------------------------------------------------- regex


class Regex(_ScorerBase):
    """Output matches a regular expression (search by default)."""

    type: Literal["regex"]
    pattern: str
    mode: Literal["search", "fullmatch"] = "search"
    ignore_case: bool = False
    multiline: bool = False
    dotall: bool = False
    negate: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from None
        return value

    def _compiled(self) -> re.Pattern[str]:
        flags = 0
        if self.ignore_case:
            flags |= re.IGNORECASE
        if self.multiline:
            flags |= re.MULTILINE
        if self.dotall:
            flags |= re.DOTALL
        return re.compile(self.pattern, flags)

    async def score(self, output: str, ctx: ScoreContext) -> ScoreResult:
        compiled = self._compiled()
        text = output.strip() if self.mode == "fullmatch" else output
        match = compiled.fullmatch(text) if self.mode == "fullmatch" else compiled.search(text)
        hit = match is not None
        passed = hit != self.negate
        details: dict[str, Any] = {"matched": hit}
        if match is not None:
            details["match"] = match.group(0)[:200]
        return self._result(1.0 if passed else 0.0, passed, **details)


# --------------------------------------------------------------- contains_all


class ContainsAll(_ScorerBase):
    """Output contains every keyword. Score is the fraction found."""

    type: Literal["contains_all"]
    keywords: list[str] | None = None
    case_sensitive: bool = False

    @field_validator("keywords")
    @classmethod
    def _non_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("keywords must not be empty")
        return value

    def _keywords(self, expected: Any) -> list[str] | None:
        if self.keywords:
            return self.keywords
        if isinstance(expected, list) and expected and all(isinstance(v, str) for v in expected):
            return expected
        return None

    def check_case(self, expected: Any) -> str | None:
        if self._keywords(expected) is None:
            return (
                "contains_all needs 'keywords' on the scorer or a list of strings in the "
                "case 'expected' field"
            )
        return None

    async def score(self, output: str, ctx: ScoreContext) -> ScoreResult:
        keywords = self._keywords(ctx.expected) or []
        haystack = output if self.case_sensitive else output.casefold()
        found, missing = [], []
        for keyword in keywords:
            needle = keyword if self.case_sensitive else keyword.casefold()
            (found if needle in haystack else missing).append(keyword)
        fraction = len(found) / len(keywords) if keywords else 0.0
        return self._result(fraction, not missing, found=found, missing=missing)


# ----------------------------------------------------------------- json_valid


class JsonValid(_ScorerBase):
    """Output parses as JSON and, optionally, validates against a JSON Schema."""

    type: Literal["json_valid"]
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    extract: bool = True

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("json_schema")
    @classmethod
    def _schema_ok(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            try:
                jsonschema.Draft202012Validator.check_schema(value)
            except jsonschema.SchemaError as exc:
                raise ValueError(f"invalid JSON Schema: {exc.message}") from None
        return value

    async def score(self, output: str, ctx: ScoreContext) -> ScoreResult:
        try:
            data = parse_json(output, extract=self.extract)
        except JSONExtractError as exc:
            return self._result(0.0, False, error=str(exc))
        if self.json_schema is None:
            return self._result(1.0, True)
        validator = jsonschema.Draft202012Validator(self.json_schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if errors:
            messages = [
                f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                for e in errors[:5]
            ]
            return self._result(
                0.0, False, error="schema validation failed", schema_errors=messages
            )
        return self._result(1.0, True)


# ---------------------------------------------------------------- field_match


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[,\s$€£¥]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


class FieldMatch(_ScorerBase):
    """Parse the output as JSON and compare chosen fields to the expected object.

    Score is the fraction of fields that match.
    """

    type: Literal["field_match"]
    fields: list[str] | None = None
    case_sensitive: bool = False
    numeric_tolerance: float = Field(default=0.0, ge=0)
    coerce_numbers: bool = True
    extract: bool = True
    pass_threshold: float = Field(default=1.0, ge=0, le=1)

    def check_case(self, expected: Any) -> str | None:
        if not isinstance(expected, dict):
            return "field_match needs a JSON object in the case 'expected' field"
        missing = [f for f in (self.fields or []) if not get_path(expected, f)[0]]
        if missing:
            return f"field_match: expected object has no value for field(s) {missing}"
        if not self.fields and not expected:
            return "field_match: expected object is empty and no 'fields' were given"
        return None

    def _equal(self, got: Any, want: Any) -> bool:
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            got_num = (
                _as_number(got)
                if self.coerce_numbers
                else (
                    float(got)
                    if isinstance(got, (int, float)) and not isinstance(got, bool)
                    else None
                )
            )
            if got_num is None:
                return False
            return math.isclose(
                got_num, float(want), rel_tol=0.0, abs_tol=max(self.numeric_tolerance, 1e-9)
            )
        if isinstance(want, str):
            if not isinstance(got, str):
                got = "" if got is None else str(got)
            return normalize_text(got, case_sensitive=self.case_sensitive) == normalize_text(
                want, case_sensitive=self.case_sensitive
            )
        return got == want

    async def score(self, output: str, ctx: ScoreContext) -> ScoreResult:
        expected = ctx.expected if isinstance(ctx.expected, dict) else {}
        fields = self.fields or list(expected.keys())
        try:
            data = parse_json(output, extract=self.extract)
        except JSONExtractError as exc:
            return self._result(0.0, False, error=str(exc), fields={f: False for f in fields})
        per_field: dict[str, bool] = {}
        mismatches: dict[str, dict[str, Any]] = {}
        for field in fields:
            _, want = get_path(expected, field)
            found, got = get_path(data, field)
            ok = found and self._equal(got, want)
            per_field[field] = ok
            if not ok:
                mismatches[field] = {"expected": want, "got": got if found else "<missing>"}
        fraction = sum(per_field.values()) / len(fields) if fields else 0.0
        return self._result(
            fraction, fraction >= self.pass_threshold, fields=per_field, mismatches=mismatches
        )


# ------------------------------------------------------------------ llm_judge

DEFAULT_JUDGE_SYSTEM = (
    "You are a strict, consistent grader. You compare a model response against a rubric "
    "and reply with JSON only."
)

DEFAULT_JUDGE_TEMPLATE = """Grade the RESPONSE using the RUBRIC.

RUBRIC:
{rubric}

TASK INPUT:
{prompt}

REFERENCE ANSWER (may be empty):
{expected}

RESPONSE:
{output}

Reply with a single JSON object and nothing else:
{{"score": <integer from 1 to 5>, "reason": "<one sentence>"}}"""


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    score: int = Field(ge=1, le=5)
    reason: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _integral(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("score must be an integer from 1 to 5")
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("score must be an integer from 1 to 5")
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return value


def parse_judge_reply(text: str) -> JudgeVerdict:
    """Parse a judge reply. Raises ValueError with a readable reason on failure."""
    try:
        data = parse_json(text, extract=True)
    except JSONExtractError as exc:
        raise ValueError(f"judge reply is not JSON ({exc})") from None
    if not isinstance(data, dict):
        raise ValueError("judge reply JSON is not an object")
    try:
        return JudgeVerdict.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<root>"
        raise ValueError(f"judge reply invalid at '{loc}': {first['msg']}") from None


class LLMJudge(_ScorerBase):
    """Send a rubric prompt to a judge model; the 1 to 5 score maps to 0..1."""

    type: Literal["llm_judge"]
    judge: str
    rubric: str = Field(min_length=1)
    pass_threshold: int = Field(default=4, ge=1, le=5)
    include_expected: bool = True
    prompt_template: str | None = None
    system_prompt: str | None = None

    async def score(self, output: str, ctx: ScoreContext) -> ScoreResult:
        if ctx.judge is None:
            return self._result(0.0, False, error="judge_unavailable")
        expected = ctx.expected if self.include_expected and ctx.expected is not None else ""
        template = self.prompt_template or DEFAULT_JUDGE_TEMPLATE
        user = template.format(
            rubric=self.rubric, prompt=ctx.prompt, expected=expected, output=output
        )
        messages = [
            {"role": "system", "content": self.system_prompt or DEFAULT_JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ]
        reply = await ctx.judge(self.judge, messages)
        if reply.error is not None or reply.text is None:
            return self._result(
                0.0,
                False,
                error="judge_call_failed",
                message=reply.error,
                judge_cost_usd=reply.cost_usd,
            )
        try:
            verdict = parse_judge_reply(reply.text)
        except ValueError as exc:
            return self._result(
                0.0,
                False,
                error="judge_parse_error",
                message=str(exc),
                raw=reply.text[:500],
                judge_cost_usd=reply.cost_usd,
            )
        return self._result(
            (verdict.score - 1) / 4,
            verdict.score >= self.pass_threshold,
            judge_score=verdict.score,
            reason=verdict.reason,
            judge_cost_usd=reply.cost_usd,
        )


ScorerConfig = Annotated[
    ExactMatch | Regex | ContainsAll | JsonValid | FieldMatch | LLMJudge,
    Field(discriminator="type"),
]

SCORER_TYPES = ["exact_match", "regex", "contains_all", "json_valid", "field_match", "llm_judge"]


def combine(results: list[ScoreResult], weights: list[float]) -> tuple[float, bool]:
    """Weighted mean score and an all-passed flag."""
    if not results:
        return 0.0, False
    total_weight = sum(weights)
    score = sum(r.score * w for r, w in zip(results, weights, strict=True)) / total_weight
    return score, all(r.passed for r in results)
