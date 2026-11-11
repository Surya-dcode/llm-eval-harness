"""Task file loading and validation.

Errors are collected into a ConfigError whose message lists every problem with
a path into the YAML (for example ``models[1].price_in_per_mtok``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evalh.scorers import SCORER_TYPES, LLMJudge, ScorerConfig
from evalh.templating import TemplateError, placeholders, render

SCORER_TAGS = frozenset(SCORER_TYPES)

ProviderName = Literal["openai_compatible", "anthropic", "fake"]


class ConfigError(Exception):
    """A task file problem, with one readable line per issue."""

    def __init__(self, source: str, problems: list[str]):
        self.source = source
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"{source}: {len(problems)} problem(s)\n{lines}")


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    provider: ProviderName
    model: str = Field(min_length=1)
    price_in_per_mtok: float = Field(ge=0)
    price_out_per_mtok: float = Field(ge=0)
    params: dict[str, Any] = Field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None
    rate_group: str | None = None
    timeout_s: float | None = Field(default=None, gt=0)

    @property
    def group(self) -> str:
        return self.rate_group or self.provider


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    vars: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None
    scorers: list[ScorerConfig] | None = None
    tags: list[str] = Field(default_factory=list)


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int = Field(default=4, ge=1)
    provider_concurrency: dict[str, int] = Field(default_factory=dict)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    backoff_base_s: float = Field(default=1.0, ge=0)
    backoff_max_s: float = Field(default=30.0, ge=0)

    def limit_for(self, group: str) -> int:
        return self.provider_concurrency.get(group, self.concurrency)


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    system_prompt: str | None = None
    prompt_template: str = Field(min_length=1)
    template_syntax: Literal["double_brace", "format"] = "double_brace"
    cases: list[Case] | None = None
    cases_file: str | None = None
    repeats: int = Field(default=1, ge=1)
    scorers: list[ScorerConfig] = Field(default_factory=list)
    models: list[ModelSpec] | None = None
    models_file: str | None = None
    judge_models: list[ModelSpec] = Field(default_factory=list)
    execution: Execution = Field(default_factory=Execution)

    @model_validator(mode="after")
    def _one_source_each(self) -> TaskConfig:
        if self.cases is not None and self.cases_file is not None:
            raise ValueError("give either 'cases' or 'cases_file', not both")
        if self.cases is None and self.cases_file is None:
            raise ValueError("no cases: add 'cases' or 'cases_file'")
        if self.models is not None and self.models_file is not None:
            raise ValueError("give either 'models' or 'models_file', not both")
        if self.models is None and self.models_file is None:
            raise ValueError("no models: add 'models' or 'models_file'")
        return self

    # Helpers used after load_task() has resolved files.
    def all_cases(self) -> list[Case]:
        return self.cases or []

    def candidate_models(self) -> list[ModelSpec]:
        return self.models or []

    def scorers_for(self, case: Case) -> list:
        return case.scorers if case.scorers is not None else self.scorers

    def find_model(self, model_id: str) -> ModelSpec | None:
        for spec in [*self.candidate_models(), *self.judge_models]:
            if spec.id == model_id:
                return spec
        return None

    def build_messages(self, case: Case) -> list[dict[str, str]]:
        user = render(self.prompt_template, case.vars, self.template_syntax)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user})
        return messages


# ------------------------------------------------------------------- loading


def format_validation_error(exc: ValidationError, prefix: str = "") -> list[str]:
    problems = []
    for err in exc.errors():
        where = prefix
        for item in err["loc"]:
            if isinstance(item, int):
                where += f"[{item}]"
            elif item in SCORER_TAGS:
                continue  # discriminated-union tag, not a real key
            else:
                where += ("." if where else "") + str(item)
        where = where or "<root>"
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = "unknown field (check the spelling)"
        elif err["type"] == "missing":
            msg = "required field is missing"
        elif err["type"] == "union_tag_invalid":
            ctx = err.get("ctx", {})
            tag, expected = ctx.get("tag"), ctx.get("expected_tags")
            msg = f"unknown scorer type {tag!r}; expected one of {expected}"
        elif err["type"] == "union_tag_not_found":
            msg = "scorer is missing its 'type' field"
        msg = re.sub(r"^Value error, ", "", msg)
        problems.append(f"{where}: {msg}")
    return problems


def _read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(str(path), ["file not found"]) from None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(str(path), [f"YAML syntax error at {where}{problem}"]) from None


def _read_jsonl(path: Path, source: str) -> tuple[list[Any], list[str]]:
    rows, problems = [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], [f"cases_file: file not found: {path}"]
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name} line {number}: invalid JSON ({exc.msg})")
    return rows, problems


def parse_task(data: Any, source: str = "<task>", base_dir: Path | None = None) -> TaskConfig:
    """Validate a task mapping, resolve case/model files and cross-check references."""
    base_dir = base_dir or Path.cwd()
    if not isinstance(data, dict):
        raise ConfigError(source, ["task file must be a YAML mapping (key: value pairs)"])
    try:
        task = TaskConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(source, format_validation_error(exc)) from None

    problems: list[str] = []

    if task.cases_file is not None:
        raw_cases, file_problems = _read_jsonl(base_dir / task.cases_file, source)
        problems.extend(file_problems)
        cases = []
        for index, raw in enumerate(raw_cases):
            try:
                cases.append(Case.model_validate(raw))
            except ValidationError as exc:
                problems.extend(format_validation_error(exc, prefix=f"{task.cases_file}[{index}]"))
        task.cases = cases

    if task.models_file is not None:
        models_data = _read_yaml(base_dir / task.models_file)
        if isinstance(models_data, dict) and "models" in models_data:
            models_data = models_data["models"]
        if not isinstance(models_data, list):
            problems.append(f"models_file: {task.models_file} must contain a list of models")
            task.models = []
        else:
            models = []
            for index, raw in enumerate(models_data):
                try:
                    models.append(ModelSpec.model_validate(raw))
                except ValidationError as exc:
                    problems.extend(
                        format_validation_error(exc, prefix=f"{task.models_file}[{index}]")
                    )
            task.models = models

    problems.extend(_cross_check(task))
    if problems:
        raise ConfigError(source, problems)
    return task


def _cross_check(task: TaskConfig) -> list[str]:
    problems: list[str] = []
    cases = task.all_cases()
    models = task.candidate_models()
    if not cases:
        problems.append("cases: no cases defined")
    if not models:
        problems.append("models: no models defined")

    from evalh.providers.fake import FakeParams

    seen: set[str] = set()
    for spec in [*models, *task.judge_models]:
        if spec.id in seen:
            problems.append(f"models: duplicate model id '{spec.id}'")
        seen.add(spec.id)
        if spec.provider == "fake":
            try:
                FakeParams.model_validate(spec.params)
            except ValidationError as exc:
                problems.extend(format_validation_error(exc, prefix=f"model '{spec.id}' params"))

    case_ids: set[str] = set()
    template_vars = placeholders(task.prompt_template, task.template_syntax)
    for index, case in enumerate(cases):
        where = f"cases[{index}] (id={case.id})"
        if case.id in case_ids:
            problems.append(f"{where}: duplicate case id")
        case_ids.add(case.id)
        try:
            render(task.prompt_template, case.vars, task.template_syntax)
        except TemplateError as exc:
            problems.append(f"{where}: prompt_template uses {template_vars}, {exc}")
        scorers = task.scorers_for(case)
        if not scorers:
            problems.append(f"{where}: no scorers (set task 'scorers' or case 'scorers')")
        for scorer in scorers:
            message = scorer.check_case(case.expected)
            if message:
                problems.append(f"{where}: {message}")
            if isinstance(scorer, LLMJudge) and task.find_model(scorer.judge) is None:
                problems.append(
                    f"{where}: llm_judge refers to judge '{scorer.judge}', which is not in "
                    "'models' or 'judge_models'"
                )
    return problems


def load_task(path: str | Path) -> TaskConfig:
    path = Path(path)
    data = _read_yaml(path)
    return parse_task(data, source=str(path), base_dir=path.parent)


def missing_api_keys(models: list[ModelSpec]) -> list[str]:
    """Return readable messages for models whose API key env var is not set."""
    messages = []
    for spec in models:
        env_name = spec.api_key_env
        if spec.provider == "anthropic" and env_name is None:
            env_name = "ANTHROPIC_API_KEY"
        if env_name and not os.environ.get(env_name):
            messages.append(f"model '{spec.id}': environment variable {env_name} is not set")
    return messages
