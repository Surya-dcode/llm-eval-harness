"""Result rows: one per model call, written to results.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from evalh.scorers import ScoreResult


class ResultRow(BaseModel):
    run_id: str
    task: str
    model_id: str
    provider: str
    model: str
    case_id: str
    repeat: int
    output: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    attempts: int = 1
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_approximate: bool = False
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    score: float = 0.0
    passed: bool = False
    scores: list[ScoreResult] = Field(default_factory=list)
    timestamp: str = ""


def write_results(rows: list[ResultRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


class ResultsFileError(Exception):
    pass


def read_results(path: str | Path) -> list[ResultRow]:
    path = Path(path)
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise ResultsFileError(f"{path}: file not found") from None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(ResultRow.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            reason = exc.msg if isinstance(exc, json.JSONDecodeError) else exc.errors()[0]["msg"]
            raise ResultsFileError(
                f"{path} line {number}: not a valid result row ({reason})"
            ) from None
    if not rows:
        raise ResultsFileError(f"{path}: no result rows")
    return rows
