"""Aggregate result rows into summary.csv and report.md."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from evalh.results import ResultRow
from evalh.stats import mean, percentile, stdev


@dataclass
class ModelSummary:
    model_id: str
    provider: str
    model: str
    calls: int
    failures: int
    mean_score: float
    score_stdev: float | None
    pass_rate: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    mean_latency_ms: float | None
    input_tokens: int
    output_tokens: int
    approx_token_calls: int
    total_cost_usd: float
    cost_per_call_usd: float
    judge_cost_usd: float
    cached_calls: int


def _ordered_unique(values: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def summarize(rows: list[ResultRow]) -> list[ModelSummary]:
    """One summary per model, in the order models first appear in ``rows``."""
    by_model: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        by_model[row.model_id].append(row)
    summaries = []
    for model_id in _ordered_unique([r.model_id for r in rows]):
        group = by_model[model_id]
        latencies = [r.latency_ms for r in group if r.latency_ms is not None and r.error is None]
        scores = [r.score for r in group]
        total_cost = sum(r.cost_usd for r in group)
        summaries.append(
            ModelSummary(
                model_id=model_id,
                provider=group[0].provider,
                model=group[0].model,
                calls=len(group),
                failures=sum(1 for r in group if r.error is not None),
                mean_score=mean(scores) or 0.0,
                score_stdev=stdev(scores),
                pass_rate=sum(1 for r in group if r.passed) / len(group),
                p50_latency_ms=percentile(latencies, 50),
                p95_latency_ms=percentile(latencies, 95),
                mean_latency_ms=mean(latencies),
                input_tokens=sum(r.input_tokens for r in group),
                output_tokens=sum(r.output_tokens for r in group),
                approx_token_calls=sum(1 for r in group if r.tokens_approximate),
                total_cost_usd=total_cost,
                cost_per_call_usd=total_cost / len(group),
                judge_cost_usd=sum(r.judge_cost_usd for r in group),
                cached_calls=sum(1 for r in group if r.cached),
            )
        )
    return summaries


def write_summary_csv(summaries: list[ModelSummary], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ModelSummary.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            record = asdict(summary)
            for key, value in record.items():
                if isinstance(value, float):
                    record[key] = f"{value:.6g}" if "cost" not in key else f"{value:.8f}"
                elif value is None:
                    record[key] = ""
            writer.writerow(record)


# ----------------------------------------------------------------- report.md


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.0f} ms"


def fmt_usd(value: float) -> str:
    if value == 0:
        return "$0"
    if value >= 1:
        return f"${value:.2f}"
    if value >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in body]
    return lines


@dataclass
class CaseCell:
    runs: int
    passes: int
    mean_score: float

    def text(self) -> str:
        return f"{self.mean_score:.2f} ({self.passes}/{self.runs})"


def case_grid(
    rows: list[ResultRow],
) -> tuple[list[str], list[str], dict[tuple[str, str], CaseCell]]:
    models = _ordered_unique([r.model_id for r in rows])
    cases = _ordered_unique([r.case_id for r in rows])
    grouped: dict[tuple[str, str], list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.case_id, row.model_id)].append(row)
    grid = {
        key: CaseCell(
            runs=len(group),
            passes=sum(1 for r in group if r.passed),
            mean_score=sum(r.score for r in group) / len(group),
        )
        for key, group in grouped.items()
    }
    return models, cases, grid


def disagreement(case_id: str, models: list[str], grid: dict) -> str | None:
    """Why a case is interesting, or None if every model behaves the same."""
    cells = [grid[(case_id, m)] for m in models if (case_id, m) in grid]
    if not cells:
        return None
    reasons = []
    pass_rates = {round(c.passes / c.runs, 6) for c in cells}
    if len(models) > 1 and len(pass_rates) > 1:
        reasons.append("models differ")
    if any(0 < c.passes < c.runs for c in cells):
        reasons.append("varies across repeats")
    return ", ".join(reasons) or None


def render_report(rows: list[ResultRow]) -> str:
    summaries = summarize(rows)
    models, cases, grid = case_grid(rows)
    task = rows[0].task
    run_ids = _ordered_unique([r.run_id for r in rows])
    repeats = max(r.repeat for r in rows) + 1
    uses_fake = any(r.provider == "fake" for r in rows)

    lines = [f"# Eval report: {task}", ""]
    if uses_fake:
        lines += [
            "> This run uses the `fake` provider. Its answers, latencies and costs are",
            "> simulated, so the numbers below only show what a report looks like.",
            "",
        ]
    lines += [
        f"- Run: {', '.join(run_ids)}",
        f"- Cases: {len(cases)}, repeats per case: {repeats}, models: {len(models)}",
        f"- Calls: {len(rows)} ({sum(s.cached_calls for s in summaries)} served from cache)",
        "",
        "## Models",
        "",
        "Sorted by mean score. Pass rate counts calls where every scorer passed. "
        "Latency is wall time per successful call. Cost uses the prices in the task config.",
        "",
    ]
    ranked = sorted(summaries, key=lambda s: (-s.mean_score, s.total_cost_usd))
    body = []
    any_approx = False
    for s in ranked:
        approx = s.approx_token_calls > 0
        any_approx = any_approx or approx
        marker = " *" if approx else ""
        body.append(
            [
                f"`{s.model_id}`",
                f"{s.provider}/{s.model}",
                str(s.calls),
                f"{s.mean_score:.3f}",
                f"{s.pass_rate:.0%}",
                fmt_ms(s.p50_latency_ms),
                fmt_ms(s.p95_latency_ms),
                fmt_usd(s.total_cost_usd) + marker,
                fmt_usd(s.cost_per_call_usd) + marker,
                str(s.failures),
            ]
        )
    lines += _table(
        [
            "Model",
            "Provider/model",
            "Calls",
            "Mean score",
            "Pass rate",
            "p50 latency",
            "p95 latency",
            "Total cost",
            "Cost / case",
            "Failures",
        ],
        body,
    )
    if any_approx:
        lines += ["", "\\* Some token counts were approximated (provider returned no usage)."]
    judge_total = sum(s.judge_cost_usd for s in summaries)
    if judge_total:
        lines += ["", f"Judge model calls cost {fmt_usd(judge_total)} on top of the totals above."]

    # Per-scorer means.
    scorer_names = _ordered_unique([sc.name for r in rows for sc in r.scores])
    if scorer_names:
        lines += ["", "## Scorers", "", "Mean score per scorer (failed calls count as 0).", ""]
        body = []
        for name in scorer_names:
            cells = []
            for model_id in models:
                values = [
                    next((sc.score for sc in r.scores if sc.name == name), 0.0)
                    for r in rows
                    if r.model_id == model_id
                ]
                cells.append(f"{mean(values) or 0.0:.3f}")
            body.append([name, *cells])
        lines += _table(["Scorer", *[f"`{m}`" for m in models]], body)

    # Per-case breakdown.
    flagged = [(c, disagreement(c, models, grid)) for c in cases]
    interesting = [(c, why) for c, why in flagged if why]
    lines += [
        "",
        "## Where models disagree",
        "",
        "Each cell is the mean score with passes/runs in brackets.",
        "",
    ]
    if interesting:
        body = [
            [c, *[grid[(c, m)].text() if (c, m) in grid else "n/a" for m in models], why]
            for c, why in interesting
        ]
        lines += _table(["Case", *[f"`{m}`" for m in models], "Why"], body)
    else:
        lines.append("Every model got the same result on every case.")

    lines += ["", "## All cases", ""]
    body = [[c, *[grid[(c, m)].text() if (c, m) in grid else "n/a" for m in models]] for c in cases]
    lines += _table(["Case", *[f"`{m}`" for m in models]], body)

    failures = [r for r in rows if r.error is not None]
    if failures:
        lines += ["", "## Failed calls", ""]
        body = [
            [f"`{r.model_id}`", r.case_id, str(r.repeat), str(r.attempts), (r.error or "")[:160]]
            for r in failures
        ]
        lines += _table(["Model", "Case", "Repeat", "Attempts", "Error"], body)

    misses = [r for r in rows if r.error is None and not r.passed]
    if misses:
        lines += [
            "",
            "## Sample of failed scores",
            "",
            "Up to 10 scored outputs that did not pass.",
            "",
        ]
        body = []
        for r in misses[:10]:
            failed = [sc for sc in r.scores if not sc.passed]
            notes = "; ".join(_score_note(sc.name, sc.details) for sc in failed)
            body.append([f"`{r.model_id}`", r.case_id, str(r.repeat), notes[:200]])
        lines += _table(["Model", "Case", "Repeat", "What failed"], body)

    lines.append("")
    return "\n".join(lines)


def _score_note(name: str, details: dict) -> str:
    if details.get("mismatches"):
        parts = [
            f"{field}: expected {info['expected']!r}, got {info['got']!r}"
            for field, info in details["mismatches"].items()
        ]
        return f"{name} ({', '.join(parts)})"
    if "error" in details:
        message = details.get("message") or details["error"]
        return f"{name} ({message})"
    if details.get("missing"):
        return f"{name} (missing {details['missing']})"
    return name


def write_outputs(rows: list[ResultRow], out_dir: str | Path) -> dict[str, Path]:
    """Write results.jsonl, summary.csv and report.md into ``out_dir``."""
    from evalh.results import write_results

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": out_dir / "results.jsonl",
        "summary": out_dir / "summary.csv",
        "report": out_dir / "report.md",
    }
    write_results(rows, paths["results"])
    write_report_files(rows, out_dir)
    return paths


def write_report_files(rows: list[ResultRow], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summarize(rows), out_dir / "summary.csv")
    (out_dir / "report.md").write_text(render_report(rows), encoding="utf-8")
