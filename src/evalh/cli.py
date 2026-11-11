"""Command line interface: ``evalh run``, ``evalh report``, ``evalh validate``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from evalh import __version__
from evalh.cache import DiskCache
from evalh.config import ConfigError, load_task, missing_api_keys
from evalh.providers import ProviderError
from evalh.report import summarize, write_outputs, write_report_files
from evalh.results import ResultsFileError, read_results
from evalh.runner import Runner, select_models

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2


def _split_models(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _print_summary(rows: list) -> None:
    for s in sorted(summarize(rows), key=lambda s: -s.mean_score):
        print(
            f"  {s.model_id:<24} score {s.mean_score:.3f}  pass {s.pass_rate:>4.0%}  "
            f"failures {s.failures}  cost ${s.total_cost_usd:.6f}"
        )


def cmd_run(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    model_ids = _split_models(args.models)
    try:
        models = select_models(task, model_ids)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    missing = missing_api_keys(models)
    if missing:
        print("error: cannot start the run:", file=sys.stderr)
        for message in missing:
            print(f"  - {message}", file=sys.stderr)
        return EXIT_CONFIG

    repeats = args.repeats or task.repeats
    out_dir = Path(args.out) if args.out else Path("runs") / task.name
    cache = DiskCache(args.cache_dir, enabled=not args.no_cache)
    calls = len(models) * len(task.all_cases()) * repeats
    print(
        f"Running '{task.name}': {len(task.all_cases())} cases x {len(models)} models x "
        f"{repeats} repeats = {calls} calls",
        file=sys.stderr,
    )
    runner = Runner(task, cache=cache, progress=sys.stderr.isatty())
    try:
        rows = asyncio.run(runner.run(model_ids=model_ids, repeats=repeats))
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    paths = write_outputs(rows, out_dir)
    if cache.enabled:
        print(
            f"Cache: {cache.hits} hits, {cache.misses} misses ({cache.directory})", file=sys.stderr
        )
    _print_summary(rows)
    print(f"Wrote {paths['results']}, {paths['summary']}, {paths['report']}")
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    try:
        rows = read_results(args.results)
    except ResultsFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    out_dir = Path(args.out) if args.out else Path(args.results).parent
    write_report_files(rows, out_dir)
    _print_summary(rows)
    print(f"Wrote {out_dir / 'summary.csv'}, {out_dir / 'report.md'}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    models = task.candidate_models()
    scorer_types = sorted({s.type for case in task.all_cases() for s in task.scorers_for(case)})
    print(
        f"OK: '{task.name}' has {len(task.all_cases())} cases, {len(models)} models "
        f"({', '.join(m.id for m in models)}), repeats {task.repeats}, "
        f"scorers {', '.join(scorer_types)}"
    )
    for message in missing_api_keys(models + task.judge_models):
        print(f"warning: {message}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalh", description="Compare LLMs on your own task: quality, latency and cost."
    )
    parser.add_argument("--version", action="version", version=f"evalh {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task against its models and write a report")
    run.add_argument("task", help="path to the task YAML file")
    run.add_argument("--models", help="comma-separated model ids to run (default: all)")
    run.add_argument("--repeats", type=int, help="override the task's repeats per case")
    run.add_argument("--out", help="output directory (default: runs/<task name>)")
    run.add_argument("--no-cache", action="store_true", help="ignore and do not write the cache")
    run.add_argument(
        "--cache-dir", help="cache directory (default: .evalh_cache or $EVALH_CACHE_DIR)"
    )
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="rebuild summary.csv and report.md from results.jsonl")
    report.add_argument("results", help="path to results.jsonl")
    report.add_argument("--out", help="output directory (default: next to results.jsonl)")
    report.set_defaults(func=cmd_report)

    validate = sub.add_parser("validate", help="check a task file without calling any model")
    validate.add_argument("task", help="path to the task YAML file")
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "repeats", None) is not None and args.repeats < 1:
        parser.error("--repeats must be at least 1")
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
