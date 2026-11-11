"""Full runs through the CLI with the fake provider. No network."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from evalh.cli import main
from evalh.report import render_report
from evalh.results import read_results
from evalh.tokens import approx_message_tokens, approx_tokens, call_cost
from tests.conftest import task_dict

PRICE_IN, PRICE_OUT = 2.0, 8.0


def write_task(tmp_path: Path) -> Path:
    data = task_dict(
        repeats=2,
        scorers=[{"type": "json_valid"}, {"type": "field_match", "weight": 3}],
        models=[
            {
                "id": "good",
                "provider": "fake",
                "model": "fake-good",
                "price_in_per_mtok": PRICE_IN,
                "price_out_per_mtok": PRICE_OUT,
            },
            {
                "id": "shaky",
                "provider": "fake",
                "model": "fake-shaky",
                "price_in_per_mtok": 0.5,
                "price_out_per_mtok": 1.0,
                "params": {"malformed_cases": ["c2"], "error_cases": ["c1"]},
            },
        ],
    )
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def run_dir(tmp_path):
    task_path = write_task(tmp_path)
    out = tmp_path / "out"
    code = main(["run", str(task_path), "--out", str(out), "--cache-dir", str(tmp_path / "cache")])
    assert code == 0
    return out


def test_run_writes_all_three_files(run_dir):
    for name in ["results.jsonl", "summary.csv", "report.md"]:
        assert (run_dir / name).is_file(), name


def test_results_rows_have_expected_numbers(run_dir):
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 2 * 2  # models x cases x repeats
    good = [r for r in rows if r["model_id"] == "good"]
    assert all(r["score"] == 1.0 and r["passed"] for r in good)

    messages = [
        {"role": "system", "content": "Reply with JSON."},
        {"role": "user", "content": "Extract from: Acme, 10 USD"},
    ]
    output = json.dumps({"vendor": "Acme", "total": 10})
    first = next(r for r in good if r["case_id"] == "c1" and r["repeat"] == 0)
    assert first["output"] == output
    assert first["input_tokens"] == approx_message_tokens(messages)
    assert first["output_tokens"] == approx_tokens(output)
    assert first["cost_usd"] == pytest.approx(
        call_cost(first["input_tokens"], first["output_tokens"], PRICE_IN, PRICE_OUT)
    )
    assert first["latency_ms"] >= 0

    shaky = {(r["case_id"], r["repeat"]): r for r in rows if r["model_id"] == "shaky"}
    assert shaky[("c1", 0)]["error"].startswith("HTTP 400")
    assert shaky[("c2", 0)]["score"] == 0 and shaky[("c2", 0)]["error"] is None


def test_summary_csv_numbers(run_dir):
    with (run_dir / "summary.csv").open() as handle:
        summary = {row["model_id"]: row for row in csv.DictReader(handle)}
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    good_cost = sum(r["cost_usd"] for r in rows if r["model_id"] == "good")

    assert summary["good"]["calls"] == "4"
    assert float(summary["good"]["mean_score"]) == 1.0
    assert float(summary["good"]["pass_rate"]) == 1.0
    assert float(summary["good"]["total_cost_usd"]) == pytest.approx(good_cost, abs=1e-8)
    assert float(summary["good"]["cost_per_call_usd"]) == pytest.approx(good_cost / 4, abs=1e-8)
    assert summary["good"]["failures"] == "0"

    assert summary["shaky"]["failures"] == "2"
    assert float(summary["shaky"]["mean_score"]) == 0.0
    assert float(summary["shaky"]["pass_rate"]) == 0.0


def test_report_contents(run_dir):
    report = (run_dir / "report.md").read_text()
    assert report.startswith("# Eval report: unit")
    assert "uses the `fake` provider" in report
    assert "| `good` | fake/fake-good | 4 | 1.000 | 100% |" in report
    assert "| c1 | 1.00 (2/2) | 0.00 (0/2) | models differ |" in report
    assert "## Failed calls" in report
    assert "HTTP 400: fake bad request" in report


def test_report_command_regenerates_same_report(run_dir, tmp_path):
    original = (run_dir / "report.md").read_text()
    out = tmp_path / "regen"
    assert main(["report", str(run_dir / "results.jsonl"), "--out", str(out)]) == 0
    assert (out / "report.md").read_text() == original
    assert (out / "summary.csv").read_text() == (run_dir / "summary.csv").read_text()
    assert render_report(read_results(run_dir / "results.jsonl")) == original


def test_rerun_uses_cache(tmp_path, capsys):
    task_path = write_task(tmp_path)
    args = ["run", str(task_path), "--cache-dir", str(tmp_path / "cache")]
    assert main([*args, "--out", str(tmp_path / "a")]) == 0
    capsys.readouterr()
    assert main([*args, "--out", str(tmp_path / "b")]) == 0
    err = capsys.readouterr().err
    assert "Cache: 6 hits, 2 misses" in err  # the two error calls are not cached
    report = (tmp_path / "b" / "report.md").read_text()
    assert "(6 served from cache)" in report


def test_cli_flags_models_and_repeats(tmp_path):
    task_path = write_task(tmp_path)
    out = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    code = main(
        [
            "run",
            str(task_path),
            *["--models", "good", "--repeats", "3", "--out", str(out)],
            *["--no-cache", "--cache-dir", str(cache_dir)],
        ]
    )
    assert code == 0
    rows = read_results(out / "results.jsonl")
    assert {r.model_id for r in rows} == {"good"}
    assert len(rows) == 6
    assert not cache_dir.exists()  # --no-cache writes nothing


def test_cli_validate_and_errors(tmp_path, capsys):
    task_path = write_task(tmp_path)
    assert main(["validate", str(task_path)]) == 0
    assert "OK: 'unit' has 2 cases, 2 models" in capsys.readouterr().out

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nprompt_template: hi\ncases: []\nmodels: []\nscorers: [{type: nope}]\n")
    assert main(["validate", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "scorers[0]: unknown scorer type 'nope'" in err

    assert main(["run", str(task_path), "--models", "missing", "--no-cache"]) == 2
    assert "unknown model id" in capsys.readouterr().err

    assert main(["report", str(tmp_path / "none.jsonl")]) == 1


def test_cli_refuses_to_start_without_api_key(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TEST_EVALH_NO_KEY", raising=False)
    data = task_dict(
        models=[
            {
                "id": "real",
                "provider": "openai_compatible",
                "model": "x",
                "api_key_env": "TEST_EVALH_NO_KEY",
                "price_in_per_mtok": 0,
                "price_out_per_mtok": 0,
            }
        ]
    )
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data))
    assert main(["run", str(path), "--no-cache", "--out", str(tmp_path / "o")]) == 2
    assert "TEST_EVALH_NO_KEY is not set" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_example_task_runs(tmp_path):
    code = main(
        [
            "run",
            "examples/extract_invoice/task.yaml",
            "--out",
            str(tmp_path / "ex"),
            "--no-cache",
            "--repeats",
            "1",
        ]
    )
    assert code == 0
    rows = read_results(tmp_path / "ex" / "results.jsonl")
    assert len(rows) == 20
    quick = {r.case_id: r for r in rows if r.model_id == "fake-quick"}
    assert quick["inv-03"].score == 0  # configured as malformed
    assert not quick["inv-07"].passed  # configured as wrong
