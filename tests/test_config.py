from __future__ import annotations

import json

import pytest

from evalh.config import ConfigError, load_task, parse_task
from tests.conftest import task_dict


def problems_for(data) -> list[str]:
    with pytest.raises(ConfigError) as info:
        parse_task(data, source="task.yaml")
    return info.value.problems


def test_valid_task_loads(make_task):
    task = make_task()
    assert task.name == "unit"
    assert [c.id for c in task.all_cases()] == ["c1", "c2"]
    assert task.build_messages(task.all_cases()[0])[-1]["content"] == "Extract from: Acme, 10 USD"


def test_missing_required_field_names_the_path():
    data = task_dict()
    del data["models"][0]["price_in_per_mtok"]
    assert problems_for(data) == ["models[0].price_in_per_mtok: required field is missing"]


def test_negative_price_is_rejected():
    data = task_dict()
    data["models"][0]["price_out_per_mtok"] = -1
    (problem,) = problems_for(data)
    assert problem.startswith("models[0].price_out_per_mtok:")
    assert "greater than or equal to 0" in problem


def test_unknown_field_is_called_out():
    data = task_dict(promt_template="typo")
    assert "promt_template: unknown field (check the spelling)" in problems_for(data)


def test_unknown_provider_lists_choices():
    data = task_dict()
    data["models"][0]["provider"] = "openai"
    (problem,) = problems_for(data)
    assert problem.startswith("models[0].provider:")
    assert "openai_compatible" in problem


def test_unknown_scorer_type():
    data = task_dict(scorers=[{"type": "bleu"}])
    (problem,) = problems_for(data)
    assert problem.startswith("scorers[0]: unknown scorer type 'bleu'")


def test_scorer_option_error_has_path():
    data = task_dict(scorers=[{"type": "regex", "pattern": "("}])
    (problem,) = problems_for(data)
    assert problem.startswith("scorers[0].pattern: invalid regular expression")


def test_missing_template_variable_is_reported_per_case():
    data = task_dict(prompt_template="{{ text }} and {{ other }}")
    problems = problems_for(data)
    assert len(problems) == 2
    assert "cases[0] (id=c1)" in problems[0]
    assert "missing template variable 'other'" in problems[0]


def test_format_syntax_template():
    data = task_dict(prompt_template="Text: {text}", template_syntax="format")
    task = parse_task(data)
    assert task.build_messages(task.all_cases()[0])[-1]["content"] == "Text: Acme, 10 USD"
    data["prompt_template"] = "Text: {nope}"
    assert "missing template variable 'nope'" in problems_for(data)[0]


def test_scorer_needs_expected_value():
    data = task_dict(scorers=[{"type": "exact_match"}])
    problems = problems_for(data)
    assert len(problems) == 2
    assert "exact_match needs a string" in problems[0]


def test_unknown_judge_reference():
    data = task_dict(scorers=[{"type": "llm_judge", "judge": "nobody", "rubric": "Is it right?"}])
    assert "judge 'nobody'" in problems_for(data)[0]


def test_duplicate_ids():
    data = task_dict()
    data["cases"][1]["id"] = "c1"
    data["models"].append(dict(data["models"][0]))
    problems = problems_for(data)
    assert any("duplicate case id" in p for p in problems)
    assert any("duplicate model id 'fake-a'" in p for p in problems)


def test_fake_params_typo():
    data = task_dict()
    data["models"][0]["params"] = {"acuracy": 0.5}
    assert problems_for(data) == [
        "model 'fake-a' params.acuracy: unknown field (check the spelling)"
    ]


def test_needs_cases_and_models():
    data = task_dict()
    del data["cases"]
    assert problems_for(data) == ["<root>: no cases: add 'cases' or 'cases_file'"]


def test_yaml_syntax_error_has_line(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text("name: x\nprompt_template: [unclosed\n")
    with pytest.raises(ConfigError) as info:
        load_task(path)
    assert "YAML syntax error at line" in info.value.problems[0]


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError) as info:
        load_task(tmp_path / "nope.yaml")
    assert info.value.problems == ["file not found"]


def test_cases_and_models_files(tmp_path):
    (tmp_path / "cases.jsonl").write_text(
        json.dumps({"id": "a", "vars": {"text": "x"}, "expected": {"v": 1}})
        + "\n\nnot json\n"
        + json.dumps({"id": "b", "vars": {"text": "y"}, "expectd": {}})
        + "\n"
    )
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  - {id: m, provider: fake, model: m, price_in_per_mtok: 0, price_out_per_mtok: 0}\n"
    )
    data = task_dict(cases_file="cases.jsonl", models_file="models.yaml")
    del data["cases"], data["models"]
    with pytest.raises(ConfigError) as info:
        parse_task(data, base_dir=tmp_path)
    problems = info.value.problems
    assert "cases.jsonl line 3: invalid JSON (Expecting value)" in problems
    assert "cases.jsonl[1].expectd: unknown field (check the spelling)" in problems


def test_error_message_is_readable():
    data = task_dict()
    del data["models"][0]["model"]
    with pytest.raises(ConfigError) as info:
        parse_task(data, source="my_task.yaml")
    assert (
        str(info.value)
        == "my_task.yaml: 1 problem(s)\n  - models[0].model: required field is missing"
    )


def test_example_task_validates():
    task = load_task("examples/extract_invoice/task.yaml")
    assert len(task.all_cases()) == 10
    assert [m.id for m in task.candidate_models()] == ["fake-careful", "fake-quick"]
