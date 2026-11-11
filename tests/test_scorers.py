from __future__ import annotations

import asyncio

import pytest
from pydantic import TypeAdapter, ValidationError

from evalh.jsonutil import JSONExtractError, parse_json
from evalh.scorers import (
    JudgeReply,
    ScoreContext,
    ScorerConfig,
    combine,
    parse_judge_reply,
)

adapter = TypeAdapter(ScorerConfig)


def run(config: dict, output: str, expected=None, judge=None):
    scorer = adapter.validate_python(config)
    ctx = ScoreContext(case_id="c", expected=expected, prompt="the prompt", judge=judge)
    return asyncio.run(scorer.score(output, ctx))


# exact_match


def test_exact_match_normalizes_case_and_whitespace():
    result = run({"type": "exact_match"}, "  Paris \n", expected="paris")
    assert (result.score, result.passed) == (1.0, True)


def test_exact_match_case_sensitive():
    assert run({"type": "exact_match", "case_sensitive": True}, "Paris", "paris").score == 0.0


def test_exact_match_punctuation_option():
    assert run({"type": "exact_match"}, "Paris.", "Paris").score == 0.0
    assert run({"type": "exact_match", "strip_punctuation": True}, "Paris.", "Paris").score == 1.0


def test_exact_match_list_of_answers_and_override():
    assert run({"type": "exact_match"}, "NYC", ["New York", "NYC"]).passed
    assert run({"type": "exact_match", "expected": "yes"}, "Yes", expected=None).passed


def test_exact_match_empty_output():
    result = run({"type": "exact_match"}, "", "x")
    assert (result.score, result.passed) == (0.0, False)


def test_exact_match_check_case():
    scorer = adapter.validate_python({"type": "exact_match"})
    assert scorer.check_case({"a": 1}) is not None
    assert scorer.check_case("ok") is None


# regex


def test_regex_search_and_fullmatch():
    assert run({"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"}, "Date: 2026-01-02").passed
    assert not run(
        {"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}", "mode": "fullmatch"}, "Date: 2026-01-02"
    ).passed
    assert run({"type": "regex", "pattern": r"\d+", "mode": "fullmatch"}, " 42\n").passed


def test_regex_flags_and_negate():
    assert run({"type": "regex", "pattern": "hello", "ignore_case": True}, "HELLO").passed
    result = run({"type": "regex", "pattern": "sorry", "negate": True}, "Sorry, I can't")
    assert result.passed  # no case-sensitive match, so negated check passes
    assert not run({"type": "regex", "pattern": "(?i)sorry", "negate": True}, "Sorry").passed


def test_regex_invalid_pattern_rejected():
    with pytest.raises(ValidationError, match="invalid regular expression"):
        adapter.validate_python({"type": "regex", "pattern": "[a-"})


# contains_all


def test_contains_all_partial_credit():
    result = run({"type": "contains_all", "keywords": ["alpha", "Beta", "gamma"]}, "ALPHA and beta")
    assert result.score == pytest.approx(2 / 3)
    assert not result.passed
    assert result.details["missing"] == ["gamma"]


def test_contains_all_uses_expected_list_and_case_sensitivity():
    assert run({"type": "contains_all"}, "a b", expected=["a", "b"]).passed
    assert not run(
        {"type": "contains_all", "case_sensitive": True}, "A b", expected=["a", "b"]
    ).passed


def test_contains_all_empty_keywords_rejected():
    with pytest.raises(ValidationError, match="keywords must not be empty"):
        adapter.validate_python({"type": "contains_all", "keywords": []})


# json_valid

SCHEMA = {
    "type": "object",
    "required": ["total"],
    "properties": {"total": {"type": "number"}},
}


def test_json_valid_plain_and_fenced():
    assert run({"type": "json_valid"}, '{"a": 1}').passed
    assert run({"type": "json_valid"}, 'Sure:\n```json\n{"a": 1}\n```\nDone.').passed
    assert run({"type": "json_valid"}, 'The answer is {"a": [1, 2]} as asked.').passed


def test_json_valid_strict_mode():
    assert not run({"type": "json_valid", "extract": False}, 'Here: {"a": 1}').passed


def test_json_valid_invalid_and_empty():
    result = run({"type": "json_valid"}, '{"a": 1')
    assert (result.score, result.passed) == (0.0, False)
    assert "no valid JSON found" in result.details["error"]
    assert not run({"type": "json_valid"}, "").passed


def test_json_valid_schema():
    assert run({"type": "json_valid", "schema": SCHEMA}, '{"total": 3.5}').passed
    result = run({"type": "json_valid", "schema": SCHEMA}, '{"total": "3.5"}')
    assert not result.passed
    assert result.details["schema_errors"] == ["total: '3.5' is not of type 'number'"]
    missing = run({"type": "json_valid", "schema": SCHEMA}, "{}")
    assert "<root>: 'total' is a required property" in missing.details["schema_errors"]


def test_json_valid_bad_schema_rejected():
    with pytest.raises(ValidationError, match="invalid JSON Schema"):
        adapter.validate_python({"type": "json_valid", "schema": {"type": "nonsense"}})


# field_match

EXPECTED = {"vendor": "Acme Corp", "total": 1250.0, "currency": "USD", "meta": {"page": 2}}


def test_field_match_all_fields():
    output = '{"vendor": "acme  corp", "total": 1250, "currency": "USD", "meta": {"page": 2}}'
    result = run({"type": "field_match"}, output, EXPECTED)
    assert (result.score, result.passed) == (1.0, True)


def test_field_match_partial_score_and_mismatch_details():
    output = '{"vendor": "Acme", "total": 1250, "currency": "EUR"}'
    result = run(
        {"type": "field_match", "fields": ["vendor", "total", "currency"]}, output, EXPECTED
    )
    assert result.score == pytest.approx(1 / 3)
    assert not result.passed
    assert result.details["mismatches"]["currency"] == {"expected": "USD", "got": "EUR"}


def test_field_match_missing_field_and_nested_path():
    result = run(
        {"type": "field_match", "fields": ["meta.page", "vendor"]},
        '{"meta": {"page": 2}}',
        EXPECTED,
    )
    assert result.score == 0.5
    assert result.details["mismatches"]["vendor"]["got"] == "<missing>"


def test_field_match_number_coercion_and_tolerance():
    exp = {"total": 1250.0}
    assert run({"type": "field_match"}, '{"total": "$1,250.00"}', exp).passed
    assert not run(
        {"type": "field_match", "coerce_numbers": False}, '{"total": "1250"}', exp
    ).passed
    assert not run({"type": "field_match"}, '{"total": 1250.01}', exp).passed
    assert run({"type": "field_match", "numeric_tolerance": 0.05}, '{"total": 1250.01}', exp).passed


def test_field_match_booleans_are_not_numbers():
    assert not run({"type": "field_match"}, '{"total": true}', {"total": 1}).passed


def test_field_match_pass_threshold():
    output = '{"vendor": "Acme Corp", "total": 1, "currency": "USD"}'
    config = {
        "type": "field_match",
        "fields": ["vendor", "total", "currency"],
        "pass_threshold": 0.6,
    }
    result = run(config, output, EXPECTED)
    assert result.score == pytest.approx(2 / 3)
    assert result.passed


def test_field_match_unparseable_output():
    result = run({"type": "field_match"}, "I could not find it.", EXPECTED)
    assert (result.score, result.passed) == (0.0, False)
    assert "error" in result.details


def test_field_match_check_case():
    scorer = adapter.validate_python({"type": "field_match", "fields": ["x"]})
    assert "needs a JSON object" in scorer.check_case("text")
    assert "no value for field" in scorer.check_case({"y": 1})


# llm_judge


def judge_returning(text=None, error=None):
    calls = []

    async def judge(model_id, messages):
        calls.append((model_id, messages))
        return JudgeReply(text=text, error=error, cost_usd=0.001)

    return judge, calls


JUDGE = {"type": "llm_judge", "judge": "j", "rubric": "Is the answer correct and concise?"}


def test_llm_judge_valid_reply():
    judge, calls = judge_returning('{"score": 4, "reason": "Mostly right."}')
    result = run(JUDGE, "Paris", expected="Paris", judge=judge)
    assert result.score == 0.75
    assert result.passed
    assert result.details["judge_score"] == 4
    assert result.details["judge_cost_usd"] == 0.001
    user_prompt = calls[0][1][-1]["content"]
    assert "Is the answer correct and concise?" in user_prompt
    assert "Paris" in user_prompt and "the prompt" in user_prompt


def test_llm_judge_threshold():
    judge, _ = judge_returning('{"score": 3, "reason": "ok"}')
    assert not run(JUDGE, "x", judge=judge).passed
    assert run({**JUDGE, "pass_threshold": 3}, "x", judge=judge).passed


@pytest.mark.parametrize(
    "reply",
    [
        "I think it deserves a 4.",
        '{"score": 7, "reason": "great"}',
        '{"score": 3.5, "reason": "between"}',
        '{"reason": "no score"}',
        "[4]",
        "",
    ],
)
def test_llm_judge_unparseable_reply_is_a_scored_failure(reply):
    judge, _ = judge_returning(reply)
    result = run(JUDGE, "x", judge=judge)
    assert (result.score, result.passed) == (0.0, False)
    assert result.details["error"] == "judge_parse_error"
    assert result.details["message"]


def test_llm_judge_call_failure_is_a_scored_failure():
    judge, _ = judge_returning(error="HTTP 500: boom")
    result = run(JUDGE, "x", judge=judge)
    assert result.details["error"] == "judge_call_failed"
    assert not result.passed


def test_parse_judge_reply_accepts_fenced_and_string_scores():
    assert parse_judge_reply('```json\n{"score": "5", "reason": "r"}\n```').score == 5
    assert parse_judge_reply('{"score": 2.0}').score == 2


# helpers


def test_combine_weighted():
    a = run({"type": "regex", "pattern": "x"}, "x")
    b = run({"type": "regex", "pattern": "y"}, "x")
    score, passed = combine([a, b], [3, 1])
    assert score == 0.75
    assert not passed


def test_parse_json_errors():
    with pytest.raises(JSONExtractError):
        parse_json("nothing here")
