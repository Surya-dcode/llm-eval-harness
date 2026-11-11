import pytest

from evalh.templating import TemplateError, placeholders, render


def test_double_brace_substitution_and_dotted_access():
    out = render("Hi {{ user.name }}, item {{id}}", {"user": {"name": "Ana"}, "id": 3})
    assert out == "Hi Ana, item 3"


def test_double_brace_serializes_structures_as_json():
    assert render("{{ x }}", {"x": {"a": [1, 2]}}) == '{"a": [1, 2]}'


def test_double_brace_leaves_single_braces_alone():
    assert render('JSON like {"a": 1} and {{ v }}', {"v": "ok"}) == 'JSON like {"a": 1} and ok'


def test_missing_variable_raises():
    with pytest.raises(TemplateError, match="missing template variable 'nope'"):
        render("{{ nope }}", {})


def test_format_syntax():
    assert render("{a}-{b}", {"a": 1, "b": 2}, syntax="format") == "1-2"
    with pytest.raises(TemplateError):
        render("{a}", {}, syntax="format")


def test_placeholders():
    assert placeholders("{{ a }} {{ b.c }} {{ a }}") == ["a", "b"]
    assert placeholders("{a} {b[0]}", "format") == ["a", "b"]
