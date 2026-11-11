import pytest

from evalh.stats import mean, percentile, stdev
from evalh.tokens import approx_message_tokens, approx_tokens, call_cost


def test_call_cost_per_million_tokens():
    assert call_cost(1_000_000, 0, 3.0, 15.0) == pytest.approx(3.0)
    assert call_cost(0, 1_000_000, 3.0, 15.0) == pytest.approx(15.0)
    assert call_cost(1200, 300, 0.5, 1.5) == pytest.approx((1200 * 0.5 + 300 * 1.5) / 1e6)
    assert call_cost(1000, 1000, 0, 0) == 0


def test_approx_tokens():
    assert approx_tokens("") == 0
    assert approx_tokens("abcd") == 1
    assert approx_tokens("abcde") == 2
    messages = [{"role": "system", "content": "abcd"}, {"role": "user", "content": "abcdefgh"}]
    assert approx_message_tokens(messages) == (4 + 1) + (4 + 2)


def test_percentile_linear_interpolation():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 0) == 10
    assert percentile(values, 100) == 50
    assert percentile(values, 95) == pytest.approx(48.0)
    assert percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)  # unsorted input
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_percentile_edge_cases():
    assert percentile([], 50) is None
    assert percentile([7.5], 95) == 7.5
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_mean_and_stdev():
    assert mean([]) is None
    assert mean([1, 2, 3]) == 2
    assert stdev([5]) is None
    assert stdev([1, 3]) == 1.0
