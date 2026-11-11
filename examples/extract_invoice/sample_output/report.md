# Eval report: extract_invoice

> This run uses the `fake` provider. Its answers, latencies and costs are
> simulated, so the numbers below only show what a report looks like.

- Run: 20261111-055857
- Cases: 10, repeats per case: 3, models: 2
- Calls: 60 (0 served from cache)

## Models

Sorted by mean score. Pass rate counts calls where every scorer passed. Latency is wall time per successful call. Cost uses the prices in the task config.

| Model | Provider/model | Calls | Mean score | Pass rate | p50 latency | p95 latency | Total cost | Cost / case | Failures |
|---|---|---|---|---|---|---|---|---|---|
| `fake-careful` | fake/fake-careful-v1 | 30 | 0.988 | 93% | 597 ms | 859 ms | $0.006754 | $0.000225 | 0 |
| `fake-quick` | fake/fake-quick-v1 | 30 | 0.850 | 63% | 187 ms | 236 ms | $0.000761 | $0.000025 | 0 |

## Scorers

Mean score per scorer (failed calls count as 0).

| Scorer | `fake-careful` | `fake-quick` |
|---|---|---|
| json_valid | 1.000 | 0.900 |
| field_match | 0.983 | 0.833 |

## Where models disagree

Each cell is the mean score with passes/runs in brackets.

| Case | `fake-careful` | `fake-quick` | Why |
|---|---|---|---|
| inv-03 | 0.94 (2/3) | 0.00 (0/3) | models differ, varies across repeats |
| inv-04 | 1.00 (3/3) | 0.94 (2/3) | models differ, varies across repeats |
| inv-05 | 1.00 (3/3) | 0.94 (2/3) | models differ, varies across repeats |
| inv-06 | 1.00 (3/3) | 0.94 (2/3) | models differ, varies across repeats |
| inv-07 | 1.00 (3/3) | 0.81 (0/3) | models differ |
| inv-08 | 1.00 (3/3) | 0.94 (2/3) | models differ, varies across repeats |
| inv-09 | 0.94 (2/3) | 0.94 (2/3) | varies across repeats |

## All cases

| Case | `fake-careful` | `fake-quick` |
|---|---|---|
| inv-01 | 1.00 (3/3) | 1.00 (3/3) |
| inv-02 | 1.00 (3/3) | 1.00 (3/3) |
| inv-03 | 0.94 (2/3) | 0.00 (0/3) |
| inv-04 | 1.00 (3/3) | 0.94 (2/3) |
| inv-05 | 1.00 (3/3) | 0.94 (2/3) |
| inv-06 | 1.00 (3/3) | 0.94 (2/3) |
| inv-07 | 1.00 (3/3) | 0.81 (0/3) |
| inv-08 | 1.00 (3/3) | 0.94 (2/3) |
| inv-09 | 0.94 (2/3) | 0.94 (2/3) |
| inv-10 | 1.00 (3/3) | 1.00 (3/3) |

## Sample of failed scores

Up to 10 scored outputs that did not pass.

| Model | Case | Repeat | What failed |
|---|---|---|---|
| `fake-careful` | inv-03 | 1 | field_match (date: expected '2026-01-27', got '2026-01-28') |
| `fake-careful` | inv-09 | 0 | field_match (vendor: expected 'Greenleaf Garden Supply', got 'Greenleaf Garden') |
| `fake-quick` | inv-03 | 0 | json_valid (no valid JSON found: Unterminated string starting at (position 88)); field_match (no valid JSON found: Unterminated string starting at (position 88)) |
| `fake-quick` | inv-03 | 1 | json_valid (no valid JSON found: Unterminated string starting at (position 88)); field_match (no valid JSON found: Unterminated string starting at (position 88)) |
| `fake-quick` | inv-03 | 2 | json_valid (no valid JSON found: Unterminated string starting at (position 88)); field_match (no valid JSON found: Unterminated string starting at (position 88)) |
| `fake-quick` | inv-04 | 0 | field_match (total: expected 238.8, got 263.68) |
| `fake-quick` | inv-05 | 0 | field_match (currency: expected 'USD', got 'EUR') |
| `fake-quick` | inv-06 | 1 | field_match (vendor: expected 'Kestrel Bike Repairs', got 'Kestrel Bike') |
| `fake-quick` | inv-07 | 0 | field_match (date: expected '2026-07-09', got '2026-09-07') |
| `fake-quick` | inv-07 | 1 | field_match (total: expected 61200, got 67321.0) |
