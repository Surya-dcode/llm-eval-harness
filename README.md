# llm-eval-harness

[![CI](https://github.com/Surya-dcode/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Surya-dcode/llm-eval-harness/actions/workflows/ci.yml)

`evalh` runs the same set of test cases against several LLMs and tells you how each one did on quality, latency and cost. You describe the task, the cases, the scorers and the models in one YAML file, then `evalh run task.yaml` writes a per-call results file, a CSV summary and a Markdown report.

## Why

Public leaderboards rank models on someone else's tasks. When you have to pick a model for your own job (pulling fields out of documents, classifying tickets, answering from a fixed context), what matters is how each model does on examples that look like your data, how long it takes, and what it costs at your volume.

This tool is for that decision. You write 10 to a few hundred cases that represent your task, list the models you are considering with the prices you actually pay, and get a table you can compare side by side. Repeats show how much a model's answers vary between runs, and the per-case breakdown shows exactly where models disagree, which is usually where the interesting failures are.

## Quick start (no API keys needed)

```bash
git clone https://github.com/Surya-dcode/llm-eval-harness.git
cd llm-eval-harness
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

evalh validate examples/extract_invoice/task.yaml
evalh run examples/extract_invoice/task.yaml --out runs/demo
```

The example uses the built-in `fake` provider, which answers from each case's expected value and can be told to get some cases wrong. It makes no network calls. The run writes:

```
runs/demo/results.jsonl   one row per model call
runs/demo/summary.csv     one row per model
runs/demo/report.md       the report
```

To rebuild the CSV and report from an existing results file without calling any model:

```bash
evalh report runs/demo/results.jsonl
```

`python -m evalh` works the same as `evalh`.

## Switching to real providers

Three providers are built in:

| `provider` | Covers | Needs |
|---|---|---|
| `openai_compatible` | OpenAI, Groq, Together, OpenRouter, and local servers such as Ollama or vLLM. Anything that serves `POST /chat/completions`. | `base_url`, and `api_key_env` unless the server is local |
| `anthropic` | Anthropic Messages API | `pip install -e ".[anthropic]"` and `ANTHROPIC_API_KEY` (or `api_key_env`) |
| `fake` | Deterministic, seedable fake for tests and demos | nothing |

1. Copy `.env.example` to `.env`, fill in the keys you need, and load it into your shell: `set -a; source .env; set +a`. `evalh` reads keys from environment variables and never reads `.env` itself.
2. Add the models to your task (or to a separate `models_file`). The example's `models.yaml` has commented-out entries to start from.
3. Fill in the prices. **evalh ships no prices.** Look up the current price per 1M input and output tokens for each model and put them in `price_in_per_mtok` and `price_out_per_mtok`. For a local model, use 0 or your own estimate of hardware cost.

```yaml
models:
  - id: groq-llama
    provider: openai_compatible
    model: your-groq-model-name
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
    rate_group: groq            # shares a concurrency limit with other groq models
    price_in_per_mtok: 0.0      # fill in the current price
    price_out_per_mtok: 0.0     # fill in the current price
    params: {temperature: 0, max_tokens: 300}

  - id: local-ollama
    provider: openai_compatible
    model: llama3.2
    base_url: http://localhost:11434/v1
    price_in_per_mtok: 0
    price_out_per_mtok: 0
```

Before any call is made, `evalh run` checks that every selected model's key variable is set and stops with a list of what is missing. Run a subset with `--models groq-llama,local-ollama`.

## Task file reference

```yaml
name: extract_invoice                 # required; used in the report and default output dir
description: optional free text
system_prompt: optional system message
prompt_template: |                    # required; filled from each case's vars
  Extract the fields from: {{ invoice }}
template_syntax: double_brace         # double_brace ({{ name }}) or format ({name})

cases_file: cases.jsonl               # or `cases:` inline, one of the two
repeats: 3                            # calls per case per model (default 1)

scorers:                              # applied to every case without its own scorers
  - type: json_valid
  - type: field_match
    weight: 3

models_file: models.yaml              # or `models:` inline, one of the two
judge_models: []                      # models used only by llm_judge, never evaluated

execution:
  concurrency: 4                      # default in-flight calls per rate group
  provider_concurrency: {groq: 2}     # per rate group overrides
  timeout_s: 60                       # per attempt
  max_retries: 3                      # on 408, 429, 5xx, timeouts, connection errors
  backoff_base_s: 1.0                 # exponential backoff with jitter
  backoff_max_s: 30                   # also caps a server's Retry-After
```

Paths in `cases_file` and `models_file` are relative to the task file. `double_brace` placeholders are Jinja-style variable substitution only (`{{ name }}`, `{{ customer.name }}`). There are no loops or filters. Dicts and lists are inserted as JSON.

CLI flags: `evalh run task.yaml [--models a,b] [--repeats N] [--out dir] [--no-cache] [--cache-dir dir]`. The default output directory is `runs/<task name>`.

**Cases.** Each case is an object with these keys. In a JSONL file, put one per line.

| Key | Meaning |
|---|---|
| `id` | Unique case id (required) |
| `vars` | Values for the prompt template |
| `expected` | Reference answer. A string, list or object, depending on the scorers |
| `scorers` | Optional list that replaces the task-level scorers for this case |
| `tags` | Optional labels, kept for your own filtering |

**Models.**

| Key | Meaning |
|---|---|
| `id` | Name used in reports and `--models` (required, unique) |
| `provider` | `openai_compatible`, `anthropic` or `fake` |
| `model` | Model name sent to the provider |
| `price_in_per_mtok`, `price_out_per_mtok` | Your prices in USD per 1M tokens (required) |
| `params` | Passed through to the API (`temperature`, `max_tokens`, ...). For `fake`, its own options |
| `base_url` | API root for `openai_compatible` (default `https://api.openai.com/v1`) |
| `api_key_env` | Name of the environment variable that holds the key |
| `rate_group` | Concurrency group (default: the provider name) |
| `timeout_s` | Per-model override of `execution.timeout_s` |

`evalh validate task.yaml` loads everything, renders every prompt, checks that each scorer has what it needs from each case, and reports every problem it finds with its location. It calls no model. For example:

```
error: task.yaml: 2 problem(s)
  - models[1].price_in_per_mtok: required field is missing
  - cases[3] (id=inv-04): prompt_template uses ['invoice'], missing template variable 'invoice'
```

**Fake provider options** (under `params`): `seed`, `accuracy` (0 to 1, chance each call is right), `wrong_cases`, `malformed_cases` (broken JSON), `error_cases` (HTTP 400), `fail_first` (HTTP 429 this many times before succeeding), `latency_ms` (number or `[min, max]`), `prose_wrap` (wrap JSON in a sentence and code fence), `report_usage` (false to exercise the approximate token path), `fixed_response`, `judge_min_score`. The same seed, model, case and repeat always give the same answer.

## Scorers

Every scorer returns a score from 0 to 1, a pass flag and details. A call's score is the weighted mean of its scorers (`weight`, default 1), and a call passes only if every scorer passes. Give a scorer a `name` to tell two of the same type apart.

| `type` | Passes when | Options |
|---|---|---|
| `exact_match` | Output equals `expected` (a string, or any of a list of strings) after normalization | `expected` (override), `case_sensitive` (false), `strip_whitespace` (true), `collapse_whitespace` (true), `strip_punctuation` (false) |
| `regex` | The pattern matches | `pattern`, `mode` (`search` or `fullmatch`), `ignore_case`, `multiline`, `dotall`, `negate` |
| `contains_all` | Every keyword appears. Score is the fraction found | `keywords` (or a list in `expected`), `case_sensitive` (false) |
| `json_valid` | Output parses as JSON and, if given, matches the JSON Schema | `schema`, `extract` (true: unwrap code fences and find the first JSON value) |
| `field_match` | Parsed JSON fields equal the fields of the `expected` object. Score is the fraction that match | `fields` (default: all keys of `expected`, dotted paths allowed), `case_sensitive` (false), `numeric_tolerance` (0), `coerce_numbers` (true: `"$1,250.00"` equals `1250`), `extract`, `pass_threshold` (1.0) |
| `llm_judge` | A judge model's 1 to 5 score is at least `pass_threshold`. Score is `(s - 1) / 4` | `judge` (a model id from `models` or `judge_models`), `rubric`, `pass_threshold` (4), `include_expected` (true), `prompt_template`, `system_prompt` |

The judge must reply with `{"score": <1-5>, "reason": "..."}`. Code fences around it are fine. A reply that is not JSON, has no score, or has a score that is fractional or outside 1 to 5 becomes a failed score with `judge_parse_error` and the raw reply in the details. It does not stop the run. Judge calls go through the same cache, retries and concurrency limits, and their cost is reported separately.

```yaml
judge_models:
  - id: judge
    provider: openai_compatible
    model: your-judge-model-name
    api_key_env: OPENAI_API_KEY
    price_in_per_mtok: 0.0      # fill in
    price_out_per_mtok: 0.0     # fill in
    params: {temperature: 0}

scorers:
  - type: llm_judge
    judge: judge
    rubric: |
      5: answers the question fully and only uses facts from the context.
      3: mostly right but misses a detail or adds an unsupported claim.
      1: wrong or off topic.
```

## How cost and latency are measured

**Latency** is the wall time of the successful attempt, measured around the provider call with `time.perf_counter()`. It includes network time and any queueing on the provider's side. It excludes time spent waiting for a local concurrency slot and time spent in retry backoff. p50 and p95 use linear interpolation over successful calls.

Caveats:

- Latency depends on where you run from, the time of day, and your concurrency setting. Compare models from the same machine in the same run.
- Calls are not streamed, so this is time to the full response, not time to first token.
- With a small number of calls, p95 is close to the maximum and moves a lot between runs.
- Cached rows report the latency recorded when the call was first made.

**Tokens** come from the provider's `usage` field. If a provider returns no usage, evalh estimates tokens as characters / 4 (plus a small per-message overhead) and marks the row `tokens_approximate: true`. The report marks those costs with `*`. The estimate can be far off for code, non-English text or unusual tokenizers.

**Cost** is `input_tokens * price_in_per_mtok / 1e6 + output_tokens * price_out_per_mtok / 1e6`, using the prices in your config. It does not account for prompt-caching discounts, batch pricing, free tiers, minimum charges or taxes. Failed calls are counted as zero cost. "Cost / case" in the report is total cost divided by the number of calls. Cached rows are priced from their recorded token counts with the prices in the current config, so a re-run report shows the cost of running the task, and you can change prices and re-run without paying again.

## Caching

Every successful response is stored under `.evalh_cache/` (or `--cache-dir`, or `$EVALH_CACHE_DIR`). The key is a SHA-256 hash of the provider, model, base URL, params, messages and repeat index. The repeat index is part of the key so `repeats: 3` makes three real calls the first time and then reuses them.

Re-running the same task with the same models and prompts makes no new API calls. Change the prompt, a param or a case and only the affected calls run again. Errors are never cached, so a re-run retries only what failed. Use `--no-cache` to skip the cache entirely (no reads, no writes). Delete the directory to clear it.

The cache stores model outputs as plain JSON. Do not commit it if your cases contain private data. It is in `.gitignore` by default.

## Sample report

The report below was generated with the fake provider from `examples/extract_invoice/` (`evalh run examples/extract_invoice/task.yaml`). Both models are fake: their answers are derived from the expected values, their latency is a simulated delay, and their prices are placeholders. The numbers only show what a report looks like and say nothing about any real model. The full report is in [`examples/extract_invoice/sample_output/report.md`](examples/extract_invoice/sample_output/report.md).

| Model | Provider/model | Calls | Mean score | Pass rate | p50 latency | p95 latency | Total cost | Cost / case | Failures |
|---|---|---|---|---|---|---|---|---|---|
| `fake-careful` | fake/fake-careful-v1 | 30 | 0.988 | 93% | 597 ms | 859 ms | $0.006754 | $0.000225 | 0 |
| `fake-quick` | fake/fake-quick-v1 | 30 | 0.850 | 63% | 187 ms | 236 ms | $0.000761 | $0.000025 | 0 |

Part of the per-case breakdown, where each cell is the mean score with passes/runs:

| Case | `fake-careful` | `fake-quick` | Why |
|---|---|---|---|
| inv-03 | 0.94 (2/3) | 0.00 (0/3) | models differ, varies across repeats |
| inv-07 | 1.00 (3/3) | 0.81 (0/3) | models differ |
| inv-09 | 0.94 (2/3) | 0.94 (2/3) | varies across repeats |

The full report also has a per-scorer table, a list of failed calls, and a sample of failed scores with the field mismatches, such as `field_match (total: expected 238.8, got 263.68)`.

## Limitations

- Text in, text out. No images, tools or function calling, and no streaming.
- One system prompt and one user turn per case. Multi-turn conversations are not supported.
- Token estimates without provider usage are rough, and costs are only as accurate as the prices you enter.
- Retries cover 408, 429, 5xx, timeouts and connection errors. There is no client-side rate limiter beyond the concurrency limit.
- `llm_judge` scores are only as reliable as the judge model and rubric. Read a sample of judge reasons before trusting them, and avoid using a model as the judge of its own outputs.
- No significance testing. With few cases, a difference of one or two cases between models may be noise. Use repeats and more cases before drawing conclusions.
- The Anthropic adapter is tested against a stand-in client, not the live API.

## Running tests

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

The tests make no network calls. HTTP behavior (retries on 429, Retry-After, status mapping, auth headers) is tested with `httpx.MockTransport`, and full runs use the fake provider. CI runs the same commands on Python 3.11 and 3.12.

## License

MIT. See [LICENSE](LICENSE).
