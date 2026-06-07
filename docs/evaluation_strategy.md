# Ground-Truth Evaluation Strategy

This benchmark evaluates each agent final answer against a task-specific ground-truth answer. The implementation is recorded in `ground_truth_evaluation.py`. The evaluator first normalizes both the ground truth and the model's final answer into a common payload shape, then selects a scoring strategy from the declared `answer_type`. The three structured answer types used for quantitative outputs are `scalar`, `timeseries`, and `chart`.

All three structured strategies are deterministic. They do not use an LLM judge. Each task receives a binary ground-truth score: `1` when the required structure and values match the ground truth, and `0` otherwise. The result also records diagnostics such as parse errors, parser used, expected and actual values, tolerances, and mismatches.

## Payload Normalization

Before scoring, the evaluator parses the raw answer text as JSON, an embedded JSON fragment, or YAML. Markdown JSON code fences are stripped when present. If a parsed object follows the benchmark response format:

```json
{
  "status": "success",
  "answer": {}
}
```

only the nested `answer` object is compared. This allows the evaluator to ignore superficial response wrapping while still preserving parse diagnostics and status metadata.

The declared `answer_type` controls the main dispatch:

- `scalar` uses the scalar deterministic evaluator.
- `timeseries` and `chart` both use the series deterministic evaluator.
- `text` uses the separate judge-based text evaluator.

If no answer type is declared, the evaluator falls back to payload shape: objects with `value` are treated as scalar, objects with `series` are treated as timeseries-like series, and objects with `labels` plus `datasets` are treated as chart-like series.

## Scalar Answers

Scalar answers are expected to contain one numeric value and, where applicable, a unit:

```json
{
  "value": 28845,
  "unit": "posts"
}
```

The scalar evaluator extracts `value` as a floating-point number and normalizes common unit aliases. For example, `post`, `posts`, `count`, `records`, and `entries` are normalized to `posts`; `%`, `percentage`, and `percent` are normalized to `percent`.

A scalar answer passes only when both conditions hold:

- The actual value is numerically close to the ground-truth value.
- The actual unit is compatible with the ground-truth unit.

Numeric closeness is computed with Python's `math.isclose`. The default scalar tolerance is:

- absolute tolerance: `0.5`
- relative tolerance: `1e-6`

These defaults allow minor integer rounding or negligible data backfill differences while still treating count-like answers as essentially exact. Task metadata can override them with `scalar_abs_tolerance` and `scalar_rel_tolerance`, or with shared `abs_tolerance` and `rel_tolerance`.

If the ground truth omits a unit, any actual unit is accepted. If the ground truth specifies a unit, the final answer must provide the same normalized unit.

## Timeseries Answers

Timeseries answers are expected to contain one or more ordered series:

```json
{
  "x_axis_type": "time",
  "granularity": "day",
  "series": [
    {
      "label": "posts",
      "points": [
        {"x": "2024-01-01", "value": 120}
      ]
    }
  ]
}
```

The evaluator normalizes each series into a list of `(x, value)` points. The `x` value is compared as a string, and `value` is compared as a floating-point number.

Timeseries scoring checks:

- Granularity, when both expected and actual granularity are provided.
- Number of series.
- Alignment between expected and actual series.
- Number of points in each matched series.
- Exact x-label ordering within each matched series.
- Numeric value closeness for every corresponding point.

Series alignment is order-tolerant at the series level. If an answer contains multiple series, the evaluator finds the best mapping between expected and actual series by minimizing missing series, point-count differences, x-label mismatches, value mismatches, label mismatches, and total absolute error. After alignment, the points inside each matched series are compared in order. This means the evaluator can tolerate a different ordering of multiple named series, but it does not tolerate reordering time points within a series.

The default timeseries tolerance is:

- absolute tolerance: `0.5`
- relative tolerance: `1e-6`

Task metadata can override these with `series_abs_tolerance` and `series_rel_tolerance`, or with shared `abs_tolerance` and `rel_tolerance`.

## Chart Answers

Chart answers use the same deterministic series evaluator as timeseries answers, but are normalized from a chart-oriented payload:

```json
{
  "chart_type": "bar",
  "labels": ["2024-01", "2024-02"],
  "datasets": [
    {
      "label": "posts",
      "data": [100, 120]
    }
  ]
}
```

The evaluator converts each dataset into a series by pairing each entry in `labels` with the corresponding numeric entry in `data`. For example, the first label is paired with the first data value, the second label with the second data value, and so on.

Chart scoring checks:

- Chart type, when both expected and actual chart types are provided.
- Number of datasets after normalization.
- Dataset alignment, using the same best-match procedure as timeseries series.
- Number of points per dataset.
- Exact label ordering within each dataset.
- Numeric value closeness for every plotted value.

This design evaluates the semantic data encoded in the chart rather than the rendered image itself. A chart passes if it represents the correct chart type and plots the correct labels, dataset structure, and values within tolerance. Cosmetic rendering details such as colors, dimensions, and chart URL are not part of the deterministic score.

## Rationale

The benchmark separates structured quantitative tasks from free-text tasks. For `scalar`, `timeseries`, and `chart` answers, correctness can be defined over explicit JSON fields, so deterministic scoring is preferable to LLM judging. This makes these scores reproducible, auditable, and insensitive to wording differences in the final response. The evaluator still records detailed diagnostics for failed cases, which supports later error analysis by distinguishing parsing failures, missing values, unit mismatches, structural mismatches, ordering errors, and numerical errors.
