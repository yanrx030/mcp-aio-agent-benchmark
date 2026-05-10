from __future__ import annotations

import asyncio
import json
import math
import re
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml
from openai import OpenAI

from prompts import normalize_answer_type, Evaluator_PROMPT


_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_EMBEDDED_JSON_CODE_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_DIRECT_OPENROUTER_KWARGS = {
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stream",
}
# Default tolerance policy for count-like outputs:
# - small absolute wiggle room for low-volume counts
# - tiny relative wiggle room for high-volume counts with minor backfill
_SERIES_ABS_TOLERANCE_DEFAULT = 1.0
_SERIES_REL_TOLERANCE_DEFAULT = 1e-6
_SCALAR_ABS_TOLERANCE_DEFAULT = 1.0
_SCALAR_REL_TOLERANCE_DEFAULT = 1e-6


@dataclass(slots=True)
class EvaluationModelConfig:
    openrouter_api_key: str
    judge_model: str | None = None
    judge_openrouter_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedPayload:
    raw: str | None
    parsed: Any | None
    answer_payload: Any | None
    status: str | None
    parse_error: str | None = None
    parser_used: str | None = None


@dataclass(slots=True)
class EvaluationContext:
    task_id: str
    prompt: str
    answer_type: str | None
    normalized_answer_type: str | None
    dispatch_type: str
    metadata: dict[str, Any]
    ground_truth: NormalizedPayload
    final_answer: NormalizedPayload
    model_config: EvaluationModelConfig


def build_evaluation_context(
    *,
    task_id: str,
    prompt: str,
    answer_type: str | None,
    ground_truth_raw: str | None,
    final_answer_raw: str | None,
    metadata: Mapping[str, Any] | None,
    openrouter_api_key: str,
    judge_model: str | None,
    judge_openrouter_params: Mapping[str, Any] | None,
) -> EvaluationContext:
    # Normalize both sides once so every evaluator receives a consistent shape.
    normalized_answer_type = normalize_answer_type(answer_type)
    ground_truth = _normalize_payload(ground_truth_raw)
    final_answer = _normalize_payload(final_answer_raw)
    # Choose evaluator strategy from declared answer type, with payload-shape fallback.
    dispatch_type = _resolve_dispatch_type(normalized_answer_type, ground_truth.answer_payload)

    return EvaluationContext(
        task_id=task_id,
        prompt=prompt,
        answer_type=answer_type,
        normalized_answer_type=normalized_answer_type,
        dispatch_type=dispatch_type,
        metadata=dict(metadata or {}),
        ground_truth=ground_truth,
        final_answer=final_answer,
        model_config=EvaluationModelConfig(
            openrouter_api_key=openrouter_api_key,
            judge_model=judge_model,
            judge_openrouter_params=dict(judge_openrouter_params or {}),
        ),
    )


async def evaluate_ground_truth(context: EvaluationContext) -> dict[str, Any]:
    # Central dispatcher: each evaluator returns diagnostics, then we collapse to a binary score.
    if context.dispatch_type == "scalar":
        score, diagnostics = _evaluate_scalar(context)
        strategy = "scalar_deterministic"
    elif context.dispatch_type == "series":
        score, diagnostics = _evaluate_series(context)
        strategy = "series_deterministic"
    else:
        score, diagnostics = await _evaluate_text(context)
        strategy = "text_judge"

    diagnostics = {
        "ground_truth_parse_error": context.ground_truth.parse_error,
        "final_answer_parse_error": context.final_answer.parse_error,
        "ground_truth_parser_used": context.ground_truth.parser_used,
        "final_answer_parser_used": context.final_answer.parser_used,
        **diagnostics,
    }

    return {
        "score": int(score),
        "passed": bool(score),
        "strategy": strategy,
        "answer_type": context.answer_type,
        "normalized_answer_type": context.normalized_answer_type,
        "dispatch_type": context.dispatch_type,
        "diagnostics": diagnostics,
    }


def _normalize_payload(raw: str | None) -> NormalizedPayload:
    # Final answers may be wrapped in markdown fences; strip and parse permissively.
    cleaned = _strip_code_fences(raw)
    parsed, parse_error, parser_used = _parse_any_structured(cleaned)
    # If payload follows {"status": ..., "answer": ...}, compare only the answer body.
    answer_payload, status = _extract_answer_payload(parsed)
    return NormalizedPayload(
        raw=raw,
        parsed=parsed,
        answer_payload=answer_payload,
        status=status,
        parse_error=parse_error,
        parser_used=parser_used,
    )


def _strip_code_fences(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    match = _JSON_CODE_FENCE_RE.fullmatch(text)
    if match:
        return match.group(1).strip()
    embedded_match = _EMBEDDED_JSON_CODE_FENCE_RE.search(text)
    if embedded_match:
        return embedded_match.group(1).strip()
    return text


def _parse_any_structured(raw: str | None) -> tuple[Any | None, str | None, str | None]:
    if raw is None:
        return None, "missing payload", None

    try:
        return json.loads(raw), None, "json"
    except Exception as json_exc:
        json_error = f"{type(json_exc).__name__}: {json_exc}"

    embedded_json_error: str | None = None
    extracted = _extract_first_balanced_json(raw)
    if extracted is not None:
        extracted_fragment, parsed_fragment = extracted
        if isinstance(parsed_fragment, (Mapping, list)):
            return parsed_fragment, None, "json_fragment"
        embedded_json_error = (
            f"fragment parsed to unsupported top-level type {type(parsed_fragment).__name__}: "
            f"{extracted_fragment[:120]!r}"
        )

    try:
        return yaml.safe_load(raw), None, "yaml"
    except Exception as yaml_exc:
        yaml_error = f"{type(yaml_exc).__name__}: {yaml_exc}"

    details = [f"json_parse={json_error}"]
    if embedded_json_error:
        details.append(f"embedded_json_parse={embedded_json_error}")
    details.append(f"yaml_parse={yaml_error}")
    return None, "; ".join(details), None


def _extract_first_balanced_json(raw: str) -> tuple[str, Any] | None:
    for start_index, opener, closer in _iter_json_start_tokens(raw):
        candidate = _scan_balanced_json(raw, start_index, opener, closer)
        if candidate is None:
            continue
        try:
            return candidate, json.loads(candidate)
        except Exception:
            continue
    return None


def _iter_json_start_tokens(raw: str) -> list[tuple[int, str, str]]:
    starts: list[tuple[int, str, str]] = []
    for index, char in enumerate(raw):
        if char == "{":
            starts.append((index, "{", "}"))
        elif char == "[":
            starts.append((index, "[", "]"))
    return starts


def _scan_balanced_json(raw: str, start_index: int, opener: str, closer: str) -> str | None:
    depth = 0
    in_string = False
    escape = False

    for index in range(start_index, len(raw)):
        char = raw[index]

        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
            continue
        if char == closer:
            depth -= 1
            if depth == 0:
                return raw[start_index : index + 1]
            if depth < 0:
                return None

    return None


def _extract_answer_payload(parsed: Any) -> tuple[Any | None, str | None]:
    if isinstance(parsed, Mapping):
        status = _as_clean_str(parsed.get("status"))
        if "answer" in parsed:
            return parsed.get("answer"), status
        return parsed, status
    return parsed, None


def _resolve_dispatch_type(answer_type: str | None, ground_truth_answer_payload: Any) -> str:
    if answer_type == "scalar":
        return "scalar"
    if answer_type in {"timeseries", "chart"}:
        return "series"
    if answer_type == "text":
        return "text"

    if isinstance(ground_truth_answer_payload, Mapping):
        if "value" in ground_truth_answer_payload:
            return "scalar"
        if "series" in ground_truth_answer_payload:
            return "series"
        if "labels" in ground_truth_answer_payload and "datasets" in ground_truth_answer_payload:
            return "series"
    return "text"


def _evaluate_scalar(context: EvaluationContext) -> tuple[int, dict[str, Any]]:
    expected_raw = context.ground_truth.answer_payload
    actual_raw = context.final_answer.answer_payload

    expected = _normalize_scalar(expected_raw)
    actual = _normalize_scalar(actual_raw)

    abs_tol = _get_float_config(
        context.metadata,
        keys=("scalar_abs_tolerance", "abs_tolerance"),
        default=_SCALAR_ABS_TOLERANCE_DEFAULT,
    )
    rel_tol = _get_float_config(
        context.metadata,
        keys=("scalar_rel_tolerance", "rel_tolerance"),
        default=_SCALAR_REL_TOLERANCE_DEFAULT,
    )

    # Scalar scoring is deterministic: value match within tolerance + unit compatibility.
    if expected["value"] is None or actual["value"] is None:
        return 0, {
            "reason": "missing scalar value in expected or actual payload",
            "expected": expected,
            "actual": actual,
            "tolerance": {"abs": abs_tol, "rel": rel_tol},
        }

    expected_value = float(expected["value"])
    actual_value = float(actual["value"])
    abs_error = abs(actual_value - expected_value)
    rel_error = abs_error / abs(expected_value) if expected_value != 0 else None
    value_match = math.isclose(actual_value, expected_value, abs_tol=abs_tol, rel_tol=rel_tol)
    unit_match = _units_compatible(expected.get("unit"), actual.get("unit"))

    score = int(value_match and unit_match)
    return score, {
        "expected": expected,
        "actual": actual,
        "tolerance": {"abs": abs_tol, "rel": rel_tol},
        "value_match": value_match,
        "unit_match": unit_match,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def _normalize_scalar(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"value": None, "unit": None}

    value = _as_float(payload.get("value"))
    unit = _normalize_unit(_as_clean_str(payload.get("unit")))
    return {
        "value": value,
        "unit": unit,
    }


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    cleaned = unit.strip().lower()
    if not cleaned:
        return None
    aliases = {
        "post": "posts",
        "posts": "posts",
        "count": "posts",
        "counts": "posts",
        "%": "percent",
        "percentage": "percent",
        "percent": "percent",
    }
    return aliases.get(cleaned, cleaned)


def _units_compatible(expected_unit: str | None, actual_unit: str | None) -> bool:
    if expected_unit is None:
        return True
    if actual_unit is None:
        return False
    return expected_unit == actual_unit


def _evaluate_series(context: EvaluationContext) -> tuple[int, dict[str, Any]]:
    expected = _normalize_series(context.ground_truth.answer_payload)
    actual = _normalize_series(context.final_answer.answer_payload)

    abs_tol = _get_float_config(
        context.metadata,
        keys=("series_abs_tolerance", "abs_tolerance"),
        default=_SERIES_ABS_TOLERANCE_DEFAULT,
    )
    rel_tol = _get_float_config(
        context.metadata,
        keys=("series_rel_tolerance", "rel_tolerance"),
        default=_SERIES_REL_TOLERANCE_DEFAULT,
    )

    # Series scoring is deterministic over structure (granularity/order/length) and values.
    mismatches: list[str] = []

    if expected["mode"] == "timeseries":
        expected_granularity = _as_clean_str(expected.get("granularity"))
        actual_granularity = _as_clean_str(actual.get("granularity"))
        if (
            expected_granularity
            and actual_granularity
            and _normalized_token(expected_granularity) != _normalized_token(actual_granularity)
        ):
            mismatches.append(
                f"granularity mismatch: expected '{expected_granularity}', got '{actual_granularity}'"
            )

    if expected["mode"] == "chart":
        expected_chart_type = _as_clean_str(expected.get("chart_type"))
        actual_chart_type = _as_clean_str(actual.get("chart_type"))
        if (
            expected_chart_type
            and actual_chart_type
            and _normalized_token(expected_chart_type) != _normalized_token(actual_chart_type)
        ):
            mismatches.append(
                f"chart_type mismatch: expected '{expected_chart_type}', got '{actual_chart_type}'"
            )

    expected_series = expected["series"]
    actual_series = actual["series"]
    if len(expected_series) != len(actual_series):
        mismatches.append(
            f"series count mismatch: expected {len(expected_series)}, got {len(actual_series)}"
        )

    series_mapping = _align_series(expected_series, actual_series, abs_tol=abs_tol, rel_tol=rel_tol)
    mapping_by_expected = {expected_index: actual_index for expected_index, actual_index in series_mapping}

    for series_index, expected_item in enumerate(expected_series):
        actual_index = mapping_by_expected.get(series_index)
        if actual_index is None:
            mismatches.append(f"series[{series_index}] missing from actual payload")
            continue
        actual_item = actual_series[actual_index]
        mismatches.extend(
            _compare_series_points(
                expected_item,
                actual_item,
                series_index=series_index,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        )

    score = int(len(mismatches) == 0)
    return score, {
        "expected_mode": expected["mode"],
        "actual_mode": actual["mode"],
        "tolerance": {"abs": abs_tol, "rel": rel_tol},
        "series_mapping": [
            {"expected_index": expected_index, "actual_index": actual_index}
            for expected_index, actual_index in series_mapping
        ],
        "unmatched_actual_series": [
            actual_index
            for actual_index in range(len(actual_series))
            if actual_index not in {mapped_actual for _, mapped_actual in series_mapping}
        ],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
    }


def _align_series(
    expected_series: list[dict[str, Any]],
    actual_series: list[dict[str, Any]],
    *,
    abs_tol: float,
    rel_tol: float,
) -> list[tuple[int, int]]:
    pair_summaries: dict[tuple[int, int], dict[str, Any]] = {}
    for expected_index, expected_item in enumerate(expected_series):
        for actual_index, actual_item in enumerate(actual_series):
            pair_summaries[(expected_index, actual_index)] = _summarize_series_pair(
                expected_item,
                actual_item,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )

    @lru_cache(maxsize=None)
    def _search(expected_index: int, used_mask: int) -> tuple[tuple[Any, ...], tuple[tuple[int, int], ...]]:
        if expected_index >= len(expected_series):
            unmatched_actual = len(actual_series) - used_mask.bit_count()
            return (unmatched_actual, 0, 0, 0, 0, 0.0), ()

        best_cost: tuple[Any, ...] | None = None
        best_mapping: tuple[tuple[int, int], ...] = ()

        skip_cost, skip_mapping = _search(expected_index + 1, used_mask)
        skip_total = (skip_cost[0] + 1, *skip_cost[1:])
        best_cost = skip_total
        best_mapping = skip_mapping

        for actual_index in range(len(actual_series)):
            bit = 1 << actual_index
            if used_mask & bit:
                continue
            pair_summary = pair_summaries[(expected_index, actual_index)]
            remaining_cost, remaining_mapping = _search(expected_index + 1, used_mask | bit)
            total_cost = (
                remaining_cost[0],
                remaining_cost[1] + pair_summary["point_count_delta"],
                remaining_cost[2] + pair_summary["x_mismatch_count"],
                remaining_cost[3] + pair_summary["value_issue_count"],
                remaining_cost[4] + pair_summary["label_mismatch"],
                remaining_cost[5] + pair_summary["abs_error_sum"],
            )
            candidate_mapping = ((expected_index, actual_index),) + remaining_mapping
            if best_cost is None or total_cost < best_cost:
                best_cost = total_cost
                best_mapping = candidate_mapping

        return best_cost or (0, 0, 0, 0, 0, 0.0), best_mapping

    _, mapping = _search(0, 0)
    return list(mapping)


def _summarize_series_pair(
    expected_item: Mapping[str, Any],
    actual_item: Mapping[str, Any],
    *,
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    expected_points = expected_item["points"]
    actual_points = actual_item["points"]
    compare_count = min(len(expected_points), len(actual_points))

    x_mismatch_count = 0
    value_issue_count = 0
    abs_error_sum = 0.0

    for point_index in range(compare_count):
        expected_point = expected_points[point_index]
        actual_point = actual_points[point_index]
        if expected_point["x"] != actual_point["x"]:
            x_mismatch_count += 1

        expected_value = expected_point["value"]
        actual_value = actual_point["value"]
        if expected_value is None or actual_value is None:
            value_issue_count += 1
            continue

        abs_error_sum += abs(actual_value - expected_value)
        if not math.isclose(actual_value, expected_value, abs_tol=abs_tol, rel_tol=rel_tol):
            value_issue_count += 1

    expected_label = _normalized_token(_as_clean_str(expected_item.get("label")))
    actual_label = _normalized_token(_as_clean_str(actual_item.get("label")))
    label_mismatch = int(
        expected_label is not None and actual_label is not None and expected_label != actual_label
    )

    return {
        "point_count_delta": abs(len(expected_points) - len(actual_points)),
        "x_mismatch_count": x_mismatch_count,
        "value_issue_count": value_issue_count,
        "label_mismatch": label_mismatch,
        "abs_error_sum": abs_error_sum,
    }


def _compare_series_points(
    expected_item: Mapping[str, Any],
    actual_item: Mapping[str, Any],
    *,
    series_index: int,
    abs_tol: float,
    rel_tol: float,
) -> list[str]:
    mismatches: list[str] = []
    expected_points = expected_item["points"]
    actual_points = actual_item["points"]

    if len(expected_points) != len(actual_points):
        mismatches.append(
            f"series[{series_index}] point count mismatch: expected {len(expected_points)}, got {len(actual_points)}"
        )

    compare_count = min(len(expected_points), len(actual_points))
    for point_index in range(compare_count):
        expected_point = expected_points[point_index]
        actual_point = actual_points[point_index]
        if expected_point["x"] != actual_point["x"]:
            mismatches.append(
                f"series[{series_index}] ordering/x mismatch at point[{point_index}]: "
                f"expected '{expected_point['x']}', got '{actual_point['x']}'"
            )

        expected_value = expected_point["value"]
        actual_value = actual_point["value"]
        if expected_value is None or actual_value is None:
            mismatches.append(f"series[{series_index}] value missing at point[{point_index}]")
            continue

        if not math.isclose(actual_value, expected_value, abs_tol=abs_tol, rel_tol=rel_tol):
            mismatches.append(
                f"series[{series_index}] value mismatch at point[{point_index}]: "
                f"expected {expected_value}, got {actual_value}"
            )

    return mismatches


def _normalize_series(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"mode": "unknown", "series": []}

    if "series" in payload:
        series: list[dict[str, Any]] = []
        for raw_series in payload.get("series") or []:
            if not isinstance(raw_series, Mapping):
                continue
            points = _normalize_points(raw_series.get("points"))
            series.append(
                {
                    "label": _as_clean_str(raw_series.get("label")),
                    "points": points,
                }
            )
        return {
            "mode": "timeseries",
            "granularity": _as_clean_str(payload.get("granularity")),
            "x_axis_type": _as_clean_str(payload.get("x_axis_type")),
            "series": series,
        }

    labels = payload.get("labels")
    datasets = payload.get("datasets")
    if isinstance(labels, list) and isinstance(datasets, list):
        label_values = [str(value) for value in labels]
        series_items: list[dict[str, Any]] = []
        for dataset in datasets:
            if not isinstance(dataset, Mapping):
                continue
            data_values = dataset.get("data")
            if not isinstance(data_values, list):
                continue
            points: list[dict[str, Any]] = []
            for index, label in enumerate(label_values):
                if index >= len(data_values):
                    break
                points.append(
                    {
                        "x": label,
                        "value": _as_float(data_values[index]),
                    }
                )
            series_items.append(
                {
                    "label": _as_clean_str(dataset.get("label")),
                    "points": points,
                }
            )
        return {
            "mode": "chart",
            "chart_type": _as_clean_str(payload.get("chart_type")),
            "series": series_items,
        }

    return {"mode": "unknown", "series": []}


def _normalize_points(points_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(points_payload, list):
        return []
    points: list[dict[str, Any]] = []
    for item in points_payload:
        if not isinstance(item, Mapping):
            continue
        x_value = item.get("x")
        if x_value is None:
            continue
        points.append(
            {
                "x": str(x_value),
                "value": _as_float(item.get("value")),
            }
        )
    return points


async def _evaluate_text(context: EvaluationContext) -> tuple[int, dict[str, Any]]:
    # Text scoring is semantic and delegated to a judge model; still collapsed to 0/1.
    ground_truth = _normalize_text_payload(context.ground_truth.answer_payload)
    agent_answer = _normalize_text_payload(context.final_answer.answer_payload)

    judge_model = context.model_config.judge_model
    if not judge_model:
        return 0, {
            "reason": "judge model is not configured",
            "expected_text": ground_truth,
            "actual_text": agent_answer,
        }

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=context.model_config.openrouter_api_key,
    )

    system_prompt = Evaluator_PROMPT
    user_payload = {
        "task": context.prompt,
        "ground_truth": ground_truth,
        "agent_answer": agent_answer,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    completion_kwargs = _build_openrouter_completion_kwargs(
        model=judge_model,
        messages=messages,
        openrouter_params=context.model_config.judge_openrouter_params,
    )

    try:
        response = await asyncio.to_thread(client.chat.completions.create, **completion_kwargs)
        content = (response.choices[0].message.content or "").strip() if response.choices else ""
        parsed, parse_error, parser_used = _parse_any_structured(_strip_code_fences(content))
        if not isinstance(parsed, Mapping):
            return 0, {
                "reason": "judge returned non-object response",
                "judge_raw_response": content,
                "judge_parse_error": parse_error,
            }

        raw_rating = parsed.get("score")
        # collapse to binary score: only perfect (2) is passing, everything else is failing but we keep the granularity in diagnostics.
        score = 1 if str(raw_rating).strip() == "2" else 0
        return score, {
            "raw_rating": raw_rating,
            "reason": parsed.get("reason") or "no reason provided",
            "judge_raw_response": content,
            "expected_text": ground_truth,
            "actual_text": agent_answer,
        }
    except Exception as exc:
        return 0, {
            "reason": "judge evaluation failed",
            "judge_model": judge_model,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "expected_text": ground_truth,
            "actual_text": agent_answer,
        }


def _normalize_text_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):


        raw_summary = summary = _as_clean_str(payload.get("summary"))
        raw_key_points = payload.get("key_points")
        if not isinstance(raw_key_points, list):
            raw_key_points = []

        # Normalize fact and key_point elements to non-empty strings.
        def _normalize_list_items(items: Any) -> list[str]:
            if not isinstance(items, list):
                return []
            out: list[str] = []
            for it in items:
                if it is None:
                    continue
                s = str(it).strip()
                if s:
                    out.append(s)
            return out

        key_points = _normalize_list_items(raw_key_points)
        

        return {
            "summary": summary,
            "key_points": key_points,
        }

    if payload is None:
        return {
            "summary": None,
            "key_points": [],
        }

    return {
        "summary": str(payload),
        "key_points": [],
    }


def _build_openrouter_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    openrouter_params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = dict(openrouter_params or {})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    extra_body: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if key in _DIRECT_OPENROUTER_KWARGS:
            kwargs[key] = value
        else:
            extra_body[key] = value
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _as_clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_token(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower() or None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except Exception:
            return None
    return None


def _get_float_config(
    metadata: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    default: float,
) -> float:
    for key in keys:
        if key not in metadata:
            continue
        value = _as_float(metadata.get(key))
        if value is not None:
            return value
    return default
