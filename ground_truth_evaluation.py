from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml
from openai import OpenAI

from prompts import normalize_answer_type


_JSON_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_DIRECT_OPENROUTER_KWARGS = {
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stream",
}
_SERIES_ABS_TOLERANCE_DEFAULT = 0.0
_SERIES_REL_TOLERANCE_DEFAULT = 0.0
_SCALAR_ABS_TOLERANCE_DEFAULT = 0.0
_SCALAR_REL_TOLERANCE_DEFAULT = 0.0


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
    match = _JSON_CODE_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def _parse_any_structured(raw: str | None) -> tuple[Any | None, str | None, str | None]:
    if raw is None:
        return None, "missing payload", None

    try:
        return json.loads(raw), None, "json"
    except Exception as json_exc:
        json_error = f"{type(json_exc).__name__}: {json_exc}"

    try:
        return yaml.safe_load(raw), None, "yaml"
    except Exception as yaml_exc:
        yaml_error = f"{type(yaml_exc).__name__}: {yaml_exc}"

    return None, f"json_parse={json_error}; yaml_parse={yaml_error}", None


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

    for series_index, expected_item in enumerate(expected_series):
        if series_index >= len(actual_series):
            break
        actual_item = actual_series[series_index]
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
                mismatches.append(
                    f"series[{series_index}] value missing at point[{point_index}]"
                )
                continue

            if not math.isclose(actual_value, expected_value, abs_tol=abs_tol, rel_tol=rel_tol):
                mismatches.append(
                    f"series[{series_index}] value mismatch at point[{point_index}]: "
                    f"expected {expected_value}, got {actual_value}"
                )

    score = int(len(mismatches) == 0)
    return score, {
        "expected_mode": expected["mode"],
        "actual_mode": actual["mode"],
        "tolerance": {"abs": abs_tol, "rel": rel_tol},
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
    }


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
    expected_text = _normalize_text_payload(context.ground_truth.answer_payload)
    actual_text = _normalize_text_payload(context.final_answer.answer_payload)

    judge_model = context.model_config.judge_model
    if not judge_model:
        return 0, {
            "reason": "judge model is not configured",
            "expected_text": expected_text,
            "actual_text": actual_text,
        }

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=context.model_config.openrouter_api_key,
    )

    system_prompt = (
        "You are a strict semantic evaluator. Compare candidate answer with reference facts. "
        "Return ONLY valid JSON object: "
        '{"score":0|1,"verdict":"string","matched_facts":["string"],"missing_or_incorrect":["string"]}. '
        "Score 1 only if candidate answer is materially correct and not contradictory."
    )
    user_payload = {
        "task_id": context.task_id,
        "task_prompt": context.prompt,
        "reference": expected_text,
        "candidate": actual_text,
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

        raw_score = parsed.get("score")
        score = 1 if str(raw_score).strip() == "1" else 0
        return score, {
            "judge_model": judge_model,
            "judge_parser_used": parser_used,
            "judge_verdict": _as_clean_str(parsed.get("verdict")),
            "judge_matched_facts": parsed.get("matched_facts"),
            "judge_missing_or_incorrect": parsed.get("missing_or_incorrect"),
            "expected_text": expected_text,
            "actual_text": actual_text,
        }
    except Exception as exc:
        return 0, {
            "reason": "judge evaluation failed",
            "judge_model": judge_model,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "expected_text": expected_text,
            "actual_text": actual_text,
        }


def _normalize_text_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        reference_answer = None
        for key in ("reference answer", "reference_answer", "ref_answer", "summary"):
            reference_answer = _as_clean_str(payload.get(key))
            if reference_answer:
                break

        facts = payload.get("facts")
        if not isinstance(facts, list):
            facts = []

        summary = _as_clean_str(payload.get("summary"))
        key_points = payload.get("key_points")
        if not isinstance(key_points, list):
            key_points = []

        return {
            "reference_answer": reference_answer,
            "facts": facts,
            "summary": summary,
            "key_points": key_points,
            "raw_payload": payload,
        }

    if payload is None:
        return {
            "reference_answer": None,
            "facts": [],
            "summary": None,
            "key_points": [],
            "raw_payload": None,
        }

    return {
        "reference_answer": None,
        "facts": [],
        "summary": str(payload),
        "key_points": [],
        "raw_payload": payload,
    }


def _build_openrouter_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    openrouter_params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = dict(openrouter_params or {})
    params.setdefault("temperature", 0.0)
    params.pop("tool_choice", None)
    params.pop("parallel_tool_calls", None)

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
