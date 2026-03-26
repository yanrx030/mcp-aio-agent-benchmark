from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping


_ALLOWED_VALUES_BY_TOOL: dict[str, dict[str, set[str]]] = {
    "aggregate_by_time": {
        "aggregation_level": {"day", "month"},
    },
    "aggregate_seasonality": {
        "aggregation_level": {"dayofweek", "hourofday"},
    },
    "generate_chart": {
        "format": {"png", "jpg", "svg", "webp"},
    },
}

_DATE_FIELDS_BY_TOOL: dict[str, tuple[str, ...]] = {
    "aggregate_by_time": ("startdate", "enddate"),
    "aggregate_seasonality": ("startdate", "enddate"),
    "analyze_terms_in_collection": ("startdate", "enddate"),
    "get_all_terms": ("startdate", "enddate"),
    "get_term_daily_counts": ("startdate", "enddate"),
    "get_nlp_terms_for_day": ("day",),
    "get_nlp_term_analysis": ("day",),
    "get_nlp_topics": ("startdate", "enddate"),
    "get_topic_groupings": ("startdate", "enddate"),
    "get_nlp_metadata": ("startdate", "enddate"),
}

_MAX_INTERVAL_DAYS_BY_TOOL: dict[str, int] = {
    "aggregate_by_time": 365,
    "aggregate_seasonality": 365,
    "analyze_terms_in_collection": 28,
    "get_all_terms": 28,
    "get_term_daily_counts": 28,
    "get_nlp_topics": 28,
    "get_topic_groupings": 28,
    "get_nlp_metadata": 28,
}

_YESTERDAY_CUTOFF_TOOLS: set[str] = {
    "aggregate_by_time",
    "aggregate_seasonality",
    "analyze_terms_in_collection",
    "get_all_terms",
    "get_term_daily_counts",
    "get_nlp_topics",
    "get_topic_groupings",
    "get_nlp_metadata",
}

_CONSTRAINED_TOOLS: set[str] = (
    set(_ALLOWED_VALUES_BY_TOOL)
    | set(_DATE_FIELDS_BY_TOOL)
    | set(_MAX_INTERVAL_DAYS_BY_TOOL)
    | set(_YESTERDAY_CUTOFF_TOOLS)
)


@dataclass(slots=True)
class ToolArgumentValidation:
    schema_valid: bool | None
    schema_errors: list[str]
    constraint_valid: bool | None


def has_explicit_constraints(tool_name: str) -> bool:
    return tool_name in _CONSTRAINED_TOOLS


def validate_tool_arguments(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    openai_tool_def: Mapping[str, Any] | None,
    reference_date: date | None = None,
) -> ToolArgumentValidation:
    effective_reference_date = reference_date or date.today()
    schema_valid, schema_errors = validate_schema(arguments, openai_tool_def)
    constraint_valid, _constraint_checks = validate_constraints(
        tool_name=tool_name,
        arguments=arguments,
        openai_tool_def=openai_tool_def,
        reference_date=effective_reference_date,
    )
    return ToolArgumentValidation(
        schema_valid=schema_valid,
        schema_errors=schema_errors,
        constraint_valid=constraint_valid,
    )


def validate_schema(
    arguments: dict[str, Any],
    openai_tool_def: Mapping[str, Any] | None,
) -> tuple[bool | None, list[str]]:
    if openai_tool_def is None:
        return None, []

    parameters = (openai_tool_def.get("function") or {}).get("parameters") or {}
    if parameters.get("type") != "object":
        return None, []

    if not isinstance(arguments, dict):
        return False, ["Tool arguments must decode to a JSON object."]

    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    errors: list[str] = []

    for field_name in required:
        if field_name not in arguments:
            errors.append(f"Missing required field '{field_name}'.")

    for field_name, value in arguments.items():
        field_schema = properties.get(field_name)
        if not field_schema:
            continue
        if not _matches_schema(field_schema, value):
            expected = _describe_schema_types(field_schema)
            actual = type(value).__name__
            errors.append(
                f"Field '{field_name}' expected {expected}, got {actual}."
            )

    return len(errors) == 0, errors


def validate_constraints(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    openai_tool_def: Mapping[str, Any] | None,
    reference_date: date,
) -> tuple[bool | None, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    for field_name, allowed_values in _ALLOWED_VALUES_BY_TOOL.get(tool_name, {}).items():
        default_value = _get_property_default(openai_tool_def, field_name)
        checks.append(
            _check_allowed_values(
                arguments=arguments,
                field_name=field_name,
                allowed_values=allowed_values,
                default_value=default_value,
            )
        )

    for field_name in _DATE_FIELDS_BY_TOOL.get(tool_name, ()):
        checks.append(_check_date_format(arguments=arguments, field_name=field_name))

    max_interval_days = _MAX_INTERVAL_DAYS_BY_TOOL.get(tool_name)
    if max_interval_days is not None:
        checks.append(
            _check_date_interval(
                arguments=arguments,
                start_key="startdate",
                end_key="enddate",
                max_days=max_interval_days,
            )
        )

    if tool_name in _YESTERDAY_CUTOFF_TOOLS:
        checks.append(
            _check_enddate_not_after_yesterday(
                arguments=arguments,
                end_key="enddate",
                reference_date=reference_date,
            )
        )

    if not checks:
        return None, []

    statuses = [check["status"] for check in checks]
    return all(status == "pass" for status in statuses), checks


def _get_property_default(
    openai_tool_def: Mapping[str, Any] | None,
    field_name: str,
) -> Any:
    if openai_tool_def is None:
        return None
    parameters = (openai_tool_def.get("function") or {}).get("parameters") or {}
    properties = parameters.get("properties") or {}
    field_schema = properties.get(field_name) or {}
    return field_schema.get("default")


def _check_allowed_values(
    *,
    arguments: dict[str, Any],
    field_name: str,
    allowed_values: set[str],
    default_value: Any,
) -> dict[str, Any]:
    value = arguments.get(field_name, default_value)
    if value is None:
        return _make_check(
            rule=f"{field_name}_allowed_values",
            status="na",
            message=f"Cannot evaluate '{field_name}' allowed values because no value was provided.",
        )
    if not isinstance(value, str):
        return _make_check(
            rule=f"{field_name}_allowed_values",
            status="fail",
            message=f"Field '{field_name}' must be a string from {sorted(allowed_values)}.",
        )
    if value not in allowed_values:
        return _make_check(
            rule=f"{field_name}_allowed_values",
            status="fail",
            message=f"Field '{field_name}' must be one of {sorted(allowed_values)}; got '{value}'.",
        )
    return _make_check(
        rule=f"{field_name}_allowed_values",
        status="pass",
        message=f"Field '{field_name}' is within the allowed values.",
    )


def _check_date_format(*, arguments: dict[str, Any], field_name: str) -> dict[str, Any]:
    if field_name not in arguments or arguments.get(field_name) is None:
        return _make_check(
            rule=f"{field_name}_date_format",
            status="na",
            message=f"Cannot evaluate '{field_name}' date format because no value was provided.",
        )
    parsed = _parse_iso_date(arguments[field_name])
    if parsed is None:
        return _make_check(
            rule=f"{field_name}_date_format",
            status="fail",
            message=f"Field '{field_name}' must use YYYY-MM-DD format.",
        )
    return _make_check(
        rule=f"{field_name}_date_format",
        status="pass",
        message=f"Field '{field_name}' uses YYYY-MM-DD format.",
    )


def _check_date_interval(
    *,
    arguments: dict[str, Any],
    start_key: str,
    end_key: str,
    max_days: int,
) -> dict[str, Any]:
    if arguments.get(start_key) is None or arguments.get(end_key) is None:
        return _make_check(
            rule="date_interval_limit",
            status="na",
            message=f"Cannot evaluate the date interval limit without both '{start_key}' and '{end_key}'.",
        )

    start_date = _parse_iso_date(arguments[start_key])
    end_date = _parse_iso_date(arguments[end_key])
    if start_date is None or end_date is None:
        return _make_check(
            rule="date_interval_limit",
            status="fail",
            message=f"Fields '{start_key}' and '{end_key}' must use YYYY-MM-DD format.",
        )
    if end_date < start_date:
        return _make_check(
            rule="date_interval_limit",
            status="fail",
            message=f"Field '{end_key}' cannot be earlier than '{start_key}'.",
        )

    inclusive_days = (end_date - start_date).days + 1
    if inclusive_days > max_days:
        return _make_check(
            rule="date_interval_limit",
            status="fail",
            message=f"Date interval is {inclusive_days} days inclusive; maximum allowed is {max_days}.",
        )
    return _make_check(
        rule="date_interval_limit",
        status="pass",
        message=f"Date interval is {inclusive_days} days inclusive, within the {max_days}-day limit.",
    )


def _check_enddate_not_after_yesterday(
    *,
    arguments: dict[str, Any],
    end_key: str,
    reference_date: date,
) -> dict[str, Any]:
    if arguments.get(end_key) is None:
        return _make_check(
            rule="enddate_not_after_yesterday",
            status="na",
            message=f"Cannot evaluate yesterday cutoff without '{end_key}'.",
        )

    end_date = _parse_iso_date(arguments[end_key])
    if end_date is None:
        return _make_check(
            rule="enddate_not_after_yesterday",
            status="fail",
            message=f"Field '{end_key}' must use YYYY-MM-DD format.",
        )

    yesterday = reference_date - timedelta(days=1)
    if end_date > yesterday:
        return _make_check(
            rule="enddate_not_after_yesterday",
            status="fail",
            message=f"Field '{end_key}' must not be later than {yesterday.isoformat()}.",
        )
    return _make_check(
        rule="enddate_not_after_yesterday",
        status="pass",
        message=f"Field '{end_key}' is not later than {yesterday.isoformat()}.",
    )


def _make_check(*, rule: str, status: str, message: str) -> dict[str, Any]:
    return {
        "rule": rule,
        "status": status,
        "message": message,
    }


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _matches_schema(schema: Mapping[str, Any], value: Any) -> bool:
    allowed_types: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        allowed_types.append(schema_type)

    for option in schema.get("anyOf") or []:
        option_type = option.get("type")
        if isinstance(option_type, str):
            allowed_types.append(option_type)

    if not allowed_types:
        return True

    return any(_matches_single_type(type_name, value) for type_name in allowed_types)


def _matches_single_type(type_name: str, value: Any) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, Mapping)
    if type_name == "null":
        return value is None
    if type_name == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, list)
    return True


def _describe_schema_types(schema: Mapping[str, Any]) -> str:
    allowed_types: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        allowed_types.append(schema_type)
    for option in schema.get("anyOf") or []:
        option_type = option.get("type")
        if isinstance(option_type, str):
            allowed_types.append(option_type)
    if not allowed_types:
        return "a valid value"
    return " or ".join(dict.fromkeys(allowed_types))
