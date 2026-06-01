from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


FULL_REPORT_FILENAME = "task_run_report.json"
SIMPLE_REPORT_FILENAME = "task_run_report_simple.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate task_run_report.json and task_run_report_simple.json summaries "
            "from the current per-task report content."
        )
    )
    parser.add_argument(
        "run_dir",
        help=(
            "Run directory containing task_run_report.json and task_run_report_simple.json, "
            "or the path to task_run_report.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the old and new summaries without writing files.",
    )
    parser.add_argument(
        "--keep-generated-at",
        action="store_true",
        help="Leave generated_at unchanged instead of setting it to the current UTC time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_report_path = resolve_full_report_path(Path(args.run_dir))
    simple_report_path = full_report_path.with_name(SIMPLE_REPORT_FILENAME)

    full_payload = load_json_object(full_report_path)
    results_payload = full_payload.get("results")
    if not isinstance(results_payload, list):
        raise RuntimeError(f"Expected list field 'results' in {full_report_path}")

    if not all(isinstance(item, dict) for item in results_payload):
        raise RuntimeError(
            f"{full_report_path} contains non-object entries in 'results'; refusing to summarize."
        )
    task_results = list(results_payload)

    recalculated_summary = summarize_results(task_results)
    full_summary = {
        **recalculated_summary,
        **extract_summary_extras(full_payload.get("summary"), recalculated_summary),
    }
    judge_token_usage = summarize_judge_token_usage(task_results)
    generated_at = (
        full_payload.get("generated_at") if args.keep_generated_at else utc_now_iso()
    )

    updated_full_payload = {
        **full_payload,
        "generated_at": generated_at,
        "judge_token_usage": judge_token_usage,
        "summary": full_summary,
    }

    updated_simple_payload = None
    if simple_report_path.exists():
        simple_payload = load_json_object(simple_report_path)
        updated_simple_payload = {
            **simple_payload,
            "generated_at": (
                simple_payload.get("generated_at") if args.keep_generated_at else generated_at
            ),
            "source_csv": updated_full_payload.get("source_csv"),
            "judge_token_usage": judge_token_usage,
            "summary": build_simple_summary(full_summary),
        }

    print_summary_delta("full", full_payload.get("summary"), updated_full_payload["summary"])
    if updated_simple_payload is not None:
        simple_payload = load_json_object(simple_report_path)
        print_summary_delta("simple", simple_payload.get("summary"), updated_simple_payload["summary"])
    else:
        print(f"simple: skipped; {simple_report_path} does not exist")

    if args.dry_run:
        print("dry-run: no files written")
        return

    write_json_atomic(full_report_path, updated_full_payload)
    if updated_simple_payload is not None:
        write_json_atomic(simple_report_path, updated_simple_payload)

    print(f"updated: {full_report_path}")
    if updated_simple_payload is not None:
        print(f"updated: {simple_report_path}")


def resolve_full_report_path(path: Path) -> Path:
    if path.is_dir():
        return path / FULL_REPORT_FILENAME
    if path.name != FULL_REPORT_FILENAME:
        raise RuntimeError(
            f"Expected a run directory or {FULL_REPORT_FILENAME}, got: {path}"
        )
    return path


def extract_summary_extras(
    original_summary: Any,
    recalculated_summary: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(original_summary, dict):
        return {}
    return {
        key: value
        for key, value in original_summary.items()
        if key not in recalculated_summary
    }


def summarize_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result_list = list(results)
    finished_with_result = sum(1 for result in result_list if result.get("status") == "completed")
    failed = sum(1 for result in result_list if result.get("status") == "failed")
    judge_token_usage = summarize_judge_token_usage(result_list)

    tool_appropriateness_scores = [
        score
        for result in result_list
        if isinstance(result.get("tool_appropriateness"), Mapping)
        for score in [result["tool_appropriateness"].get("average_score")]
        if score is not None
    ]
    ground_truth_scores = [
        int(score)
        for result in result_list
        if isinstance(result.get("ground_truth_validation"), Mapping)
        for score in [result["ground_truth_validation"].get("score")]
        if score in {0, 1, True, False}
    ]
    parameter_schema_valid_rates = non_null_values(
        result.get("parameter_schema_valid_rate") for result in result_list
    )
    constraint_compliance_rates = non_null_values(
        result.get("constraint_compliance_rate") for result in result_list
    )
    total_tokens_values = non_null_values(result.get("total_tokens") for result in result_list)
    total_steps_values = non_null_values(result.get("total_steps") for result in result_list)
    total_tool_calls_values = non_null_values(
        result.get("total_tool_calls") for result in result_list
    )
    latency_values = non_null_values(result.get("latency") for result in result_list)
    end_to_end_latency_values = non_null_values(
        result.get("end_to_end_latency") for result in result_list
    )
    attempt_counts = [
        result.get("attempt_count", 1)
        for result in result_list
        if result.get("attempt_count", 1) is not None
    ]
    retried_tasks = [
        result for result in result_list if numeric_or_default(result.get("attempt_count"), 1) > 1
    ]
    auth_refreshed_tasks = [result for result in result_list if result.get("auth_refreshed")]
    cleanup_warning_tasks = [result for result in result_list if result.get("cleanup_warning")]

    return {
        "total_tasks": len(result_list),
        "finished_with_result_tasks": finished_with_result,
        "failed_tasks": failed,
        "tasks_retried": len(retried_tasks),
        "tasks_with_auth_refresh": len(auth_refreshed_tasks),
        "tasks_with_cleanup_warning": len(cleanup_warning_tasks),
        "Evaluated tasks": len(ground_truth_scores),
        "Task Completion Rate (TCR)": (
            sum(ground_truth_scores) / len(ground_truth_scores)
            if ground_truth_scores
            else None
        ),
        "average_tool_appropriateness": average_or_none(tool_appropriateness_scores),
        "macro_average_parameter_schema_valid_rate(PSV)": average_or_none(
            parameter_schema_valid_rates
        ),
        "macro_average_constraint_compliance_rates(CCR)": average_or_none(
            constraint_compliance_rates
        ),
        "average_total_tokens": average_or_none(total_tokens_values),
        "judge_token_usage": judge_token_usage,
        "average_judge_total_tokens": (
            judge_token_usage["total_tokens"] / judge_token_usage["evaluations_with_usage"]
            if judge_token_usage["evaluations_with_usage"]
            else None
        ),
        "average_total_steps(temp)": average_or_none(total_steps_values),
        "average_total_tool_calls": average_or_none(total_tool_calls_values),
        "average_latency": average_or_none(latency_values),
        "average_end_to_end_latency": average_or_none(end_to_end_latency_values),
        "average_attempt_count": average_or_none(attempt_counts),
    }


def summarize_judge_token_usage(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "text_judge_evaluations": 0,
        "evaluations_with_usage": 0,
    }

    for result in results:
        validation = result.get("ground_truth_validation")
        if not isinstance(validation, Mapping) or validation.get("strategy") != "text_judge":
            continue

        totals["text_judge_evaluations"] += 1
        diagnostics = validation.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue

        usage = diagnostics.get("judge_token_usage")
        if not isinstance(usage, Mapping):
            continue

        has_usage = False
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if value is None:
                continue
            try:
                totals[key] += int(value)
            except (TypeError, ValueError):
                continue
            has_usage = True

        if has_usage:
            totals["evaluations_with_usage"] += 1

    return totals


def non_null_values(values: Iterable[Any]) -> list[Any]:
    return [value for value in values if value is not None]


def average_or_none(values: list[Any]) -> float | None:
    return sum(values) / len(values) if values else None


def numeric_or_default(value: Any, default: int | float) -> int | float:
    if value is None:
        return default
    return value


def build_simple_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": full_summary.get("model"),
        "total_tasks": full_summary.get("total_tasks"),
        "finished_with_result_tasks": full_summary.get("finished_with_result_tasks"),
        "Task Completion Rate (TCR)": full_summary.get("Task Completion Rate (TCR)"),
        "average_tool_appropriateness": full_summary.get("average_tool_appropriateness"),
        "macro_average_parameter_schema_valid_rate(PSV)": full_summary.get(
            "macro_average_parameter_schema_valid_rate(PSV)"
        ),
        "macro_average_constraint_compliance_rates(CCR)": full_summary.get(
            "macro_average_constraint_compliance_rates(CCR)"
        ),
        "average_total_tokens": full_summary.get("average_total_tokens"),
        "judge_token_usage": full_summary.get("judge_token_usage"),
        "average_judge_total_tokens": full_summary.get("average_judge_total_tokens"),
        "average_total_steps(temp)": full_summary.get("average_total_steps(temp)"),
        "average_total_tool_calls": full_summary.get("average_total_tool_calls"),
        "average_latency": full_summary.get("average_latency"),
    }


def print_summary_delta(label: str, old_summary: Any, new_summary: dict[str, Any]) -> None:
    old_summary = old_summary if isinstance(old_summary, dict) else {}
    changed_keys = [
        key
        for key, value in new_summary.items()
        if old_summary.get(key) != value
    ]
    print(f"{label}: {len(changed_keys)} summary fields changed")
    for key in changed_keys:
        print(f"  {key}: {old_summary.get(key)!r} -> {new_summary[key]!r}")


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    main()
