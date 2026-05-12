from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ground_truth_evaluation import (
    build_evaluation_context,
    build_task_failure_evaluation,
    evaluate_ground_truth,
)
from manifest_loader import resolve_runner_manifest
from prompts import normalize_answer_type
from task_runner import (
    BenchmarkTask,
    TaskRunResult,
    load_benchmark_tasks,
    select_tasks,
    summarize_judge_token_usage,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run ground-truth evaluation against an existing task_run_report.json "
            "and write a new task_run_report-style JSON file."
        )
    )
    parser.add_argument(
        "--report-file",
        required=True,
        help="Path to an existing task_run_report.json file.",
    )
    parser.add_argument(
        "--task-file",
        default=None,
        help="Benchmark CSV path. Defaults to source_csv from the report file.",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.yaml",
        help="Runner manifest YAML path for resolving judge OpenRouter settings.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Manifest profile to use when resolving judge settings.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override the judge model id used for text-task evaluation.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Re-evaluate only the specified task id. Repeat to select multiple tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Re-evaluate only the first N selected tasks.",
    )
    parser.add_argument(
        "--only-text",
        action="store_true",
        help="Re-evaluate only text tasks.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum number of concurrent ground-truth evaluations.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <report_dir>/new_task_run_report.json.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    report_path = Path(args.report_file)
    report_payload = _load_json_object(report_path)
    report_results = report_payload.get("results")
    if not isinstance(report_results, list):
        raise RuntimeError("Report file must contain a top-level 'results' array.")

    task_file = _resolve_task_file(args.task_file, report_payload, report_path)
    tasks = load_benchmark_tasks(task_file)
    selected_tasks = select_tasks(tasks, task_ids=set(args.task_ids or []), limit=args.limit)
    if args.only_text:
        selected_tasks = [
            task for task in selected_tasks if normalize_answer_type(task.answer_type) == "text"
        ]
    if not selected_tasks:
        raise RuntimeError("No tasks selected. Check --task-id, --limit, --task-file, or --only-text.")

    task_by_id = {task.task_id: task for task in tasks}
    selected_task_ids = {task.task_id for task in selected_tasks}
    source_result_by_task_id = {
        str(item.get("task_id")): item for item in report_results if isinstance(item, dict)
    }
    missing_in_report = [task_id for task_id in selected_task_ids if task_id not in source_result_by_task_id]
    if missing_in_report:
        print(
            "Warning: selected task ids were not found in the report and will be skipped: "
            + ", ".join(sorted(missing_in_report))
        )

    judge_model, judge_openrouter_params, config_source = _resolve_judge_config(
        manifest_path=Path(args.manifest),
        profile=args.profile,
        explicit_judge_model=args.judge_model,
    )
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    print(
        f"Loaded {len(tasks)} task(s) from {task_file}. "
        f"Selected {len(selected_task_ids)} task(s) for re-evaluation."
    )
    print(
        f"Using judge: {judge_model or 'none'}, "
        f"parameters: {judge_openrouter_params}, source: {config_source}"
    )

    updated_results = copy.deepcopy(report_results)
    rerun_count = await _rerun_ground_truth_validation(
        results=updated_results,
        task_by_id=task_by_id,
        selected_task_ids=selected_task_ids,
        openrouter_api_key=openrouter_api_key,
        judge_model=judge_model,
        judge_openrouter_params=judge_openrouter_params,
        max_concurrency=args.max_concurrency,
    )

    updated_task_results = [
        _task_run_result_from_payload(item)
        for item in updated_results
        if isinstance(item, dict)
    ]
    summary = summarize_results(updated_task_results)
    summary = {**summary, **_extract_summary_extras(report_payload.get("summary"), summary)}
    judge_token_usage = summarize_judge_token_usage(updated_task_results)

    extra_top_level_fields = {
        key: value
        for key, value in report_payload.items()
        if key not in {"generated_at", "source_csv", "judge_token_usage", "summary", "results"}
    }
    new_report_payload = {
        "generated_at": _utc_now_iso(),
        "source_csv": report_payload.get("source_csv") or str(task_file),
        "judge_token_usage": judge_token_usage,
        "summary": summary,
        "results": updated_results,
        **extra_top_level_fields,
    }

    output_path = _resolve_output_path(args.output, report_path)
    output_path.write_text(
        json.dumps(new_report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Re-ran ground-truth evaluation for {rerun_count} task(s).")
    print(f"Wrote task_run_report-style output to {output_path}")


async def _rerun_ground_truth_validation(
    *,
    results: list[Any],
    task_by_id: dict[str, BenchmarkTask],
    selected_task_ids: set[str],
    openrouter_api_key: str,
    judge_model: str | None,
    judge_openrouter_params: dict[str, Any],
    max_concurrency: int,
) -> int:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    rerun_counter = 0
    rerun_counter_lock = asyncio.Lock()

    async def run_single(result: dict[str, Any]) -> None:
        nonlocal rerun_counter

        task_id = _clean_str(result.get("task_id"))
        if not task_id or task_id not in selected_task_ids:
            return

        task = task_by_id.get(task_id)
        if task is None:
            print(f"[task {task_id}] skipped: task not found in benchmark CSV")
            return

        status = _clean_str(result.get("status"))
        if status != "completed":
            print(f"[task {task_id}] counting source failure: status is {status or 'missing'}")
            result["ground_truth_validation"] = _build_failure_validation(
                task,
                openrouter_api_key=openrouter_api_key,
                judge_model=judge_model,
                judge_openrouter_params=judge_openrouter_params,
                reason="source result failed before final answer was produced",
                failure_category="task_execution_failed",
                error_type=_clean_str(result.get("error_type")),
                error_message=_clean_str(result.get("error_message")),
                source_status=status or "missing",
            )
            async with rerun_counter_lock:
                rerun_counter += 1
            return

        query_trace = result.get("query_trace")
        if not isinstance(query_trace, dict):
            print(f"[task {task_id}] counting source failure: missing query_trace")
            result["ground_truth_validation"] = _build_failure_validation(
                task,
                openrouter_api_key=openrouter_api_key,
                judge_model=judge_model,
                judge_openrouter_params=judge_openrouter_params,
                reason="source result does not contain query_trace",
                failure_category="missing_query_trace",
                source_status=status or "completed",
            )
            async with rerun_counter_lock:
                rerun_counter += 1
            return

        final_answer_raw = _coerce_raw_payload(query_trace.get("final_answer"))
        if not final_answer_raw:
            print(f"[task {task_id}] counting source failure: missing final_answer")
            result["ground_truth_validation"] = _build_failure_validation(
                task,
                openrouter_api_key=openrouter_api_key,
                judge_model=judge_model,
                judge_openrouter_params=judge_openrouter_params,
                reason="source result does not contain final_answer",
                failure_category="missing_final_answer",
                source_status=status or "completed",
            )
            async with rerun_counter_lock:
                rerun_counter += 1
            return

        async with semaphore:
            print(f"[task {task_id}] re-running ground-truth evaluation")
            context = build_evaluation_context(
                task_id=task.task_id,
                prompt=task.prompt,
                answer_type=task.answer_type,
                ground_truth_raw=task.ground_truth,
                final_answer_raw=final_answer_raw,
                metadata=task.metadata,
                openrouter_api_key=openrouter_api_key,
                judge_model=judge_model,
                judge_openrouter_params=judge_openrouter_params,
            )
            result["ground_truth_validation"] = await evaluate_ground_truth(context)
            async with rerun_counter_lock:
                rerun_counter += 1

    await asyncio.gather(*(run_single(item) for item in results if isinstance(item, dict)))
    return rerun_counter


def _resolve_judge_config(
    *,
    manifest_path: Path,
    profile: str,
    explicit_judge_model: str | None,
) -> tuple[str | None, dict[str, Any], str]:
    if manifest_path.exists():
        try:
            manifest_config = resolve_runner_manifest(
                manifest_path,
                profile=profile,
                judge_model_override=explicit_judge_model,
            )
        except RuntimeError:
            if explicit_judge_model:
                return explicit_judge_model, {}, "direct"
            raise
        if manifest_config.judge:
            return (
                manifest_config.judge.model_id,
                manifest_config.judge.openrouter_params,
                f"manifest:{manifest_path}",
            )

    if explicit_judge_model:
        return explicit_judge_model, {}, "direct"

    return None, {}, "none"


def _build_failure_validation(
    task: BenchmarkTask,
    *,
    openrouter_api_key: str,
    judge_model: str | None,
    judge_openrouter_params: dict[str, Any],
    reason: str,
    failure_category: str,
    error_type: str | None = None,
    error_message: str | None = None,
    source_status: str | None = None,
) -> dict[str, Any]:
    context = build_evaluation_context(
        task_id=task.task_id,
        prompt=task.prompt,
        answer_type=task.answer_type,
        ground_truth_raw=task.ground_truth,
        final_answer_raw=None,
        metadata=task.metadata,
        openrouter_api_key=openrouter_api_key,
        judge_model=judge_model,
        judge_openrouter_params=judge_openrouter_params,
    )
    return build_task_failure_evaluation(
        context,
        reason=reason,
        failure_category=failure_category,
        error_type=error_type,
        error_message=error_message,
        source_status=source_status,
    )


def _extract_summary_extras(
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


def _task_run_result_from_payload(payload: dict[str, Any]) -> TaskRunResult:
    field_names = {field.name for field in fields(TaskRunResult)}
    filtered = {key: value for key, value in payload.items() if key in field_names}

    required_defaults = {
        "task_id": _clean_str(payload.get("task_id")) or "",
        "prompt": _clean_str(payload.get("prompt")) or "",
        "status": _clean_str(payload.get("status")) or "unknown",
        "started_at": _clean_str(payload.get("started_at")) or "",
        "finished_at": _clean_str(payload.get("finished_at")) or "",
    }
    for key, value in required_defaults.items():
        filtered.setdefault(key, value)

    return TaskRunResult(**filtered)


def _resolve_task_file(
    cli_task_file: str | None,
    report_payload: dict[str, Any],
    report_path: Path,
) -> Path:
    if cli_task_file:
        return Path(cli_task_file)

    source_csv = report_payload.get("source_csv")
    if not isinstance(source_csv, str) or not source_csv.strip():
        raise RuntimeError(
            "Task file was not provided and the report does not contain a usable 'source_csv' value."
        )

    candidate = Path(source_csv)
    if candidate.exists():
        return candidate

    sibling_candidate = report_path.parent / source_csv
    if sibling_candidate.exists():
        return sibling_candidate

    raise RuntimeError(
        f"Unable to resolve task CSV from report source_csv={source_csv!r}. "
        "Pass --task-file explicitly."
    )


def _resolve_output_path(output_arg: str | None, report_path: Path) -> Path:
    if output_arg:
        return Path(output_arg)
    return report_path.parent / "new_task_run_report.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _coerce_raw_payload(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return json.dumps(value, ensure_ascii=False)


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    asyncio.run(main())
