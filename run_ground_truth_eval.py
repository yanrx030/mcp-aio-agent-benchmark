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

_REPORT_FILENAME = "task_run_report.json"
_SIMPLE_REPORT_FILENAME = "task_run_report_simple.json"
_DEFAULT_BATCH_GLOB = "*/*/task_run_report.json"
_EXCLUDED_REPORT_ROOT_NAMES = frozenset({"results"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run ground-truth evaluation against existing task_run_report JSON files "
            "without re-running the tool-using agent."
        )
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Path to a single existing task_run_report.json file.",
    )
    parser.add_argument(
        "--reports-root",
        default=None,
        help="Root directory for batch discovery of production reports.",
    )
    parser.add_argument(
        "--report-glob",
        default=_DEFAULT_BATCH_GLOB,
        help=(
            "Glob pattern, relative to --reports-root, used to discover full report files. "
            f"Default: {_DEFAULT_BATCH_GLOB}"
        ),
    )
    parser.add_argument(
        "--task-file",
        default=None,
        help="Benchmark CSV path. Defaults to source_csv from each report file.",
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
        help="Maximum number of concurrent ground-truth evaluations per report.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path for single-report mode. Defaults to <report_dir>/new_task_run_report.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite each source task_run_report.json in place.",
    )
    parser.add_argument(
        "--rewrite-simple-report",
        action="store_true",
        help="Also rewrite the corresponding task_run_report_simple.json file.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    report_paths = _resolve_report_paths(args)

    if args.overwrite and args.output:
        raise RuntimeError("Do not combine --overwrite with --output.")
    if args.output and len(report_paths) != 1:
        raise RuntimeError("--output can only be used when exactly one report is selected.")

    judge_model, judge_openrouter_params, config_source = _resolve_judge_config(
        manifest_path=Path(args.manifest),
        profile=args.profile,
        explicit_judge_model=args.judge_model,
    )
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    print(
        f"Selected {len(report_paths)} report(s). "
        f"Using judge: {judge_model or 'none'}, parameters: {judge_openrouter_params}, source: {config_source}"
    )

    task_context_cache: dict[Path, tuple[list[BenchmarkTask], dict[str, BenchmarkTask], set[str]]] = {}
    total_rerun_count = 0

    for index, report_path in enumerate(report_paths, start=1):
        print(f"[report {index}/{len(report_paths)}] loading {report_path}")
        report_payload = _load_json_object(report_path)
        report_results = report_payload.get("results")
        if not isinstance(report_results, list):
            raise RuntimeError(f"Report file must contain a top-level 'results' array: {report_path}")

        task_file = _resolve_task_file(args.task_file, report_payload, report_path)
        task_context = task_context_cache.get(task_file)
        if task_context is None:
            task_context = _load_task_context(
                task_file=task_file,
                task_ids=set(args.task_ids or []),
                limit=args.limit,
                only_text=args.only_text,
            )
            task_context_cache[task_file] = task_context

        tasks, task_by_id, selected_task_ids = task_context
        if not selected_task_ids:
            raise RuntimeError(
                "No tasks selected. Check --task-id, --limit, --task-file, or --only-text."
            )

        source_result_by_task_id = {
            str(item.get("task_id")): item for item in report_results if isinstance(item, dict)
        }
        missing_in_report = [
            task_id for task_id in selected_task_ids if task_id not in source_result_by_task_id
        ]
        if missing_in_report:
            print(
                f"[report {index}/{len(report_paths)}] warning: selected task ids were not found and will be skipped: "
                + ", ".join(sorted(missing_in_report))
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
        total_rerun_count += rerun_count

        generated_at = _utc_now_iso()
        new_report_payload, updated_task_results = _build_full_report_payload(
            report_payload=report_payload,
            task_file=task_file,
            updated_results=updated_results,
            generated_at=generated_at,
        )

        output_path = report_path if args.overwrite else _resolve_output_path(args.output, report_path)
        _write_json_atomic(output_path, new_report_payload)
        print(
            f"[report {index}/{len(report_paths)}] re-ran {rerun_count} task(s); "
            f"wrote full report to {output_path}"
        )

        if args.rewrite_simple_report:
            simple_output_path = _resolve_simple_output_path(output_path, overwrite=args.overwrite)
            simple_payload = _build_simple_report_payload(
                full_report_payload=new_report_payload,
                updated_results=updated_results,
                generated_at=generated_at,
            )
            _write_json_atomic(simple_output_path, simple_payload)
            print(
                f"[report {index}/{len(report_paths)}] wrote simple report to {simple_output_path}"
            )

    print(f"Re-ran ground-truth evaluation for {total_rerun_count} task(s) across {len(report_paths)} report(s).")


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


def _build_full_report_payload(
    *,
    report_payload: dict[str, Any],
    task_file: Path,
    updated_results: list[Any],
    generated_at: str,
) -> tuple[dict[str, Any], list[TaskRunResult]]:
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
        "generated_at": generated_at,
        "source_csv": str(task_file),
        "judge_token_usage": judge_token_usage,
        "summary": summary,
        "results": updated_results,
        **extra_top_level_fields,
    }
    return new_report_payload, updated_task_results


def _build_simple_report_payload(
    *,
    full_report_payload: dict[str, Any],
    updated_results: list[Any],
    generated_at: str,
) -> dict[str, Any]:
    summary = full_report_payload.get("summary") if isinstance(full_report_payload.get("summary"), dict) else {}
    simple_summary = {
        "model": _clean_str(summary.get("model")) or _resolve_report_model_from_results(updated_results),
        "total_tasks": summary.get("total_tasks"),
        "finished_with_result_tasks": summary.get("finished_with_result_tasks"),
        "Task Completion Rate (TCR)": summary.get("Task Completion Rate (TCR)"),
        "average_tool_appropriateness": summary.get("average_tool_appropriateness"),
        "macro_average_parameter_schema_valid_rate(PSV)": summary.get(
            "macro_average_parameter_schema_valid_rate(PSV)"
        ),
        "macro_average_constraint_compliance_rates(CCR)": summary.get(
            "macro_average_constraint_compliance_rates(CCR)"
        ),
        "average_total_tokens": summary.get("average_total_tokens"),
        "judge_token_usage": summary.get("judge_token_usage"),
        "average_judge_total_tokens": summary.get("average_judge_total_tokens"),
        "average_total_steps(temp)": summary.get("average_total_steps(temp)"),
        "average_total_tool_calls": summary.get("average_total_tool_calls"),
        "average_latency": summary.get("average_latency"),
    }
    return {
        "generated_at": generated_at,
        "source_csv": full_report_payload.get("source_csv"),
        "judge_token_usage": full_report_payload.get("judge_token_usage"),
        "summary": simple_summary,
        "results": [
            _build_simple_result_payload(item) for item in updated_results if isinstance(item, dict)
        ],
    }


def _build_simple_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    query_trace = result.get("query_trace")
    if not isinstance(query_trace, dict):
        query_trace = {}
    validation = result.get("ground_truth_validation")
    if not isinstance(validation, dict):
        validation = {}

    return {
        "task_id": result.get("task_id"),
        "parameter_schema_valid_rate": result.get("parameter_schema_valid_rate"),
        "constraint_compliance_rate": result.get("constraint_compliance_rate"),
        "total_tokens": result.get("total_tokens"),
        "total_steps": result.get("total_steps"),
        "total_tool_calls": result.get("total_tool_calls"),
        "latency": result.get("latency"),
        "tools_used": query_trace.get("tools_used") or [],
        "passed": validation.get("passed"),
        "answer_type": result.get("answer_type"),
        "final_answer": query_trace.get("final_answer"),
    }


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


def _load_task_context(
    *,
    task_file: Path,
    task_ids: set[str],
    limit: int | None,
    only_text: bool,
) -> tuple[list[BenchmarkTask], dict[str, BenchmarkTask], set[str]]:
    tasks = load_benchmark_tasks(task_file)
    selected_tasks = select_tasks(tasks, task_ids=task_ids, limit=limit)
    if only_text:
        selected_tasks = [
            task for task in selected_tasks if normalize_answer_type(task.answer_type) == "text"
        ]
    task_by_id = {task.task_id: task for task in tasks}
    selected_task_ids = {task.task_id for task in selected_tasks}
    return tasks, task_by_id, selected_task_ids


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


def _resolve_report_paths(args: argparse.Namespace) -> list[Path]:
    report_file = _clean_str(args.report_file)
    reports_root = _clean_str(args.reports_root)

    if bool(report_file) == bool(reports_root):
        raise RuntimeError("Pass exactly one of --report-file or --reports-root.")

    if report_file:
        return [Path(report_file)]

    root = Path(reports_root or "")
    if not root.exists():
        raise RuntimeError(f"Reports root does not exist: {root}")

    report_paths: list[Path] = []
    for candidate in root.glob(args.report_glob):
        if not candidate.is_file():
            continue
        if candidate.name != _REPORT_FILENAME:
            continue
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) != 3:
            continue
        if relative.parts[0] in _EXCLUDED_REPORT_ROOT_NAMES:
            continue
        report_paths.append(candidate)

    report_paths = sorted(set(report_paths))
    if not report_paths:
        raise RuntimeError(
            f"No production reports matched glob {args.report_glob!r} under {root}"
        )
    return report_paths


def _resolve_output_path(output_arg: str | None, report_path: Path) -> Path:
    if output_arg:
        return Path(output_arg)
    return report_path.parent / "new_task_run_report.json"


def _resolve_simple_output_path(full_output_path: Path, *, overwrite: bool) -> Path:
    if overwrite:
        return full_output_path.parent / _SIMPLE_REPORT_FILENAME
    if full_output_path.name == "new_task_run_report.json":
        return full_output_path.parent / "new_task_run_report_simple.json"
    return full_output_path.with_name(f"{full_output_path.stem}_simple{full_output_path.suffix}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _resolve_report_model_from_results(results: list[Any]) -> str | None:
    for result in results:
        if not isinstance(result, dict):
            continue
        query_trace = result.get("query_trace")
        if not isinstance(query_trace, dict):
            continue
        model = _clean_str(query_trace.get("model"))
        if model:
            return model
    return None


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
