from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from MCPClient import (
    MCPClient,
    JwtTokenManager,
    is_authentication_error,
    is_transport_error,
)
from eval_logger import JSONLLogger, utc_now_iso
from prompts import resolve_system_prompt


@dataclass(slots=True)
class BenchmarkTask:
    task_id: str
    prompt: str
    answer_type: str | None = None
    note: str | None = None
    ground_truth_tool_call_raw: str | None = None
    ground_truth_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ground_truth: str | None = None
    aux_toolset: list[str] = field(default_factory=list)
    core_toolset: list[str] = field(default_factory=list)
    category: str | None = None
    source_scope: str | None = None
    temporal_req: str | None = None
    complexity: str | None = None
    feasibility: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskRunResult:
    task_id: str
    prompt: str
    status: str
    started_at: str
    finished_at: str
    answer_type: str | None = None
    parameter_schema_valid_rate: float | None = None
    constraint_compliance_rate: float | None = None
    total_tokens: int | None = None
    total_steps: int | None = None
    total_tool_calls: int | None = None
    latency: float | None = None
    attempt_count: int = 1
    auth_refreshed: bool = False
    cleanup_warning: str | None = None
    query_trace: dict[str, Any] | None = None
    ground_truth_validation: dict[str, Any] | None = None
    tool_appropriateness: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


def load_benchmark_tasks(csv_path: str | Path) -> list[BenchmarkTask]:
    path = Path(csv_path)
    tasks: list[BenchmarkTask] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task_id = _clean_field(row.get("task_id"))
            prompt = _clean_field(row.get("prompt"))
            if not task_id or not prompt:
                continue

            raw_tool_calls = _clean_field(row.get("ref_tool_call")) or _clean_field(
                row.get("ground_truth_tool_call")
            )
            task = BenchmarkTask(
                task_id=task_id,
                prompt=prompt,
                answer_type=_clean_field(row.get("answer_type")),
                note=_clean_field(row.get("note")),
                ground_truth_tool_call_raw=raw_tool_calls,
                ground_truth_tool_calls=_parse_ground_truth_tool_calls(raw_tool_calls),
                ground_truth=_clean_field(row.get("ground_truth")),
                aux_toolset=_parse_toolset_names(row.get("aux_toolset")),
                core_toolset=_parse_toolset_names(row.get("core_toolset")),
                category=_clean_field(row.get("category")),
                source_scope=_clean_field(row.get("source_scope")),
                temporal_req=_clean_field(row.get("temporal_req")),
                complexity=_clean_field(row.get("complexity")),
                feasibility=_clean_field(row.get("feasibility")),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "task_id",
                        "prompt",
                        "answer_type",
                        "note",
                        "ref_tool_call",
                        "ground_truth_tool_call",
                        "ground_truth",
                        "aux_toolset",
                        "core_toolset",
                        "category",
                        "source_scope",
                        "temporal_req",
                        "complexity",
                        "feasibility",
                    }
                    and _clean_field(value) is not None
                },
            )
            tasks.append(task)

    return tasks


async def run_benchmark_tasks(
    tasks: Iterable[BenchmarkTask],
    *,
    auth_key: str,
    openrouter_api_key: str,
    openrouter_model: str,
    toolset_path: str,
    jwt_manager: JwtTokenManager,
    shared_logger: JSONLLogger,
    openrouter_params: dict[str, Any] | None = None,
    max_concurrency: int = 5,
    max_retries: int = 1,
) -> list[TaskRunResult]:
    task_list = list(tasks)
    if not task_list:
        return []

    concurrency = max(1, max_concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    results_by_index: list[TaskRunResult | None] = [None] * len(task_list)
    retry_attempts = max(0, max_retries)
    total_attempts = retry_attempts + 1

    async def run_single(index: int, task: BenchmarkTask) -> None:
        async with semaphore:
            started_at = utc_now_iso()
            started_perf = time.perf_counter()
            print(f"[task {task.task_id}] started")
            attempt_count = 0
            auth_refreshed = False
            cleanup_warnings: list[str] = []
            final_exc: Exception | None = None
            query_trace: dict[str, Any] | None = None
            system_prompt, resolved_answer_type = resolve_system_prompt(task.answer_type)

            for attempt in range(1, total_attempts + 1):
                attempt_count = attempt
                token = await jwt_manager.get_token()
                client = MCPClient(
                    auth_key=auth_key,
                    openrouter_api_key=openrouter_api_key,
                    openrouter_model=openrouter_model,
                    openrouter_params=openrouter_params,
                    logger=shared_logger,
                    jwt_token=token,
                )
                client.load_tools(toolset_path)
                attempt_exc: Exception | None = None

                try:
                    await client.connect_to_server()
                    query_trace = await client.process_query(
                        task.prompt,
                        task_id=task.task_id,
                        return_trace=True,
                        system_prompt=system_prompt,
                        system_prompt_label=resolved_answer_type or "default",
                    )
                except Exception as exc:
                    attempt_exc = exc
                finally:
                    cleanup_warning = await client.cleanup()
                    if cleanup_warning:
                        cleanup_warnings.append(f"attempt {attempt}: {cleanup_warning}")

                if attempt_exc is None and query_trace is not None:
                    validation = validate_ground_truth_placeholder(task, query_trace)
                    tool_appropriateness = calculate_tool_appropriateness(task, query_trace)
                    finished_at = utc_now_iso()
                    latency = time.perf_counter() - started_perf
                    cleanup_warning = " | ".join(cleanup_warnings) if cleanup_warnings else None
                    results_by_index[index] = TaskRunResult(
                        task_id=task.task_id,
                        prompt=task.prompt,
                        answer_type=task.answer_type,
                        status="completed",
                        started_at=started_at,
                        finished_at=finished_at,
                        parameter_schema_valid_rate=query_trace.get("parameter_schema_valid_rate"),
                        constraint_compliance_rate=query_trace.get("constraint_compliance_rate"),
                        total_tokens=query_trace.get("total_tokens"),
                        total_steps=query_trace.get("total_steps"),
                        total_tool_calls=query_trace.get("total_tool_calls"),
                        latency=latency,
                        attempt_count=attempt_count,
                        auth_refreshed=auth_refreshed,
                        cleanup_warning=cleanup_warning,
                        query_trace=query_trace,
                        ground_truth_validation=validation,
                        tool_appropriateness=tool_appropriateness,
                    )
                    print(f"[task {task.task_id}] completed (attempt {attempt_count})")
                    return

                final_exc = attempt_exc
                retryable = bool(attempt_exc) and (
                    is_authentication_error(attempt_exc) or is_transport_error(attempt_exc)
                )
                if retryable and attempt < total_attempts:
                    if attempt_exc and is_authentication_error(attempt_exc):
                        auth_refreshed = True
                        try:
                            await jwt_manager.refresh_if_stale_or_forced(
                                force=True,
                                failed_token=token,
                            )
                        except Exception as refresh_exc:
                            final_exc = refresh_exc
                            break

                    backoff_seconds = 0.75 * (2 ** (attempt - 1))
                    print(
                        f"[task {task.task_id}] retrying after {type(attempt_exc).__name__} "
                        f"(attempt {attempt + 1}/{total_attempts})"
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue

                break

            try:
                finished_at = utc_now_iso()
                latency = time.perf_counter() - started_perf
                cleanup_warning = " | ".join(cleanup_warnings) if cleanup_warnings else None
                results_by_index[index] = TaskRunResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    answer_type=task.answer_type,
                    status="failed",
                    started_at=started_at,
                    finished_at=finished_at,
                    latency=latency,
                    attempt_count=attempt_count,
                    auth_refreshed=auth_refreshed,
                    cleanup_warning=cleanup_warning,
                    error_type=type(final_exc).__name__ if final_exc else "RuntimeError",
                    error_message=str(final_exc) if final_exc else "Task failed without explicit exception.",
                )
                if final_exc:
                    print(f"[task {task.task_id}] failed: {type(final_exc).__name__}: {final_exc}")
                else:
                    print(f"[task {task.task_id}] failed: unknown error")
            except Exception as exc:
                # Keep worker alive even if result-construction has an unexpected error.
                finished_at = utc_now_iso()
                latency = time.perf_counter() - started_perf
                results_by_index[index] = TaskRunResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    answer_type=task.answer_type,
                    status="failed",
                    started_at=started_at,
                    finished_at=finished_at,
                    latency=latency,
                    attempt_count=attempt_count,
                    auth_refreshed=auth_refreshed,
                    cleanup_warning=" | ".join(cleanup_warnings) if cleanup_warnings else None,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                print(f"[task {task.task_id}] failed while building result: {type(exc).__name__}: {exc}")

    await asyncio.gather(
        *(run_single(index, task) for index, task in enumerate(task_list))
    )

    return [result for result in results_by_index if result is not None]


def validate_ground_truth_placeholder(
    task: BenchmarkTask,
    query_trace: dict[str, Any],
) -> dict[str, Any]:
    return {"placeholder":"TODO: implement llm as judge"}
    # expected_answer = (task.ground_truth or "").strip()
    # final_answer = str(query_trace.get("final_answer") or "").strip()
    # heuristic_match = bool(expected_answer) and expected_answer in final_answer

    # return {
    #     "status": "placeholder_match" if heuristic_match else "placeholder_review_required",
    #     "placeholder": True,
    #     "reason": (
    #         "Detailed ground-truth validation is not implemented yet. "
    #         "This placeholder only checks whether the expected ground-truth string appears in the final answer "
    #         "and records the parsed reference tool calls for future validation."
    #     ),
    #     "expected_ground_truth": task.ground_truth,
    #     "expected_tool_call_count": len(task.ground_truth_tool_calls),
    #     "expected_tool_calls": task.ground_truth_tool_calls,
    #     "heuristic_answer_contains_ground_truth": heuristic_match,
    # }



#==========================HELPER FUNCTIONS TO CALCULATE EVALUATION SCORES AND WRITE REPORT=============================


def calculate_tool_appropriateness(
    task: BenchmarkTask,
    query_trace: dict[str, Any],
) -> dict[str, Any]:
    core_tools = {_normalize_tool_name(name) for name in task.core_toolset}
    aux_tools = {_normalize_tool_name(name) for name in task.aux_toolset}

    tool_calls_path_raw = query_trace.get("tool_calls_path")
    tool_calls_path = Path(tool_calls_path_raw) if tool_calls_path_raw else None
    tool_names = _load_tool_names_from_jsonl(tool_calls_path)

    scored_calls: list[dict[str, Any]] = []
    total_score = 0.0

    for index, tool_name in enumerate(tool_names, start=1):
        normalized_name = _normalize_tool_name(tool_name)
        if normalized_name in core_tools:
            label = "core"
            score = 1.0
        elif normalized_name in aux_tools:
            label = "auxiliary"
            score = 0.5
        else:
            label = "inappropriate"
            score = 0.0

        total_score += score
        scored_calls.append(
            {
                "call_index": index,
                "tool_name": tool_name,
                "appropriateness": label,
                "score": score,
            }
        )

    call_count = len(scored_calls)
    average_score = total_score / call_count if call_count else None

    return {
        "metric": "tool_appropriateness",
        "formula": "TA = sum(call_scores) / number_of_calls",
        "core_toolset": task.core_toolset,
        "aux_toolset": task.aux_toolset,
        "tool_calls_path": str(tool_calls_path) if tool_calls_path else None,
        "tool_call_count": call_count,
        "total_score": total_score,
        "average_score": average_score,
        "scored_calls": scored_calls,
    }


def summarize_results(results: Iterable[TaskRunResult]) -> dict[str, Any]:
    result_list = list(results)
    completed = sum(1 for result in result_list if result.status == "completed")
    failed = sum(1 for result in result_list if result.status == "failed")

    tool_appropriateness_scores = [
        score
        for result in result_list
        if result.tool_appropriateness is not None
        for score in [result.tool_appropriateness.get("average_score")]
        if score is not None
    ]
    parameter_schema_valid_rates = [
        value
        for result in result_list
        for value in [result.parameter_schema_valid_rate]
        if value is not None
    ]
    constraint_compliance_rates = [
        value
        for result in result_list
        for value in [result.constraint_compliance_rate]
        if value is not None
    ]
    total_tokens_values = [
        value
        for result in result_list
        for value in [result.total_tokens]
        if value is not None
    ]
    total_steps_values = [
        value
        for result in result_list
        for value in [result.total_steps]
        if value is not None
    ]
    total_tool_calls_values = [
        value
        for result in result_list
        for value in [result.total_tool_calls]
        if value is not None
    ]
    latency_values = [
        value
        for result in result_list
        for value in [result.latency]
        if value is not None
    ]
    attempt_counts = [result.attempt_count for result in result_list if result.attempt_count is not None]
    retried_tasks = [result for result in result_list if result.attempt_count > 1]
    auth_refreshed_tasks = [result for result in result_list if result.auth_refreshed]
    cleanup_warning_tasks = [result for result in result_list if result.cleanup_warning]

    return {
        "total_tasks": len(result_list),
        "completed_tasks": completed,
        "failed_tasks": failed,
        "tasks_retried": len(retried_tasks),
        "tasks_with_auth_refresh": len(auth_refreshed_tasks),
        "tasks_with_cleanup_warning": len(cleanup_warning_tasks),
        "average_tool_appropriateness": (
            sum(tool_appropriateness_scores) / len(tool_appropriateness_scores)
            if tool_appropriateness_scores
            else None
        ),
        "average_parameter_schema_valid_rate": (
            sum(parameter_schema_valid_rates) / len(parameter_schema_valid_rates)
            if parameter_schema_valid_rates
            else None
        ),
        "average_constraint_compliance_rate": (
            sum(constraint_compliance_rates) / len(constraint_compliance_rates)
            if constraint_compliance_rates
            else None
        ),
        "average_total_tokens": (
            sum(total_tokens_values) / len(total_tokens_values)
            if total_tokens_values
            else None
        ),
        "average_total_steps": (
            sum(total_steps_values) / len(total_steps_values)
            if total_steps_values
            else None
        ),
        "average_total_tool_calls": (
            sum(total_tool_calls_values) / len(total_tool_calls_values)
            if total_tool_calls_values
            else None
        ),
        "average_latency": (
            sum(latency_values) / len(latency_values)
            if latency_values
            else None
        ),
        "average_attempt_count": (
            sum(attempt_counts) / len(attempt_counts)
            if attempt_counts
            else None
        ),
    }


def write_task_run_report(
    results: Iterable[TaskRunResult],
    *,
    source_csv: str | Path,
    output_dir: str | Path = "logs",
    filename: str = "task_run_report.json",
    extra_summary: Mapping[str, Any] | None = None,
) -> Path:
    result_list = list(results)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    output_path = output_root / filename
    summary = summarize_results(result_list)
    if extra_summary:
        summary = {**summary, **dict(extra_summary)}
    payload = {
        "generated_at": utc_now_iso(),
        "source_csv": str(Path(source_csv)),
        "summary": summary,
        "results": [asdict(result) for result in result_list],
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def select_tasks(
    tasks: Iterable[BenchmarkTask],
    *,
    task_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[BenchmarkTask]:
    selected = [task for task in tasks if not task_ids or task.task_id in task_ids]
    if limit is not None:
        return selected[:limit]
    return selected


def _parse_ground_truth_tool_calls(raw_value: str | None) -> list[dict[str, Any]]:
    if not raw_value:
        return []

    wrapped = f"[{raw_value}]"
    try:
        parsed = json.loads(wrapped)
    except json.JSONDecodeError:
        return []

    return [item for item in parsed if isinstance(item, dict)]


def _parse_toolset_names(raw_value: str | None) -> list[str]:
    cleaned = _clean_field(raw_value)
    if not cleaned:
        return []

    names: list[str] = []
    for chunk in cleaned.replace("\n", ",").split(","):
        normalized = chunk.strip()
        if normalized:
            names.append(normalized)
    return names


def _clean_field(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _load_tool_names_from_jsonl(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []

    tool_names: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            tool_name = _clean_field(str(record.get("tool_name") or ""))
            if tool_name:
                tool_names.append(tool_name)
    return tool_names


def _normalize_tool_name(value: str) -> str:
    return value.strip().lower()
