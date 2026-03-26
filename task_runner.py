from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from eval_logger import utc_now_iso


@dataclass(slots=True)
class BenchmarkTask:
    task_id: str
    prompt: str
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
    query_trace: dict[str, Any] | None = None
    ground_truth_validation: dict[str, Any] | None = None
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

            raw_tool_calls = _clean_field(row.get("ground_truth_tool_call"))
            task = BenchmarkTask(
                task_id=task_id,
                prompt=prompt,
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
                        "note",
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


async def run_benchmark_tasks(client: Any, tasks: Iterable[BenchmarkTask]) -> list[TaskRunResult]:
    results: list[TaskRunResult] = []

    for task in tasks:
        started_at = utc_now_iso()
        print(f"[task {task.task_id}] {task.prompt}")
        try:
            query_trace = await client.process_query(
                task.prompt,
                task_id=task.task_id,
                return_trace=True,
            )
            validation = validate_ground_truth_placeholder(task, query_trace)
            results.append(
                TaskRunResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    status="completed",
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                    query_trace=query_trace,
                    ground_truth_validation=validation,
                )
            )
            print(f"[task {task.task_id}] completed")
        except Exception as exc:
            results.append(
                TaskRunResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    status="failed",
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            print(f"[task {task.task_id}] failed: {type(exc).__name__}: {exc}")

    return results


def validate_ground_truth_placeholder(
    task: BenchmarkTask,
    query_trace: dict[str, Any],
) -> dict[str, Any]:
    expected_answer = (task.ground_truth or "").strip()
    final_answer = str(query_trace.get("final_answer") or "").strip()
    heuristic_match = bool(expected_answer) and expected_answer in final_answer

    return {
        "status": "placeholder_match" if heuristic_match else "placeholder_review_required",
        "placeholder": True,
        "reason": (
            "Detailed ground-truth validation is not implemented yet. "
            "This placeholder only checks whether the expected ground-truth string appears in the final answer "
            "and records the parsed reference tool calls for future validation."
        ),
        "expected_ground_truth": task.ground_truth,
        "expected_tool_call_count": len(task.ground_truth_tool_calls),
        "expected_tool_calls": task.ground_truth_tool_calls,
        "heuristic_answer_contains_ground_truth": heuristic_match,
    }


def summarize_results(results: Iterable[TaskRunResult]) -> dict[str, Any]:
    result_list = list(results)
    completed = sum(1 for result in result_list if result.status == "completed")
    failed = sum(1 for result in result_list if result.status == "failed")
    placeholder_matches = sum(
        1
        for result in result_list
        if (result.ground_truth_validation or {}).get("status") == "placeholder_match"
    )

    return {
        "total_tasks": len(result_list),
        "completed_tasks": completed,
        "failed_tasks": failed,
        "placeholder_matches": placeholder_matches,
        "placeholder_review_required": completed - placeholder_matches,
    }


def write_task_run_report(
    results: Iterable[TaskRunResult],
    *,
    source_csv: str | Path,
    output_dir: str | Path = "logs",
    filename: str = "task_run_report.json",
) -> Path:
    result_list = list(results)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    output_path = output_root / filename
    payload = {
        "generated_at": utc_now_iso(),
        "source_csv": str(Path(source_csv)),
        "summary": summarize_results(result_list),
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
