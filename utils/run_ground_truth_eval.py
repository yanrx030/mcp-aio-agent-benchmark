from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ground_truth_evaluation import build_evaluation_context, evaluate_ground_truth
from manifest_loader import resolve_runner_manifest
from prompts import normalize_answer_type
from task_runner import BenchmarkTask, load_benchmark_tasks, select_tasks


@dataclass(slots=True)
class JudgeConfig:
    judge_model: str | None
    judge_openrouter_params: dict[str, Any]
    config_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the ground-truth evaluator against an existing task_run_report.json "
            "without re-running the tool-using agent."
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
        action="append",
        dest="judge_models",
        help="Judge model id to test. Repeat to compare multiple judges.",
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
        help="Re-evaluate only text tasks, which are the ones affected by LLM judge choice.",
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
        help="Output JSON path. Defaults next to the source report.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    report_path = Path(args.report_file)
    report_payload = _load_json_object(report_path)
    task_file = _resolve_task_file(args.task_file, report_payload, report_path)
    tasks = load_benchmark_tasks(task_file)
    selected_tasks = select_tasks(tasks, task_ids=set(args.task_ids or []), limit=args.limit)
    if args.only_text:
        selected_tasks = [
            task for task in selected_tasks if normalize_answer_type(task.answer_type) == "text"
        ]

    if not selected_tasks:
        raise RuntimeError("No tasks selected. Check --task-id, --limit, --task-file, or --only-text.")

    report_results = report_payload.get("results")
    if not isinstance(report_results, list):
        raise RuntimeError("Report file must contain a top-level 'results' array.")

    result_by_task_id = {
        str(item.get("task_id")): item for item in report_results if isinstance(item, dict)
    }
    judge_configs = _resolve_judge_configs(
        manifest_path=Path(args.manifest),
        profile=args.profile,
        explicit_judge_models=args.judge_models or [],
    )

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model_runs: list[dict[str, Any]] = []

    print(
        f"Loaded {len(selected_tasks)} task(s) from {task_file} and {len(judge_configs)} judge configuration(s)."
    )

    for judge_config in judge_configs:
        label = judge_config.judge_model or "no_judge_model"
        print(f"Running ground-truth evaluation with judge: {label}, parameters: {judge_config.judge_openrouter_params}, source: {judge_config.config_source}")
        rows = await _evaluate_for_judge(
            tasks=selected_tasks,
            result_by_task_id=result_by_task_id,
            openrouter_api_key=openrouter_api_key,
            judge_config=judge_config,
            max_concurrency=args.max_concurrency,
        )
        model_runs.append(
            {
                "judge_model": judge_config.judge_model,
                "judge_openrouter_params": judge_config.judge_openrouter_params,
                "config_source": judge_config.config_source,
                "summary": _summarize_rows(rows),
                "results": rows,
            }
        )

    output_path = _resolve_output_path(args.output, report_path)
    payload = {
        "generated_at": _utc_now_iso(),
        "source_report": str(report_path),
        "source_csv": str(task_file),
        "manifest": str(Path(args.manifest)),
        "profile": args.profile,
        "selected_task_count": len(selected_tasks),
        "only_text": bool(args.only_text),
        "evaluations": model_runs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote ground-truth evaluation report to {output_path}")


async def _evaluate_for_judge(
    *,
    tasks: list[BenchmarkTask],
    result_by_task_id: dict[str, dict[str, Any]],
    openrouter_api_key: str,
    judge_config: JudgeConfig,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    rows: list[dict[str, Any] | None] = [None] * len(tasks)

    async def run_single(index: int, task: BenchmarkTask) -> None:
        async with semaphore:
            source_result = result_by_task_id.get(task.task_id)
            if source_result is None:
                rows[index] = _build_skipped_row(task, reason="task not found in source report")
                return

            source_status = _as_clean_str(source_result.get("status"))
            if source_status != "completed":
                rows[index] = _build_skipped_row(
                    task,
                    reason=f"source result status is {source_status or 'missing'}",
                    source_result=source_result,
                )
                return

            query_trace = source_result.get("query_trace")
            if not isinstance(query_trace, dict):
                rows[index] = _build_skipped_row(
                    task,
                    reason="source result does not contain query_trace",
                    source_result=source_result,
                )
                return

            final_answer_raw = _coerce_raw_payload(query_trace.get("final_answer"))
            if not final_answer_raw:
                rows[index] = _build_skipped_row(
                    task,
                    reason="source result does not contain final_answer",
                    source_result=source_result,
                )
                return

            context = build_evaluation_context(
                task_id=task.task_id,
                prompt=task.prompt,
                answer_type=task.answer_type,
                ground_truth_raw=task.ground_truth,
                final_answer_raw=final_answer_raw,
                metadata=task.metadata,
                openrouter_api_key=openrouter_api_key,
                judge_model=judge_config.judge_model,
                judge_openrouter_params=judge_config.judge_openrouter_params,
            )
            validation = await evaluate_ground_truth(context)

            previous_validation = source_result.get("ground_truth_validation")
            previous_score = _extract_score(previous_validation)
            current_score = _extract_score(validation)

            rows[index] = {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "answer_type": task.answer_type,
                "dispatch_type": validation.get("dispatch_type"),
                "status": "evaluated",
                "source_query_id": _extract_query_id(query_trace),
                "previous_score": previous_score,
                "current_score": current_score,
                "score_changed": (
                    previous_score is not None
                    and current_score is not None
                    and previous_score != current_score
                ),
                "previous_ground_truth_validation": previous_validation,
                "ground_truth_validation": validation,
            }

    await asyncio.gather(*(run_single(index, task) for index, task in enumerate(tasks)))
    return [row for row in rows if row is not None]


def _resolve_judge_configs(
    *,
    manifest_path: Path,
    profile: str,
    explicit_judge_models: list[str],
) -> list[JudgeConfig]:
    if explicit_judge_models:
        configs: list[JudgeConfig] = []
        for judge_model in explicit_judge_models:
            judge_model = judge_model.strip()
            if not judge_model:
                continue
            configs.append(
                _resolve_single_judge_config(
                    manifest_path=manifest_path,
                    profile=profile,
                    judge_model=judge_model,
                )
            )
        if configs:
            return configs

    if manifest_path.exists():
        manifest_config = resolve_runner_manifest(manifest_path, profile=profile)
        if manifest_config.judge:
            return [
                JudgeConfig(
                    judge_model=manifest_config.judge.model_id,
                    judge_openrouter_params=manifest_config.judge.openrouter_params,
                    config_source=f"manifest:{manifest_path}",
                )
            ]

    return [
        JudgeConfig(
            judge_model=None,
            judge_openrouter_params={},
            config_source="none",
        )
    ]


def _resolve_single_judge_config(
    *,
    manifest_path: Path,
    profile: str,
    judge_model: str,
) -> JudgeConfig:
    if manifest_path.exists():
        try:
            manifest_config = resolve_runner_manifest(
                manifest_path,
                profile=profile,
                judge_model_override=judge_model,
            )
        except RuntimeError:
            pass
        else:
            if manifest_config.judge:
                return JudgeConfig(
                    judge_model=manifest_config.judge.model_id,
                    judge_openrouter_params=manifest_config.judge.openrouter_params,
                    config_source=f"manifest:{manifest_path}",
                )

    return JudgeConfig(
        judge_model=judge_model,
        judge_openrouter_params={},
        config_source="direct",
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_rows = [row for row in rows if row.get("status") == "evaluated"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]
    text_rows = [
        row
        for row in evaluated_rows
        if normalize_answer_type(row.get("answer_type")) == "text"
        or row.get("dispatch_type") == "text"
    ]

    pass_count = sum(1 for row in evaluated_rows if row.get("current_score") == 1)
    text_pass_count = sum(1 for row in text_rows if row.get("current_score") == 1)
    changed_rows = [row for row in evaluated_rows if row.get("score_changed") is True]

    raw_ratings = Counter()
    judge_failures = 0
    parse_failures = 0
    judge_token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "text_judge_evaluations": len(text_rows),
        "evaluations_with_usage": 0,
    }
    for row in text_rows:
        validation = row.get("ground_truth_validation") or {}
        diagnostics = validation.get("diagnostics") or {}
        raw_rating = diagnostics.get("raw_rating")
        if raw_rating is not None:
            raw_ratings[str(raw_rating)] += 1
        reason = _as_clean_str(diagnostics.get("reason"))
        if reason == "judge evaluation failed":
            judge_failures += 1
        if reason == "judge returned non-object response":
            parse_failures += 1
        usage = diagnostics.get("judge_token_usage")
        if isinstance(usage, dict):
            has_usage = False
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage.get(key)
                if value is None:
                    continue
                try:
                    judge_token_usage[key] += int(value)
                except (TypeError, ValueError):
                    continue
                has_usage = True
            if has_usage:
                judge_token_usage["evaluations_with_usage"] += 1

    return {
        "total_rows": len(rows),
        "evaluated_rows": len(evaluated_rows),
        "skipped_rows": len(skipped_rows),
        "passed_rows": pass_count,
        "pass_rate": (pass_count / len(evaluated_rows)) if evaluated_rows else None,
        "text_rows": len(text_rows),
        "text_passed_rows": text_pass_count,
        "text_pass_rate": (text_pass_count / len(text_rows)) if text_rows else None,
        "score_changed_rows": len(changed_rows),
        "score_changed_task_ids": [row["task_id"] for row in changed_rows],
        "judge_failure_rows": judge_failures,
        "judge_parse_failure_rows": parse_failures,
        "judge_token_usage": judge_token_usage,
        "average_judge_total_tokens": (
            judge_token_usage["total_tokens"] / judge_token_usage["evaluations_with_usage"]
            if judge_token_usage["evaluations_with_usage"]
            else None
        ),
        "text_raw_rating_distribution": dict(sorted(raw_ratings.items())),
    }


def _build_skipped_row(
    task: BenchmarkTask,
    *,
    reason: str,
    source_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_trace = source_result.get("query_trace") if isinstance(source_result, dict) else None
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "answer_type": task.answer_type,
        "dispatch_type": normalize_answer_type(task.answer_type),
        "status": "skipped",
        "skip_reason": reason,
        "source_query_id": _extract_query_id(query_trace) if isinstance(query_trace, dict) else None,
        "previous_score": _extract_score(
            source_result.get("ground_truth_validation") if isinstance(source_result, dict) else None
        ),
        "current_score": None,
        "score_changed": False,
        "previous_ground_truth_validation": (
            source_result.get("ground_truth_validation") if isinstance(source_result, dict) else None
        ),
        "ground_truth_validation": None,
    }


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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return report_path.parent / f"ground_truth_eval_{timestamp}.json"


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


def _extract_score(validation: Any) -> int | None:
    if not isinstance(validation, dict):
        return None
    score = validation.get("score")
    if score in {0, 1, True, False}:
        return int(score)
    return None


def _extract_query_id(query_trace: dict[str, Any]) -> str | None:
    return _as_clean_str(query_trace.get("query_id"))


def _as_clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    asyncio.run(main())
