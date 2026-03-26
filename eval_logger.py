from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_id(prefix: str) -> str:
    """Generate a compact identifier suitable for query and tool-call records."""
    return f"{prefix}_{uuid4().hex[:12]}"


def generate_task_set_id() -> str:
    """Generate a readable identifier for a task-set run directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"taskset_{timestamp}_{uuid4().hex[:6]}"


def _safe_path_component(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in value.strip()
    )
    return cleaned or "unassigned"


def _normalize_record(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        normalized = asdict(record)
    elif isinstance(record, Mapping):
        normalized = dict(record)
    else:
        raise TypeError("record must be a dataclass instance or a mapping")

    normalized.setdefault("timestamp", utc_now_iso())
    return normalized


@dataclass(slots=True)
class QueryRunRecord:
    query_id: str
    user_query: str
    task_set_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    model: str | None = None
    toolset_id: str | None = None
    status: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    start_time: str = field(default_factory=utc_now_iso)
    end_time: str | None = None
    final_answer: str | None = None
    total_steps: int | None = None
    total_tool_calls: int | None = None
    parameter_schema_valid_rate: float | None = None
    ccr_applicable_calls: int | None = None
    ccr_true_calls: int | None = None
    constraint_compliance_rate: float | None = None
    tools_used: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallRecord:
    query_id: str
    tool_name: str
    step_index: int
    task_set_id: str | None = None
    task_id: str | None = None
    tool_call_id: str = field(default_factory=lambda: generate_id("call"))
    raw_arguments: str | None = None
    parsed_arguments: dict[str, Any] | None = None
    parse_success: bool | None = None
    schema_valid: bool | None = None
    schema_errors: list[str] | None = None
    constraint_valid: bool | None = None
    execution_attempted: bool | None = None
    transport_success: bool | None = None
    mcp_is_error: bool | None = None
    payload_has_error: bool | None = None
    execution_success: bool | None = None
    failure_stage: str | None = None
    http_status_code: int | None = None
    output_schema_valid: bool | None = None
    latency_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    server_error_message: str | None = None
    response_summary: Any | None = None
    content_blocks: list[dict[str, Any]] | None = None
    structured_content: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class JSONLLogger:
    """
    Append-only JSONL logger for evaluation traces.

    Layout:
      - one task-set directory per run under `logs/task_sets/<task_set_id>/`
      - one query-level record per query in `query_runs.jsonl`
      - one record per attempted tool call in `tool_calls/<task_id>.jsonl`
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        task_set_id: str | None = None,
        task_sets_dirname: str = "task_sets",
        query_runs_filename: str = "query_runs.jsonl",
        tool_calls_dirname: str = "tool_calls",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.task_set_id = task_set_id or generate_task_set_id()
        self.task_set_dir = self.log_dir / task_sets_dirname / self.task_set_id
        self.task_set_dir.mkdir(parents=True, exist_ok=True)

        self.query_runs_path = self.task_set_dir / query_runs_filename
        self.tool_calls_dir = self.task_set_dir / tool_calls_dirname
        self.tool_calls_dir.mkdir(parents=True, exist_ok=True)

        self._query_lock = threading.Lock()
        self._tool_locks: dict[Path, threading.Lock] = {}
        self._tool_locks_guard = threading.Lock()

    def new_query_id(self) -> str:
        return generate_id("query")

    def new_session_id(self) -> str:
        return generate_id("session")

    def log_query_run(self, record: QueryRunRecord | Mapping[str, Any]) -> dict[str, Any]:
        normalized = _normalize_record(record)
        self._append_jsonl(self.query_runs_path, normalized, self._query_lock)
        return normalized

    def log_tool_call(self, record: ToolCallRecord | Mapping[str, Any]) -> dict[str, Any]:
        normalized = _normalize_record(record)
        path = self.task_tool_calls_path(
            task_id=normalized.get("task_id"),
            query_id=normalized.get("query_id"),
        )
        self._append_jsonl(path, normalized, self._get_tool_lock(path))
        return normalized

    def task_tool_calls_path(
        self,
        *,
        task_id: str | None = None,
        query_id: str | None = None,
    ) -> Path:
        reference = task_id or query_id or "unassigned"
        return self.tool_calls_dir / f"{_safe_path_component(reference)}.jsonl"

    def build_query_run(
        self,
        *,
        user_query: str,
        query_id: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        model: str | None = None,
        toolset_id: str | None = None,
        **extra_fields: Any,
    ) -> QueryRunRecord:
        metadata = extra_fields.pop("metadata", {})
        if extra_fields:
            metadata = {**metadata, **extra_fields}

        return QueryRunRecord(
            query_id=query_id or self.new_query_id(),
            user_query=user_query,
            task_set_id=self.task_set_id,
            task_id=task_id,
            session_id=session_id,
            model=model,
            toolset_id=toolset_id,
            metadata=metadata,
        )

    def build_tool_call(
        self,
        *,
        query_id: str,
        task_id: str | None = None,
        tool_name: str,
        step_index: int,
        tool_call_id: str | None = None,
        **fields: Any,
    ) -> ToolCallRecord:
        metadata = fields.pop("metadata", {})
        return ToolCallRecord(
            query_id=query_id,
            task_set_id=self.task_set_id,
            task_id=task_id,
            tool_name=tool_name,
            step_index=step_index,
            tool_call_id=tool_call_id or generate_id("call"),
            metadata=metadata,
            **fields,
        )

    def _get_tool_lock(self, path: Path) -> threading.Lock:
        with self._tool_locks_guard:
            lock = self._tool_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._tool_locks[path] = lock
            return lock

    def _append_jsonl(self, path: Path, record: Mapping[str, Any], lock: threading.Lock) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
