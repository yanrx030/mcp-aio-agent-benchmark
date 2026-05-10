from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from requests.auth import HTTPBasicAuth


LOGIN_URL = "https://api.aio.eresearch.unimelb.edu.au/login"
MCP_URL = "https://mcp.aio.eresearch.unimelb.edu.au/mcp"
DEFAULT_TOOLSET_PATH = Path("toolsets/aio_mcp_toolset_v2.json")


def authenticate(auth_key: str) -> dict[str, str]:
    response = requests.post(
        LOGIN_URL,
        auth=HTTPBasicAuth("apikey", auth_key),
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(f"Authentication failed: {response.status_code} {response.text}")
    return {
        "Authorization": f"Bearer {response.text}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect success/failure response structures for all tools in a toolset."
    )
    parser.add_argument(
        "--toolset",
        default=str(DEFAULT_TOOLSET_PATH),
        help="Path to the frozen toolset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="inspection_reports",
        help="Directory where JSON and Markdown reports will be written",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation",
    )
    return parser.parse_args()


def to_plain_data(value: Any, *, depth: int = 0, max_depth: int = 8) -> Any:
    if depth >= max_depth:
        return f"<max_depth_reached type={type(value).__name__}>"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): to_plain_data(v, depth=depth + 1, max_depth=max_depth) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_plain_data(v, depth=depth + 1, max_depth=max_depth) for v in value]

    if is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            "__dataclass__": to_plain_data(asdict(value), depth=depth + 1, max_depth=max_depth),
        }

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            return {
                "__type__": type(value).__name__,
                "__model_dump__": to_plain_data(dumped, depth=depth + 1, max_depth=max_depth),
            }
        except Exception as exc:
            return {
                "__type__": type(value).__name__,
                "__model_dump_error__": str(exc),
                "__repr__": repr(value),
            }

    if hasattr(value, "__dict__"):
        data = {
            key: to_plain_data(val, depth=depth + 1, max_depth=max_depth)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
        return {
            "__type__": type(value).__name__,
            "__attrs__": data,
            "__repr__": repr(value),
        }

    return {
        "__type__": type(value).__name__,
        "__repr__": repr(value),
        "__str__": str(value),
    }


def summarize_result(result: Any) -> dict[str, Any]:
    content = getattr(result, "content", None)
    structured_content = getattr(result, "structuredContent", None)
    is_error = getattr(result, "isError", None)
    meta = getattr(result, "meta", None)
    return {
        "python_type": type(result).__name__,
        "is_error": is_error,
        "content_type": type(content).__name__ if content is not None else None,
        "content_len": len(content) if isinstance(content, list) else None,
        "structured_content_type": type(structured_content).__name__ if structured_content is not None else None,
        "meta_type": type(meta).__name__ if meta is not None else None,
        "repr": repr(result),
    }


def stringify_result(result: Any) -> str:
    content = getattr(result, "content", None)
    try:
        if isinstance(content, str):
            return content
        if content is not None:
            return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        pass
    return str(result)


def extract_http_status_code(text: str | None) -> int | None:
    if not text:
        return None
    patterns = [
        r"\b([1-5]\d{2})\s+Client Error\b",
        r"\b([1-5]\d{2})\s+Server Error\b",
        r"\bHTTP\s+([1-5]\d{2})\b",
        r"\bstatus(?:\s+code)?[:=]?\s*([1-5]\d{2})\b",
        r"\b([1-5]\d{2})\s+Bad Gateway\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_text_blocks(result: Any) -> list[str]:
    texts: list[str] = []
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                texts.append(text)
    elif isinstance(content, str):
        texts.append(content)
    return texts


def try_parse_json_text(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    object_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not object_match:
        return None
    try:
        return json.loads(object_match.group(1))
    except json.JSONDecodeError:
        return None


def flatten_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(flatten_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(flatten_strings(item))
    return found


def first_iso_date_in_texts(texts: list[str]) -> date | None:
    for text in texts:
        for match in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text):
            try:
                return date.fromisoformat(match)
            except ValueError:
                continue
    return None


def collect_candidate_terms(value: Any) -> list[str]:
    candidates: list[str] = []
    if isinstance(value, str):
        cleaned = value.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{2,30}", cleaned):
            candidates.append(cleaned)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"term", "terms", "stem", "label", "name"}:
                candidates.extend(collect_candidate_terms(item))
            else:
                candidates.extend(collect_candidate_terms(item))
    elif isinstance(value, list):
        for item in value:
            candidates.extend(collect_candidate_terms(item))
    return candidates


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def load_toolset(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        toolset = json.load(handle)
    return toolset["mcp_raw"]["tools"]


async def call_tool(session: ClientSession, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    try:
        result = await session.call_tool(tool_name, tool_args)
        result_text = stringify_result(result)
        return {
            "call_status": "returned",
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "latency_ms": int((datetime.now(UTC) - started_at).total_seconds() * 1000),
            "is_error": bool(getattr(result, "isError", False)),
            "http_status_code": extract_http_status_code(result_text),
            "summary": summarize_result(result),
            "text_blocks": extract_text_blocks(result),
            "plain_data": to_plain_data(result),
            "result_text": result_text,
        }
    except Exception as exc:
        return {
            "call_status": "exception",
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "latency_ms": int((datetime.now(UTC) - started_at).total_seconds() * 1000),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }


async def bootstrap_context(session: ClientSession) -> dict[str, Any]:
    today = datetime.now(UTC).date()
    yesterday = today - timedelta(days=1)
    context: dict[str, Any] = {
        "today": today.isoformat(),
        "yesterday": yesterday.isoformat(),
        "collection": "reddit",
        "summary_start": None,
        "summary_end": None,
        "summary_count": None,
        "term": "covid",
    }

    collections_result = await call_tool(session, "get_collections", {})
    context["collections_probe"] = collections_result

    candidate_collection = "reddit"
    if collections_result["call_status"] == "returned":
        blob = collections_result["plain_data"]
        strings = unique_preserve_order(flatten_strings(blob))
        lowercase = {s.lower(): s for s in strings}
        if "reddit" in lowercase:
            candidate_collection = lowercase["reddit"]
        elif strings:
            candidate_collection = strings[0]
    context["collection"] = candidate_collection

    summary_result = await call_tool(session, "get_collection_summary", {"collection": candidate_collection})
    context["summary_probe"] = summary_result
    if summary_result["call_status"] == "returned":
        plain = summary_result["plain_data"]
        strings = flatten_strings(plain)
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", json.dumps(plain))
        if len(dates) >= 2:
            try:
                context["summary_start"] = min(date.fromisoformat(d) for d in dates).isoformat()
                context["summary_end"] = max(date.fromisoformat(d) for d in dates).isoformat()
            except ValueError:
                pass
        count_match = re.search(r'"count"\s*:\s*(\d+)', json.dumps(plain))
        if count_match:
            context["summary_count"] = int(count_match.group(1))

    default_end = yesterday
    if context["summary_end"]:
        try:
            default_end = min(default_end, date.fromisoformat(context["summary_end"]))
        except ValueError:
            pass

    context["safe_end"] = default_end.isoformat()
    context["safe_start_7"] = max(default_end - timedelta(days=6), date(2020, 1, 1)).isoformat()
    context["safe_start_28"] = max(default_end - timedelta(days=27), date(2020, 1, 1)).isoformat()
    context["safe_start_30"] = max(default_end - timedelta(days=29), date(2020, 1, 1)).isoformat()

    all_terms_result = await call_tool(
        session,
        "get_all_terms",
        {
            "collection": candidate_collection,
            "startdate": context["safe_start_7"],
            "enddate": context["safe_end"],
        },
    )
    context["all_terms_probe"] = all_terms_result
    if all_terms_result["call_status"] == "returned" and not all_terms_result.get("is_error"):
        parsed_candidates = collect_candidate_terms(all_terms_result["plain_data"])
        terms = unique_preserve_order(parsed_candidates)
        if terms:
            context["term"] = terms[0]

    return context


def build_valid_args(tool_name: str, context: dict[str, Any]) -> dict[str, Any]:
    collection = context["collection"]
    safe_end = context["safe_end"]
    safe_start_7 = context["safe_start_7"]
    safe_start_28 = context["safe_start_28"]
    safe_start_30 = context["safe_start_30"]
    term = context["term"]

    valid_args: dict[str, dict[str, Any]] = {
        "get_api_version": {},
        "get_collections": {},
        "get_collection_summary": {"collection": collection},
        "aggregate_by_time": {
            "collection": collection,
            "aggregation_level": "month",
            "startdate": safe_start_30,
            "enddate": safe_end,
            "sentiment": False,
        },
        "aggregate_seasonality": {
            "collection": collection,
            "aggregation_level": "dayofweek",
            "startdate": safe_start_30,
            "enddate": safe_end,
            "sentiment": False,
        },
        "analyze_terms_in_collection": {
            "collection": collection,
            "startdate": safe_start_7,
            "enddate": safe_end,
            "limit": 10,
        },
        "get_all_terms": {
            "collection": collection,
            "startdate": safe_start_7,
            "enddate": safe_end,
        },
        "get_term_daily_counts": {
            "collection": collection,
            "terms": term,
            "startdate": safe_start_7,
            "enddate": safe_end,
        },
        "get_nlp_terms_for_day": {
            "collection": collection,
            "day": safe_end,
            "limit": 10,
        },
        "get_nlp_term_analysis": {
            "collection": collection,
            "day": safe_end,
            "term": term,
            "limit": 10,
        },
        "get_nlp_topics": {
            "collection": collection,
            "startdate": safe_start_7,
            "enddate": safe_end,
        },
        "get_topic_groupings": {
            "collection": collection,
            "startdate": safe_start_7,
            "enddate": safe_end,
            "threshold": 1,
        },
        "get_nlp_metadata": {
            "collection": collection,
            "startdate": safe_start_7,
            "enddate": safe_end,
        },
        "text_search": {
            "collection": collection,
            "query": "text:covid",
            "limit": 10,
        },
        "generate_chart": {
            "chart": {
                "type": "bar",
                "data": {
                    "labels": ["A", "B"],
                    "datasets": [{"label": "Counts", "data": [1, 2]}],
                },
            },
            "width": 300,
            "height": 200,
            "format": "png",
        },
    }
    return valid_args[tool_name]


def build_invalid_args(tool_name: str, context: dict[str, Any]) -> dict[str, Any] | None:
    invalid_args: dict[str, dict[str, Any] | None] = {
        "get_api_version": None,
        "get_collections": None,
        "get_collection_summary": {},
        "aggregate_by_time": {
            "collection": context["collection"],
            "aggregation_level": "month",
            "startdate": "2019-01-01",
            "enddate": context["safe_end"],
            "sentiment": False,
        },
        "aggregate_seasonality": {
            "collection": context["collection"],
            "aggregation_level": "invalid_level",
            "startdate": context["safe_start_30"],
            "enddate": context["safe_end"],
            "sentiment": False,
        },
        "analyze_terms_in_collection": {},
        "get_all_terms": {
            "collection": context["collection"],
            "startdate": "2019-01-01",
            "enddate": context["safe_end"],
        },
        "get_term_daily_counts": {
            "collection": context["collection"],
            "terms": 123,
        },
        "get_nlp_terms_for_day": {
            "collection": context["collection"],
            "day": "1900-01-01",
        },
        "get_nlp_term_analysis": {
            "collection": context["collection"],
            "day": context["safe_end"],
            "term": 123,
        },
        "get_nlp_topics": {
            "collection": context["collection"],
            "startdate": "2019-01-01",
            "enddate": context["safe_end"],
        },
        "get_topic_groupings": {
            "collection": context["collection"],
            "startdate": "2019-01-01",
            "enddate": context["safe_end"],
            "threshold": "bad",
        },
        "get_nlp_metadata": {},
        "text_search": {
            "collection": context["collection"],
            "query": 123,
        },
        "generate_chart": {
            "chart": "not-an-object",
        },
    }
    return invalid_args[tool_name]


async def inspect_all_tools(toolset_path: str, output_dir: str, indent: int) -> tuple[Path, Path]:
    tools = load_toolset(toolset_path)
    auth_key = os.environ["AIO_AUTH_KEY"]
    headers = authenticate(auth_key)

    exit_stack = AsyncExitStack()
    try:
        transport = await exit_stack.enter_async_context(
            streamablehttp_client(MCP_URL, headers=headers)
        )
        read_stream, write_stream, _ = transport
        session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        context = await bootstrap_context(session)
        results: list[dict[str, Any]] = []
        for tool in tools:
            tool_name = tool["name"]
            valid_args = build_valid_args(tool_name, context)
            invalid_args = build_invalid_args(tool_name, context)
            tool_entry: dict[str, Any] = {
                "tool_name": tool_name,
                "input_schema": tool.get("inputSchema"),
                "output_schema": tool.get("outputSchema"),
                "valid_args": valid_args,
                "invalid_args": invalid_args,
                "success_case": await call_tool(session, tool_name, valid_args),
                "failure_case": None,
            }
            if invalid_args is not None:
                tool_entry["failure_case"] = await call_tool(session, tool_name, invalid_args)
            results.append(tool_entry)

        report = {
            "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "toolset_path": str(toolset_path),
            "context": context,
            "results": results,
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        json_report = output_path / f"mcp_tool_inspection_{stamp}.json"
        markdown_report = output_path / f"mcp_tool_inspection_{stamp}.md"

        json_report.write_text(json.dumps(report, indent=indent, ensure_ascii=False, default=str), encoding="utf-8")
        markdown_report.write_text(render_markdown_report(report), encoding="utf-8")
        return json_report, markdown_report
    finally:
        await exit_stack.aclose()


def render_case_summary(case: dict[str, Any] | None) -> str:
    if case is None:
        return "not tested"
    if case["call_status"] == "exception":
        return f"exception: {case['exception_type']} - {case['exception_message']}"
    state = "mcp_error" if case.get("is_error") else "ok"
    return (
        f"{state}; is_error={case.get('is_error')}; "
        f"http_status_code={case.get('http_status_code')}; "
        f"content_type={case['summary'].get('content_type')}; "
        f"structured_content_type={case['summary'].get('structured_content_type')}"
    )


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# MCP Tool Inspection Report")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at_utc']}`")
    lines.append(f"- Toolset: `{report['toolset_path']}`")
    lines.append(f"- Collection context: `{report['context'].get('collection')}`")
    lines.append(f"- Safe end date: `{report['context'].get('safe_end')}`")
    lines.append(f"- Chosen term: `{report['context'].get('term')}`")
    lines.append("")
    lines.append("## Per-tool Summary")
    lines.append("")
    for result in report["results"]:
        lines.append(f"### `{result['tool_name']}`")
        lines.append(f"- Success case: {render_case_summary(result['success_case'])}")
        lines.append(f"- Failure case: {render_case_summary(result['failure_case'])}")
        lines.append(f"- Success args: `{json.dumps(result['valid_args'], ensure_ascii=False)}`")
        if result["invalid_args"] is not None:
            lines.append(f"- Failure args: `{json.dumps(result['invalid_args'], ensure_ascii=False)}`")
        success_texts = (result["success_case"] or {}).get("text_blocks") or []
        if success_texts:
            lines.append(f"- Success text sample: `{success_texts[0][:220]}`")
        failure_texts = (result["failure_case"] or {}).get("text_blocks") or []
        if failure_texts:
            lines.append(f"- Failure text sample: `{failure_texts[0][:220]}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    json_report, markdown_report = asyncio.run(
        inspect_all_tools(args.toolset, args.output_dir, args.indent)
    )
    print(f"JSON report written to {json_report}")
    print(f"Markdown report written to {markdown_report}")


if __name__ == "__main__":
    main()
