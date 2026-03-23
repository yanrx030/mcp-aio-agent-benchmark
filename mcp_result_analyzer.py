from __future__ import annotations

import json
import re
from typing import Any


def stringify_tool_result(result: Any) -> str:
    """Convert an MCP tool result into a stable string for logs and prompts."""
    try:
        content = getattr(result, "content", None)
        if isinstance(content, str):
            return content
        if content is not None:
            return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        pass
    return str(result)


def extract_content_blocks(result: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content = getattr(result, "content", None)
    if not isinstance(content, list):
        return blocks

    for item in content:
        blocks.append(
            {
                "type": getattr(item, "type", type(item).__name__),
                "text": getattr(item, "text", None),
                "annotations": getattr(item, "annotations", None),
                "meta": getattr(item, "meta", None),
            }
        )
    return blocks


def extract_structured_content(result: Any) -> Any:
    structured_content = getattr(result, "structuredContent", None)
    if structured_content is None:
        return None
    try:
        return json.loads(json.dumps(structured_content, ensure_ascii=False, default=str))
    except Exception:
        return structured_content


def extract_http_status_code(text: str | None) -> int | None:
    if not text:
        return None
    patterns = [
        r"\b([1-5]\d{2})\s+Client Error\b",
        r"\b([1-5]\d{2})\s+Server Error\b",
        r"\bHTTP\s+([1-5]\d{2})\b",
        r"\bstatus(?:\s+code)?[:=]?\s*([1-5]\d{2})\b",
        r"\b([1-5]\d{2})\s+Bad Gateway\b",
        r"\b([1-5]\d{2})\s+Not Found\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_json_objects(text: str) -> list[Any]:
    candidates: list[Any] = []
    try:
        candidates.append(json.loads(text))
        return candidates
    except Exception:
        pass

    for match in re.finditer(r"(\{.*?\}|\[.*?\])", text):
        try:
            candidates.append(json.loads(match.group(1)))
        except Exception:
            continue
    return candidates


def _find_error_value(value: Any) -> str | None:
    if isinstance(value, dict):
        if isinstance(value.get("error"), str):
            return value["error"]
        for item in value.values():
            found = _find_error_value(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_error_value(item)
            if found:
                return found
    return None


def detect_payload_error(structured_content: Any, content_blocks: list[dict[str, Any]], response_summary: str) -> tuple[bool, str | None]:
    error_value = _find_error_value(structured_content)
    if error_value:
        return True, error_value

    for block in content_blocks:
        text = block.get("text")
        if not isinstance(text, str):
            continue
        for candidate in _extract_json_objects(text):
            error_value = _find_error_value(candidate)
            if error_value:
                return True, error_value
        if "Input validation error" in text or "Output validation error" in text:
            return True, text

    if "Input validation error" in response_summary or "Output validation error" in response_summary:
        return True, response_summary
    return False, None


def analyze_tool_result(result: Any) -> dict[str, Any]:
    response_summary = stringify_tool_result(result)
    content_blocks = extract_content_blocks(result)
    structured_content = extract_structured_content(result)
    mcp_is_error = bool(getattr(result, "isError", False))
    payload_has_error, server_error_message = detect_payload_error(
        structured_content,
        content_blocks,
        response_summary,
    )
    http_status_code = extract_http_status_code(server_error_message or response_summary)

    failure_stage = None
    output_schema_valid = None
    if "Output validation error" in (server_error_message or response_summary):
        failure_stage = "output_validation"
        output_schema_valid = False
    elif mcp_is_error or payload_has_error:
        failure_stage = "server_response"

    execution_success = not (mcp_is_error or payload_has_error)

    return {
        "response_summary": response_summary,
        "content_blocks": content_blocks,
        "structured_content": structured_content,
        "mcp_is_error": mcp_is_error,
        "payload_has_error": payload_has_error,
        "server_error_message": server_error_message,
        "http_status_code": http_status_code,
        "output_schema_valid": output_schema_valid,
        "failure_stage": failure_stage,
        "execution_success": execution_success,
    }
