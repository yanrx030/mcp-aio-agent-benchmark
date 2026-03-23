from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import asdict, is_dataclass
from typing import Any

import requests
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from requests.auth import HTTPBasicAuth


LOGIN_URL = "https://api.aio.eresearch.unimelb.edu.au/login"
MCP_URL = "https://mcp.aio.eresearch.unimelb.edu.au/mcp"


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
        description="Inspect the raw structure of an MCP tool result."
    )
    parser.add_argument("--tool", required=True, help="MCP tool name to call")
    parser.add_argument(
        "--args",
        default="{}",
        help='Tool arguments as a JSON string, e.g. \'{"collection":"reddit"}\'',
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for JSON pretty-printing",
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
        "has_content_attr": hasattr(result, "content"),
        "has_structuredContent_attr": hasattr(result, "structuredContent"),
        "has_isError_attr": hasattr(result, "isError"),
        "has_meta_attr": hasattr(result, "meta"),
        "isError": is_error,
        "content_type": type(content).__name__ if content is not None else None,
        "content_len": len(content) if isinstance(content, list) else None,
        "structuredContent_type": type(structured_content).__name__ if structured_content is not None else None,
        "meta_type": type(meta).__name__ if meta is not None else None,
        "repr": repr(result),
    }


async def inspect_tool_result(tool_name: str, tool_args: dict[str, Any], indent: int) -> None:
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

        result = await session.call_tool(tool_name, tool_args)

        print("=== Result Summary ===")
        print(json.dumps(summarize_result(result), indent=indent, ensure_ascii=False, default=str))
        print()

        print("=== Plain Data Dump ===")
        print(json.dumps(to_plain_data(result), indent=indent, ensure_ascii=False, default=str))
    finally:
        await exit_stack.aclose()


def main() -> None:
    args = parse_args()
    try:
        tool_args = json.loads(args.args)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for --args: {exc}") from exc

    if not isinstance(tool_args, dict):
        raise SystemExit("--args must decode to a JSON object")

    asyncio.run(inspect_tool_result(args.tool, tool_args, args.indent))


if __name__ == "__main__":
    main()
