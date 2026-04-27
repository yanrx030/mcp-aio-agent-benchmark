import argparse
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

from MCPClient import MCPClient
from mcp_result_analyzer import analyze_tool_result


DEFAULT_TOOLSET = "toolsets/aio_mcp_toolset_v2.json"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_call_spec(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON input: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object.")

    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("Input must include a non-empty string field 'tool_name'.")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError("Input field 'arguments' must be a JSON object.")

    return {
        "tool_name": tool_name.strip(),
        "arguments": arguments,
    }


def _read_call_spec(args: argparse.Namespace) -> dict[str, Any]:
    sources = [bool(args.call), bool(args.call_file), not os.isatty(0)]
    if sum(sources) > 1:
        raise ValueError("Provide exactly one input source: positional JSON, --call-file, or stdin.")

    if args.call:
        return _parse_call_spec(args.call)
    if args.call_file:
        return _parse_call_spec(Path(args.call_file).read_text(encoding="utf-8"))

    raw = input().strip() if os.isatty(0) else os.sys.stdin.read().strip()
    if not raw:
        raise ValueError("No tool call JSON provided.")
    return _parse_call_spec(raw)


async def _run_tool_call(args: argparse.Namespace, call_spec: dict[str, Any]) -> dict[str, Any]:
    auth_key = os.environ.get("AIO_AUTH_KEY")
    if not auth_key:
        raise RuntimeError("AIO_AUTH_KEY is not set. Use --env-file or export it before running.")

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "unused")
    client = MCPClient(
        auth_key=auth_key,
        openrouter_api_key=openrouter_key,
        openrouter_model=DEFAULT_MODEL,
    )

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            client.authenticate()
            client.load_tools(args.toolset)
            await client.connect_to_server()

        result = await client.session.call_tool(
            call_spec["tool_name"],
            {k: v for k, v in call_spec["arguments"].items() if v is not None},
        )
        analyzed = analyze_tool_result(result)
        return {
            "tool_name": call_spec["tool_name"],
            "arguments": call_spec["arguments"],
            "execution_success": analyzed["execution_success"],
            "mcp_is_error": analyzed["mcp_is_error"],
            "payload_has_error": analyzed["payload_has_error"],
            "response_summary": analyzed["response_summary"],
            "content_blocks": analyzed["content_blocks"],
            "structured_content": analyzed["structured_content"],
            "server_error_message": analyzed["server_error_message"],
        }
    finally:
        await client.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute one MCP tool call from an explicit JSON tool spec.",
    )
    parser.add_argument(
        "call",
        nargs="?",
        help='JSON string like {"tool_name":"text_search","arguments":{...}}',
    )
    parser.add_argument(
        "--call-file",
        help="Path to a file containing the JSON tool spec.",
    )
    parser.add_argument(
        "--toolset",
        default=DEFAULT_TOOLSET,
        help=f"Path to the frozen toolset JSON. Default: {DEFAULT_TOOLSET}",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file to preload. Default: .env",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON.",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))

    try:
        call_spec = _read_call_spec(args)
        payload = asyncio.run(_run_tool_call(args, call_spec))
    except Exception as exc:
        error_payload = {
            "execution_success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        return 1

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
