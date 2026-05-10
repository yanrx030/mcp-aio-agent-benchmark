import argparse
import asyncio
import contextlib
import io
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from MCPClient import MCPClient
from mcp_result_analyzer import analyze_tool_result


DEFAULT_TOOLSET = "toolsets/aio_mcp_toolset_v2.json"
DEFAULT_MODEL = "placeholder"
DEFAULT_ENV_FILE = ".env"

__all__ = ["execute_tool_call", "main"]


class _NullLogger:
    task_set_id = None
    query_runs_path = None

    def new_session_id(self) -> str:
        return "tool_call_session"

    def task_tool_calls_path(self, **_: Any) -> Path:
        return Path("NUL")


def _to_jsonable(value: Any, *, depth: int = 0, max_depth: int = 8) -> Any:
    if depth >= max_depth:
        return f"<max_depth_reached type={type(value).__name__}>"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item, depth=depth + 1, max_depth=max_depth) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item, depth=depth + 1, max_depth=max_depth) for item in value]

    if is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            "__dataclass__": _to_jsonable(asdict(value), depth=depth + 1, max_depth=max_depth),
        }

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            return {
                "__type__": type(value).__name__,
                "__model_dump__": _to_jsonable(dumped, depth=depth + 1, max_depth=max_depth),
            }
        except Exception as exc:
            return {
                "__type__": type(value).__name__,
                "__model_dump_error__": str(exc),
                "__repr__": repr(value),
            }

    if hasattr(value, "__dict__"):
        return {
            "__type__": type(value).__name__,
            "__attrs__": {
                key: _to_jsonable(item, depth=depth + 1, max_depth=max_depth)
                for key, item in vars(value).items()
                if not key.startswith("_")
            },
            "__repr__": repr(value),
        }

    return {
        "__type__": type(value).__name__,
        "__repr__": repr(value),
        "__str__": str(value),
    }


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


def _normalize_call_spec(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("tool_name must be a non-empty string.")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object.")

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


async def _run_tool_call(
    *,
    call_spec: dict[str, Any],
    toolset: str = DEFAULT_TOOLSET,
    raw_result_file: str | Path | None = None,
) -> dict[str, Any]:
    auth_key = os.environ.get("AIO_AUTH_KEY")
    if not auth_key:
        raise RuntimeError("AIO_AUTH_KEY is not set. Use --env-file or export it before running.")

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "unused")
    client = MCPClient(
        auth_key=auth_key,
        openrouter_api_key=openrouter_key,
        openrouter_model=DEFAULT_MODEL,
        logger=_NullLogger(),
    )

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            client.authenticate()
            client.load_tools(toolset)
            await client.connect_to_server()

        result = await client.session.call_tool(
            call_spec["tool_name"],
            {k: v for k, v in call_spec["arguments"].items() if v is not None},
        )
        if raw_result_file:
            raw_result_path = Path(raw_result_file)
            raw_result_path.parent.mkdir(parents=True, exist_ok=True)
            raw_result_path.write_text(
                json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        analyzed = analyze_tool_result(result)
        payload = {
            "tool_name": call_spec["tool_name"],
            "arguments": call_spec["arguments"],
            "execution_success": analyzed["execution_success"],
            "mcp_is_error": analyzed["mcp_is_error"],
            "payload_has_error": analyzed["payload_has_error"],
            "content_blocks": analyzed["content_blocks"],
            "structured_content": analyzed["structured_content"],
            "server_error_message": analyzed["server_error_message"],
        }
        if raw_result_file:
            payload["raw_result_file"] = str(Path(raw_result_file))
        return payload
    finally:
        await client.cleanup()


def execute_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    toolset: str = DEFAULT_TOOLSET,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    raw_result_file: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one MCP tool call and return a normalized payload for agent use."""
    if env_file:
        _load_env_file(Path(env_file))

    call_spec = _normalize_call_spec(tool_name, arguments)
    return asyncio.run(
        _run_tool_call(
            call_spec=call_spec,
            toolset=toolset,
            raw_result_file=raw_result_file,
        )
    )


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
        default=DEFAULT_ENV_FILE,
        help=f"Optional env file to preload. Default: {DEFAULT_ENV_FILE}",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON.",
    )
    parser.add_argument(
        "--raw-result-file",
        help="Optional path to save the raw MCP call result as JSON.",
    )
    args = parser.parse_args()

    try:
        call_spec = _read_call_spec(args)
        payload = execute_tool_call(
            call_spec["tool_name"],
            call_spec["arguments"],
            toolset=args.toolset,
            env_file=args.env_file,
            raw_result_file=args.raw_result_file,
        )
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
