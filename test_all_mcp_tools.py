import argparse
import asyncio
import contextlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from MCPClient import MCPClient
from mcp_result_analyzer import analyze_tool_result
from tool_call_templates import TOOLS


DEFAULT_TOOLSET = "toolsets/aio_mcp_toolset_v2.json"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def _default_report_file() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("logs") / f"mcp_tool_test_report_{timestamp}.json"


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


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


async def _run_all_tools(toolset_path: str) -> dict[str, Any]:
    auth_key = os.environ.get("AIO_AUTH_KEY")
    if not auth_key:
        raise RuntimeError("AIO_AUTH_KEY is not set. Use --env-file or export it before running.")

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "unused")
    client = MCPClient(
        auth_key=auth_key,
        openrouter_api_key=openrouter_key,
        openrouter_model=DEFAULT_MODEL,
    )

    results: list[dict[str, Any]] = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            client.authenticate()
            client.load_tools(toolset_path)
            await client.connect_to_server()

        for tool_name, tool_template in TOOLS.items():
            arguments = _drop_none_values(tool_template.get("example", {}))
            try:
                result = await client.session.call_tool(tool_name, arguments)
                analyzed = analyze_tool_result(result)
                results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "execution_success": analyzed["execution_success"],
                        "mcp_is_error": analyzed["mcp_is_error"],
                        "payload_has_error": analyzed["payload_has_error"],
                        "http_status_code": analyzed["http_status_code"],
                        "server_error_message": analyzed["server_error_message"],
                        "failure_stage": analyzed["failure_stage"],
                        "response_summary": analyzed["response_summary"],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "execution_success": False,
                        "mcp_is_error": True,
                        "payload_has_error": True,
                        "http_status_code": None,
                        "server_error_message": f"{type(exc).__name__}: {exc}",
                        "failure_stage": "client_exception",
                        "response_summary": f"{type(exc).__name__}: {exc}",
                    }
                )

        failures = [
            item
            for item in results
            if not item["execution_success"]
            or item["mcp_is_error"]
            or item["payload_has_error"]
            or item["http_status_code"] is not None
        ]

        return {
            "toolset_path": toolset_path,
            "total_tools": len(results),
            "failed_tools": len(failures),
            "failures": [
                {
                    "tool_name": item["tool_name"],
                    "http_status_code": item["http_status_code"],
                    "server_error_message": item["server_error_message"],
                    "failure_stage": item["failure_stage"],
                }
                for item in failures
            ],
            "results": results,
        }
    finally:
        await client.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call every MCP tool once using the example payloads in tool_call_templates.py.",
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
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Print only the failure summary and failed tool entries.",
    )
    parser.add_argument(
        "--report-file",
        help="Path to write the full JSON report. Default: logs/mcp_tool_test_report_<utc>.json",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))

    try:
        payload = asyncio.run(_run_all_tools(args.toolset))
    except Exception as exc:
        error_payload = {
            "execution_success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        return 1

    report_path = Path(args.report_file) if args.report_file else _default_report_file()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.failures_only:
        payload = {
            "toolset_path": payload["toolset_path"],
            "total_tools": payload["total_tools"],
            "failed_tools": payload["failed_tools"],
            "failures": payload["failures"],
            "report_file": str(report_path),
        }
    else:
        payload = {
            **payload,
            "report_file": str(report_path),
        }

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())