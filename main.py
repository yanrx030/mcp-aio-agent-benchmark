import argparse
import asyncio
import json
import os
from pathlib import Path

from MCPClient import MCPClient
from manifest_loader import resolve_runner_manifest
from prompts import resolve_system_prompt
from task_runner import (
    load_benchmark_tasks,
    run_benchmark_tasks,
    select_tasks,
    summarize_results,
    write_task_run_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIRED evaluation tasks.")
    parser.add_argument(
        "--manifest",
        default="manifest.yaml",
        help="Runner manifest YAML path.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Manifest profile to use.",
    )
    parser.add_argument(
        "--task-file",
        default="task/tasksmall.csv",
        nargs="?",
        const="task/tasksmall.csv",
        help="CSV file containing benchmark tasks.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Run only the specified task id. Repeat to select multiple tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected tasks.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Run a single ad hoc prompt instead of the task CSV.",
    )
    parser.add_argument(
        "--answer-type",
        default=None,
        help="Optional answer type for --prompt (scalar, timeseries, text, chart).",
    )
    parser.add_argument(
        "--toolset",
        default="toolsets/aio_mcp_toolset_v2.json",
        help="Frozen toolset JSON to load before running tasks.",
    )
    parser.add_argument(
        "--tool-agent-model",
        default=None,
        help="Override selected tool-agent model id from manifest.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override selected judge model id from manifest.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Legacy alias for --tool-agent-model.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    default_tool_agent_model = "nvidia/nemotron-3-super-120b-a12b:free"
    tool_agent_override = args.tool_agent_model or args.model
    selected_model = tool_agent_override or default_tool_agent_model
    selected_judge_model: str | None = args.judge_model
    selected_openrouter_params: dict | None = None

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest_config = resolve_runner_manifest(
            manifest_path,
            profile=args.profile,
            tool_agent_model_override=tool_agent_override,
            judge_model_override=args.judge_model,
        )
        selected_model = manifest_config.tool_agent.model_id
        selected_openrouter_params = manifest_config.tool_agent.openrouter_params
        selected_judge_model = manifest_config.judge.model_id if manifest_config.judge else None

        print(
            f"Using manifest profile '{manifest_config.profile}' from {manifest_config.source_path}"
        )
        print(f"Selected tool-agent model: {selected_model}")
        if selected_judge_model:
            print(f"Selected judge model: {selected_judge_model}")
    elif args.manifest != "manifest.yaml":
        raise RuntimeError(f"Manifest file not found: {manifest_path}")
    elif selected_model:
        print(f"Manifest not found. Falling back to tool-agent model: {selected_model}")

    aio_key = os.environ["AIO_AUTH_KEY"]
    openrouter_key = os.environ["OPENROUTER_API_KEY"]
    client = MCPClient(
        auth_key=aio_key,
        openrouter_api_key=openrouter_key,
        openrouter_model=selected_model,
        openrouter_params=selected_openrouter_params,
    )

    try:
        client.authenticate()
        client.load_tools(args.toolset)
        await client.connect_to_server()

        if args.prompt:
            system_prompt, resolved_answer_type = resolve_system_prompt(args.answer_type)
            result = await client.process_query(
                args.prompt,
                return_trace=True,
                system_prompt=system_prompt,
                system_prompt_label=resolved_answer_type or "default",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        tasks = load_benchmark_tasks(args.task_file)
        selected_tasks = select_tasks(
            tasks,
            task_ids=set(args.task_ids or []),
            limit=args.limit,
        )

        if not selected_tasks:
            raise RuntimeError("No tasks selected. Check --task-id, --limit, or the input CSV.")

        print(f"Loaded {len(tasks)} tasks from {args.task_file}. Running {len(selected_tasks)} task(s).")
        results = await run_benchmark_tasks(client, selected_tasks)
        report_path = write_task_run_report(
            results,
            source_csv=args.task_file,
            output_dir=client.logger.task_set_dir,
        )

        print(json.dumps(summarize_results(results), ensure_ascii=False, indent=2))
        print(f"Detailed report written to {report_path}")
        print(f"Query run log written to {client.logger.query_runs_path}")
        print(f"Per-task tool call logs written under {client.logger.tool_calls_dir}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
