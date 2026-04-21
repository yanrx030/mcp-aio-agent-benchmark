import argparse
import asyncio
import json
import os

from MCPClient import MCPClient
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
        "--task-file",
        default="task/taskv3.csv",
        nargs="?",
        const="task/taskv3.csv",
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
        "--model",
        default="nvidia/nemotron-3-super-120b-a12b:free",
        help="OpenRouter model name.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    aio_key = os.environ["AIO_AUTH_KEY"]
    openrouter_key = os.environ["OPENROUTER_API_KEY"]
    client = MCPClient(
        auth_key=aio_key,
        openrouter_api_key=openrouter_key,
        openrouter_model=args.model,
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
