import argparse
import asyncio
import json
import os
from pathlib import Path

from MCPClient import MCPClient, JwtTokenManager, is_authentication_error, is_transport_error
from eval_logger import JSONLLogger
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
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum number of benchmark tasks to run concurrently.",
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
    selected_judge_openrouter_params: dict | None = None
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
        selected_judge_openrouter_params = (
            manifest_config.judge.openrouter_params if manifest_config.judge else None
        )

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
    shared_logger = JSONLLogger(task_set_prefix=selected_model)
    shared_logger.write_run_metadata(
        tool_agent_model=selected_model,
        judge_model=selected_judge_model,
    )
    jwt_manager = JwtTokenManager(auth_key=aio_key, login_url=MCPClient.LOGIN_URL)
    await jwt_manager.initialize()

    if args.prompt:
        system_prompt, resolved_answer_type = resolve_system_prompt(args.answer_type)
        max_prompt_attempts = 2
        for attempt in range(1, max_prompt_attempts + 1):
            token = await jwt_manager.get_token()
            prompt_client = MCPClient(
                auth_key=aio_key,
                openrouter_api_key=openrouter_key,
                openrouter_model=selected_model,
                openrouter_params=selected_openrouter_params,
                logger=shared_logger,
                jwt_token=token,
            )
            cleanup_warning: str | None = None
            try:
                prompt_client.load_tools(args.toolset)
                await prompt_client.connect_to_server()
                result = await prompt_client.process_query(
                    args.prompt,
                    return_trace=True,
                    system_prompt=system_prompt,
                    system_prompt_label=resolved_answer_type or "default",
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return
            except Exception as exc:
                retryable = is_authentication_error(exc) or is_transport_error(exc)
                if retryable and attempt < max_prompt_attempts:
                    if is_authentication_error(exc):
                        await jwt_manager.refresh_if_stale_or_forced(force=True, failed_token=token)
                    await asyncio.sleep(0.75 * (2 ** (attempt - 1)))
                    continue
                raise
            finally:
                cleanup_warning = await prompt_client.cleanup()
                if cleanup_warning:
                    print(f"Prompt cleanup warning: {cleanup_warning}")

    tasks = load_benchmark_tasks(args.task_file)
    selected_tasks = select_tasks(
        tasks,
        task_ids=set(args.task_ids or []),
        limit=args.limit,
    )

    if not selected_tasks:
        raise RuntimeError("No tasks selected. Check --task-id, --limit, or the input CSV.")

    print(f"Loaded {len(tasks)} tasks from {args.task_file}. Running {len(selected_tasks)} task(s).")
    results = await run_benchmark_tasks(
        selected_tasks,
        auth_key=aio_key,
        openrouter_api_key=openrouter_key,
        openrouter_model=selected_model,
        judge_model=selected_judge_model,
        judge_openrouter_params=selected_judge_openrouter_params,
        toolset_path=args.toolset,
        jwt_manager=jwt_manager,
        shared_logger=shared_logger,
        openrouter_params=selected_openrouter_params,
        max_concurrency=args.max_concurrency,
        max_retries=1,
    )
    report_path = write_task_run_report(
        results,
        source_csv=args.task_file,
        output_dir=shared_logger.task_set_dir,
        extra_summary={
            "model": selected_model,
            "jwt_login_count": jwt_manager.login_count,
            "jwt_refresh_count": jwt_manager.refresh_count,
        },
    )

    print(json.dumps(summarize_results(results), ensure_ascii=False, indent=2))
    print(f"Detailed report written to {report_path}")
    print(f"Query run log written to {shared_logger.query_runs_path}")
    print(f"Per-task tool call logs written under {shared_logger.tool_calls_dir}")
    print(
        f"JWT login count: {jwt_manager.login_count}, refresh count: {jwt_manager.refresh_count}"
    )


if __name__ == "__main__":
    asyncio.run(main())
