# AIRED Eval Runner

Lightweight benchmark runner for evaluating a tool-using LLM agent against AIRED tasks.

## Prerequisites

- Python 3.11+
- `uv` installed
- Access to AIRED auth key and OpenRouter API key

## Installation

```bash
uv sync
```

## Environment Variables

Create a `.env` file in the repo root:

```env
AIO_AUTH_KEY=your_aio_auth_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

These are required by `main.py` and `manual_ground_truth_collector.py`.

## Quick Start

Run the benchmark with defaults:

```bash
uv run --env-file .env main.py
```

Default behavior:
- Task file: `task/taskv3.csv`
- Toolset: `toolsets/aio_mcp_toolset_v2.json`
- Model: `nvidia/nemotron-3-super-120b-a12b:free`

## Common Commands

Run first N tasks:

```bash
uv run --env-file .env main.py --limit 2
```

Run specific task IDs:

```bash
uv run --env-file .env main.py --task-id 1 --task-id 4
```

Use a custom task CSV:

```bash
uv run --env-file .env main.py --task-file task/taskv3.csv
```

Run a single ad hoc prompt:

```bash
uv run --env-file .env main.py --prompt "How many Reddit posts are available from January to March 2025?"
```

Run ad hoc prompt with explicit output format schema:

```bash
uv run --env-file .env main.py --prompt "Show monthly post counts for 2025." --answer-type timeseries
```

Supported `--answer-type` values:
- `scalar`
- `timeseries`
- `text`
- `chart`

Override model/toolset:

```bash
uv run --env-file .env main.py --model "openai/gpt-4.1-mini" --toolset toolsets/aio_mcp_toolset_v2.json
```

## Task CSV Format

At minimum, each row needs:
- `task_id`
- `prompt`

Recommended fields for evaluation:
- `answer_type` (used to select output JSON schema prompt)
- `ref_tool_call` (reference tool call JSON)
- `ground_truth`
- other metadata columns as needed

Notes:
- `ref_tool_call` is preferred.
- `ground_truth_tool_call` is still accepted for backward compatibility.

## Output and Logs

Each run creates a task-set directory under:

- `logs/task_sets/<task_set_id>/`

Key files:
- `query_runs.jsonl`: one record per query
- `tool_calls/*.jsonl`: per-task tool call logs
- `task_run_report.json`: task summary + per-task results

`main.py` prints the exact paths at the end of each run.

## Manual Tool Probing

For manual tool argument testing:

```bash
uv run --env-file .env manual_ground_truth_collector.py
```

This launches an interactive prompt to select tools and provide JSON arguments.

## Troubleshooting

- `RuntimeError: No tasks selected`
- Check `--task-id`, `--limit`, and CSV structure.

- Missing environment variable errors
- Confirm `.env` contains both required keys and you used `--env-file .env`.

- Dependency/import issues
- Run `uv sync` again and ensure Python version is 3.11+.



