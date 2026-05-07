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

These are required by the benchmark runner, evaluator, and manual tool-call helpers.

## Quick Start

Run the benchmark with defaults:

```bash
uv run --env-file .env main.py
```

Default behavior:

- Task file: `task/tasksmall.csv`
- Toolset: `toolsets/aio_mcp_toolset_v2.json`
- Manifest: `manifest.yaml`
- Profile: `default`
- Tool-agent model: first enabled model in `profiles.default.tool_agent_candidates`

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
uv run --env-file .env main.py --task-file task/task30.csv
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

Run with a specific manifest profile:

```bash
uv run --env-file .env main.py --profile default
```

Override tool-agent model:

```bash
uv run --env-file .env main.py --tool-agent-model "openai/gpt-5.4-mini"
```

Override judge model:

```bash
uv run --env-file .env main.py --judge-model "openai/gpt-5.4-mini"
```

Re-run only the ground-truth evaluator against an existing report:

```bash
uv run --env-file .env run_ground_truth_eval.py --report-file logs/results/<run_dir>/task_run_report.json
```

Compare multiple LLM judges on just the text tasks:

```bash
uv run --env-file .env run_ground_truth_eval.py \
  --report-file logs/results/<run_dir>/task_run_report.json \
  --only-text \
  --judge-model "openai/gpt-5.4-mini" \
  --judge-model "openai/gpt-4.1-mini"
```

Notes:

- `run_ground_truth_eval.py` reuses the original `task_run_report.json` final answers, so it does not re-run the tool-using agent.
- If `--task-file` is omitted, it uses the `source_csv` recorded in the report.
- When a `--judge-model` exists in `manifest.yaml`, the script also reuses that model's OpenRouter/provider settings.

Use a non-default manifest path:

```bash
uv run --env-file .env main.py --manifest path/to/manifest.yaml
```

Override toolset:

```bash
uv run --env-file .env main.py --toolset toolsets/aio_mcp_toolset_v2.json
```

## Manifest-Driven Model Selection

`main.py` now reads model and OpenRouter runtime settings from `manifest.yaml`:

- `openrouter.defaults`
- `model_catalog[*].openrouter`
- `profiles.<name>.openrouter_override`

Runtime merge order:

1. `openrouter.defaults`
2. selected model `openrouter`
3. selected profile `openrouter_override`

If `manifest.yaml` is missing, runner falls back to `nvidia/nemotron-3-super-120b-a12b:free` unless `--tool-agent-model`/`--model` is provided.

## Task CSV Format

Each row needs:

- `task_id`
- `prompt`
- `answer_type` (used to select output JSON schema prompt)
- `ground_truth`
- other metadata columns

- `ref_tool_call` (reference tool call JSON)
- `notes`

Useful metadata columns for deterministic evaluators:

- `scalar_abs_tolerance`, `scalar_rel_tolerance`
- `series_abs_tolerance`, `series_rel_tolerance`
- or shared `abs_tolerance`, `rel_tolerance`

The evaluator uses `math.isclose`, so a value passes when the difference is within either the absolute tolerance or the relative tolerance window.

Notes:

- `ground_truth_tool_call` is still accepted for backward compatibility.

## Output and Logs

Each run creates a task-set directory under:

- `logs/results/<run_dir>/`

Key files:

- `query_runs.jsonl`: one record per query
- `tool_calls/*.jsonl`: per-task tool call logs
- `task_run_report.json`: task summary + per-task results

`main.py` prints the exact paths at the end of each run.

## Manual Tool Probing

For manual tool argument testing, place a JSON tool-call spec in `call.json`:

```json
{
  "tool_name": "text_search",
  "arguments": {}
}
```

Then run:

```bash
uv run python tool_call.py --call-file call.json --pretty
uv run python tool_call.py --call-file call.json --pretty --raw-result-file logs/raw_result.json
```

You can also pass the tool-call JSON directly as the positional argument instead of using `--call-file`.

## Troubleshooting

- `RuntimeError: No tasks selected`
- Check `--task-id`, `--limit`, and CSV structure.

- Missing environment variable errors
- Confirm `.env` contains both required keys and you used `--env-file .env`.

- Dependency/import issues
- Run `uv sync` again and ensure Python version is 3.11+.
