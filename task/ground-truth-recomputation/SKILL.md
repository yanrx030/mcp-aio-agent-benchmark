---
name: ground-truth-recomputation
description: Recomputes and validates benchmark ground truth for AIO MCP evaluation tasks. Use when checking whether stored raw outputs or normalized ground truth have changed after re-executing fixed reference MCP tool plans, especially in live or changing data environments.
compatibility: Designed for Codex or similar coding agents working in the AIO MCP benchmark repository. Requires access to the benchmark dataset, MCP execution module, and Python environment used by the benchmark runner.
metadata:
  author: rosa-yan
  version: "0.1.0"
---

# Ground Truth Recomputation Skill

## Purpose

Use this skill to support semi-automated recomputation of benchmark ground truth for AIO MCP evaluation tasks.

The goal is to detect whether stored benchmark ground truth has become stale because the AIO data infrastructure has changed, including new harvesting, backfilling, NLP pipeline updates, or API/tool behaviour changes.

This skill must preserve the methodological rule:

> The fixed reference tool plan is the source of execution. The coding agent may generate candidate updates, but final acceptance requires human review.

## When to Use

Use this skill when the user explicitly asks to recompute ground truth for the benchmark
Do not use this skill for normal agent benchmark runs

## Inputs

Expected input files:

- benchmark task CSV file
- task records containing:
  - `task_id`
  - `prompt`
  - `category`
  - `raw_output`
  - `ref_tool_call`
  - `answer_type`
  - `core_toolset`
  - `ground_truth`


- MCP tool call execution module - tool_call.py
- evaluator schemas for scalar, timeseries, chart, and text answers.

## Core Principles

1. Never invent tool outputs.
   Always re-execute the stored `ref_tool_call` sequence using the repository's MCP execution module.

2. Do not modify `ref_tool_call` .
   The reference tool plan is fixed during recomputation. If the tool plan appears invalid, flag the task for human review.

3. Do not silently overwrite accepted ground truth.
   Produce candidate updates in a separate output file.

4. Preserve the existing answer schema.
   If a task already has a scalar, timeseries, chart, or text schema, regenerated `ground_truth` must follow the same schema. 
   For text answers, follows the same wording and formatting rules, but update the content based on new tool outputs.

5. Distinguish raw-output changes from answer-level changes.
   Some raw MCP outputs may differ without changing the final expected answer.

6. Classify the cause of change where possible.
   Use categories such as:
   - `no_change`
   - `raw_changed_answer_same`
   - `legitimate_data_update`
   - `tool_or_api_error`
   - `missing_data`
   - `schema_change`
   - `ambiguous_change`
   - `manual_review_required`

7. Human verification is mandatory for accepted updates.
   The agent may propose candidate updates, but the final benchmark file should only be changed after manual approval.

## Workflow

### Step 1: Load benchmark records

Read the benchmark task file and identify records with `ref_tool_call`.

Validate that each selected task has enough information for recomputation:

- task identifier;
- answer type;
- reference tool call sequence;
- stored raw output;
- stored normalized ground truth.

If required fields are missing, do not guess. Mark the task as `manual_review_required`.

### Step 2: Re-execute fixed reference tool calls

Use the repository's `tool_call.py` script as the trusted execution wrapper for collecting current MCP outputs from stored reference tool calls.

This wrapper exposes both:

1. an importable Python function:
2. a command-line interface (not needed for this skill, but useful for manual checks).

```python
from tool_call import execute_tool_call
result = execute_tool_call(
    tool_name="get_collection_summary",
    arguments={"collection": "bluesky"},
    env_file=".env",
)
```

tool_call.py is responsible only for executing one explicit MCP tool call and returning a normalized payload.
The recomputation workflow should treat tool_call.py as the raw-output collection layer.
Do not call alternative tools unless the user explicitly requests a repair attempt.

Multi-Call Task Handling

Many benchmark tasks contain a ref_tool_call list rather than a single call.

For each task:

1. execute calls in their stored order;
2. assign a call_index to each returned result;
3. preserve the original tool_name and arguments;
4. store the normalized result for each call;

### Step 3: Compare new outputs with stored `raw_output`

Compare the newly returned raw output with the stored `raw_output`.


### Step 4: Determine whether the final answer changes

If raw output changed, inspect whether the normalized `ground_truth` should change.

Examples:

- A collection count changes: likely answer-level change for scalar metadata tasks.
- Raw response contains additional metadata not used in answer: likely `raw_changed_answer_same`.
- A topic list changes for an exploratory NLP task: likely answer-level change.
- Tool returns 404 where it previously returned data: likely `missing_data` or `tool_or_api_error`.
- Tool response shape changes: likely `schema_change`.

### Step 5: Generate candidate `ground_truth`

If the final answer is affected by a legitimate data update, generate a candidate replacement for `ground_truth`.

Rules:

- Keep the same top-level schema as the existing `ground_truth`.
- Keep the same `answer_type`.
- Preserve field names.
- Preserve units.
- Preserve ordering rules, especially for time-series points.
- Do not add explanatory fields unless the existing schema allows them.
- For text answers, update the summary only using evidence from the newly executed tool outputs.

### Step 6: Produce an audit report

For every checked task, output an audit entry containing:

- `task_id`
- `status`
- `change_classification`
- `raw_output_changed`
- `ground_truth_changed`
- `requires_manual_review`
- `old_ground_truth`
- `candidate_ground_truth`
- `evidence`
- `notes`

The audit report should make it easy for the researcher to review and accept or reject candidate updates.


### Step 7: Output a new csv file with new candidate ground truth and new raw outputs
if the user requests it, produce a new CSV file that includes candidate ground truth and new raw outputs for all tasks that were re-executed. This file should be separate from the original benchmark file to avoid confusion.