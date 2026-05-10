from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_call import execute_tool_calls


def parse_json_objects(raw: str | None) -> list[dict[str, Any]]:
    if not raw or not raw.strip():
        return []

    text = raw.strip()
    if text.startswith('"') and text.endswith('"') and '""' in text:
        text = text[1:-1].replace('""', '"').strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, str):
            text = parsed.strip()
            if text.startswith('"') and text.endswith('"') and '""' in text:
                text = text[1:-1].replace('""', '"').strip()
        elif isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at offset {idx}")
        objects.append(obj)
        idx = end
    return objects


def parse_json_payload(raw: str | None) -> Any:
    if not raw or not raw.strip():
        return None
    return json.loads(raw)


def relevant_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": payload.get("tool_name"),
        "arguments": payload.get("arguments") or {},
        "execution_success": payload.get("execution_success"),
        "mcp_is_error": payload.get("mcp_is_error"),
        "payload_has_error": payload.get("payload_has_error"),
        "structured_content": payload.get("structured_content"),
        "content_blocks": payload.get("content_blocks"),
        "server_error_message": payload.get("server_error_message"),
    }


def signature(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": payload.get("tool_name"),
            "arguments": payload.get("arguments") or {},
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def comparable_by_signature(payloads: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[signature(payload)].append(relevant_payload(payload))
    return dict(grouped)


def first_structured_value(payloads: list[dict[str, Any]], key: str) -> Any:
    for payload in payloads:
        structured = payload.get("structured_content")
        if isinstance(structured, dict) and key in structured:
            return structured[key]
        if isinstance(structured, dict) and isinstance(structured.get("result"), list):
            for item in structured["result"]:
                if isinstance(item, dict) and key in item:
                    return item[key]
    return None


def tool_failed(payloads: list[dict[str, Any]]) -> bool:
    return any(
        payload.get("execution_success") is False
        or payload.get("mcp_is_error") is True
        or payload.get("payload_has_error") is True
        or payload.get("server_error_message")
        for payload in payloads
    )


def shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def infer_candidate_ground_truth(
    *,
    answer_type: str,
    old_ground_truth: Any,
    old_payloads: list[dict[str, Any]],
    new_payloads: list[dict[str, Any]],
) -> tuple[Any, bool, str]:
    if answer_type == "scalar" and isinstance(old_ground_truth, dict) and "value" in old_ground_truth:
        old_count = first_structured_value(old_payloads, "count")
        new_count = first_structured_value(new_payloads, "count")
        if old_count == old_ground_truth.get("value") and new_count is not None:
            candidate = dict(old_ground_truth)
            candidate["value"] = new_count
            return candidate, candidate != old_ground_truth, "scalar count was refreshed from structured_content.count"
    return old_ground_truth, False, "no automated ground-truth rewrite rule was applied"


def summarize_changes(old_payloads: list[dict[str, Any]], new_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old_by_sig = comparable_by_signature(old_payloads)
    new_by_sig = comparable_by_signature(new_payloads)
    changes = []
    for sig in sorted(set(old_by_sig) | set(new_by_sig)):
        old_value = old_by_sig.get(sig)
        new_value = new_by_sig.get(sig)
        if old_value == new_value:
            continue
        changes.append(
            {
                "call": json.loads(sig),
                "old_shape": shape(old_value),
                "new_shape": shape(new_value),
                "old_excerpt": old_value,
                "new_excerpt": new_value,
            }
        )
    return changes


def compact_raw_output(payloads: list[dict[str, Any]]) -> str:
    if len(payloads) == 1:
        return json.dumps(payloads[0], ensure_ascii=False, indent=2)
    return "\n\n".join(json.dumps(payload, ensure_ascii=False) for payload in payloads)


def recompute(csv_path: Path, env_file: str, toolset: str) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = csv_path.parent / f"recompute_{csv_path.stem}_{timestamp}.json"
    changed_csv_path = csv_path.parent / f"recompute_{csv_path.stem}_{timestamp}.csv"

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    audit_entries = []
    changed_rows = []

    for row in rows:
        task_id = row.get("task_id", "")
        try:
            calls = parse_json_objects(row.get("ref_tool_call"))
            if not calls:
                raise ValueError("missing ref_tool_call")
        except Exception as exc:
            entry = {
                "task_id": task_id,
                "change_classification": "manual_review_required",
                "raw_output_changed": None,
                "ground_truth_changed": False,
                "requires_manual_review": True,
                "old_ground_truth": row.get("ground_truth"),
                "candidate_ground_truth": row.get("ground_truth"),
                "evidence": {"parse_error": str(exc)},
                "notes": "Could not parse task record fields.",
            }
            audit_entries.append(entry)
            changed_rows.append(csv_changed_row(row, "", row.get("ground_truth"), entry))
            continue

        note_parse_error = None
        try:
            old_payloads = parse_json_objects(row.get("note"))
        except Exception as exc:
            old_payloads = []
            note_parse_error = str(exc)

        try:
            old_ground_truth = parse_json_payload(row.get("ground_truth"))
        except Exception as exc:
            entry = {
                "task_id": task_id,
                "change_classification": "manual_review_required",
                "raw_output_changed": None,
                "ground_truth_changed": False,
                "requires_manual_review": True,
                "old_ground_truth": row.get("ground_truth"),
                "candidate_ground_truth": row.get("ground_truth"),
                "evidence": {"parse_error": str(exc), "field": "ground_truth"},
                "notes": "Could not parse task ground_truth field.",
            }
            audit_entries.append(entry)
            changed_rows.append(csv_changed_row(row, "", row.get("ground_truth"), entry))
            continue

        print(f"recomputing {task_id} ({len(calls)} call(s))", flush=True)
        try:
            new_payloads = execute_tool_calls(calls, env_file=env_file, toolset=toolset)
        except Exception as exc:
            entry = {
                "task_id": task_id,
                "change_classification": "tool_or_api_error",
                "raw_output_changed": None,
                "ground_truth_changed": False,
                "requires_manual_review": True,
                "old_ground_truth": old_ground_truth,
                "candidate_ground_truth": old_ground_truth,
                "evidence": {"execution_error": f"{type(exc).__name__}: {exc}"},
                "notes": "Reference calls could not be re-executed.",
            }
            audit_entries.append(entry)
            changed_rows.append(csv_changed_row(row, "", old_ground_truth, entry))
            continue

        raw_output_changed = (
            None
            if note_parse_error
            else comparable_by_signature(old_payloads) != comparable_by_signature(new_payloads)
        )
        candidate, gt_changed, candidate_note = infer_candidate_ground_truth(
            answer_type=(row.get("answer_type") or "").strip(),
            old_ground_truth=old_ground_truth,
            old_payloads=old_payloads,
            new_payloads=new_payloads,
        )

        if note_parse_error:
            classification = "manual_review_required"
            requires_manual_review = True
            gt_changed = False
            candidate = old_ground_truth
        elif not raw_output_changed:
            classification = "no_change"
            requires_manual_review = False
            gt_changed = False
            candidate = old_ground_truth
        elif tool_failed(new_payloads):
            classification = "tool_or_api_error"
            requires_manual_review = True
        elif gt_changed:
            classification = "legitimate_data_update"
            requires_manual_review = True
        else:
            old_shapes = [shape(item.get("structured_content")) for item in old_payloads]
            new_shapes = [shape(item.get("structured_content")) for item in new_payloads]
            classification = "schema_change" if old_shapes != new_shapes else "ambiguous_change"
            requires_manual_review = True

        evidence = {
            "calls_executed": calls,
            "changed_calls": summarize_changes(old_payloads, new_payloads) if raw_output_changed else [],
            "candidate_note": candidate_note,
        }
        if note_parse_error:
            evidence["stored_note_parse_error"] = note_parse_error
        entry = {
            "task_id": task_id,
            "change_classification": classification,
            "raw_output_changed": raw_output_changed,
            "ground_truth_changed": gt_changed,
            "requires_manual_review": requires_manual_review,
            "old_ground_truth": old_ground_truth,
            "candidate_ground_truth": candidate,
            "evidence": evidence,
            "notes": "Original benchmark CSV was not modified.",
        }
        audit_entries.append(entry)

        if raw_output_changed or gt_changed or note_parse_error:
            changed_rows.append(csv_changed_row(row, compact_raw_output(new_payloads), candidate, entry))

    counts = Counter(entry["change_classification"] for entry in audit_entries)
    audit_payload = {
        "source_csv": str(csv_path),
        "generated_at": timestamp,
        "task_count": len(audit_entries),
        "classification_counts": dict(sorted(counts.items())),
        "entries": audit_entries,
    }
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with changed_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "task_id",
            "prompt",
            "new_raw_output",
            "candidate_ground_truth",
            "change_classification",
            "requires_manual_review",
            "evidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changed_rows)

    return audit_path, changed_csv_path


def csv_changed_row(
    source_row: dict[str, str],
    new_raw_output: str,
    candidate_ground_truth: Any,
    audit_entry: dict[str, Any],
) -> dict[str, str]:
    return {
        "task_id": source_row.get("task_id", ""),
        "prompt": source_row.get("prompt", ""),
        "new_raw_output": new_raw_output,
        "candidate_ground_truth": json.dumps(candidate_ground_truth, ensure_ascii=False, indent=2),
        "change_classification": audit_entry["change_classification"],
        "requires_manual_review": str(audit_entry["requires_manual_review"]).lower(),
        "evidence": json.dumps(audit_entry["evidence"], ensure_ascii=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute task ground truth audit artifacts.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--toolset", default="toolsets/aio_mcp_toolset_v2.json")
    args = parser.parse_args()

    audit_path, changed_csv_path = recompute(args.csv_path, args.env_file, args.toolset)
    print(f"Wrote audit JSON: {audit_path}")
    print(f"Wrote changed CSV: {changed_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
