#!/usr/bin/env python3
"""Rank terms from a raw MCP result by count and report the top N."""

import argparse
import json
from pathlib import Path
from typing import Any


def extract_terms(result: dict[str, Any]) -> dict[str, int]:
    """Extract the terms dictionary from the raw result structure."""
    # Navigate the nested structure: __model_dump__ -> structuredContent -> terms
    try:
        model_dump = result.get("__model_dump__", {})
        structured_content = model_dump.get("structuredContent", {})
        terms = structured_content.get("terms", {})
        return terms
    except (TypeError, AttributeError):
        return {}


def rank_terms(terms: dict[str, int], limit: int = 20) -> list[tuple[str, int]]:
    """Sort terms by count descending and return top N."""
    return sorted(terms.items(), key=lambda x: x[1], reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank terms from a raw MCP result by count."
    )
    parser.add_argument(
        "input_file",
        help="Path to the raw result JSON file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top results to display. Default: 20",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format. Default: table",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        result = json.load(f)

    terms = extract_terms(result)
    if not terms:
        print("No terms found in result.")
        return

    ranked = rank_terms(terms, args.top)

    if args.format == "table":
        print(f"Top {len(ranked)} Terms by Count:")
        print("-" * 50)
        print(f"{'Rank':<6}{'Term':<30}{'Count':<10}")
        print("-" * 50)
        for rank, (term, count) in enumerate(ranked, start=1):
            print(f"{rank:<6}{term:<30}{count:<10}")
        print("-" * 50)
        print(f"Total unique terms: {len(terms)}")

    elif args.format == "json":
        output = [{"rank": rank, "term": term, "count": count} for rank, (term, count) in enumerate(ranked, start=1)]
        print(json.dumps(output, ensure_ascii=False, indent=2))

    elif args.format == "csv":
        print("rank,term,count")
        for rank, (term, count) in enumerate(ranked, start=1):
            # Escape quotes in term
            term_escaped = term.replace('"', '""')
            print(f'{rank},"{term_escaped}",{count}')


if __name__ == "__main__":
    main()
