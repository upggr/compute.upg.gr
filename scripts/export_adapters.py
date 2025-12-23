#!/usr/bin/env python3
"""
Export adapter for upg-strings results.

Usage:
  python scripts/export_adapters.py --input results.json --format cytools --output cytools.json
"""

import argparse
import json
from pathlib import Path


def export_candidates(results):
    return results.get("top_results", [])


def export_json(results):
    return json.dumps(results, indent=2)


def export_csv(results):
    candidates = export_candidates(results)
    if not candidates:
        return ""
    headers = sorted({key for candidate in candidates for key in candidate.keys()})
    lines = [",".join(headers)]
    for candidate in candidates:
        row = []
        for header in headers:
            value = candidate.get(header, "")
            value_str = "" if value is None else str(value).replace('"', '""')
            row.append(f'"{value_str}"')
        lines.append(",".join(row))
    return "\n".join(lines)


def export_wrapped(results, schema_name):
    payload = {
        "schema": schema_name,
        "run_metadata": results.get("run_metadata", {}),
        "candidates": export_candidates(results)
    }
    return json.dumps(payload, indent=2)


def export_sage(results):
    payload = export_candidates(results)
    return "candidates = " + repr(payload)


def export_mathematica(results):
    candidates = export_candidates(results)

    def format_value(value):
        if isinstance(value, bool):
            return "True" if value else "False"
        if value is None:
            return "Null"
        if isinstance(value, (int, float)):
            return str(value)
        return f"\"{str(value)}\""

    rows = []
    for candidate in candidates:
        items = [f'"{key}" -> {format_value(value)}' for key, value in candidate.items()]
        rows.append("<|" + ", ".join(items) + "|>")
    return "candidates = {" + ", ".join(rows) + "};"


def main():
    parser = argparse.ArgumentParser(description="Export upg-strings results to tool-friendly formats.")
    parser.add_argument("--input", required=True, help="Path to results JSON")
    parser.add_argument("--format", required=True, choices=["json", "csv", "cytools", "cymetric", "sage", "mathematica"])
    parser.add_argument("--output", required=True, help="Output file path")
    args = parser.parse_args()

    results_path = Path(args.input)
    results = json.loads(results_path.read_text())

    if args.format == "json":
        content = export_json(results)
    elif args.format == "csv":
        content = export_csv(results)
    elif args.format == "cytools":
        content = export_wrapped(results, "cytools-candidates-v1")
    elif args.format == "cymetric":
        content = export_wrapped(results, "cymetric-candidates-v1")
    elif args.format == "sage":
        content = export_sage(results)
    elif args.format == "mathematica":
        content = export_mathematica(results)
    else:
        raise ValueError("Unsupported format")

    Path(args.output).write_text(content)


if __name__ == "__main__":
    main()
