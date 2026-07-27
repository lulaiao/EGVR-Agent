"""Build compact tables for offline biomedical generalization slices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .biomedical_benchmark_runner import run_biomedical_benchmark


BIOMEDICAL_GENERALIZATION_COLUMNS = [
    "benchmark_id",
    "domain",
    "task_count",
    "workflow_success_rate",
    "tool_selection_accuracy",
    "mean_evidence_coverage",
    "mean_provenance_coverage",
    "verifier_expectation_match_rate",
    "false_success_count",
    "missing_evidence_task_count",
    "mean_tool_call_count",
]


def build_biomedical_generalization_table(benchmark_paths: list[str | Path]) -> dict[str, Any]:
    """Run offline biomedical slices and return paper-friendly table rows."""

    rows: list[dict[str, Any]] = []
    for benchmark_path in benchmark_paths:
        summary = run_biomedical_benchmark(benchmark_path)
        detail_rows = list(summary.get("rows") or [])
        domains = sorted({str(row.get("domain") or "") for row in detail_rows if row.get("domain")})
        rows.append(
            {
                "benchmark_id": summary.get("benchmark_id"),
                "domain": ", ".join(domains),
                "task_count": summary.get("task_count"),
                "workflow_success_rate": summary.get("workflow_success_rate"),
                "tool_selection_accuracy": summary.get("tool_selection_accuracy"),
                "mean_evidence_coverage": summary.get("mean_evidence_coverage"),
                "mean_provenance_coverage": summary.get("mean_provenance_coverage"),
                "verifier_expectation_match_rate": summary.get("verifier_expectation_match_rate"),
                "false_success_count": summary.get("false_success_count"),
                "missing_evidence_task_count": sum(
                    1 for row in detail_rows if int(row.get("missing_evidence_count") or 0) > 0
                ),
                "mean_tool_call_count": _mean(row.get("tool_call_count") for row in detail_rows),
            }
        )
    return {
        "table_id": "biomedical_generalization_table_v1",
        "columns": list(BIOMEDICAL_GENERALIZATION_COLUMNS),
        "row_count": len(rows),
        "rows": rows,
        "notes": (
            "Offline supporting evidence slices for biomedical workflow generalization; "
            "not clinical prediction, DTI, repurposing, or drug-discovery SOTA evidence."
        ),
    }


def write_biomedical_generalization_table(table: dict[str, Any], output_path: str | Path) -> None:
    """Write table JSON and sibling CSV using a stable column order."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BIOMEDICAL_GENERALIZATION_COLUMNS)
        writer.writeheader()
        writer.writerows(table.get("rows") or [])


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a biomedical generalization table from offline slices.")
    parser.add_argument("--benchmark", action="append", required=True, help="Benchmark JSONL path. Repeatable.")
    parser.add_argument("--output", required=True, help="Output JSON path. A sibling CSV is also written.")
    args = parser.parse_args()
    table = build_biomedical_generalization_table(args.benchmark)
    write_biomedical_generalization_table(table, args.output)
    print(json.dumps({key: value for key, value in table.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
