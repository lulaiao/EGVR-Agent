"""Run lightweight offline biomedical generalization benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .domain_router import execute_and_verify_domain, parse_domain_task, plan_domain_workflow


def run_biomedical_benchmark(benchmark_path: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in _read_jsonl(benchmark_path):
        parsed = parse_domain_task(
            str(task.get("raw_user_query") or ""),
            task_id=task.get("task_id"),
            metadata=task.get("metadata") or {},
        )
        workflow = plan_domain_workflow(parsed)
        tool_calls, _candidates, verifier = execute_and_verify_domain(parsed, workflow)
        expected_tools = list(task.get("expected_tools") or [])
        selected_tools = list(workflow.selected_tools)
        provenance_coverage = _provenance_coverage(verifier.metadata.get("evidence_records") or [])
        rows.append(
            {
                "task_id": parsed.task_id,
                "domain": parsed.metadata.get("domain"),
                "task_type": parsed.task_type,
                "selected_tools": selected_tools,
                "expected_tools": expected_tools,
                "tool_selection_match": selected_tools == expected_tools if expected_tools else None,
                "task_success": verifier.success,
                "evidence_coverage": verifier.metrics.get("evidence_coverage"),
                "provenance_coverage": provenance_coverage,
                "missing_evidence_count": verifier.metrics.get("missing_evidence_count"),
                "verifier_expectation_match": verifier.success == bool(task.get("should_succeed", True)),
                "false_success": verifier.success and not bool(task.get("should_succeed", True)),
                "tool_call_count": len(tool_calls),
                "failure_reason": verifier.failure_reason,
            }
        )
    return {
        "benchmark_id": Path(benchmark_path).stem,
        "task_count": len(rows),
        "workflow_success_rate": _rate(row["task_success"] for row in rows),
        "tool_selection_accuracy": _rate(row["tool_selection_match"] for row in rows if row["tool_selection_match"] is not None),
        "mean_evidence_coverage": _mean(row["evidence_coverage"] for row in rows),
        "mean_provenance_coverage": _mean(row["provenance_coverage"] for row in rows),
        "verifier_expectation_match_rate": _rate(row["verifier_expectation_match"] for row in rows),
        "false_success_count": sum(1 for row in rows if row["false_success"]),
        "rows": rows,
        "notes": "Offline biomedical generalization slice; no clinical prediction, DTI, or repurposing SOTA claim.",
    }


def write_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    rows = list(summary.get("rows") or [])
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _rate(values) -> float | None:
    flags = list(values)
    if not flags:
        return None
    return sum(1 for value in flags if value) / len(flags)


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _provenance_coverage(evidence_records: list[dict[str, Any]]) -> float | None:
    if not evidence_records:
        return None
    covered = 0
    for record in evidence_records:
        has_value = record.get("value") not in (None, "")
        has_provenance = bool(record.get("source") or record.get("provenance"))
        if has_value and has_provenance:
            covered += 1
    return covered / len(evidence_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline clinical/drug-target generalization slices.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_biomedical_benchmark(args.benchmark)
    write_summary(summary, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
