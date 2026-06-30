"""Run multiple planner baselines and emit table-ready summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .baseline_planners import SUPPORTED_BASELINES
from .benchmark_runner import BenchmarkRunner

SUMMARY_COLUMNS = [
    "benchmark_id",
    "benchmark_path",
    "execution_mode",
    "planner_baseline",
    "result_path",
    "total",
    "task_success_count",
    "failed_task_count",
    "false_success_count",
    "repair_attempt_count",
    "repair_success_count",
    "repair_attempt_rate",
    "repair_success_rate",
    "parser_accuracy",
    "planner_tool_coverage_rate",
    "planner_tool_precision",
    "planner_tool_recall",
    "planner_tool_f1",
    "mean_selected_tool_count",
    "mean_tool_sequence_length",
    "mean_extra_tool_count",
    "mean_tool_call_count",
    "failed_tool_call_count",
    "tool_call_failure_rate",
    "verifier_expectation_match",
    "task_success_rate",
]


def run_baseline_suite(
    benchmark_path: str | Path,
    *,
    output_dir: str | Path,
    planner_baselines: list[str] | tuple[str, ...] | None = None,
    execution_mode: str = "mock",
    trace_log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run each requested planner baseline and write JSON/CSV summaries."""

    baselines = _validate_baselines(planner_baselines or SUPPORTED_BASELINES)
    benchmark_file = Path(benchmark_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for baseline in baselines:
        baseline_result_path = output_path / f"{benchmark_file.stem}.{baseline}.json"
        baseline_trace_dir = Path(trace_log_dir) / baseline if trace_log_dir else None
        payload = BenchmarkRunner(
            execution_mode=execution_mode,
            planner_baseline=baseline,
            trace_log_dir=baseline_trace_dir,
        ).run_file(benchmark_file, output_path=baseline_result_path)
        row = flatten_baseline_summary(
            payload,
            benchmark_path=benchmark_file,
            execution_mode=execution_mode,
            result_path=baseline_result_path,
        )
        rows.append(row)
        results[baseline] = {
            "result_path": str(baseline_result_path),
            "summary": payload["summary"],
        }

    summary_json_path = output_path / f"{benchmark_file.stem}.baseline_summary.json"
    summary_csv_path = output_path / f"{benchmark_file.stem}.baseline_summary.csv"
    suite_payload = {
        "benchmark_id": benchmark_file.stem,
        "benchmark_path": str(benchmark_file),
        "execution_mode": execution_mode,
        "planner_baselines": list(baselines),
        "artifacts": {
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
        },
        "rows": rows,
        "results": results,
    }
    summary_json_path.write_text(
        json.dumps(suite_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_csv(rows, summary_csv_path)
    return suite_payload


def flatten_baseline_summary(
    payload: dict[str, Any],
    *,
    benchmark_path: str | Path,
    execution_mode: str,
    result_path: str | Path,
) -> dict[str, Any]:
    """Convert one BenchmarkRunner payload into a single table row."""

    benchmark_file = Path(benchmark_path)
    summary = payload.get("summary", {})
    results = payload.get("results", [])
    task_success_count = sum(1 for item in results if item.get("task_success"))
    total = int(summary.get("total", len(results)))
    return {
        "benchmark_id": benchmark_file.stem,
        "benchmark_path": str(benchmark_file),
        "execution_mode": execution_mode,
        "planner_baseline": payload.get("planner_baseline"),
        "result_path": str(result_path),
        "total": total,
        "task_success_count": task_success_count,
        "failed_task_count": total - task_success_count,
        "false_success_count": summary.get("false_success_count", 0),
        "repair_attempt_count": summary.get("repair_attempt_count", 0),
        "repair_success_count": summary.get("repair_success_count", 0),
        "repair_attempt_rate": summary.get("repair_attempt_rate", 0.0),
        "repair_success_rate": summary.get("repair_success_rate", 0.0),
        "parser_accuracy": summary.get("parser_accuracy", 0.0),
        "planner_tool_coverage_rate": summary.get("planner_tool_coverage_rate", 0.0),
        "planner_tool_precision": summary.get("planner_tool_precision", 0.0),
        "planner_tool_recall": summary.get("planner_tool_recall", 0.0),
        "planner_tool_f1": summary.get("planner_tool_f1", 0.0),
        "mean_selected_tool_count": summary.get("mean_selected_tool_count", 0.0),
        "mean_tool_sequence_length": summary.get("mean_tool_sequence_length", 0.0),
        "mean_extra_tool_count": summary.get("mean_extra_tool_count", 0.0),
        "mean_tool_call_count": summary.get("mean_tool_call_count", 0.0),
        "failed_tool_call_count": summary.get("failed_tool_call_count", 0),
        "tool_call_failure_rate": summary.get("tool_call_failure_rate", 0.0),
        "verifier_expectation_match": summary.get("verifier_expectation_match", 0.0),
        "task_success_rate": summary.get("task_success_rate", 0.0),
    }


def write_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write suite rows as a stable CSV table."""

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _validate_baselines(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    baselines: list[str] = []
    for value in values:
        if value not in SUPPORTED_BASELINES:
            raise ValueError(f"Unsupported planner baseline: {value}")
        if value not in seen:
            baselines.append(value)
            seen.add(value)
    return tuple(baselines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple CAi planner baselines for one benchmark JSONL.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-baseline outputs and summary tables.")
    parser.add_argument("--execution-mode", choices=["mock", "real"], default="mock")
    parser.add_argument(
        "--planner-baselines",
        nargs="+",
        choices=SUPPORTED_BASELINES,
        default=list(SUPPORTED_BASELINES),
        help="Planner baselines to run in order.",
    )
    parser.add_argument("--trace-log-dir", help="Optional trace log root. Each baseline gets a subdirectory.")
    args = parser.parse_args()
    result = run_baseline_suite(
        args.benchmark,
        output_dir=args.output_dir,
        planner_baselines=args.planner_baselines,
        execution_mode=args.execution_mode,
        trace_log_dir=args.trace_log_dir,
    )
    print(
        json.dumps(
            {
                "summary_json": result["artifacts"]["summary_json"],
                "summary_csv": result["artifacts"]["summary_csv"],
                "rows": result["rows"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
