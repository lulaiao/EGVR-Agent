"""Run and summarize task-family generalization slices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .baseline_planners import RULE_BASED_PLANNER, SUPPORTED_BASELINES
from .benchmark_runner import BenchmarkRunner, load_benchmark_cases


TASK_GENERALIZATION_SUMMARY_COLUMNS = [
    "task_type",
    "benchmark_id",
    "dataset",
    "execution_mode",
    "planner_baseline",
    "task_count",
    "task_success_rate",
    "valid_candidate_rate",
    "verifier_expectation_match",
    "mean_elapsed_sec",
    "tools",
    "notes",
]


def run_task_generalization_summary(
    benchmark_path: str | Path,
    *,
    output_dir: str | Path,
    planner_baseline: str = RULE_BASED_PLANNER,
    execution_mode: str = "mock",
    trace_log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one task-generalization JSONL and aggregate metrics by parsed task type."""

    if planner_baseline not in SUPPORTED_BASELINES:
        raise ValueError(f"Unsupported planner baseline: {planner_baseline}")
    benchmark_file = Path(benchmark_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_json_path = output_path / f"{benchmark_file.stem}.{planner_baseline}.json"
    result_csv_path = output_path / "task_generalization_summary.csv"
    result_summary_path = output_path / "task_generalization_summary.json"
    cases = load_benchmark_cases(benchmark_file)
    case_metadata = {case.task_id: case.metadata for case in cases}
    payload = BenchmarkRunner(
        execution_mode=execution_mode,
        planner_baseline=planner_baseline,
        trace_log_dir=trace_log_dir,
    ).run_file(benchmark_file, output_path=result_json_path)
    rows = _rows_by_task_type(
        payload.get("results", []),
        benchmark_id=benchmark_file.stem,
        execution_mode=execution_mode,
        planner_baseline=planner_baseline,
        case_metadata=case_metadata,
    )
    summary = {
        "benchmark_id": benchmark_file.stem,
        "benchmark_path": str(benchmark_file),
        "execution_mode": execution_mode,
        "planner_baseline": planner_baseline,
        "artifacts": {
            "result_json": str(result_json_path),
            "summary_json": str(result_summary_path),
            "summary_csv": str(result_csv_path),
        },
        "rows": rows,
    }
    result_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(rows, result_csv_path)
    return summary


def _rows_by_task_type(
    results: list[dict[str, Any]],
    *,
    benchmark_id: str,
    execution_mode: str,
    planner_baseline: str,
    case_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if isinstance(result, dict):
            grouped.setdefault(str(result.get("parsed_task_type") or "unknown"), []).append(result)

    rows: list[dict[str, Any]] = []
    for task_type in sorted(grouped):
        items = grouped[task_type]
        task_count = len(items)
        task_success_count = sum(1 for item in items if item.get("task_success"))
        verifier_match_count = sum(1 for item in items if item.get("verifier_matched_expectation"))
        candidate_counts = [_metric_number(item, "candidate_count") for item in items]
        valid_counts = [_metric_number(item, "valid_smiles_count") for item in items]
        elapsed = [_as_float(item.get("total_elapsed_sec")) for item in items]
        tools = sorted({tool for item in items for tool in item.get("selected_tools", [])})
        datasets = {
            str(case_metadata.get(str(item.get("task_id")), {}).get("dataset"))
            for item in items
            if case_metadata.get(str(item.get("task_id")), {}).get("dataset")
        }
        generated_total = sum(value or 0 for value in candidate_counts)
        valid_total = sum(value or 0 for value in valid_counts)
        valid_candidate_rate = None
        if task_type != "docking_evaluation" and generated_total:
            valid_candidate_rate = valid_total / generated_total
        rows.append(
            {
                "task_type": task_type,
                "benchmark_id": benchmark_id,
                "dataset": ", ".join(sorted(datasets)) or "task_generalization",
                "execution_mode": execution_mode,
                "planner_baseline": planner_baseline,
                "task_count": task_count,
                "task_success_rate": task_success_count / task_count if task_count else 0.0,
                "valid_candidate_rate": valid_candidate_rate,
                "verifier_expectation_match": verifier_match_count / task_count if task_count else 0.0,
                "mean_elapsed_sec": _mean(elapsed),
                "tools": ", ".join(tools),
                "notes": _notes_for_execution_mode(execution_mode),
            }
        )
    return rows


def _metric_number(item: dict[str, Any], key: str) -> float | None:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
    return _as_float(metrics.get(key))


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def _notes_for_execution_mode(execution_mode: str) -> str:
    if execution_mode == "real":
        return "Real tool-server slice checks workflow coverage across task families; not a drug-discovery quality claim."
    return "Mock slice checks planner coverage across task families; not a drug-discovery quality claim."


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_GENERALIZATION_SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EGVR task-generalization benchmark and summarize by task type.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for task-generalization artifacts.")
    parser.add_argument("--execution-mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--planner-baseline", choices=SUPPORTED_BASELINES, default=RULE_BASED_PLANNER)
    parser.add_argument("--trace-log-dir", help="Optional trace log directory.")
    args = parser.parse_args()
    result = run_task_generalization_summary(
        args.benchmark,
        output_dir=args.output_dir,
        planner_baseline=args.planner_baseline,
        execution_mode=args.execution_mode,
        trace_log_dir=args.trace_log_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
