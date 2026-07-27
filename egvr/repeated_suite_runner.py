"""Repeat a baseline suite and aggregate robustness statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .baseline_planners import EGVR_AGENT, LEGACY_FULL_COPILOT, RULE_BASED_PLANNER, SUPPORTED_BASELINES
from .baseline_suite_runner import run_baseline_suite


REPEATED_SUMMARY_COLUMNS = [
    "benchmark_id",
    "source_benchmark_id",
    "benchmark_path",
    "execution_mode",
    "planner_baseline",
    "repeat_count",
    "task_count",
    "mean_task_success_rate",
    "std_task_success_rate",
    "mean_repair_success_rate",
    "std_repair_success_rate",
    "false_success_count",
    "mean_verifier_expectation_match",
    "std_verifier_expectation_match",
    "mean_tool_call_count",
    "std_tool_call_count",
    "notes",
]


REPEATED_DETAIL_COLUMNS = [
    "repeat_index",
    "benchmark_id",
    "execution_mode",
    "planner_baseline",
    "task_count",
    "task_success_count",
    "task_success_rate",
    "repair_attempt_count",
    "repair_success_count",
    "repair_success_rate",
    "false_success_count",
    "verifier_expectation_match",
    "mean_tool_call_count",
    "result_path",
]


def run_repeated_baseline_suite(
    benchmark_path: str | Path,
    *,
    output_dir: str | Path,
    repeats: int = 3,
    planner_baselines: list[str] | tuple[str, ...] | None = None,
    execution_mode: str = "mock",
    trace_log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a baseline suite several times and write aggregate robustness rows."""

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    baselines = tuple(planner_baselines or (RULE_BASED_PLANNER, EGVR_AGENT))
    for baseline in baselines:
        if baseline not in SUPPORTED_BASELINES:
            raise ValueError(f"Unsupported planner baseline: {baseline}")

    benchmark_file = Path(benchmark_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, Any]] = []
    repeat_payloads: list[dict[str, Any]] = []
    for repeat_index in range(1, repeats + 1):
        repeat_dir = target_dir / f"repeat_{repeat_index:02d}"
        repeat_trace_dir = Path(trace_log_dir) / f"repeat_{repeat_index:02d}" if trace_log_dir else None
        payload = run_baseline_suite(
            benchmark_file,
            output_dir=repeat_dir,
            planner_baselines=list(baselines),
            execution_mode=execution_mode,
            trace_log_dir=repeat_trace_dir,
        )
        repeat_payloads.append(payload)
        detail_rows.extend(_detail_rows(payload, repeat_index))

    summary_rows = _summary_rows(
        detail_rows,
        benchmark_path=benchmark_file,
        source_benchmark_id=benchmark_file.stem,
        execution_mode=execution_mode,
        baselines=baselines,
    )
    summary_json_path = target_dir / f"{benchmark_file.stem}.repeated_summary.json"
    summary_csv_path = target_dir / f"{benchmark_file.stem}.repeated_summary.csv"
    detail_csv_path = target_dir / f"{benchmark_file.stem}.repeated_detail.csv"
    result = {
        "benchmark_id": f"{benchmark_file.stem}_repeated",
        "source_benchmark_id": benchmark_file.stem,
        "benchmark_path": str(benchmark_file),
        "execution_mode": execution_mode,
        "repeat_count": repeats,
        "planner_baselines": list(baselines),
        "artifacts": {
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "detail_csv": str(detail_csv_path),
        },
        "rows": summary_rows,
        "detail_rows": detail_rows,
        "repeat_artifacts": [payload["artifacts"] for payload in repeat_payloads],
    }
    summary_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(summary_rows, summary_csv_path, REPEATED_SUMMARY_COLUMNS)
    _write_csv(detail_rows, detail_csv_path, REPEATED_DETAIL_COLUMNS)
    return result


def _detail_rows(payload: dict[str, Any], repeat_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "repeat_index": repeat_index,
                "benchmark_id": row.get("benchmark_id"),
                "execution_mode": row.get("execution_mode"),
                "planner_baseline": row.get("planner_baseline"),
                "task_count": row.get("total"),
                "task_success_count": row.get("task_success_count"),
                "task_success_rate": row.get("task_success_rate"),
                "repair_attempt_count": row.get("repair_attempt_count"),
                "repair_success_count": row.get("repair_success_count"),
                "repair_success_rate": row.get("repair_success_rate"),
                "false_success_count": row.get("false_success_count", 0),
                "verifier_expectation_match": row.get("verifier_expectation_match"),
                "mean_tool_call_count": row.get("mean_tool_call_count"),
                "result_path": row.get("result_path"),
            }
        )
    return rows


def _summary_rows(
    detail_rows: list[dict[str, Any]],
    *,
    benchmark_path: Path,
    source_benchmark_id: str,
    execution_mode: str,
    baselines: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        baseline_rows = [row for row in detail_rows if row.get("planner_baseline") == baseline]
        if not baseline_rows:
            continue
        task_counts = [_as_float(row.get("task_count")) for row in baseline_rows]
        task_success_rates = [_as_float(row.get("task_success_rate")) for row in baseline_rows]
        repair_success_rates = [_as_float(row.get("repair_success_rate")) for row in baseline_rows]
        verifier_matches = [_as_float(row.get("verifier_expectation_match")) for row in baseline_rows]
        mean_tool_calls = [_as_float(row.get("mean_tool_call_count")) for row in baseline_rows]
        rows.append(
            {
                "benchmark_id": f"{source_benchmark_id}_repeated",
                "source_benchmark_id": source_benchmark_id,
                "benchmark_path": str(benchmark_path),
                "execution_mode": execution_mode,
                "planner_baseline": baseline,
                "repeat_count": len(baseline_rows),
                "task_count": int(task_counts[0]) if task_counts and task_counts[0] is not None else None,
                "mean_task_success_rate": _mean(task_success_rates),
                "std_task_success_rate": _std(task_success_rates),
                "mean_repair_success_rate": _mean(repair_success_rates),
                "std_repair_success_rate": _std(repair_success_rates),
                "false_success_count": sum(int(_as_float(row.get("false_success_count")) or 0) for row in baseline_rows),
                "mean_verifier_expectation_match": _mean(verifier_matches),
                "std_verifier_expectation_match": _std(verifier_matches),
                "mean_tool_call_count": _mean(mean_tool_calls),
                "std_tool_call_count": _std(mean_tool_calls),
                "notes": _notes_for_baseline(baseline),
            }
        )
    return rows


def _notes_for_baseline(baseline: str) -> str:
    if baseline in {EGVR_AGENT, LEGACY_FULL_COPILOT}:
        return "Verifier-triggered repair is repeated without changing the failure taxonomy."
    if baseline == RULE_BASED_PLANNER:
        return "No repair is attempted; repeats test baseline stability."
    return "Repeated suite aggregate."


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def _std(values: list[float | None]) -> float:
    clean = [value for value in values if value is not None]
    return stdev(clean) if len(clean) > 1 else 0.0


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(rows: list[dict[str, Any]], path: str | Path, columns: list[str]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat an EGVR planner baseline suite and aggregate statistics.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for repeat outputs and aggregate tables.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--execution-mode", choices=["mock", "real"], default="mock")
    parser.add_argument(
        "--planner-baselines",
        nargs="+",
        choices=SUPPORTED_BASELINES,
        default=[RULE_BASED_PLANNER, EGVR_AGENT],
    )
    parser.add_argument("--trace-log-dir", help="Optional trace log root. Each repeat gets a subdirectory.")
    args = parser.parse_args()
    result = run_repeated_baseline_suite(
        args.benchmark,
        output_dir=args.output_dir,
        repeats=args.repeats,
        planner_baselines=args.planner_baselines,
        execution_mode=args.execution_mode,
        trace_log_dir=args.trace_log_dir,
    )
    print(
        json.dumps(
            {
                "summary_json": result["artifacts"]["summary_json"],
                "summary_csv": result["artifacts"]["summary_csv"],
                "detail_csv": result["artifacts"]["detail_csv"],
                "rows": result["rows"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
