"""Build a common reliability/cost table for executed planner baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


COLUMNS = [
    "method",
    "planner_family",
    "repair_policy",
    "execution_mode",
    "task_count",
    "workflow_valid_rate",
    "verified_success_rate",
    "verifier_expectation_match",
    "false_success_count",
    "repair_attempt_rate",
    "repair_success_rate",
    "mean_selected_tool_count",
    "mean_backend_call_count",
    "mean_candidate_evaluation_call_count",
    "backend_calls_per_verified_success",
    "mean_elapsed_sec",
    "elapsed_sec_per_verified_success",
    "server_unreachable_count",
    "source_path",
]


def build_executed_planner_rows(
    result_paths: Iterable[str | Path],
    *,
    project_root: str | Path = ".",
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    for raw_path in result_paths:
        path = _resolve(raw_path, root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        results = [item for item in payload.get("results", []) if isinstance(item, dict)]
        router = payload.get("router_name")
        repair_policy = payload.get("repair_mode")
        baseline = payload.get("planner_baseline")
        if router:
            method = f"{router}:{repair_policy}"
            planner_family = "llm_router"
        else:
            method = str(baseline or _common(results, "planner_type") or path.stem)
            planner_family = "structured_planner"
            repair_policy = "verifier_guided" if method in {"egvr_agent", "full_copilot"} else "none"
        verified_rate = summary.get("evidence_verified_success_rate", summary.get("task_success_rate"))
        mean_calls = summary.get("mean_tool_call_count")
        mean_elapsed = summary.get("mean_total_elapsed_sec")
        rows.append(
            {
                "method": method,
                "planner_family": planner_family,
                "repair_policy": repair_policy,
                "execution_mode": payload.get("execution_mode") or _common(results, "execution_mode"),
                "task_count": len(results),
                "workflow_valid_rate": summary.get("router_workflow_valid_rate", 1.0),
                "verified_success_rate": verified_rate,
                "verifier_expectation_match": summary.get("verifier_expectation_match"),
                "false_success_count": summary.get("false_success_count", 0),
                "repair_attempt_rate": summary.get("repair_attempt_rate"),
                "repair_success_rate": summary.get("repair_success_rate"),
                "mean_selected_tool_count": summary.get("mean_selected_tool_count"),
                "mean_backend_call_count": mean_calls,
                "mean_candidate_evaluation_call_count": summary.get("mean_candidate_evaluation_call_count"),
                "backend_calls_per_verified_success": _per_success(mean_calls, verified_rate),
                "mean_elapsed_sec": mean_elapsed,
                "elapsed_sec_per_verified_success": _per_success(mean_elapsed, verified_rate),
                "server_unreachable_count": json.dumps(results).count("Cannot reach tool server"),
                "source_path": _display(path, root),
            }
        )
    return rows


def write_executed_planner_table(
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    target = _resolve(output_dir, root)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "table_id": "executed_planner_comparison_v1",
        "row_count": len(rows),
        "rows": rows,
        "notes": [
            "LLM-router workflows are saved API responses replayed through the real executor; no new LLM call occurs during execution.",
            "Candidate evaluator fan-out is reported separately from logical tool selection and backend calls.",
            "This is a workflow-reliability comparison, not a molecular-quality or clinical-prediction benchmark.",
        ],
    }
    (target / "executed_planner_comparison_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (target / "executed_planner_comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (target / "executed_planner_comparison_table.tex").write_text(_latex(rows), encoding="utf-8")
    return payload


def _latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated executed planner comparison.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Valid wf. & Verified & False succ. & Calls/succ. & Repair & Sec./succ. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['method'])} & {_pct(row['workflow_valid_rate'])} & {_pct(row['verified_success_rate'])} & "
            f"{row['false_success_count']} & {_num(row['backend_calls_per_verified_success'])} & "
            f"{_pct(row['repair_attempt_rate'])} & {_num(row['elapsed_sec_per_verified_success'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Executed planner reliability and cost on the shared 20-task slice.}",
            "\\label{tab:executed-planner-comparison}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _common(rows: list[dict[str, Any]], key: str) -> Any:
    values = {row.get(key) for row in rows if row.get(key) is not None}
    return next(iter(values)) if len(values) == 1 else None


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _pct(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}\\%"


def _num(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _per_success(value: Any, success_rate: Any) -> float | None:
    if value is None or success_rate in {None, 0, 0.0}:
        return None
    return float(value) / float(success_rate)


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an executed planner comparison table.")
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    rows = build_executed_planner_rows(args.result, project_root=args.project_root)
    payload = write_executed_planner_table(rows, output_dir=args.output_dir, project_root=args.project_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
