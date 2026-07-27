"""Combine matched tool-menu planning and real-execution evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "menu_condition",
    "task_count",
    "mean_exposed_tools",
    "schema_validity",
    "required_tool_recall",
    "tool_precision",
    "exact_order_match_rate",
    "missing_required_input_count",
    "evidence_verified_success_rate",
    "false_success_count",
    "mean_backend_call_count",
    "mean_elapsed_sec",
]


def build_tool_menu_execution_summary(
    *,
    planning_summary_path: str | Path,
    all_tool_execution_path: str | Path,
    task_conditioned_execution_path: str | Path,
) -> dict[str, Any]:
    planning = _read_json(planning_summary_path)
    execution_by_condition = {
        "all_tool": _read_json(all_tool_execution_path),
        "task_conditioned": _read_json(task_conditioned_execution_path),
    }
    planning_by_condition = {
        str(row["menu_condition"]): row
        for row in planning.get("rows", [])
        if isinstance(row, dict) and row.get("menu_condition")
    }

    rows: list[dict[str, Any]] = []
    for condition in ("all_tool", "task_conditioned"):
        planning_row = planning_by_condition.get(condition)
        if planning_row is None:
            raise ValueError(f"Planning summary has no {condition!r} row")
        execution = execution_by_condition[condition]
        execution_summary = execution.get("summary", {})
        planning_count = int(planning_row.get("task_count", 0))
        execution_count = int(execution_summary.get("executed_task_count", len(execution.get("results", []))))
        if planning_count != execution_count:
            raise ValueError(
                f"{condition} task-count mismatch: planning={planning_count}, execution={execution_count}"
            )
        rows.append(
            {
                "menu_condition": condition,
                "task_count": planning_count,
                "mean_exposed_tools": planning_row.get("mean_exposed_tools"),
                "schema_validity": planning_row.get("schema_validity"),
                "required_tool_recall": planning_row.get("required_tool_recall"),
                "tool_precision": planning_row.get("tool_precision"),
                "exact_order_match_rate": planning_row.get("exact_order_match_rate"),
                "missing_required_input_count": planning_row.get("missing_required_input_count"),
                "evidence_verified_success_rate": execution_summary.get(
                    "evidence_verified_success_rate",
                    execution_summary.get("task_success_rate"),
                ),
                "false_success_count": execution_summary.get("false_success_count", 0),
                "mean_backend_call_count": execution_summary.get("mean_backend_call_count"),
                "mean_elapsed_sec": execution_summary.get("mean_total_elapsed_sec"),
            }
        )

    return {
        "table_id": "strict_tool_menu_execution_comparison_v1",
        "model": planning.get("model"),
        "router_mode": planning.get("router_mode"),
        "task_count": planning.get("task_count"),
        "rows": rows,
        "notes": [
            "Matched 20-task fixed run; only the exposed tool menu differs between conditions.",
            "Saved plans are executed through the same real-tool executor and verifier without repair.",
            "This measures planning-interface control and evidence-verified workflow completion, not molecular quality.",
        ],
        "source_paths": {
            "planning_summary": str(planning_summary_path),
            "all_tool_execution": str(all_tool_execution_path),
            "task_conditioned_execution": str(task_conditioned_execution_path),
        },
    }


def write_tool_menu_execution_summary(payload: dict[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "strict_tool_menu_execution_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (target / "strict_tool_menu_execution_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["rows"])
    (target / "strict_tool_menu_execution_table.tex").write_text(
        _latex(payload["rows"]),
        encoding="utf-8",
    )


def _latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated matched tool-menu planning and execution comparison.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Menu & Exposed & Schema & Recall & Precision & Order & Missing input & Verified & Calls \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['menu_condition'])} & {_num(row['mean_exposed_tools'])} & "
            f"{_pct(row['schema_validity'])} & {_pct(row['required_tool_recall'])} & "
            f"{_pct(row['tool_precision'])} & {_pct(row['exact_order_match_rate'])} & "
            f"{row['missing_required_input_count']} & {_pct(row['evidence_verified_success_rate'])} & "
            f"{_num(row['mean_backend_call_count'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Matched Gemini tool-menu comparison on the shared 20-task slice.}",
            "\\label{tab:gemini-tool-menu-executed}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}\\%"


def _num(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a matched planning/execution tool-menu table.")
    parser.add_argument("--planning-summary", required=True)
    parser.add_argument("--all-tool-execution", required=True)
    parser.add_argument("--task-conditioned-execution", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = build_tool_menu_execution_summary(
        planning_summary_path=args.planning_summary,
        all_tool_execution_path=args.all_tool_execution,
        task_conditioned_execution_path=args.task_conditioned_execution,
    )
    write_tool_menu_execution_summary(payload, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
