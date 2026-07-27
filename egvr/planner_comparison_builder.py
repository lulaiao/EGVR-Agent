"""Build a common 65-task planning table for LLM and structured planners."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .llm_router_baseline_runner import (
    DEFAULT_BENCHMARKS,
    _summarize_task_results,
    evaluate_router_task,
)
from .tool_registry import build_default_tool_registry


COLUMNS = [
    "method",
    "planner_family",
    "task_count",
    "valid_json_rate",
    "valid_schema_rate",
    "required_tool_recall",
    "tool_precision",
    "exact_dependency_order_rate",
    "missing_required_input_count",
    "hallucinated_tool_count",
    "mean_selected_tool_count",
    "source",
]


def build_planner_comparison(
    *,
    llm_results: Iterable[tuple[str, str | Path]],
    benchmark_paths: Iterable[str | Path] = DEFAULT_BENCHMARKS,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    task_rows: dict[str, list[dict[str, Any]]] = {}

    for method, raw_path in llm_results:
        path = _resolve(raw_path, root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload.get("row") or {}
        results = [item for item in payload.get("task_results", []) if isinstance(item, dict)]
        rows.append(_llm_row(method, row, source=_display(path, root)))
        task_rows[method] = results

    structured_results = _evaluate_structured_planner(benchmark_paths, root)
    structured_summary = _summarize_task_results(structured_results, router_mode="structured")
    for method in ("structured_planner", "egvr_agent"):
        rows.append(
            _summary_row(
                method=method,
                planner_family="structured",
                summary=structured_summary,
                source="deterministic_rule_planner_on_locked_65_tasks",
            )
        )
        task_rows[method] = structured_results

    target = _resolve(output_dir, root)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "table_id": "planner_comparison_65_v1",
        "recorded_at": _utc_timestamp(),
        "task_count": structured_summary["task_count"],
        "rows": rows,
        "task_results": task_rows,
        "notes": [
            "LLM rows are fixed saved-plan runs; no LLM calls are made by this builder.",
            "Structured planner and EGVR-Agent share the same deterministic initial plan.",
            "EGVR-Agent differs after execution through verification and repair, not at initial planning.",
            "This table reports planning-interface quality only; executed reliability is reported separately.",
        ],
    }
    (target / "planner_comparison_65_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(target / "planner_comparison_65_table.csv", rows, COLUMNS)
    (target / "planner_comparison_65_table.tex").write_text(_latex(rows), encoding="utf-8")
    return payload


def _evaluate_structured_planner(
    benchmark_paths: Iterable[str | Path],
    root: Path,
) -> list[dict[str, Any]]:
    registry = build_default_tool_registry()
    rows: list[dict[str, Any]] = []
    for raw_path in benchmark_paths:
        path = _resolve(raw_path, root)
        for task in _read_jsonl(path):
            rows.append(
                evaluate_router_task(
                    task,
                    benchmark_path=path,
                    registry=registry,
                    responses={},
                    router_mode="heuristic",
                    project_root=root,
                )
            )
    return rows


def _llm_row(method: str, summary: dict[str, Any], *, source: str) -> dict[str, Any]:
    return _summary_row(
        method=method,
        planner_family="llm_router",
        summary=summary,
        source=source,
    )


def _summary_row(
    *,
    method: str,
    planner_family: str,
    summary: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    task_count = int(summary.get("task_count") or 0)
    return {
        "method": method,
        "planner_family": planner_family,
        "task_count": task_count,
        "valid_json_rate": _ratio(summary.get("valid_json_count"), task_count),
        "valid_schema_rate": _ratio(summary.get("valid_schema_count"), task_count),
        "required_tool_recall": summary.get("tool_recall"),
        "tool_precision": summary.get("tool_precision"),
        "exact_dependency_order_rate": summary.get("workflow_order_match_rate"),
        "missing_required_input_count": summary.get("missing_required_input_count"),
        "hallucinated_tool_count": summary.get("hallucinated_tool_count"),
        "mean_selected_tool_count": summary.get("mean_selected_tool_count"),
        "source": source,
    }


def _latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated 65-task planner comparison.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Planner & Schema & Recall & Precision & Exact order & Missing input & Halluc. & Sel. tools & Tasks \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['method'])} & {_pct(row['valid_schema_rate'])} & "
            f"{_pct(row['required_tool_recall'])} & {_pct(row['tool_precision'])} & "
            f"{_pct(row['exact_dependency_order_rate'])} & "
            f"{row['missing_required_input_count']} & {row['hallucinated_tool_count']} & "
            f"{_num(row['mean_selected_tool_count'])} & {row['task_count']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Planning quality on the shared 65-task benchmark. LLM routers are fixed single runs.}",
            "\\label{tab:planner-comparison-65}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--llm-result must be METHOD=PATH")
    method, path = value.split("=", 1)
    if not method.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--llm-result must be METHOD=PATH")
    return method.strip(), path.strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ratio(numerator: Any, denominator: int) -> float | None:
    if not denominator:
        return None
    return float(numerator or 0) / denominator


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


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the common 65-task planner comparison.")
    parser.add_argument("--llm-result", action="append", type=_parse_named_path, required=True)
    parser.add_argument("--benchmark", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    payload = build_planner_comparison(
        llm_results=args.llm_result,
        benchmark_paths=args.benchmark or DEFAULT_BENCHMARKS,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps({"table_id": payload["table_id"], "rows": payload["rows"]}, indent=2))


if __name__ == "__main__":
    main()
