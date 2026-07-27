"""Build repair-quality and cost-accounting tables from benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


REPAIR_QUALITY_COLUMNS = [
    "benchmark_id",
    "execution_mode",
    "planner_baseline",
    "task_count",
    "recoverable_case_count",
    "irrecoverable_case_count",
    "healthy_case_count",
    "repair_attempt_count",
    "repair_attempt_rate",
    "repair_recovery_count",
    "repair_recovery_rate",
    "repair_precision",
    "unnecessary_repair_count",
    "unnecessary_repair_rate",
    "false_repair_count",
    "fallback_count",
    "fallback_rate",
    "irrecoverable_preserved_count",
    "irrecoverable_preservation_rate",
    "false_success_count",
    "initial_backend_call_count",
    "repair_backend_call_count",
    "candidate_evaluation_call_count",
    "mean_elapsed_sec_per_verified_success",
    "mean_calls_per_verified_success",
    "source_path",
]

COST_NORMALIZED_COLUMNS = [
    "benchmark_id",
    "execution_mode",
    "planner_baseline",
    "task_count",
    "verified_success_count",
    "mean_exposed_tool_count",
    "mean_planned_logical_steps",
    "initial_backend_call_count",
    "candidate_evaluation_call_count",
    "repair_backend_call_count",
    "fallback_count",
    "total_backend_call_count",
    "mean_backend_calls_per_task",
    "backend_calls_per_verified_success",
    "mean_elapsed_sec_per_task",
    "elapsed_sec_per_verified_success",
    "llm_token_usage",
    "source_path",
]

FAILURE_FAMILY_COLUMNS = [
    "benchmark_id",
    "execution_mode",
    "planner_baseline",
    "failure_family",
    "task_count",
    "healthy_count",
    "recoverable_count",
    "irrecoverable_count",
    "agent_claimed_success_rate",
    "evidence_verified_success_rate",
    "repair_attempt_rate",
    "repair_recovery_rate",
    "fallback_rate",
    "false_success_count",
    "mean_backend_calls",
    "mean_elapsed_sec",
    "source_path",
]


def build_repair_quality_rows(
    result_paths: Iterable[str | Path],
    *,
    project_root: str | Path = ".",
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        path = _resolve(result_path, root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = [item for item in payload.get("results", []) if isinstance(item, dict)]
        if not results:
            continue
        rows.append(
            _aggregate_result_payload(
                payload,
                results,
                source_path=_display_path(path, root),
            )
        )
    return rows


def build_and_write_repair_quality_table(
    result_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    result_paths = list(result_paths)
    target_dir = _resolve(output_dir, root)
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = build_repair_quality_rows(result_paths, project_root=root)
    payload = {
        "table_id": "repair_quality_v1",
        "row_count": len(rows),
        "rows": rows,
        "notes": [
            "Repair precision is successful repair divided by executed repair attempts.",
            "False repair means a repair was unnecessary or violated benchmark-declared repair tools/actions.",
            "Rates are undefined when their denominator is zero and are serialized as null.",
        ],
    }
    (target_dir / "repair_quality_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(target_dir / "repair_quality_table.csv", rows, REPAIR_QUALITY_COLUMNS)
    (target_dir / "repair_quality_table.tex").write_text(_to_latex(rows), encoding="utf-8")
    cost_rows = build_cost_normalized_rows(result_paths, project_root=root)
    family_rows = build_failure_family_rows(result_paths, project_root=root)
    _write_artifact_set(
        target_dir,
        "cost_normalized_table",
        cost_rows,
        COST_NORMALIZED_COLUMNS,
        _cost_latex(cost_rows),
        notes=[
            "Backend invocations are separated from candidate-level evaluator fan-out.",
            "LLM token usage is null when the provider or execution path does not report it.",
        ],
    )
    _write_artifact_set(
        target_dir,
        "failure_taxonomy_family_table",
        family_rows,
        FAILURE_FAMILY_COLUMNS,
        _family_latex(family_rows),
        notes=[
            "Rows aggregate independent task variants within each controlled failure family.",
            "The controlled taxonomy is a mechanism test, not an estimate of natural failure prevalence.",
        ],
    )
    payload["artifacts"] = {
        "repair_quality": [
            str(target_dir / "repair_quality_table.csv"),
            str(target_dir / "repair_quality_table.json"),
            str(target_dir / "repair_quality_table.tex"),
        ],
        "cost_normalized": [
            str(target_dir / "cost_normalized_table.csv"),
            str(target_dir / "cost_normalized_table.json"),
            str(target_dir / "cost_normalized_table.tex"),
        ],
        "failure_family": [
            str(target_dir / "failure_taxonomy_family_table.csv"),
            str(target_dir / "failure_taxonomy_family_table.json"),
            str(target_dir / "failure_taxonomy_family_table.tex"),
        ],
    }
    (target_dir / "repair_quality_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def build_cost_normalized_rows(
    result_paths: Iterable[str | Path],
    *,
    project_root: str | Path = ".",
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        path = _resolve(result_path, root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = [item for item in payload.get("results", []) if isinstance(item, dict)]
        if not results:
            continue
        verified = [item for item in results if item.get("task_success")]
        total_calls = sum(int(_number(item.get("tool_call_count"))) for item in results)
        total_elapsed = sum(_number(item.get("total_elapsed_sec")) for item in results)
        token_values = [item.get("llm_token_usage") for item in results if item.get("llm_token_usage") is not None]
        rows.append(
            {
                "benchmark_id": payload.get("benchmark_id") or _benchmark_id(results),
                "execution_mode": payload.get("execution_mode") or _common(results, "execution_mode"),
                "planner_baseline": payload.get("planner_baseline") or _common(results, "planner_type"),
                "task_count": len(results),
                "verified_success_count": len(verified),
                "mean_exposed_tool_count": _rate(
                    sum(int(_number(item.get("selected_tool_count", len(item.get("selected_tools") or [])))) for item in results),
                    len(results),
                ),
                "mean_planned_logical_steps": _rate(
                    sum(int(_number(item.get("tool_sequence_length"))) for item in results), len(results)
                ),
                "initial_backend_call_count": sum(
                    int(_number(item.get("initial_tool_call_count", item.get("tool_call_count")))) for item in results
                ),
                "candidate_evaluation_call_count": sum(
                    int(_number(item.get("candidate_evaluation_call_count"))) for item in results
                ),
                "repair_backend_call_count": sum(
                    int(_number(item.get("repair_tool_call_count"))) for item in results
                ),
                "fallback_count": sum(bool(item.get("fallback_executed")) for item in results),
                "total_backend_call_count": total_calls,
                "mean_backend_calls_per_task": _rate(total_calls, len(results)),
                "backend_calls_per_verified_success": _rate(total_calls, len(verified)),
                "mean_elapsed_sec_per_task": _rate(total_elapsed, len(results)),
                "elapsed_sec_per_verified_success": _rate(total_elapsed, len(verified)),
                "llm_token_usage": sum(_number(value) for value in token_values) if token_values else None,
                "source_path": _display_path(path, root),
            }
        )
    return rows


def build_failure_family_rows(
    result_paths: Iterable[str | Path],
    *,
    project_root: str | Path = ".",
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        path = _resolve(result_path, root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = [item for item in payload.get("results", []) if isinstance(item, dict)]
        if not results:
            continue
        by_family: dict[str, list[dict[str, Any]]] = {}
        for item in results:
            family = str(item.get("failure_family") or item.get("metadata", {}).get("failure_family") or "unspecified")
            by_family.setdefault(family, []).append(item)
        for family, items in sorted(by_family.items()):
            classes = [_repairability(item) for item in items]
            attempted = [item for item in items if item.get("repair_executed")]
            recoverable = [item for item, label in zip(items, classes) if label == "recoverable"]
            recovered = [item for item in recoverable if item.get("repair_executed") and item.get("task_success")]
            rows.append(
                {
                    "benchmark_id": payload.get("benchmark_id") or _benchmark_id(items),
                    "execution_mode": payload.get("execution_mode") or _common(items, "execution_mode"),
                    "planner_baseline": payload.get("planner_baseline") or _common(items, "planner_type"),
                    "failure_family": family,
                    "task_count": len(items),
                    "healthy_count": classes.count("healthy"),
                    "recoverable_count": classes.count("recoverable"),
                    "irrecoverable_count": classes.count("irrecoverable"),
                    "agent_claimed_success_rate": _rate(
                        sum(bool(item.get("agent_claimed_success", item.get("task_success"))) for item in items), len(items)
                    ),
                    "evidence_verified_success_rate": _rate(sum(bool(item.get("task_success")) for item in items), len(items)),
                    "repair_attempt_rate": _rate(len(attempted), len(items)),
                    "repair_recovery_rate": _rate(len(recovered), len(recoverable)),
                    "fallback_rate": _rate(sum(bool(item.get("fallback_executed")) for item in attempted), len(attempted)),
                    "false_success_count": sum(
                        bool(item.get("agent_claimed_success", item.get("task_success")))
                        and not bool(item.get("evidence_verified_success", item.get("task_success")))
                        for item in items
                    ),
                    "mean_backend_calls": _rate(sum(int(_number(item.get("tool_call_count"))) for item in items), len(items)),
                    "mean_elapsed_sec": _rate(sum(_number(item.get("total_elapsed_sec")) for item in items), len(items)),
                    "source_path": _display_path(path, root),
                }
            )
    return rows


def _aggregate_result_payload(
    payload: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    source_path: str,
) -> dict[str, Any]:
    classes = [_repairability(item) for item in results]
    recoverable = [item for item, label in zip(results, classes) if label == "recoverable"]
    irrecoverable = [item for item, label in zip(results, classes) if label == "irrecoverable"]
    healthy = [item for item, label in zip(results, classes) if label == "healthy"]
    attempted = [item for item in results if item.get("repair_executed")]
    recovered = [item for item in recoverable if item.get("repair_executed") and item.get("task_success")]
    successful_repairs = [item for item in attempted if item.get("repair_success")]
    unnecessary = [item for item in attempted if item.get("initial_evidence_verified_success")]
    false_repairs = [item for item in attempted if _is_false_repair(item)]
    fallbacks = [item for item in attempted if item.get("fallback_executed")]
    preserved = [item for item in irrecoverable if not item.get("task_success")]
    false_successes = [
        item
        for item in results
        if item.get("agent_claimed_success", item.get("task_success"))
        and not item.get("evidence_verified_success", item.get("task_success"))
    ]
    verified_successes = [item for item in results if item.get("task_success")]
    total_elapsed = sum(_number(item.get("total_elapsed_sec")) for item in results)
    total_calls = sum(int(_number(item.get("tool_call_count"))) for item in results)
    summary = payload.get("summary", {})
    return {
        "benchmark_id": payload.get("benchmark_id") or summary.get("benchmark_id") or _benchmark_id(results),
        "execution_mode": payload.get("execution_mode") or summary.get("execution_mode") or _common(results, "execution_mode"),
        "planner_baseline": payload.get("planner_baseline") or _common(results, "planner_type"),
        "task_count": len(results),
        "recoverable_case_count": len(recoverable),
        "irrecoverable_case_count": len(irrecoverable),
        "healthy_case_count": len(healthy),
        "repair_attempt_count": len(attempted),
        "repair_attempt_rate": _rate(len(attempted), len(results)),
        "repair_recovery_count": len(recovered),
        "repair_recovery_rate": _rate(len(recovered), len(recoverable)),
        "repair_precision": _rate(len(successful_repairs), len(attempted)),
        "unnecessary_repair_count": len(unnecessary),
        "unnecessary_repair_rate": _rate(len(unnecessary), len(healthy)),
        "false_repair_count": len(false_repairs),
        "fallback_count": len(fallbacks),
        "fallback_rate": _rate(len(fallbacks), len(attempted)),
        "irrecoverable_preserved_count": len(preserved),
        "irrecoverable_preservation_rate": _rate(len(preserved), len(irrecoverable)),
        "false_success_count": len(false_successes),
        "initial_backend_call_count": sum(
            int(_number(item.get("initial_tool_call_count", item.get("tool_call_count")))) for item in results
        ),
        "repair_backend_call_count": sum(int(_number(item.get("repair_tool_call_count"))) for item in results),
        "candidate_evaluation_call_count": sum(
            int(_number(item.get("candidate_evaluation_call_count"))) for item in results
        ),
        "mean_elapsed_sec_per_verified_success": _rate(total_elapsed, len(verified_successes)),
        "mean_calls_per_verified_success": _rate(total_calls, len(verified_successes)),
        "source_path": source_path,
    }


def _repairability(item: dict[str, Any]) -> str:
    declared = item.get("repairability")
    if declared in {"recoverable", "irrecoverable", "healthy"}:
        return str(declared)
    if not item.get("expected_success"):
        return "irrecoverable"
    if item.get("initial_evidence_verified_success"):
        return "healthy"
    return "recoverable"


def _is_false_repair(item: dict[str, Any]) -> bool:
    if item.get("initial_evidence_verified_success"):
        return True
    plan = item.get("repair_plan") or {}
    actions = [action for action in plan.get("actions", []) if isinstance(action, dict)]
    allowed_tools = set(item.get("expected_repair_tools") or [])
    allowed_types = set(item.get("expected_repair_action_types") or [])
    if allowed_tools and any(action.get("tool_name") and action.get("tool_name") not in allowed_tools for action in actions):
        return True
    if allowed_types and any(action.get("action_type") not in allowed_types for action in actions):
        return True
    return False


def _benchmark_id(results: list[dict[str, Any]]) -> str | None:
    task_id = results[0].get("task_id") if results else None
    return str(task_id).split("_")[0] if task_id else None


def _common(rows: list[dict[str, Any]], key: str) -> Any:
    values = {row.get(key) for row in rows if row.get(key) is not None}
    return next(iter(values)) if len(values) == 1 else None


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rate(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_artifact_set(
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    latex: str,
    *,
    notes: list[str],
) -> None:
    _write_csv(output_dir / f"{name}.csv", rows, columns)
    (output_dir / f"{name}.json").write_text(
        json.dumps(
            {"table_id": name, "row_count": len(rows), "rows": rows, "notes": notes},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / f"{name}.tex").write_text(latex, encoding="utf-8")


def _to_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated repair-quality table.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & Tasks & Attempts & Recovery & Precision & False repair & Fallback & Preserved & False success \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row.get('planner_baseline'))} & {row.get('task_count')} & {row.get('repair_attempt_count')} & "
            f"{_pct(row.get('repair_recovery_rate'))} & {_pct(row.get('repair_precision'))} & "
            f"{row.get('false_repair_count')} & {_pct(row.get('fallback_rate'))} & "
            f"{_pct(row.get('irrecoverable_preservation_rate'))} & {row.get('false_success_count')} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Repair quality under controlled evidence failures. Undefined rates are shown as --.}",
            "\\label{tab:repair-quality}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _cost_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated cost-normalized reliability table.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & Tasks & Verified & Exposed & Steps & Calls/task & Calls/success & Sec./success \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row.get('planner_baseline'))} & {row.get('task_count')} & {row.get('verified_success_count')} & "
            f"{_fmt(row.get('mean_exposed_tool_count'))} & {_fmt(row.get('mean_planned_logical_steps'))} & "
            f"{_fmt(row.get('mean_backend_calls_per_task'))} & {_fmt(row.get('backend_calls_per_verified_success'))} & "
            f"{_fmt(row.get('elapsed_sec_per_verified_success'))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Cost-normalized controlled reliability. Candidate fan-out is accounted for separately in the JSON/CSV artifact.}",
            "\\label{tab:cost-normalized-reliability}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _family_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated controlled failure-family table.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Method & Family & Tasks & Verified & Repair & Recovery & False success & Calls \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row.get('planner_baseline'))} & {_tex(row.get('failure_family'))} & {row.get('task_count')} & "
            f"{_pct(row.get('evidence_verified_success_rate'))} & {_pct(row.get('repair_attempt_rate'))} & "
            f"{_pct(row.get('repair_recovery_rate'))} & {row.get('false_success_count')} & "
            f"{_fmt(row.get('mean_backend_calls'))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Controlled reliability by failure family. This mechanism benchmark does not estimate natural failure prevalence.}",
            "\\label{tab:failure-taxonomy-v3-family}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}\\%"


def _fmt(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _tex(value: Any) -> str:
    return str(value or "--").replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build repair-quality tables from benchmark result JSON files.")
    parser.add_argument("--result", action="append", required=True, help="Benchmark result JSON. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    payload = build_and_write_repair_quality_table(
        args.result,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
