"""Run and summarize explicit verifier-guided repair budgets."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .benchmark_runner import BenchmarkRunner


DEFAULT_BUDGETS = (0, 1, 2)
ACTION_FAMILIES = {
    "retry_with_reduced_generation_count": "targeted_retry",
    "retry_evaluator_for_missing_evidence": "missing_evaluator_rerun",
    "fallback_tool": "declared_fallback",
}


def run_repair_budget_experiment(
    *,
    benchmark_path: str | Path,
    output_dir: str | Path,
    execution_mode: str = "mock",
    budgets: tuple[int, ...] | list[int] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    benchmark_for_run = _prepare_budget_benchmark(
        Path(benchmark_path),
        output / "repair_budget_benchmark.jsonl",
        max_budget=max(int(value) for value in budgets),
    )
    summaries: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []

    for budget in budgets:
        trace_dir = output / f"traces_budget_{budget}"
        result_path = output / f"repair_budget_{budget}.json"
        payload = BenchmarkRunner(
            execution_mode=execution_mode,
            planner_baseline="egvr_agent",
            trace_log_dir=trace_dir,
            repair_budget=int(budget),
        ).run_file(benchmark_for_run, output_path=result_path)
        results = payload["results"]
        summary = _budget_summary(int(budget), results, execution_mode=execution_mode)
        summaries.append(summary)
        action_rows.extend(_action_summary(int(budget), results))
        per_task_rows.extend(_per_task_rows(int(budget), results))

    result = {
        "experiment_id": "repair_budget_v1",
        "execution_mode": execution_mode,
        "budgets": list(budgets),
        "rows": summaries,
        "action_rows": action_rows,
        "source": "new_run",
        "notes": [
            "Repair budget is the maximum number of verifier-guided repair rounds per task.",
            "Calls/success uses concrete backend invocation records and evidence-verified success.",
            "Action-family call cost is task-level repair-call cost for tasks containing that action and may overlap across families.",
        ],
    }
    _write_json(output / "repair_budget_summary.json", result)
    _write_csv(output / "repair_budget_summary.csv", summaries)
    _write_csv(output / "repair_budget_action_breakdown.csv", action_rows)
    _write_json(output / "repair_budget_action_breakdown.json", {"rows": action_rows})
    _write_csv(output / "repair_budget_per_task.csv", per_task_rows)
    _write_json(output / "repair_budget_per_task.json", {"rows": per_task_rows})
    (output / "repair_budget_summary.tex").write_text(_budget_latex(summaries), encoding="utf-8")
    (output / "repair_budget_action_breakdown.tex").write_text(_action_latex(action_rows), encoding="utf-8")
    return result


def summarize_repair_budget_results(
    *,
    result_paths: dict[int, str | Path],
    output_dir: str | Path,
    execution_mode: str = "real",
) -> dict[str, Any]:
    """Rebuild final tables from complete per-budget result JSON files."""

    output = Path(output_dir)
    summaries: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    for budget in sorted(result_paths):
        payload = json.loads(Path(result_paths[budget]).read_text(encoding="utf-8"))
        results = payload.get("results") or []
        task_ids = [str(row.get("task_id")) for row in results]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"Duplicate task IDs in repair budget {budget}")
        summaries.append(_budget_summary(budget, results, execution_mode=execution_mode))
        action_rows.extend(_action_summary(budget, results))
        per_task_rows.extend(_per_task_rows(budget, results))
    result = {
        "experiment_id": "repair_budget_v1",
        "execution_mode": execution_mode,
        "budgets": sorted(result_paths),
        "rows": summaries,
        "action_rows": action_rows,
        "source": "new_run",
        "notes": [
            "Final tables rebuilt from complete per-budget result JSON files.",
            "Repair budget is the maximum number of verifier-guided repair rounds per task.",
            "Calls/success uses concrete backend invocation records and evidence-verified success.",
        ],
    }
    _write_json(output / "repair_budget_summary.json", result)
    _write_csv(output / "repair_budget_summary.csv", summaries)
    _write_csv(output / "repair_budget_action_breakdown.csv", action_rows)
    _write_json(output / "repair_budget_action_breakdown.json", {"rows": action_rows})
    _write_csv(output / "repair_budget_per_task.csv", per_task_rows)
    _write_json(output / "repair_budget_per_task.json", {"rows": per_task_rows})
    (output / "repair_budget_summary.tex").write_text(_budget_latex(summaries), encoding="utf-8")
    (output / "repair_budget_action_breakdown.tex").write_text(_action_latex(action_rows), encoding="utf-8")
    return result


def _prepare_budget_benchmark(source: Path, target: Path, *, max_budget: int) -> Path:
    """Extend irrecoverable failure injections across the full repair horizon."""

    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    required_call_index = max_budget + 1
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("repairability") != "irrecoverable":
            continue
        injections = metadata.get("failure_injections")
        if not isinstance(injections, dict):
            continue
        for tool_name, raw_plan in injections.items():
            plan = raw_plan if isinstance(raw_plan, list) else [raw_plan]
            clean = [item for item in plan if isinstance(item, dict)]
            if not clean or not _is_failure_injection(clean[-1]):
                continue
            existing = {int(item.get("call_index", index + 1)) for index, item in enumerate(clean)}
            template = clean[-1]
            for call_index in range(1, required_call_index + 1):
                if call_index in existing:
                    continue
                item = copy.deepcopy(template)
                item["call_index"] = call_index
                clean.append(item)
            clean.sort(key=lambda item: int(item.get("call_index", 0)))
            injections[tool_name] = clean
        metadata["repair_budget_injection_horizon"] = required_call_index
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return target


def _is_failure_injection(item: dict[str, Any]) -> bool:
    if str(item.get("mode", "")).lower() in {"error", "exception", "empty", "empty_generation"}:
        return True
    output = item.get("output")
    if isinstance(output, dict):
        return output.get("success") is False or not output.get("molecules_smiles", [True])
    return False


def _budget_summary(budget: int, results: list[dict[str, Any]], *, execution_mode: str) -> dict[str, Any]:
    total = len(results)
    successes = sum(bool(row.get("evidence_verified_success")) for row in results)
    initial_failures = [row for row in results if not row.get("initial_evidence_verified_success")]
    recovered = sum(bool(row.get("evidence_verified_success")) for row in initial_failures)
    repair_tasks = [row for row in results if row.get("repair_executed")]
    unnecessary = [row for row in repair_tasks if row.get("repairability") == "healthy"]
    false_repair = [
        row for row in results
        if row.get("repairability") == "irrecoverable" and row.get("evidence_verified_success")
    ]
    total_calls = sum(int(row.get("tool_call_count") or 0) for row in results)
    total_elapsed = sum(float(row.get("total_elapsed_sec") or 0.0) for row in results)
    return {
        "repair_budget": budget,
        "task_count": total,
        "evidence_verified_success_count": successes,
        "evidence_verified_success_rate": successes / total if total else None,
        "repair_attempt_task_count": len(repair_tasks),
        "repair_attempt_rate": len(repair_tasks) / total if total else None,
        "repair_recovery_count": recovered,
        "repair_recovery_rate": recovered / len(initial_failures) if initial_failures else None,
        "unnecessary_repair_count": len(unnecessary),
        "false_repair_count": len(false_repair),
        "false_success_count": sum(
            bool(row.get("agent_claimed_success")) and not bool(row.get("evidence_verified_success"))
            for row in results
        ),
        "fallback_task_count": sum(bool(row.get("fallback_executed")) for row in results),
        "fallback_rate": (
            sum(bool(row.get("fallback_executed")) for row in repair_tasks) / len(repair_tasks)
            if repair_tasks else 0.0
        ),
        "mean_calls_per_task": total_calls / total if total else None,
        "calls_per_verified_success": total_calls / successes if successes else None,
        "mean_elapsed_sec_per_task": total_elapsed / total if total else None,
        "elapsed_sec_per_verified_success": total_elapsed / successes if successes else None,
        "execution_mode": execution_mode,
        "source": "new_run",
    }


def _action_summary(budget: int, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        families = _action_families(row)
        for family in families:
            grouped[family].append(row)
    output = []
    for family in ("targeted_retry", "missing_evaluator_rerun", "declared_fallback"):
        items = grouped.get(family, [])
        recoveries = sum(
            not row.get("initial_evidence_verified_success") and row.get("evidence_verified_success")
            for row in items
        )
        output.append(
            {
                "repair_budget": budget,
                "action_family": family,
                "task_count": len(items),
                "recovery_count": recoveries,
                "recovery_rate": recoveries / len(items) if items else None,
                "mean_repair_calls": (
                    sum(int(row.get("repair_tool_call_count") or 0) for row in items) / len(items)
                    if items else None
                ),
                "false_repair_count": sum(
                    row.get("repairability") == "irrecoverable" and row.get("evidence_verified_success")
                    for row in items
                ),
                "source": "new_run",
            }
        )
    return output


def _action_families(row: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    history = row.get("repair_plan_history") or []
    for plan in history:
        for action in plan.get("actions") or []:
            family = ACTION_FAMILIES.get(str(action.get("action_type")))
            if family:
                families.add(family)
    return families


def _per_task_rows(budget: int, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        rows.append(
            {
                "repair_budget": budget,
                "task_id": item.get("task_id"),
                "scenario_template_id": item.get("scenario_template_id"),
                "variant_id": item.get("variant_id"),
                "failure_family": item.get("failure_family"),
                "repairability": item.get("repairability"),
                "initial_evidence_verified_success": item.get("initial_evidence_verified_success"),
                "evidence_verified_success": item.get("evidence_verified_success"),
                "repair_executed": item.get("repair_executed"),
                "repair_rounds_executed": item.get("repair_rounds_executed"),
                "repair_action_families": sorted(_action_families(item)),
                "fallback_executed": item.get("fallback_executed"),
                "initial_tool_call_count": item.get("initial_tool_call_count"),
                "repair_tool_call_count": item.get("repair_tool_call_count"),
                "tool_call_count": item.get("tool_call_count"),
                "total_elapsed_sec": item.get("total_elapsed_sec"),
                "failure_reason": item.get("failure_reason"),
                "source": "new_run",
            }
        )
    return rows


def _budget_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated repair-budget summary.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Budget & Verified & Recovery & Calls/task & Calls/success & False repair \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['repair_budget']} & {_pct(row['evidence_verified_success_rate'])} & "
            f"{_pct(row['repair_recovery_rate'])} & {_num(row['mean_calls_per_task'])} & "
            f"{_num(row['calls_per_verified_success'])} & {row['false_repair_count']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Reliability and backend-call cost under explicit repair budgets.}",
            "\\label{tab:repair-budget}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _action_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated repair action breakdown.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{rlrrrr}",
        "\\toprule",
        "Budget & Action & Tasks & Recovery & Repair calls & False repair \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['repair_budget']} & {_tex(row['action_family'])} & {row['task_count']} & "
            f"{_pct(row['recovery_rate'])} & {_num(row['mean_repair_calls'])} & "
            f"{row['false_repair_count']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Recovery and call cost by repair action family.}",
            "\\label{tab:repair-action-breakdown}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}\\%"


def _num(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit repair budgets over one benchmark.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    args = parser.parse_args()
    payload = run_repair_budget_experiment(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        execution_mode=args.execution_mode,
        budgets=args.budgets,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
