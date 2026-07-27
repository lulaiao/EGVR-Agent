"""Execute replayed LLM-router workflows through the EGVR executor/verifier.

This runner is intentionally small and audit-oriented. It does not call an LLM.
It consumes saved planning-only router outputs, converts valid router JSON into
``PlannedWorkflow`` objects, executes those workflows with the existing
``WorkflowExecutor``, and summarizes whether execution evidence satisfies the
task verifier.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark_runner import (
    BenchmarkCase,
    _case_result,
    _execute_verifier_guided_repair,
    _tool_functions_for_case,
    load_benchmark_cases,
    summarize_results,
)
from .executor import WorkflowExecutor
from .llm_router_baseline_runner import _parse_json_object, _workflow_from_response
from .result_normalizer import rank_candidates
from .repair import suggest_repair
from .task_parser import parse_task
from .task_schema import PlannedWorkflow, VerifierResult
from .trace_logger import JSONLTraceLogger
from .verifier import verify_workflow


DEFAULT_OUTPUT = (
    "logs/baseline_runs/llm_as_router_executed_audit_v1/"
    "llm_router_executed_audit_summary.json"
)


def run_llm_router_execution_audit(
    *,
    benchmark_paths: list[str | Path],
    router_summary_path: str | Path,
    router_name: str,
    output_path: str | Path = DEFAULT_OUTPUT,
    execution_mode: str = "mock",
    trace_log_dir: str | Path | None = None,
    limit: int | None = None,
    project_root: str | Path | None = None,
    repair_mode: str = "none",
) -> dict[str, Any]:
    """Execute saved LLM-router workflows and write an audit summary."""

    if execution_mode not in {"mock", "real"}:
        raise ValueError("execution_mode must be 'mock' or 'real'")
    if repair_mode not in {"none", "verifier_guided"}:
        raise ValueError("repair_mode must be 'none' or 'verifier_guided'")

    root = Path(project_root or ".").resolve()
    response_map = _load_router_response_map(_resolve(router_summary_path, root))
    cases: list[BenchmarkCase] = []
    for benchmark_path in benchmark_paths:
        cases.extend(load_benchmark_cases(_resolve(benchmark_path, root)))
        if limit is not None and len(cases) >= limit:
            cases = cases[:limit]
            break

    trace_logger = JSONLTraceLogger(trace_log_dir) if trace_log_dir else None
    results = [
        _run_case(
            case,
            router_name=router_name,
            raw_response=response_map.get(case.task_id),
            execution_mode=execution_mode,
            trace_logger=trace_logger,
            repair_mode=repair_mode,
        )
        for case in cases
    ]

    summary = summarize_results(results)
    summary.update(
        {
            "executed_task_count": len(results),
            "valid_router_workflow_count": sum(1 for item in results if item.get("router_workflow_valid")),
            "router_workflow_valid_rate": _rate(item.get("router_workflow_valid") for item in results),
            "missing_router_response_count": sum(1 for item in results if item.get("router_response_missing")),
            "execution_failure_count": sum(1 for item in results if item.get("failed_tool_call_count", 0) > 0),
        }
    )

    out_path = _resolve(output_path, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_id": "llm_as_router_executed_audit_v1",
        "router_name": router_name,
        "execution_mode": execution_mode,
        "repair_mode": repair_mode,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "router_summary_path": _display_path(_resolve(router_summary_path, root), root),
        "benchmark_paths": [_display_path(_resolve(path, root), root) for path in benchmark_paths],
        "summary": summary,
        "results": results,
        "notes": [
            "Executed audit over saved LLM-router workflows; no LLM API calls are made by this runner.",
            f"Repair mode: {repair_mode}.",
            "This is a small supporting baseline, not a replacement for the controlled repair benchmark.",
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    out_path.with_suffix(".csv").write_text(_summary_csv(payload), encoding="utf-8")
    return payload


def _run_case(
    case: BenchmarkCase,
    *,
    router_name: str,
    raw_response: str | None,
    execution_mode: str,
    trace_logger: JSONLTraceLogger | None,
    repair_mode: str,
) -> dict[str, Any]:
    parsed_task = parse_task(case.raw_user_query, task_id=case.task_id, metadata=case.metadata)
    workflow, router_error = _workflow_from_raw_response(case.task_id, raw_response)
    if workflow is None:
        verifier_result = VerifierResult(
            success=False,
            checks={"router_workflow_valid": False},
            metrics={},
            failure_reason=router_error or "router_workflow_invalid",
        )
        empty_workflow = PlannedWorkflow(
            task_id=case.task_id,
            planner_type=f"llm_as_router_executed:{router_name}",
            selected_tools=[],
            tool_sequence=[],
            expected_outputs=[],
            notes=[router_error or "Router workflow could not be parsed."],
        )
        return {
            **_case_result(
                case,
                parsed_task,
                empty_workflow,
                verifier_result,
                repair_plan=None,
                tool_calls=[],
                repair_executed=False,
                repair_success=False,
                planner_baseline=f"llm_as_router_executed:{router_name}:{repair_mode}",
            ),
            "router_name": router_name,
            "router_workflow_valid": False,
            "router_response_missing": raw_response is None,
            "router_error": router_error,
        }

    workflow.planner_type = f"llm_as_router_executed:{router_name}"
    executor = WorkflowExecutor(tool_functions=_tool_functions_for_case(case, execution_mode))
    tool_calls, candidates = executor.execute(parsed_task, workflow)
    candidates = rank_candidates(candidates) if parsed_task.constraints.require_ranking else candidates
    verifier_result = verify_workflow(parsed_task, workflow, tool_calls, candidates)
    initial_tool_call_count = len(tool_calls)
    initial_verifier_success = verifier_result.success
    repair_trigger_checks = [name for name, passed in verifier_result.checks.items() if passed is False]
    repair_plan = None
    repair_executed = False
    repair_success = False
    if repair_mode == "verifier_guided" and (
        not verifier_result.success or any(not record.success for record in tool_calls)
    ):
        repair_plan = suggest_repair(parsed_task, workflow, tool_calls, candidates, verifier_result)
        if repair_plan.should_retry:
            workflow, tool_calls, candidates, verifier_result, repair_executed = _execute_verifier_guided_repair(
                parsed_task,
                workflow,
                executor,
                tool_calls,
                candidates,
                repair_plan,
                planner_type=f"llm_as_router_executed:{router_name}:verifier_guided",
            )
            repair_success = repair_executed and verifier_result.success
    if trace_logger:
        trace_logger.log_trace(
            parsed_task=parsed_task,
            planned_workflow=workflow,
            tool_calls=tool_calls,
            candidate_records=candidates,
            verifier_result=verifier_result,
            metadata={
                "benchmark_case": case.task_id,
                "execution_mode": execution_mode,
                "planner_baseline": f"llm_as_router_executed:{router_name}:{repair_mode}",
                "router_name": router_name,
                "repair_mode": repair_mode,
                "repair_executed": repair_executed,
                "repair_plan": repair_plan.to_dict() if repair_plan else None,
                "initial_tool_call_count": initial_tool_call_count,
                "repair_tool_call_count": max(0, len(tool_calls) - initial_tool_call_count),
                "repair_trigger_checks": repair_trigger_checks,
            },
        )
    return {
        **_case_result(
            case,
            parsed_task,
            workflow,
            verifier_result,
            repair_plan=repair_plan,
            tool_calls=tool_calls,
            repair_executed=repair_executed,
            repair_success=repair_success,
            initial_tool_call_count=initial_tool_call_count,
            initial_verifier_success=initial_verifier_success,
            repair_trigger_checks=repair_trigger_checks,
            planner_baseline=f"llm_as_router_executed:{router_name}:{repair_mode}",
        ),
        "router_name": router_name,
        "router_workflow_valid": True,
        "router_response_missing": False,
        "router_error": None,
        "repair_mode": repair_mode,
    }


def _workflow_from_raw_response(task_id: str, raw_response: str | None) -> tuple[PlannedWorkflow | None, str | None]:
    if raw_response is None:
        return None, "missing_router_response"
    parsed_response, json_error = _parse_json_object(raw_response)
    if json_error:
        return None, f"invalid_json:{json_error}"
    workflow, schema_error = _workflow_from_response(parsed_response)
    if schema_error or workflow is None:
        return None, f"invalid_schema:{schema_error or 'unknown'}"
    if workflow.task_id != task_id:
        workflow.task_id = task_id
    return workflow, None


def _load_router_response_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    response_map: dict[str, str] = {}
    for item in payload.get("task_results", []):
        task_id = item.get("task_id")
        raw_response = item.get("raw_response")
        if task_id and raw_response:
            response_map[str(task_id)] = str(raw_response)
    return response_map


def _rate(values) -> float | None:
    flags = [bool(value) for value in values]
    if not flags:
        return None
    return sum(flags) / len(flags)


def _summary_csv(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    fields = [
        "benchmark_id",
        "router_name",
        "execution_mode",
        "repair_mode",
        "executed_task_count",
        "router_workflow_valid_rate",
        "task_success_rate",
        "verifier_expectation_match",
        "false_success_count",
        "mean_tool_call_count",
        "tool_call_failure_rate",
        "mean_total_elapsed_sec",
    ]
    row = {
        "benchmark_id": payload["benchmark_id"],
        "router_name": payload["router_name"],
        "execution_mode": payload["execution_mode"],
        "repair_mode": payload["repair_mode"],
        **{key: summary.get(key) for key in fields},
    }
    return ",".join(fields) + "\n" + ",".join(str(row.get(field, "")) for field in fields) + "\n"


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute saved LLM-router workflows through the EGVR verifier.")
    parser.add_argument("--benchmark", action="append", required=True, help="Benchmark JSONL path. Repeatable.")
    parser.add_argument("--router-summary", required=True, help="Planning-only router summary JSON with task_results.")
    parser.add_argument("--router-name", required=True, help="Short router/model name for reporting.")
    parser.add_argument("--execution-mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--repair-mode", choices=["none", "verifier_guided"], default="none")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-log-dir")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    payload = run_llm_router_execution_audit(
        benchmark_paths=args.benchmark,
        router_summary_path=args.router_summary,
        router_name=args.router_name,
        output_path=args.output,
        execution_mode=args.execution_mode,
        trace_log_dir=args.trace_log_dir,
        limit=args.limit,
        project_root=args.project_root,
        repair_mode=args.repair_mode,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
