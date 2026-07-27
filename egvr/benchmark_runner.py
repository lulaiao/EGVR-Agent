"""Benchmark runner for chemistry-aware planner baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from .baseline_planners import (
    EGVR_AGENT,
    FULL_COPILOT,
    LEGACY_FULL_COPILOT,
    RULE_BASED_PLANNER,
    SCHEDULED_FALLBACK_NO_VERIFIER,
    SUPPORTED_BASELINES,
    TOOL_STATUS_ONLY,
    VERIFIER_ONLY_NO_REPAIR,
    VERIFIER_TARGETED_RETRY_NO_FALLBACK,
    plan_for_baseline,
)
from .executor import WorkflowExecutor
from .repair import RepairAction, RepairPlan, suggest_repair
from .result_normalizer import rank_candidates
from .task_parser import parse_task
from .task_schema import CandidateRecord, ParsedTask, PlannedToolCall, PlannedWorkflow, ToolCallRecord, VerifierResult
from .trace_logger import JSONLTraceLogger
from .verifier import verify_workflow


@dataclass
class BenchmarkCase:
    """One benchmark task loaded from JSONL."""

    task_id: str
    raw_user_query: str
    expected_task_type: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    should_succeed: bool = True
    mock_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkCase":
        return cls(
            task_id=str(payload["task_id"]),
            raw_user_query=str(payload["raw_user_query"]),
            expected_task_type=payload.get("expected_task_type"),
            expected_tools=list(payload.get("expected_tools", [])),
            should_succeed=bool(payload.get("should_succeed", True)),
            mock_outputs=dict(payload.get("mock_outputs", {})),
            metadata=dict(payload.get("metadata", {})),
        )


class BenchmarkRunner:
    """Run parser -> planner -> executor -> verifier on JSONL benchmark cases."""

    def __init__(
        self,
        *,
        execution_mode: str = "mock",
        planner_baseline: str = RULE_BASED_PLANNER,
        trace_log_dir: str | Path | None = None,
        repair_budget: int | None = None,
    ) -> None:
        if execution_mode not in {"mock", "real"}:
            raise ValueError("execution_mode must be 'mock' or 'real'")
        if planner_baseline not in SUPPORTED_BASELINES:
            raise ValueError(f"planner_baseline must be one of: {', '.join(SUPPORTED_BASELINES)}")
        if repair_budget is not None and repair_budget < 0:
            raise ValueError("repair_budget must be non-negative")
        self.execution_mode = execution_mode
        self.planner_baseline = planner_baseline
        self.repair_budget = repair_budget
        self.trace_logger = JSONLTraceLogger(trace_log_dir) if trace_log_dir else None

    def run_file(self, benchmark_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
        cases = load_benchmark_cases(benchmark_path)
        results = [self.run_case(case) for case in cases]
        summary = summarize_results(results)
        payload = {
            "benchmark_id": Path(benchmark_path).stem,
            "execution_mode": self.execution_mode,
            "planner_baseline": self.planner_baseline,
            "summary": summary,
            "results": results,
        }
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def run_case(self, case: BenchmarkCase) -> dict[str, Any]:
        parsed_task = parse_task(case.raw_user_query, task_id=case.task_id, metadata=case.metadata)
        workflow = plan_for_baseline(parsed_task, self.planner_baseline)
        workflow = _prepare_controlled_workflow(case, workflow)
        initial_workflow = _clone_workflow(workflow)
        initial_planned_workflow = initial_workflow.to_dict()
        initial_workflow_signature = _workflow_signature(initial_workflow)
        declared_fallbacks = _declared_fallbacks(initial_workflow)
        deferred_fallbacks = [item for item in declared_fallbacks if item.get("deferred")]
        executor = WorkflowExecutor(tool_functions=_tool_functions_for_case(case, self.execution_mode))
        tool_calls, candidates = executor.execute(parsed_task, workflow)
        candidates = rank_candidates(candidates) if parsed_task.constraints.require_ranking else candidates
        verifier_result = verify_workflow(parsed_task, workflow, tool_calls, candidates)
        initial_verifier_result = verifier_result
        initial_tool_call_count = len(tool_calls)
        initial_verifier_success = verifier_result.success
        repair_trigger_checks = _required_failed_checks(verifier_result)
        nonrequired_failed_checks = _nonrequired_failed_checks(verifier_result, repair_trigger_checks)
        repair_plan = None
        repair_plan_history: list[dict[str, Any]] = []
        proposed_repair_actions: list[dict[str, Any]] = []
        authorized_repair_actions: list[dict[str, Any]] = []
        rejected_repair_actions: list[dict[str, Any]] = []
        executed_repair_actions: list[dict[str, Any]] = []
        repair_executed = False
        repair_success = False
        if not verifier_result.success or any(not record.success for record in tool_calls):
            proposed_plan = suggest_repair(parsed_task, workflow, tool_calls, candidates, verifier_result)
            if self.planner_baseline == SCHEDULED_FALLBACK_NO_VERIFIER:
                proposed_plan = suggest_repair(parsed_task, workflow, tool_calls, candidates, None)
            repair_plan, decision_audit = _authorize_repair_plan(
                proposed_plan,
                policy=self.planner_baseline,
                initial_workflow=initial_workflow,
                initial_tool_calls=tool_calls[:initial_tool_call_count],
                candidates=candidates,
                required_failed_checks=repair_trigger_checks,
                deferred_mode=_uses_deferred_declared_fallback(case),
                repair_round=1,
            )
            proposed_repair_actions.extend(decision_audit["proposed"])
            authorized_repair_actions.extend(decision_audit["authorized"])
            rejected_repair_actions.extend(decision_audit["rejected"])
            if _is_egvr_agent(self.planner_baseline):
                budget = 1 if self.repair_budget is None else self.repair_budget
                rounds = 0
                while repair_plan.should_retry and rounds < budget:
                    repair_plan_history.append(repair_plan.to_dict())
                    call_count_before_repair = len(tool_calls)
                    workflow, tool_calls, candidates, verifier_result, round_executed = _execute_verifier_guided_repair(
                        parsed_task,
                        workflow,
                        executor,
                        tool_calls,
                        candidates,
                        repair_plan,
                        planner_type=self.planner_baseline,
                    )
                    if not round_executed:
                        break
                    executed_repair_actions.extend(
                        _executed_action_records(
                            repair_plan.actions,
                            tool_calls[call_count_before_repair:],
                            policy=self.planner_baseline,
                            repair_round=rounds + 1,
                            initial_workflow=initial_workflow,
                            required_failed_checks=repair_trigger_checks,
                        )
                    )
                    repair_executed = True
                    rounds += 1
                    if verifier_result.success:
                        break
                    if rounds < budget:
                        proposed_plan = suggest_repair(parsed_task, workflow, tool_calls, candidates, verifier_result)
                        repair_plan, decision_audit = _authorize_repair_plan(
                            proposed_plan,
                            policy=self.planner_baseline,
                            initial_workflow=initial_workflow,
                            initial_tool_calls=tool_calls[:initial_tool_call_count],
                            candidates=candidates,
                            required_failed_checks=_required_failed_checks(verifier_result),
                            deferred_mode=_uses_deferred_declared_fallback(case),
                            repair_round=rounds + 1,
                        )
                        proposed_repair_actions.extend(decision_audit["proposed"])
                        authorized_repair_actions.extend(decision_audit["authorized"])
                        rejected_repair_actions.extend(decision_audit["rejected"])
                repair_success = repair_executed and verifier_result.success
            elif self.planner_baseline == VERIFIER_TARGETED_RETRY_NO_FALLBACK:
                if repair_plan.should_retry:
                    repair_plan_history.append(repair_plan.to_dict())
                    call_count_before_repair = len(tool_calls)
                    workflow, tool_calls, candidates, verifier_result, repair_executed = _execute_verifier_guided_repair(
                        parsed_task,
                        workflow,
                        executor,
                        tool_calls,
                        candidates,
                        repair_plan,
                        planner_type=VERIFIER_TARGETED_RETRY_NO_FALLBACK,
                    )
                    if repair_executed:
                        executed_repair_actions.extend(
                            _executed_action_records(
                                repair_plan.actions,
                                tool_calls[call_count_before_repair:],
                                policy=self.planner_baseline,
                                repair_round=1,
                                initial_workflow=initial_workflow,
                                required_failed_checks=repair_trigger_checks,
                            )
                        )
                    repair_success = repair_executed and verifier_result.success
            elif self.planner_baseline == SCHEDULED_FALLBACK_NO_VERIFIER:
                if repair_plan.should_retry:
                    repair_plan_history.append(repair_plan.to_dict())
                    call_count_before_repair = len(tool_calls)
                    workflow, tool_calls, candidates, verifier_result, repair_executed = _execute_scheduled_repair(
                        parsed_task,
                        workflow,
                        executor,
                        tool_calls,
                        candidates,
                        repair_plan,
                    )
                    if repair_executed:
                        executed_repair_actions.extend(
                            _executed_action_records(
                                repair_plan.actions,
                                tool_calls[call_count_before_repair:],
                                policy=self.planner_baseline,
                                repair_round=1,
                                initial_workflow=initial_workflow,
                                required_failed_checks=repair_trigger_checks,
                            )
                        )
                    repair_success = repair_executed and verifier_result.success
        fallback_executed = _fallback_executed(
            repair_plan,
            repair_executed,
            tool_calls[initial_tool_call_count:],
        )
        fallback_authorization_source = _fallback_authorization_source(
            self.planner_baseline,
            fallback_executed,
            case,
        )
        if self.trace_logger:
            self.trace_logger.log_trace(
                parsed_task=parsed_task,
                planned_workflow=workflow,
                tool_calls=tool_calls,
                candidate_records=candidates,
                verifier_result=verifier_result,
                metadata={
                    "benchmark_case": case.task_id,
                    "execution_mode": self.execution_mode,
                    "planner_baseline": self.planner_baseline,
                    "failure_injections": _failure_injections(case),
                    "repair_executed": repair_executed,
                    "repair_plan": repair_plan.to_dict() if repair_plan else None,
                    "repair_plan_history": repair_plan_history,
                    "repair_budget": self.repair_budget,
                    "repair_rounds_executed": len(repair_plan_history) if repair_executed else 0,
                    "initial_tool_call_count": initial_tool_call_count,
                    "repair_tool_call_count": max(0, len(tool_calls) - initial_tool_call_count),
                    "initial_backend_call_count": _backend_call_count(tool_calls[:initial_tool_call_count]),
                    "repair_backend_call_count": _backend_call_count(tool_calls[initial_tool_call_count:]),
                    "agent_claimed_success": _agent_claimed_success(
                        self.planner_baseline, tool_calls[:initial_tool_call_count], verifier_result
                    ),
                    "evidence_verified_success": verifier_result.success,
                    "initial_verifier_result": initial_verifier_result.to_dict(),
                    "fallback_executed": fallback_executed,
                    "fallback_authorization_source": fallback_authorization_source,
                    "repair_trigger_checks": repair_trigger_checks,
                    "nonrequired_failed_checks": nonrequired_failed_checks,
                    "proposed_repair_actions": proposed_repair_actions,
                    "authorized_repair_actions": authorized_repair_actions,
                    "rejected_repair_actions": rejected_repair_actions,
                    "executed_repair_actions": executed_repair_actions,
                    "controlled_protocol_id": case.metadata.get("controlled_protocol_id"),
                    "initial_planned_workflow": initial_planned_workflow,
                    "initial_workflow_signature": initial_workflow_signature,
                    "declared_fallbacks": declared_fallbacks,
                    "deferred_fallbacks": deferred_fallbacks,
                },
            )
        return _case_result(
            case,
            parsed_task,
            workflow,
            verifier_result,
            repair_plan,
            tool_calls=tool_calls,
            repair_executed=repair_executed,
            repair_success=repair_success,
            initial_tool_call_count=initial_tool_call_count,
            initial_verifier_success=initial_verifier_success,
            initial_verifier_result=initial_verifier_result,
            repair_trigger_checks=repair_trigger_checks,
            nonrequired_failed_checks=nonrequired_failed_checks,
            proposed_repair_actions=proposed_repair_actions,
            authorized_repair_actions=authorized_repair_actions,
            rejected_repair_actions=rejected_repair_actions,
            executed_repair_actions=executed_repair_actions,
            planner_baseline=self.planner_baseline,
            repair_budget=self.repair_budget,
            repair_plan_history=repair_plan_history,
            initial_planned_workflow=initial_planned_workflow,
            initial_workflow_signature=initial_workflow_signature,
            declared_fallbacks=declared_fallbacks,
            deferred_fallbacks=deferred_fallbacks,
            fallback_authorization_source=fallback_authorization_source,
        )


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            payload = json.loads(stripped)
            try:
                cases.append(BenchmarkCase.from_dict(payload))
            except KeyError as exc:
                raise ValueError(f"Missing required field {exc} on line {line_no}") from exc
    return cases


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    parser_correct = sum(1 for result in results if result["parser_correct"])
    planner_correct = sum(1 for result in results if result["planner_correct"])
    verifier_matched = sum(1 for result in results if result["verifier_matched_expectation"])
    task_successes = sum(1 for result in results if result["task_success"])
    false_success_count = sum(
        1
        for result in results
        if result.get("agent_claimed_success", result["task_success"])
        and not result.get("evidence_verified_success", result["task_success"])
    )
    repair_attempt_count = sum(1 for result in results if result["repair_executed"])
    repair_success_count = sum(1 for result in results if result["repair_success"])
    planner_precision = sum(result["planner_tool_precision"] for result in results)
    planner_recall = sum(result["planner_tool_recall"] for result in results)
    planner_f1 = sum(result["planner_tool_f1"] for result in results)
    selected_tool_count = sum(result.get("selected_tool_count", 0) for result in results)
    tool_sequence_length = sum(result.get("tool_sequence_length", 0) for result in results)
    extra_tool_count = sum(result.get("extra_tool_count", 0) for result in results)
    tool_call_count = sum(result.get("tool_call_count", 0) for result in results)
    backend_call_count = sum(result.get("backend_call_count", result.get("tool_call_count", 0)) for result in results)
    failed_tool_call_count = sum(result.get("failed_tool_call_count", 0) for result in results)
    agent_claim_successes = sum(1 for result in results if result.get("agent_claimed_success", result["task_success"]))
    initial_tool_call_count = sum(result.get("initial_tool_call_count", result.get("tool_call_count", 0)) for result in results)
    initial_backend_call_count = sum(
        result.get("initial_backend_call_count", result.get("initial_tool_call_count", 0)) for result in results
    )
    repair_tool_call_count = sum(result.get("repair_tool_call_count", 0) for result in results)
    repair_backend_call_count = sum(result.get("repair_backend_call_count", result.get("repair_tool_call_count", 0)) for result in results)
    repair_round_count = sum(result.get("repair_rounds_executed", 0) for result in results)
    candidate_evaluation_call_count = sum(result.get("candidate_evaluation_call_count", 0) for result in results)
    fallback_count = sum(1 for result in results if result.get("fallback_executed"))
    elapsed_values = [
        float(result["total_elapsed_sec"])
        for result in results
        if isinstance(result.get("total_elapsed_sec"), int | float)
    ]
    return {
        "total": total,
        "repair_attempt_count": repair_attempt_count,
        "repair_success_count": repair_success_count,
        "false_success_count": false_success_count,
        "repair_attempt_rate": repair_attempt_count / total if total else 0.0,
        "repair_success_rate": repair_success_count / repair_attempt_count if repair_attempt_count else 0.0,
        "parser_accuracy": parser_correct / total if total else 0.0,
        "planner_tool_coverage_rate": planner_correct / total if total else 0.0,
        "planner_tool_precision": planner_precision / total if total else 0.0,
        "planner_tool_recall": planner_recall / total if total else 0.0,
        "planner_tool_f1": planner_f1 / total if total else 0.0,
        "verifier_expectation_match": verifier_matched / total if total else 0.0,
        "task_success_rate": task_successes / total if total else 0.0,
        "agent_claim_success_rate": agent_claim_successes / total if total else 0.0,
        "evidence_verified_success_rate": task_successes / total if total else 0.0,
        "mean_selected_tool_count": selected_tool_count / total if total else 0.0,
        "mean_tool_sequence_length": tool_sequence_length / total if total else 0.0,
        "mean_extra_tool_count": extra_tool_count / total if total else 0.0,
        "mean_tool_call_count": tool_call_count / total if total else 0.0,
        "mean_backend_call_count": backend_call_count / total if total else 0.0,
        "mean_initial_tool_call_count": initial_tool_call_count / total if total else 0.0,
        "mean_initial_backend_call_count": initial_backend_call_count / total if total else 0.0,
        "mean_repair_tool_call_count": repair_tool_call_count / total if total else 0.0,
        "mean_repair_backend_call_count": repair_backend_call_count / total if total else 0.0,
        "mean_repair_round_count": repair_round_count / total if total else 0.0,
        "mean_candidate_evaluation_call_count": candidate_evaluation_call_count / total if total else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / repair_attempt_count if repair_attempt_count else 0.0,
        "failed_tool_call_count": failed_tool_call_count,
        "tool_call_failure_rate": failed_tool_call_count / tool_call_count if tool_call_count else 0.0,
        "mean_total_elapsed_sec": mean(elapsed_values) if elapsed_values else 0.0,
        "median_total_elapsed_sec": median(elapsed_values) if elapsed_values else 0.0,
    }


def _case_result(
    case: BenchmarkCase,
    parsed_task: ParsedTask,
    workflow,
    verifier_result: VerifierResult,
    repair_plan,
    *,
    tool_calls: list[ToolCallRecord],
    repair_executed: bool = False,
    repair_success: bool = False,
    initial_tool_call_count: int | None = None,
    initial_verifier_success: bool | None = None,
    initial_verifier_result: VerifierResult | None = None,
    repair_trigger_checks: list[str] | None = None,
    nonrequired_failed_checks: list[str] | None = None,
    proposed_repair_actions: list[dict[str, Any]] | None = None,
    authorized_repair_actions: list[dict[str, Any]] | None = None,
    rejected_repair_actions: list[dict[str, Any]] | None = None,
    executed_repair_actions: list[dict[str, Any]] | None = None,
    planner_baseline: str | None = None,
    repair_budget: int | None = None,
    repair_plan_history: list[dict[str, Any]] | None = None,
    initial_planned_workflow: dict[str, Any] | None = None,
    initial_workflow_signature: str | None = None,
    declared_fallbacks: list[dict[str, Any]] | None = None,
    deferred_fallbacks: list[dict[str, Any]] | None = None,
    fallback_authorization_source: str | None = None,
) -> dict[str, Any]:
    parser_correct = case.expected_task_type in {None, parsed_task.task_type}
    planner_correct = set(case.expected_tools).issubset(set(workflow.selected_tools))
    tool_scores = _tool_set_scores(case.expected_tools, workflow.selected_tools)
    verifier_matched = verifier_result.success is case.should_succeed
    total_elapsed_sec = _total_elapsed_sec(tool_calls)
    initial_count = len(tool_calls) if initial_tool_call_count is None else initial_tool_call_count
    claimed_success = _agent_claimed_success(
        planner_baseline or workflow.planner_type,
        tool_calls[:initial_count],
        verifier_result,
    )
    return {
        "task_id": case.task_id,
        "scenario_template_id": case.metadata.get("scenario_template_id") or case.metadata.get("failure_scenario"),
        "variant_id": case.metadata.get("variant_id"),
        "failure_family": case.metadata.get("failure_family") or case.metadata.get("evidence_family"),
        "repairability": case.metadata.get("repairability"),
        "expected_repair_tools": list(case.metadata.get("expected_repair_tools", [])),
        "expected_repair_action_types": list(case.metadata.get("expected_repair_action_types", [])),
        "parsed_task_type": parsed_task.task_type,
        "expected_task_type": case.expected_task_type,
        "parser_correct": parser_correct,
        "planner_type": workflow.planner_type,
        "selected_tools": workflow.selected_tools,
        "selected_tool_count": len(workflow.selected_tools),
        "tool_sequence_length": len(workflow.tool_sequence),
        "tool_call_count": len(tool_calls),
        "backend_call_count": _backend_call_count(tool_calls),
        "initial_tool_call_count": initial_count,
        "initial_backend_call_count": _backend_call_count(tool_calls[:initial_count]),
        "repair_tool_call_count": max(0, len(tool_calls) - initial_count),
        "repair_backend_call_count": _backend_call_count(tool_calls[initial_count:]),
        "candidate_evaluation_call_count": sum(
            1 for record in tool_calls if "candidate_index" in (record.metadata or {})
        ),
        "failed_tool_call_count": sum(1 for record in tool_calls if not record.success),
        "total_elapsed_sec": total_elapsed_sec,
        "tool_elapsed_sec": _tool_elapsed_rollup(tool_calls),
        "expected_tools": case.expected_tools,
        "planner_correct": planner_correct,
        "missing_tools": tool_scores["missing_tools"],
        "extra_tools": tool_scores["extra_tools"],
        "missing_tool_count": len(tool_scores["missing_tools"]),
        "extra_tool_count": len(tool_scores["extra_tools"]),
        "planner_tool_precision": tool_scores["precision"],
        "planner_tool_recall": tool_scores["recall"],
        "planner_tool_f1": tool_scores["f1"],
        "task_success": verifier_result.success,
        "agent_claimed_success": claimed_success,
        "evidence_verified_success": verifier_result.success,
        "initial_evidence_verified_success": (
            verifier_result.success if initial_verifier_success is None else initial_verifier_success
        ),
        "initial_verifier_result": (
            initial_verifier_result.to_dict() if initial_verifier_result else verifier_result.to_dict()
        ),
        "expected_success": case.should_succeed,
        "verifier_matched_expectation": verifier_matched,
        "failure_reason": verifier_result.failure_reason,
        "repair_plan": repair_plan.to_dict() if repair_plan else None,
        "repair_plan_history": list(repair_plan_history or []),
        "repair_budget": repair_budget,
        "repair_rounds_executed": len(repair_plan_history or []) if repair_executed else 0,
        "repair_executed": repair_executed,
        "repair_success": repair_success,
        "fallback_executed": _fallback_executed(repair_plan, repair_executed, tool_calls[initial_count:]),
        "fallback_authorization_source": fallback_authorization_source,
        "repair_trigger_checks": list(repair_trigger_checks or []),
        "nonrequired_failed_checks": list(nonrequired_failed_checks or []),
        "proposed_repair_actions": list(proposed_repair_actions or []),
        "authorized_repair_actions": list(authorized_repair_actions or []),
        "rejected_repair_actions": list(rejected_repair_actions or []),
        "executed_repair_actions": list(executed_repair_actions or []),
        "controlled_protocol_id": case.metadata.get("controlled_protocol_id"),
        "initial_planned_workflow": dict(initial_planned_workflow or {}),
        "initial_workflow_signature": initial_workflow_signature,
        "declared_fallbacks": list(declared_fallbacks or []),
        "deferred_fallbacks": list(deferred_fallbacks or []),
        "metrics": verifier_result.metrics,
    }


def _total_elapsed_sec(tool_calls: list[ToolCallRecord]) -> float | None:
    values = [record.elapsed_time_sec for record in tool_calls if record.elapsed_time_sec is not None]
    return sum(values) if values else None


def _tool_elapsed_rollup(tool_calls: list[ToolCallRecord]) -> dict[str, float]:
    rollup: dict[str, float] = {}
    for record in tool_calls:
        if record.elapsed_time_sec is None:
            continue
        rollup[record.tool_name] = rollup.get(record.tool_name, 0.0) + float(record.elapsed_time_sec)
    return rollup


def _tool_set_scores(expected_tools: list[str], selected_tools: list[str]) -> dict[str, Any]:
    expected = set(expected_tools)
    selected = set(selected_tools)
    overlap = expected & selected
    precision = len(overlap) / len(selected) if selected else (1.0 if not expected else 0.0)
    recall = len(overlap) / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing_tools": sorted(expected - selected),
        "extra_tools": sorted(selected - expected),
    }


_EVALUATOR_TOOLS = {"vina", "scscore", "toxicity", "pmic"}
_RETRY_ACTION_TYPES = {
    "fallback_tool",
    "retry_with_reduced_generation_count",
    "retry_evaluator_for_missing_evidence",
}


def _execute_verifier_guided_repair(
    parsed_task: ParsedTask,
    workflow: PlannedWorkflow,
    executor: WorkflowExecutor,
    tool_calls: list[ToolCallRecord],
    candidates: list[CandidateRecord],
    repair_plan: RepairPlan,
    *,
    planner_type: str = EGVR_AGENT,
) -> tuple[PlannedWorkflow, list[ToolCallRecord], list[CandidateRecord], VerifierResult, bool]:
    repair_workflow = _repair_execution_workflow(parsed_task, workflow, repair_plan)
    if not repair_workflow.tool_sequence:
        verifier_result = verify_workflow(parsed_task, workflow, tool_calls, candidates)
        return workflow, tool_calls, candidates, verifier_result, False

    repair_calls, repaired_candidates = executor.execute(
        parsed_task,
        repair_workflow,
        initial_candidates=candidates,
    )
    combined_calls = [*tool_calls, *repair_calls]
    final_candidates = rank_candidates(repaired_candidates) if parsed_task.constraints.require_ranking else repaired_candidates
    combined_workflow = _combine_workflows(parsed_task, workflow, repair_workflow, planner_type=planner_type)
    verifier_result = verify_workflow(parsed_task, combined_workflow, combined_calls, final_candidates)
    return combined_workflow, combined_calls, final_candidates, verifier_result, True


def _without_fallback_actions(repair_plan: RepairPlan) -> RepairPlan:
    actions = [action for action in repair_plan.actions if action.action_type != "fallback_tool"]
    return _repair_plan_with_actions(repair_plan, actions)


def _agent_claimed_success(
    planner_baseline: str,
    initial_tool_calls: list[ToolCallRecord],
    verifier_result: VerifierResult,
) -> bool:
    if planner_baseline == TOOL_STATUS_ONLY:
        return bool(initial_tool_calls) and all(record.success for record in initial_tool_calls)
    return verifier_result.success


def _required_failed_checks(verifier_result: VerifierResult) -> list[str]:
    if not verifier_result.failure_reason:
        return []
    return [
        item.strip()
        for item in verifier_result.failure_reason.split(",")
        if item.strip() and verifier_result.checks.get(item.strip()) is False
    ]


def _nonrequired_failed_checks(
    verifier_result: VerifierResult,
    required_failed_checks: list[str],
) -> list[str]:
    required = set(required_failed_checks)
    return [
        check_name
        for check_name, passed in verifier_result.checks.items()
        if passed is False and check_name not in required
    ]


def _authorize_repair_plan(
    proposed_plan: RepairPlan,
    *,
    policy: str,
    initial_workflow: PlannedWorkflow,
    initial_tool_calls: list[ToolCallRecord],
    candidates: list[CandidateRecord],
    required_failed_checks: list[str],
    deferred_mode: bool,
    repair_round: int,
) -> tuple[RepairPlan, dict[str, list[dict[str, Any]]]]:
    authorized_plan = proposed_plan
    if policy in {TOOL_STATUS_ONLY, VERIFIER_ONLY_NO_REPAIR}:
        authorized_plan = _repair_plan_with_actions(proposed_plan, [])
    elif policy == SCHEDULED_FALLBACK_NO_VERIFIER and deferred_mode:
        authorized_plan = _scheduled_failed_or_empty_repair(
            proposed_plan,
            initial_workflow,
            initial_tool_calls,
            candidates,
        )
    else:
        if deferred_mode:
            authorized_plan = _restrict_repair_plan_to_initial_workflow(proposed_plan, initial_workflow)
        if policy == VERIFIER_TARGETED_RETRY_NO_FALLBACK:
            authorized_plan = _without_fallback_actions(authorized_plan)

    proposed_signatures = {_action_signature(action) for action in proposed_plan.actions}
    authorized_signatures = {_action_signature(action) for action in authorized_plan.actions}
    proposed_records = [
        _action_audit_record(
            action,
            policy=policy,
            repair_round=repair_round,
            initial_workflow=initial_workflow,
            required_failed_checks=required_failed_checks,
            status="proposed",
        )
        for action in proposed_plan.actions
    ]
    authorized_records = [
        _action_audit_record(
            action,
            policy=policy,
            repair_round=repair_round,
            initial_workflow=initial_workflow,
            required_failed_checks=required_failed_checks,
            status="authorized",
        )
        for action in authorized_plan.actions
    ]
    rejected_records = []
    for action in proposed_plan.actions:
        if _action_signature(action) in authorized_signatures:
            continue
        rejected_records.append(
            _action_audit_record(
                action,
                policy=policy,
                repair_round=repair_round,
                initial_workflow=initial_workflow,
                required_failed_checks=required_failed_checks,
                status="rejected",
                rejection_reason=_action_rejection_reason(
                    action,
                    policy=policy,
                    initial_workflow=initial_workflow,
                    deferred_mode=deferred_mode,
                    proposed_signatures=proposed_signatures,
                    authorized_signatures=authorized_signatures,
                ),
            )
        )
    return authorized_plan, {
        "proposed": proposed_records,
        "authorized": authorized_records,
        "rejected": rejected_records,
    }


def _action_rejection_reason(
    action: RepairAction,
    *,
    policy: str,
    initial_workflow: PlannedWorkflow,
    deferred_mode: bool,
    proposed_signatures: set[str],
    authorized_signatures: set[str],
) -> str:
    del proposed_signatures, authorized_signatures
    declared_fallback_tools = {item["tool_name"] for item in _declared_fallbacks(initial_workflow)}
    initial_tools = set(initial_workflow.selected_tools)
    if deferred_mode and action.action_type == "fallback_tool" and action.tool_name not in declared_fallback_tools:
        return "undeclared_in_initial_workflow"
    if deferred_mode and action.action_type in _RETRY_ACTION_TYPES and action.tool_name not in initial_tools:
        return "undeclared_in_initial_workflow"
    if policy in {TOOL_STATUS_ONLY, VERIFIER_ONLY_NO_REPAIR}:
        return "policy_disallows_repair"
    if policy == VERIFIER_TARGETED_RETRY_NO_FALLBACK and action.action_type == "fallback_tool":
        return "policy_disallows_fallback"
    if policy == SCHEDULED_FALLBACK_NO_VERIFIER:
        if action.action_type != "fallback_tool":
            return "policy_disallows_nonfallback_action"
        return "failed_or_empty_trigger_not_met"
    return "not_authorized_by_policy_boundary"


def _action_audit_record(
    action: RepairAction,
    *,
    policy: str,
    repair_round: int,
    initial_workflow: PlannedWorkflow,
    required_failed_checks: list[str],
    status: str,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    declared_fallback_tools = {item["tool_name"] for item in _declared_fallbacks(initial_workflow)}
    initial_tools = set(initial_workflow.selected_tools)
    declared = (
        action.tool_name in declared_fallback_tools
        if action.action_type == "fallback_tool"
        else action.tool_name in initial_tools
    )
    record = {
        **action.to_dict(),
        "status": status,
        "declared_in_initial_workflow": bool(declared),
        "required_by_failed_checks": _action_required_by_failed_checks(action, required_failed_checks),
        "policy": policy,
        "repair_round": repair_round,
    }
    if rejection_reason:
        record["rejection_reason"] = rejection_reason
    return record


def _action_required_by_failed_checks(action: RepairAction, required_failed_checks: list[str]) -> bool:
    failed = set(required_failed_checks)
    if action.tool_name == "toxicity":
        return "passes_toxicity" in failed
    if action.tool_name == "scscore":
        return "passes_synthesizability" in failed
    if action.tool_name == "vina":
        return "has_docking_scores" in failed
    if action.action_type in {"fallback_tool", "retry_with_reduced_generation_count"}:
        return bool(failed & {"has_tool_success", "has_valid_smiles", "has_unique_molecules"})
    return False


def _action_signature(action: RepairAction) -> str:
    return json.dumps(action.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _executed_action_records(
    actions: list[RepairAction],
    repair_calls: list[ToolCallRecord],
    *,
    policy: str,
    repair_round: int,
    initial_workflow: PlannedWorkflow,
    required_failed_checks: list[str],
) -> list[dict[str, Any]]:
    executed_tools = {
        record.tool_name
        for record in repair_calls
        if not (record.metadata or {}).get("skipped")
    }
    return [
        _action_audit_record(
            action,
            policy=policy,
            repair_round=repair_round,
            initial_workflow=initial_workflow,
            required_failed_checks=required_failed_checks,
            status="executed",
        )
        for action in actions
        if action.tool_name in executed_tools
    ]


def _uses_deferred_declared_fallback(case: BenchmarkCase) -> bool:
    return case.metadata.get("defer_declared_fallback_until_repair") is True


def _prepare_controlled_workflow(case: BenchmarkCase, workflow: PlannedWorkflow) -> PlannedWorkflow:
    if not _uses_deferred_declared_fallback(case):
        return workflow
    steps: list[PlannedToolCall] = []
    for step in workflow.tool_sequence:
        cloned = _clone_step(step)
        if cloned.parameters.get("execute_if") and cloned.parameters.get("fallback_for"):
            cloned.parameters["defer_until_repair"] = True
        steps.append(cloned)
    return PlannedWorkflow(
        task_id=workflow.task_id,
        planner_type=workflow.planner_type,
        selected_tools=list(workflow.selected_tools),
        tool_sequence=steps,
        expected_outputs=list(workflow.expected_outputs),
        notes=[
            *workflow.notes,
            "Controlled benchmark defers declared fallback until the policy decision phase.",
        ],
    )


def _clone_workflow(workflow: PlannedWorkflow) -> PlannedWorkflow:
    return PlannedWorkflow(
        task_id=workflow.task_id,
        planner_type=workflow.planner_type,
        selected_tools=list(workflow.selected_tools),
        tool_sequence=[_clone_step(step) for step in workflow.tool_sequence],
        expected_outputs=list(workflow.expected_outputs),
        notes=list(workflow.notes),
    )


def _workflow_signature(workflow: PlannedWorkflow) -> str:
    signature_payload = {
        "selected_tools": list(workflow.selected_tools),
        "tool_sequence": [step.to_dict() for step in workflow.tool_sequence],
        "expected_outputs": list(workflow.expected_outputs),
    }
    encoded = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _declared_fallbacks(workflow: PlannedWorkflow) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": step.tool_name,
            "fallback_for": step.parameters.get("fallback_for"),
            "execute_if": step.parameters.get("execute_if"),
            "deferred": step.parameters.get("defer_until_repair") is True,
        }
        for step in workflow.tool_sequence
        if step.parameters.get("fallback_for") and step.parameters.get("execute_if")
    ]


def _restrict_repair_plan_to_initial_workflow(
    repair_plan: RepairPlan,
    initial_workflow: PlannedWorkflow,
) -> RepairPlan:
    declared_fallback_tools = {item["tool_name"] for item in _declared_fallbacks(initial_workflow)}
    initial_tools = set(initial_workflow.selected_tools)
    actions = []
    for action in repair_plan.actions:
        if action.action_type == "fallback_tool":
            if action.tool_name in declared_fallback_tools:
                actions.append(action)
            continue
        if action.action_type in _RETRY_ACTION_TYPES:
            if action.tool_name in initial_tools and action.tool_name not in declared_fallback_tools:
                actions.append(action)
            continue
        actions.append(action)
    return _repair_plan_with_actions(repair_plan, actions)


def _scheduled_failed_or_empty_repair(
    repair_plan: RepairPlan,
    initial_workflow: PlannedWorkflow,
    initial_tool_calls: list[ToolCallRecord],
    candidates: list[CandidateRecord],
) -> RepairPlan:
    eligible_fallbacks: set[str] = set()
    for declaration in _declared_fallbacks(initial_workflow):
        upstream = str(declaration["fallback_for"])
        upstream_calls = [
            record
            for record in initial_tool_calls
            if record.tool_name == upstream and not (record.metadata or {}).get("skipped")
        ]
        upstream_failed = bool(upstream_calls and not upstream_calls[-1].success)
        upstream_empty = bool(upstream_calls and not candidates)
        if upstream_failed or upstream_empty:
            eligible_fallbacks.add(str(declaration["tool_name"]))
    actions = [
        action
        for action in repair_plan.actions
        if action.action_type == "fallback_tool" and action.tool_name in eligible_fallbacks
    ]
    return _repair_plan_with_actions(repair_plan, actions)


def _repair_plan_with_actions(repair_plan: RepairPlan, actions: list) -> RepairPlan:
    should_retry = any(action.action_type in _RETRY_ACTION_TYPES for action in actions)
    repaired_workflow = None
    if should_retry and repair_plan.repaired_workflow:
        authorized_steps: list[PlannedToolCall] = []
        for action in actions:
            if action.action_type not in _RETRY_ACTION_TYPES or not action.tool_name:
                continue
            for step in repair_plan.repaired_workflow.tool_sequence:
                if step.tool_name != action.tool_name:
                    continue
                if not all(step.parameters.get(key) == value for key, value in action.parameters.items()):
                    continue
                authorized_steps.append(_clone_step(step))
                break
        repaired_workflow = PlannedWorkflow(
            task_id=repair_plan.repaired_workflow.task_id,
            planner_type=repair_plan.repaired_workflow.planner_type,
            selected_tools=_unique([step.tool_name for step in authorized_steps]),
            tool_sequence=authorized_steps,
            expected_outputs=list(repair_plan.repaired_workflow.expected_outputs),
            notes=list(repair_plan.repaired_workflow.notes),
        )
    return RepairPlan(
        task_id=repair_plan.task_id,
        should_retry=should_retry,
        actions=list(actions),
        repaired_workflow=repaired_workflow,
        failure_reason=repair_plan.failure_reason,
    )


def _backend_call_count(tool_calls: list[ToolCallRecord]) -> int:
    return sum(1 for record in tool_calls if not (record.metadata or {}).get("skipped"))


def _fallback_authorization_source(
    planner_baseline: str,
    fallback_executed: bool,
    case: BenchmarkCase,
) -> str | None:
    if not fallback_executed:
        return None
    if _uses_deferred_declared_fallback(case):
        if _is_egvr_agent(planner_baseline):
            return "verifier_failure_reason_and_declared_plan"
        if planner_baseline == SCHEDULED_FALLBACK_NO_VERIFIER:
            return "predeclared_failed_or_empty_trigger"
    return "legacy_repair_policy"


def _fallback_executed(
    repair_plan: RepairPlan | None,
    repair_executed: bool,
    repair_calls: list[ToolCallRecord],
) -> bool:
    if not repair_executed or not repair_plan:
        return False
    fallback_tools = {
        action.tool_name
        for action in repair_plan.actions
        if action.action_type == "fallback_tool" and action.tool_name
    }
    return any(
        record.tool_name in fallback_tools and not (record.metadata or {}).get("skipped")
        for record in repair_calls
    )


def _is_egvr_agent(planner_baseline: str) -> bool:
    return planner_baseline in {EGVR_AGENT, FULL_COPILOT, LEGACY_FULL_COPILOT}


def _execute_scheduled_repair(
    parsed_task: ParsedTask,
    workflow: PlannedWorkflow,
    executor: WorkflowExecutor,
    tool_calls: list[ToolCallRecord],
    candidates: list[CandidateRecord],
    repair_plan: RepairPlan,
) -> tuple[PlannedWorkflow, list[ToolCallRecord], list[CandidateRecord], VerifierResult, bool]:
    repair_workflow = _repair_execution_workflow(parsed_task, workflow, repair_plan)
    if not repair_workflow.tool_sequence:
        verifier_result = verify_workflow(parsed_task, workflow, tool_calls, candidates)
        return workflow, tool_calls, candidates, verifier_result, False

    repair_calls, repaired_candidates = executor.execute(
        parsed_task,
        repair_workflow,
        initial_candidates=candidates,
    )
    combined_calls = [*tool_calls, *repair_calls]
    final_candidates = rank_candidates(repaired_candidates) if parsed_task.constraints.require_ranking else repaired_candidates
    combined_workflow = _combine_workflows(
        parsed_task,
        workflow,
        repair_workflow,
        planner_type=SCHEDULED_FALLBACK_NO_VERIFIER,
        note="Scheduled fallback executed one repair workflow without verifier-conditioned repair selection.",
    )
    verifier_result = verify_workflow(parsed_task, combined_workflow, combined_calls, final_candidates)
    return combined_workflow, combined_calls, final_candidates, verifier_result, True


def _repair_execution_workflow(
    parsed_task: ParsedTask,
    workflow: PlannedWorkflow,
    repair_plan: RepairPlan,
) -> PlannedWorkflow:
    if not repair_plan.repaired_workflow:
        return PlannedWorkflow(task_id=parsed_task.task_id, planner_type="verifier_guided_repair")

    retry_steps = _repair_retry_steps(repair_plan)
    if not retry_steps:
        return PlannedWorkflow(task_id=parsed_task.task_id, planner_type="verifier_guided_repair")

    retry_tool_names = {step.tool_name for step in retry_steps}
    evaluator_steps = [
        _clone_evaluator_step(step)
        for step in workflow.tool_sequence
        if step.tool_name in _EVALUATOR_TOOLS and step.tool_name not in retry_tool_names
    ]
    return PlannedWorkflow(
        task_id=parsed_task.task_id,
        planner_type="verifier_guided_repair",
        tool_sequence=[*retry_steps, *evaluator_steps],
        expected_outputs=workflow.expected_outputs,
        notes=["Execution-guided repair reruns fallback/retry generation before downstream evaluators."],
    )


def _repair_retry_steps(repair_plan: RepairPlan) -> list[PlannedToolCall]:
    retry_steps: list[PlannedToolCall] = []
    repaired_steps = repair_plan.repaired_workflow.tool_sequence if repair_plan.repaired_workflow else []
    for action in repair_plan.actions:
        if action.action_type not in _RETRY_ACTION_TYPES or not action.tool_name:
            continue
        for step in repaired_steps:
            if step.tool_name == action.tool_name and all(step.parameters.get(key) == value for key, value in action.parameters.items()):
                retry_steps.append(_clone_step(step))
                break
    return retry_steps


def _combine_workflows(
    parsed_task: ParsedTask,
    workflow: PlannedWorkflow,
    repair_workflow: PlannedWorkflow,
    *,
    planner_type: str = EGVR_AGENT,
    note: str = "EGVR-Agent executed one repair workflow after initial verification failure.",
) -> PlannedWorkflow:
    steps = [*workflow.tool_sequence, *repair_workflow.tool_sequence]
    selected_tools = _unique([step.tool_name for step in steps])
    return PlannedWorkflow(
        task_id=parsed_task.task_id,
        planner_type=planner_type,
        selected_tools=selected_tools,
        tool_sequence=steps,
        expected_outputs=_unique([*workflow.expected_outputs, *repair_workflow.expected_outputs]),
        notes=[
            *workflow.notes,
            *repair_workflow.notes,
            note,
        ],
    )


def _clone_evaluator_step(step: PlannedToolCall) -> PlannedToolCall:
    cloned = _clone_step(step)
    cloned.parameters["input_source"] = "generated_smiles"
    if cloned.required_inputs:
        cloned.required_inputs = ["generated_smiles" if value == "input_smiles" else value for value in cloned.required_inputs]
    return cloned


def _clone_step(step: PlannedToolCall) -> PlannedToolCall:
    return PlannedToolCall(
        tool_name=step.tool_name,
        reason=step.reason,
        action=step.action,
        expected_outputs=list(step.expected_outputs),
        required_inputs=list(step.required_inputs),
        optional_inputs=list(step.optional_inputs),
        parameters=dict(step.parameters),
    )


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _tool_functions_for_case(case: BenchmarkCase, execution_mode: str):
    if execution_mode == "mock":
        functions = _mock_tool_functions(case)
    else:
        functions = {}
    injections = _failure_injections(case)
    if injections:
        return _with_failure_injections(functions, injections)
    return functions or None


def _mock_tool_functions(case: BenchmarkCase):
    outputs = {tool: _copy_mock_output(output) for tool, output in case.mock_outputs.items()}

    def _default_generation(**kwargs):
        return {"success": True, "molecules_smiles": ["CCO", "CCC"]}

    def _default_scscore(**kwargs):
        smiles_list = kwargs.get("smiles_list") or ["CCO", "CCC"]
        return {
            "success": True,
            "results": [
                {"input_smiles": smiles, "canonical_smiles": smiles, "scscore": 2.0 + idx * 0.1}
                for idx, smiles in enumerate(smiles_list)
            ],
        }

    def _default_toxicity(**kwargs):
        return {"success": True, "smiles": kwargs.get("smiles"), "toxicity_probability": 0.1, "verdict": "Non-Toxic"}

    def _default_pmic(**kwargs):
        return {"success": True, "smiles": kwargs.get("smiles"), "pMIC_value": 5.5, "estimated_MIC_uM": 3.2}

    def _default_vina(**kwargs):
        return {"success": True, "best_docking_score_kcal_mol": -7.5}

    defaults = {
        "rxnflow": _default_generation,
        "reinvent4_denovo": _default_generation,
        "reinvent4_mol2mol": _default_generation,
        "reinvent4_libinvent": _default_generation,
        "scaffold": _default_generation,
        "libinvent": _default_generation,
        "scscore": _default_scscore,
        "toxicity": _default_toxicity,
        "pmic": _default_pmic,
        "vina": _default_vina,
    }

    def _make(tool_name: str, default_func):
        def _func(**kwargs):
            output = outputs.get(tool_name)
            if isinstance(output, list):
                if output:
                    return dict(output.pop(0))
                return dict(default_func(**kwargs))
            return dict(output or default_func(**kwargs))

        return _func

    return {tool_name: _make(tool_name, func) for tool_name, func in defaults.items()}


def _with_failure_injections(base_functions: dict[str, Any], injections: dict[str, list[dict[str, Any]]]):
    injected_functions = dict(base_functions)
    real_executor = WorkflowExecutor()
    for tool_name, injection_plan in injections.items():
        base_func = injected_functions.get(tool_name)
        call_state = {"count": 0}

        def _make(tool=tool_name, plan=injection_plan, state=call_state, base=base_func):
            def _func(**kwargs):
                state["count"] += 1
                injected = _injected_output_for_call(tool, plan, state["count"])
                if injected is not None:
                    return injected
                if base is not None:
                    return base(**kwargs)
                return real_executor._resolve_function(tool)(**kwargs)

            return _func

        injected_functions[tool_name] = _make()
    return injected_functions


def _failure_injections(case: BenchmarkCase) -> dict[str, list[dict[str, Any]]]:
    raw = case.metadata.get("failure_injections") or case.metadata.get("failure_injection") or {}
    if not isinstance(raw, dict):
        return {}
    injections: dict[str, list[dict[str, Any]]] = {}
    for tool_name, value in raw.items():
        items = value if isinstance(value, list) else [value]
        clean_items: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            clean_item = dict(item)
            clean_item.setdefault("call_index", index)
            clean_items.append(clean_item)
        if clean_items:
            injections[str(tool_name)] = clean_items
    return injections


def _injected_output_for_call(tool_name: str, plan: list[dict[str, Any]], call_index: int) -> dict[str, Any] | None:
    for item in plan:
        if int(item.get("call_index", -1)) != call_index:
            continue
        if "output" in item and isinstance(item["output"], dict):
            return dict(item["output"])
        mode = str(item.get("mode", "error"))
        reason = str(item.get("error") or item.get("reason") or f"injected {mode} failure for {tool_name}")
        if mode == "exception":
            raise RuntimeError(reason)
        if mode in {"empty_generation", "empty"}:
            return {"success": True, "molecules_smiles": []}
        if mode == "error":
            return {"success": False, "error": reason, "injected_failure": True}
    return None


def _copy_mock_output(output: Any) -> Any:
    if isinstance(output, list):
        return [dict(item) for item in output]
    if isinstance(output, dict):
        return dict(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EGVR-Agent benchmark JSONL.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--execution-mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--planner-baseline", choices=SUPPORTED_BASELINES, default=RULE_BASED_PLANNER)
    parser.add_argument("--trace-log-dir", help="Optional trace log directory.")
    parser.add_argument("--repair-budget", type=int, help="Maximum verifier-guided repair rounds for EGVR-Agent.")
    args = parser.parse_args()
    result = BenchmarkRunner(
        execution_mode=args.execution_mode,
        planner_baseline=args.planner_baseline,
        trace_log_dir=args.trace_log_dir,
        repair_budget=args.repair_budget,
    ).run_file(
        args.benchmark,
        output_path=args.output,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
