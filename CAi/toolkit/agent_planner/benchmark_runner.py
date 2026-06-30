"""Benchmark runner for chemistry-aware planner baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from .baseline_planners import (
    FULL_COPILOT,
    RULE_BASED_PLANNER,
    SCHEDULED_FALLBACK_NO_VERIFIER,
    SUPPORTED_BASELINES,
    plan_for_baseline,
)
from .executor import WorkflowExecutor
from .repair import RepairPlan, suggest_repair
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
    ) -> None:
        if execution_mode not in {"mock", "real"}:
            raise ValueError("execution_mode must be 'mock' or 'real'")
        if planner_baseline not in SUPPORTED_BASELINES:
            raise ValueError(f"planner_baseline must be one of: {', '.join(SUPPORTED_BASELINES)}")
        self.execution_mode = execution_mode
        self.planner_baseline = planner_baseline
        self.trace_logger = JSONLTraceLogger(trace_log_dir) if trace_log_dir else None

    def run_file(self, benchmark_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
        cases = load_benchmark_cases(benchmark_path)
        results = [self.run_case(case) for case in cases]
        summary = summarize_results(results)
        payload = {"planner_baseline": self.planner_baseline, "summary": summary, "results": results}
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def run_case(self, case: BenchmarkCase) -> dict[str, Any]:
        parsed_task = parse_task(case.raw_user_query, task_id=case.task_id, metadata=case.metadata)
        workflow = plan_for_baseline(parsed_task, self.planner_baseline)
        executor = WorkflowExecutor(tool_functions=_tool_functions_for_case(case, self.execution_mode))
        tool_calls, candidates = executor.execute(parsed_task, workflow)
        candidates = rank_candidates(candidates) if parsed_task.constraints.require_ranking else candidates
        verifier_result = verify_workflow(parsed_task, workflow, tool_calls, candidates)
        repair_plan = None
        repair_executed = False
        repair_success = False
        if not verifier_result.success or any(not record.success for record in tool_calls):
            repair_plan = suggest_repair(parsed_task, workflow, tool_calls, candidates, verifier_result)
            if self.planner_baseline == FULL_COPILOT and repair_plan.should_retry:
                workflow, tool_calls, candidates, verifier_result, repair_executed = _execute_full_copilot_repair(
                    parsed_task,
                    workflow,
                    executor,
                    tool_calls,
                    candidates,
                    repair_plan,
                )
                repair_success = repair_executed and verifier_result.success
            elif self.planner_baseline == SCHEDULED_FALLBACK_NO_VERIFIER:
                scheduled_repair = suggest_repair(parsed_task, workflow, tool_calls, candidates, None)
                if scheduled_repair.should_retry:
                    workflow, tool_calls, candidates, verifier_result, repair_executed = _execute_scheduled_repair(
                        parsed_task,
                        workflow,
                        executor,
                        tool_calls,
                        candidates,
                        scheduled_repair,
                    )
                    repair_plan = scheduled_repair
                    repair_success = repair_executed and verifier_result.success
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
    false_success_count = sum(1 for result in results if result["task_success"] and not result.get("expected_success"))
    repair_attempt_count = sum(1 for result in results if result["repair_executed"])
    repair_success_count = sum(1 for result in results if result["repair_success"])
    planner_precision = sum(result["planner_tool_precision"] for result in results)
    planner_recall = sum(result["planner_tool_recall"] for result in results)
    planner_f1 = sum(result["planner_tool_f1"] for result in results)
    selected_tool_count = sum(result.get("selected_tool_count", 0) for result in results)
    tool_sequence_length = sum(result.get("tool_sequence_length", 0) for result in results)
    extra_tool_count = sum(result.get("extra_tool_count", 0) for result in results)
    tool_call_count = sum(result.get("tool_call_count", 0) for result in results)
    failed_tool_call_count = sum(result.get("failed_tool_call_count", 0) for result in results)
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
        "mean_selected_tool_count": selected_tool_count / total if total else 0.0,
        "mean_tool_sequence_length": tool_sequence_length / total if total else 0.0,
        "mean_extra_tool_count": extra_tool_count / total if total else 0.0,
        "mean_tool_call_count": tool_call_count / total if total else 0.0,
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
) -> dict[str, Any]:
    parser_correct = case.expected_task_type in {None, parsed_task.task_type}
    planner_correct = set(case.expected_tools).issubset(set(workflow.selected_tools))
    tool_scores = _tool_set_scores(case.expected_tools, workflow.selected_tools)
    verifier_matched = verifier_result.success is case.should_succeed
    total_elapsed_sec = _total_elapsed_sec(tool_calls)
    return {
        "task_id": case.task_id,
        "parsed_task_type": parsed_task.task_type,
        "expected_task_type": case.expected_task_type,
        "parser_correct": parser_correct,
        "planner_type": workflow.planner_type,
        "selected_tools": workflow.selected_tools,
        "selected_tool_count": len(workflow.selected_tools),
        "tool_sequence_length": len(workflow.tool_sequence),
        "tool_call_count": len(tool_calls),
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
        "expected_success": case.should_succeed,
        "verifier_matched_expectation": verifier_matched,
        "failure_reason": verifier_result.failure_reason,
        "repair_plan": repair_plan.to_dict() if repair_plan else None,
        "repair_executed": repair_executed,
        "repair_success": repair_success,
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


def _execute_full_copilot_repair(
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
    combined_workflow = _combine_workflows(parsed_task, workflow, repair_workflow)
    verifier_result = verify_workflow(parsed_task, combined_workflow, combined_calls, final_candidates)
    return combined_workflow, combined_calls, final_candidates, verifier_result, True


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
        return PlannedWorkflow(task_id=parsed_task.task_id, planner_type="full_copilot_repair")

    retry_steps = _repair_retry_steps(repair_plan)
    if not retry_steps:
        return PlannedWorkflow(task_id=parsed_task.task_id, planner_type="full_copilot_repair")

    retry_tool_names = {step.tool_name for step in retry_steps}
    evaluator_steps = [
        _clone_evaluator_step(step)
        for step in workflow.tool_sequence
        if step.tool_name in _EVALUATOR_TOOLS and step.tool_name not in retry_tool_names
    ]
    return PlannedWorkflow(
        task_id=parsed_task.task_id,
        planner_type="full_copilot_repair",
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
    planner_type: str = FULL_COPILOT,
    note: str = "Full copilot executed one repair workflow after initial verification failure.",
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
    parser = argparse.ArgumentParser(description="Run CAi agent_planner benchmark JSONL.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--execution-mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--planner-baseline", choices=SUPPORTED_BASELINES, default=RULE_BASED_PLANNER)
    parser.add_argument("--trace-log-dir", help="Optional trace log directory.")
    args = parser.parse_args()
    result = BenchmarkRunner(
        execution_mode=args.execution_mode,
        planner_baseline=args.planner_baseline,
        trace_log_dir=args.trace_log_dir,
    ).run_file(
        args.benchmark,
        output_path=args.output,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
