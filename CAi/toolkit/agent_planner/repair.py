"""Simple repair and fallback suggestions for failed planned workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from .task_schema import CandidateRecord, ParsedTask, PlannedToolCall, PlannedWorkflow, ToolCallRecord, VerifierResult
from .tool_registry import ChemistryToolRegistry, build_default_tool_registry


@dataclass
class RepairAction:
    """One conservative repair action proposed after execution."""

    action_type: str
    reason: str
    tool_name: str | None = None
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "parameters": dict(self.parameters),
        }


@dataclass
class RepairPlan:
    """Structured repair result that can be logged or benchmarked."""

    task_id: str
    should_retry: bool
    actions: list[RepairAction] = field(default_factory=list)
    repaired_workflow: PlannedWorkflow | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "should_retry": self.should_retry,
            "actions": [action.to_dict() for action in self.actions],
            "repaired_workflow": self.repaired_workflow.to_dict() if self.repaired_workflow else None,
            "failure_reason": self.failure_reason,
        }


class SimpleRepairPlanner:
    """Conservative first-stage fallback planner."""

    planner_type = "simple_repair"

    def __init__(self, registry: ChemistryToolRegistry | None = None) -> None:
        self.registry = registry or build_default_tool_registry()

    def suggest(
        self,
        parsed_task: ParsedTask,
        planned_workflow: PlannedWorkflow,
        tool_call_records: list[ToolCallRecord],
        candidate_records: list[CandidateRecord],
        verifier_result: VerifierResult | None = None,
    ) -> RepairPlan:
        failed_tools = [record.tool_name for record in tool_call_records if not record.success]
        actions: list[RepairAction] = []
        repaired_steps = list(planned_workflow.tool_sequence)

        has_valid_candidates = any(candidate.smiles and candidate.is_valid for candidate in candidate_records)
        generation_failed = bool(failed_tools and any(tool in _GENERATION_TOOLS for tool in failed_tools))
        all_generation_failed = _has_generation_step(planned_workflow) and not has_valid_candidates

        if "rxnflow" in failed_tools or ("rxnflow" in planned_workflow.selected_tools and all_generation_failed):
            action = self._fallback_denovo_action("rxnflow failed or returned no valid candidates.")
            actions.append(action)
            _append_step_if_missing(repaired_steps, self._action_to_step(action))

        if "reinvent4_mol2mol" in failed_tools:
            reduced = _reduced_num_variants(planned_workflow, "reinvent4_mol2mol")
            retry_action = RepairAction(
                action_type="retry_with_reduced_generation_count",
                tool_name="reinvent4_mol2mol",
                reason="Mol2Mol failed; retry once with a smaller generation count before falling back.",
                parameters={"num_variants": reduced},
            )
            fallback_action = self._fallback_denovo_action("Mol2Mol failed after seed-based optimization.")
            actions.extend([retry_action, fallback_action])
            _append_step_if_missing(repaired_steps, self._action_to_step(retry_action))
            _append_step_if_missing(repaired_steps, self._action_to_step(fallback_action))

        if "reinvent4_denovo" in failed_tools:
            reduced = _reduced_num_variants(planned_workflow, "reinvent4_denovo")
            retry_action = RepairAction(
                action_type="retry_with_reduced_generation_count",
                tool_name="reinvent4_denovo",
                reason="De novo generation failed; retry once with a smaller generation count.",
                parameters={"num_variants": reduced},
            )
            actions.append(retry_action)
            _append_step_if_missing(repaired_steps, self._action_to_step(retry_action))

        if (
            "reinvent4_denovo" in planned_workflow.selected_tools
            and all_generation_failed
            and "reinvent4_denovo" not in failed_tools
        ):
            reduced = _reduced_num_variants(planned_workflow, "reinvent4_denovo")
            retry_action = RepairAction(
                action_type="retry_with_reduced_generation_count",
                tool_name="reinvent4_denovo",
                reason="De novo generation returned no valid candidates; retry once with a smaller generation count.",
                parameters={"num_variants": reduced},
            )
            actions.append(retry_action)
            _append_step_if_missing(repaired_steps, self._action_to_step(retry_action))

        if "vina" in failed_tools:
            actions.append(
                RepairAction(
                    action_type="mark_missing_docking",
                    tool_name="vina",
                    reason="Vina failed; keep candidates but mark docking evidence as missing.",
                )
            )

        for evaluator in ("toxicity", "scscore"):
            if evaluator in failed_tools:
                actions.append(
                    RepairAction(
                        action_type="mark_incomplete_evaluation",
                        tool_name=evaluator,
                        reason=f"{evaluator} failed; workflow cannot be marked fully successful without this evidence.",
                    )
                )

        failed_checks = _failed_required_checks(verifier_result)
        if has_valid_candidates and "passes_synthesizability" in failed_checks and "scscore" not in failed_tools:
            action = self._retry_evaluator_action(
                "scscore",
                "Verifier found missing synthesizability evidence after a nominally successful run.",
            )
            actions.append(action)
            _append_step_if_missing(repaired_steps, self._action_to_step(action))

        if has_valid_candidates and "passes_toxicity" in failed_checks and "toxicity" not in failed_tools:
            action = self._retry_evaluator_action(
                "toxicity",
                "Verifier found missing toxicity evidence after a nominally successful run.",
            )
            actions.append(action)
            _append_step_if_missing(repaired_steps, self._action_to_step(action))

        if "has_docking_scores" in failed_checks and "vina" in planned_workflow.selected_tools and "vina" not in failed_tools:
            action = self._retry_evaluator_action(
                "vina",
                "Verifier found missing docking-score evidence after a nominally successful docking run.",
            )
            actions.append(action)
            _append_step_if_missing(repaired_steps, self._action_to_step(action))

        if all_generation_failed and not actions:
            actions.append(
                RepairAction(
                    action_type="return_failure",
                    reason="All generation steps failed or returned no candidates.",
                )
            )

        should_retry = any(action.action_type in _RETRY_ACTIONS for action in actions)
        repaired_workflow = None
        if should_retry:
            repaired_workflow = PlannedWorkflow(
                task_id=planned_workflow.task_id,
                planner_type=self.planner_type,
                tool_sequence=repaired_steps,
                expected_outputs=planned_workflow.expected_outputs,
                notes=[
                    *planned_workflow.notes,
                    "Repair plan appended conservative fallback/retry steps.",
                ],
            )
        return RepairPlan(
            task_id=parsed_task.task_id,
            should_retry=should_retry,
            actions=actions,
            repaired_workflow=repaired_workflow,
            failure_reason=verifier_result.failure_reason if verifier_result else None,
        )

    def _fallback_denovo_action(self, reason: str) -> RepairAction:
        return RepairAction(
            action_type="fallback_tool",
            tool_name="reinvent4_denovo",
            reason=reason,
            parameters={"fallback": True},
        )

    def _retry_evaluator_action(self, tool_name: str, reason: str) -> RepairAction:
        return RepairAction(
            action_type="retry_evaluator_for_missing_evidence",
            tool_name=tool_name,
            reason=reason,
            parameters={"repair_retry": True, "retry_reason": "missing_verifier_evidence"},
        )

    def _action_to_step(self, action: RepairAction) -> PlannedToolCall:
        if not action.tool_name:
            raise ValueError("Repair action has no tool_name")
        tool = self.registry.require(action.tool_name)
        return PlannedToolCall(
            tool_name=action.tool_name,
            reason=action.reason,
            action=tool.backend_action,
            expected_outputs=tool.outputs,
            required_inputs=tool.required_inputs,
            optional_inputs=tool.optional_inputs,
            parameters=action.parameters,
        )


def suggest_repair(
    parsed_task: ParsedTask,
    planned_workflow: PlannedWorkflow,
    tool_call_records: list[ToolCallRecord],
    candidate_records: list[CandidateRecord],
    verifier_result: VerifierResult | None = None,
    *,
    registry: ChemistryToolRegistry | None = None,
) -> RepairPlan:
    """Return a simple conservative repair plan."""

    return SimpleRepairPlanner(registry=registry).suggest(
        parsed_task,
        planned_workflow,
        tool_call_records,
        candidate_records,
        verifier_result,
    )


_GENERATION_TOOLS = {
    "rxnflow",
    "reinvent4_denovo",
    "reinvent4_mol2mol",
    "reinvent4_libinvent",
    "scaffold",
    "libinvent",
}
_RETRY_ACTIONS = {
    "fallback_tool",
    "retry_with_reduced_generation_count",
    "retry_evaluator_for_missing_evidence",
}


def _failed_required_checks(verifier_result: VerifierResult | None) -> set[str]:
    if not verifier_result or not verifier_result.failure_reason:
        return set()
    return {
        item.strip()
        for item in verifier_result.failure_reason.split(",")
        if item.strip()
    }


def _has_generation_step(workflow: PlannedWorkflow) -> bool:
    return any(step.tool_name in _GENERATION_TOOLS for step in workflow.tool_sequence)


def _append_step_if_missing(steps: list[PlannedToolCall], step: PlannedToolCall) -> None:
    if any(existing.tool_name == step.tool_name and existing.parameters == step.parameters for existing in steps):
        return
    steps.append(step)


def _reduced_num_variants(workflow: PlannedWorkflow, tool_name: str) -> int:
    for step in workflow.tool_sequence:
        if step.tool_name == tool_name:
            value = step.parameters.get("num_variants")
            if isinstance(value, int) and value > 1:
                return max(1, value // 2)
    return 10
