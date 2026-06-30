"""Baseline planner entry points for benchmark comparison."""

from __future__ import annotations

from typing import Any

from .rule_planner import plan_workflow
from .task_schema import ParsedTask, PlannedToolCall, PlannedWorkflow
from .tool_registry import ChemistryToolRegistry, build_default_tool_registry

RULE_BASED_PLANNER = "rule_based_planner"
FIXED_PIPELINE = "fixed_pipeline"
FULL_COPILOT = "full_copilot"
ALL_TOOL_AGENT = "all_tool_agent"
SCHEDULED_FALLBACK_NO_VERIFIER = "scheduled_fallback_no_verifier"
VERIFIER_ONLY_NO_REPAIR = "verifier_only_no_repair"
SUPPORTED_BASELINES = (
    ALL_TOOL_AGENT,
    FIXED_PIPELINE,
    RULE_BASED_PLANNER,
    FULL_COPILOT,
    SCHEDULED_FALLBACK_NO_VERIFIER,
    VERIFIER_ONLY_NO_REPAIR,
)


def plan_for_baseline(
    parsed_task: ParsedTask,
    baseline: str = RULE_BASED_PLANNER,
    *,
    registry: ChemistryToolRegistry | None = None,
) -> PlannedWorkflow:
    """Return a structured workflow for a named benchmark baseline."""

    if baseline == RULE_BASED_PLANNER:
        return plan_workflow(parsed_task, registry=registry)
    if baseline == FULL_COPILOT:
        base_workflow = plan_workflow(parsed_task, registry=registry)
        return PlannedWorkflow(
            task_id=base_workflow.task_id,
            planner_type=FULL_COPILOT,
            selected_tools=base_workflow.selected_tools,
            tool_sequence=base_workflow.tool_sequence,
            expected_outputs=base_workflow.expected_outputs,
            notes=[
                *base_workflow.notes,
                "Full copilot baseline enables one execution-guided repair attempt after verification failure.",
            ],
        )
    if baseline == SCHEDULED_FALLBACK_NO_VERIFIER:
        base_workflow = plan_workflow(parsed_task, registry=registry)
        return PlannedWorkflow(
            task_id=base_workflow.task_id,
            planner_type=SCHEDULED_FALLBACK_NO_VERIFIER,
            selected_tools=base_workflow.selected_tools,
            tool_sequence=base_workflow.tool_sequence,
            expected_outputs=base_workflow.expected_outputs,
            notes=[
                *base_workflow.notes,
                "Scheduled fallback baseline may retry/fallback from execution failure without using verifier failure reasons.",
            ],
        )
    if baseline == VERIFIER_ONLY_NO_REPAIR:
        base_workflow = plan_workflow(parsed_task, registry=registry)
        return PlannedWorkflow(
            task_id=base_workflow.task_id,
            planner_type=VERIFIER_ONLY_NO_REPAIR,
            selected_tools=base_workflow.selected_tools,
            tool_sequence=base_workflow.tool_sequence,
            expected_outputs=base_workflow.expected_outputs,
            notes=[
                *base_workflow.notes,
                "Verifier-only baseline records failed evidence but does not execute repair.",
            ],
        )
    if baseline == ALL_TOOL_AGENT:
        return AllToolAgentPlanner(registry=registry).plan(parsed_task)
    if baseline == FIXED_PIPELINE:
        return FixedPipelinePlanner(registry=registry).plan(parsed_task)
    raise ValueError(f"Unsupported planner baseline: {baseline}")


class AllToolAgentPlanner:
    """Static all-tools exposure baseline with safe deterministic execution.

    This models the common "give the agent every tool" baseline at the planning
    interface while reusing the rule workflow for execution, so real smoke tests
    do not trigger unrelated chemistry backends without required inputs.
    """

    planner_type = ALL_TOOL_AGENT

    def __init__(self, registry: ChemistryToolRegistry | None = None) -> None:
        self.registry = registry or build_default_tool_registry()

    def plan(self, parsed_task: ParsedTask) -> PlannedWorkflow:
        base_workflow = plan_workflow(parsed_task, registry=self.registry)
        return PlannedWorkflow(
            task_id=base_workflow.task_id,
            planner_type=self.planner_type,
            selected_tools=self.registry.names(),
            tool_sequence=base_workflow.tool_sequence,
            expected_outputs=base_workflow.expected_outputs,
            notes=[
                *base_workflow.notes,
                "All registered chemistry tools are exposed to the agent planning context.",
                "Execution reuses the deterministic rule workflow to keep real benchmark calls safe and reproducible.",
            ],
        )


class FixedPipelinePlanner:
    """Task-family fixed pipeline baseline with no execution-guided fallback step."""

    planner_type = FIXED_PIPELINE

    def __init__(self, registry: ChemistryToolRegistry | None = None) -> None:
        self.registry = registry or build_default_tool_registry()

    def plan(self, parsed_task: ParsedTask) -> PlannedWorkflow:
        steps: list[PlannedToolCall] = []
        notes: list[str] = []

        generation_tool = self._generation_tool(parsed_task)
        candidate_source = "generated_smiles" if generation_tool else "input_smiles"

        if parsed_task.task_type == "docking_evaluation":
            self._add_step(
                steps,
                "vina",
                "Fixed docking pipeline evaluates the provided receptor-ligand pair.",
                required_inputs=["protein_path", "ligand_path", "pocket_center", "box_size"],
                expected_outputs=["docking_score", "docked_pose_path", "minimized_pose_path"],
                parameters={
                    "protein_path": parsed_task.protein_path,
                    "ligand_path": parsed_task.ligand_path,
                    "pocket_center": parsed_task.pocket_center,
                    "box_size": parsed_task.box_size,
                    "input_source": candidate_source,
                },
            )
        elif generation_tool:
            self._add_generation_step(steps, parsed_task, generation_tool)
            self._add_fixed_evaluators(steps, candidate_source=candidate_source)
        elif parsed_task.input_smiles:
            self._add_fixed_evaluators(steps, candidate_source=candidate_source)
        else:
            notes.append("Fixed pipeline could not select a safe tool sequence for this task.")

        return PlannedWorkflow(
            task_id=parsed_task.task_id,
            planner_type=self.planner_type,
            tool_sequence=steps,
            expected_outputs=_expected_outputs(parsed_task, steps),
            notes=notes,
        )

    def _generation_tool(self, parsed_task: ParsedTask) -> str | None:
        if parsed_task.task_type == "pocket_conditioned_generation" or _has_pocket_without_input(parsed_task):
            return "rxnflow"
        if parsed_task.task_type == "hit_to_lead_optimization" and parsed_task.input_smiles:
            return "reinvent4_mol2mol"
        if parsed_task.task_type == "scaffold_conditioned_generation" and parsed_task.input_smiles:
            return _scaffold_generation_tool(parsed_task)
        if parsed_task.task_type == "multi_objective_screening":
            if _has_pocket_without_input(parsed_task):
                return "rxnflow"
            if parsed_task.input_smiles:
                return "reinvent4_mol2mol"
            return "reinvent4_denovo"
        if parsed_task.task_type in {"de_novo_generation", "failure_recovery"}:
            return "reinvent4_denovo"
        return None

    def _add_generation_step(
        self,
        steps: list[PlannedToolCall],
        parsed_task: ParsedTask,
        tool_name: str,
    ) -> None:
        required_inputs = {
            "rxnflow": ["protein_path", "pocket_center"],
            "reinvent4_mol2mol": ["input_smiles"],
            "reinvent4_libinvent": ["input_smiles"],
            "scaffold": ["input_smiles"],
            "reinvent4_denovo": [],
        }.get(tool_name, [])
        parameters = _generation_parameters(parsed_task, tool_name)
        if tool_name in {"reinvent4_mol2mol", "reinvent4_libinvent", "scaffold"}:
            parameters["input_smiles"] = parsed_task.input_smiles
        self._add_step(
            steps,
            tool_name,
            "Fixed generation pipeline starts with the task-family default generator.",
            required_inputs=required_inputs,
            expected_outputs=["generated_smiles", "proxy_scores", "result_csv"]
            if tool_name == "rxnflow"
            else ["generated_smiles"],
            parameters=parameters,
        )

    def _add_fixed_evaluators(self, steps: list[PlannedToolCall], *, candidate_source: str) -> None:
        self._add_step(
            steps,
            "scscore",
            "Fixed evaluation pipeline always scores synthesizability.",
            required_inputs=[candidate_source],
            expected_outputs=["scscore", "canonical_smiles"],
            parameters={"input_source": candidate_source},
        )
        self._add_step(
            steps,
            "toxicity",
            "Fixed evaluation pipeline always scores toxicity.",
            required_inputs=[candidate_source],
            expected_outputs=["toxicity_score", "toxicity_verdict"],
            parameters={"input_source": candidate_source},
        )

    def _add_step(
        self,
        steps: list[PlannedToolCall],
        tool_name: str,
        reason: str,
        *,
        required_inputs: list[str] | None = None,
        expected_outputs: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        tool = self.registry.require(tool_name)
        steps.append(
            PlannedToolCall(
                tool_name=tool_name,
                reason=reason,
                action=tool.backend_action,
                expected_outputs=expected_outputs or tool.outputs,
                required_inputs=required_inputs or tool.required_inputs,
                optional_inputs=tool.optional_inputs,
                parameters=parameters or {},
            )
        )


def _generation_parameters(parsed_task: ParsedTask, tool_name: str) -> dict[str, Any]:
    count = parsed_task.constraints.max_candidates
    if count is None:
        return {}
    if tool_name == "rxnflow":
        return {"num_samples": count}
    if tool_name in {"reinvent4_denovo", "reinvent4_mol2mol", "reinvent4_libinvent"}:
        return {"num_variants": count}
    if tool_name == "scaffold":
        return {"num_analogs": count}
    return {"max_candidates": count}


def _expected_outputs(parsed_task: ParsedTask, steps: list[PlannedToolCall]) -> list[str]:
    outputs: list[str] = []
    if any("generated_smiles" in step.expected_outputs for step in steps):
        outputs.extend(["generated_smiles", "candidate_records"])
    if any(step.tool_name in {"vina", "scscore", "toxicity", "pmic"} for step in steps):
        outputs.append("evaluated_candidates")
    if parsed_task.constraints.require_ranking or parsed_task.task_type == "multi_objective_screening":
        outputs.extend(["ranked_candidates", "pareto_candidates"])
    if not outputs and parsed_task.task_type == "docking_evaluation":
        outputs.append("docking_score")
    return _unique(outputs)


def _has_pocket_without_input(parsed_task: ParsedTask) -> bool:
    return bool(parsed_task.protein_path and parsed_task.pocket_center and not parsed_task.input_smiles)


def _has_attachment_point(parsed_task: ParsedTask) -> bool:
    return any("*" in smiles for smiles in parsed_task.input_smiles)


def _scaffold_generation_tool(parsed_task: ParsedTask) -> str:
    query = parsed_task.raw_user_query.lower()
    explicit_libinvent = "libinvent" in query or "lib-invent" in query or "reinvent4_libinvent" in query
    return "reinvent4_libinvent" if explicit_libinvent else "scaffold"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
