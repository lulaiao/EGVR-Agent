"""Rule-based chemistry tool planner baseline."""

from __future__ import annotations

from typing import Any

from .task_schema import ParsedTask, PlannedToolCall, PlannedWorkflow
from .tool_registry import ChemistryToolRegistry, build_default_tool_registry


def plan_workflow(
    parsed_task: ParsedTask,
    *,
    registry: ChemistryToolRegistry | None = None,
) -> PlannedWorkflow:
    """Plan a workflow with the default deterministic rule planner."""

    return RuleBasedPlanner(registry=registry).plan(parsed_task)


class RuleBasedPlanner:
    """Deterministic baseline planner for chemistry-aware tool orchestration."""

    planner_type = "rule_based"

    def __init__(self, registry: ChemistryToolRegistry | None = None) -> None:
        self.registry = registry or build_default_tool_registry()

    def plan(self, parsed_task: ParsedTask) -> PlannedWorkflow:
        steps: list[PlannedToolCall] = []
        notes: list[str] = []
        added_tools: set[str] = set()

        def add_step(
            tool_name: str,
            reason: str,
            *,
            required_inputs: list[str] | None = None,
            expected_outputs: list[str] | None = None,
            parameters: dict[str, Any] | None = None,
        ) -> None:
            if tool_name in added_tools:
                return
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
            added_tools.add(tool_name)

        if parsed_task.task_type == "docking_evaluation":
            self._add_docking_step(parsed_task, add_step, notes, strict=True)
            return self._workflow(parsed_task, steps, notes, ["docking_score"])

        generation_added = self._add_generation_steps(parsed_task, add_step, notes)
        candidate_source = "generated_smiles" if generation_added else "input_smiles"
        self._add_evaluation_steps(parsed_task, add_step, notes, candidate_source=candidate_source)

        expected_outputs = self._expected_outputs(parsed_task, steps)
        return self._workflow(parsed_task, steps, notes, expected_outputs)

    def _add_generation_steps(self, parsed_task: ParsedTask, add_step, notes: list[str]) -> bool:
        task_type = parsed_task.task_type

        if task_type == "pocket_conditioned_generation" or _has_pocket_without_input(parsed_task):
            add_step(
                "rxnflow",
                "Protein path and pocket center support pocket-conditioned generation.",
                required_inputs=["protein_path", "pocket_center"],
                expected_outputs=["generated_smiles", "proxy_scores", "result_csv"],
                parameters=_generation_parameters(parsed_task, "rxnflow"),
            )
            add_step(
                "reinvent4_denovo",
                "Fallback generator if rxnflow fails or returns no valid candidates.",
                required_inputs=[],
                expected_outputs=["generated_smiles"],
                parameters={
                    **_generation_parameters(parsed_task, "reinvent4_denovo"),
                    "execute_if": "rxnflow_failed_or_empty",
                    "fallback_for": "rxnflow",
                },
            )
            return True

        if task_type == "hit_to_lead_optimization" and parsed_task.input_smiles:
            add_step(
                "reinvent4_mol2mol",
                "Input hit SMILES supports Mol2Mol analog generation.",
                required_inputs=["input_smiles"],
                expected_outputs=["generated_smiles"],
                parameters={
                    **_generation_parameters(parsed_task, "reinvent4_mol2mol"),
                    "input_smiles": parsed_task.input_smiles,
                },
            )
            return True

        if task_type == "scaffold_conditioned_generation":
            if parsed_task.input_smiles:
                tool_name = _scaffold_generation_tool(parsed_task)
                add_step(
                    tool_name,
                    "Scaffold-conditioned request with input scaffold SMILES.",
                    required_inputs=["input_smiles"],
                    expected_outputs=["generated_smiles"],
                    parameters={
                        **_generation_parameters(parsed_task, tool_name),
                        "input_smiles": parsed_task.input_smiles,
                    },
                )
            else:
                notes.append("Scaffold-conditioned generation requested but no input scaffold SMILES was parsed.")
            return bool(parsed_task.input_smiles)

        if task_type == "multi_objective_screening":
            return self._add_multi_objective_generation(parsed_task, add_step)

        if task_type == "failure_recovery":
            add_step(
                "reinvent4_denovo",
                "Conservative recovery path when the failed upstream tool is not known.",
                required_inputs=[],
                expected_outputs=["generated_smiles"],
                parameters=_generation_parameters(parsed_task, "reinvent4_denovo"),
            )
            return True

        if task_type == "de_novo_generation":
            add_step(
                "reinvent4_denovo",
                "De novo generation request without a pocket or input molecule.",
                required_inputs=[],
                expected_outputs=["generated_smiles"],
                parameters=_generation_parameters(parsed_task, "reinvent4_denovo"),
            )
            return True

        if parsed_task.input_smiles and _needs_evaluation_only(parsed_task):
            return False

        if task_type == "unknown":
            notes.append("No chemistry task type could be planned conservatively.")
        return False

    def _add_multi_objective_generation(self, parsed_task: ParsedTask, add_step) -> bool:
        if _has_pocket_without_input(parsed_task):
            add_step(
                "rxnflow",
                "Multi-objective pocket task starts with pocket-conditioned generation.",
                required_inputs=["protein_path", "pocket_center"],
                expected_outputs=["generated_smiles", "proxy_scores", "result_csv"],
                parameters=_generation_parameters(parsed_task, "rxnflow"),
            )
            add_step(
                "reinvent4_denovo",
                "Fallback generator if pocket-conditioned generation fails.",
                required_inputs=[],
                expected_outputs=["generated_smiles"],
                parameters={
                    **_generation_parameters(parsed_task, "reinvent4_denovo"),
                    "execute_if": "rxnflow_failed_or_empty",
                    "fallback_for": "rxnflow",
                },
            )
            return True
        if parsed_task.input_smiles and _has_attachment_point(parsed_task):
            tool_name = _scaffold_generation_tool(parsed_task)
            add_step(
                tool_name,
                "Multi-objective scaffold task starts with scaffold decoration.",
                required_inputs=["input_smiles"],
                expected_outputs=["generated_smiles"],
                parameters={
                    **_generation_parameters(parsed_task, tool_name),
                    "input_smiles": parsed_task.input_smiles,
                },
            )
            return True
        if parsed_task.input_smiles:
            add_step(
                "reinvent4_mol2mol",
                "Multi-objective optimization starts from the provided molecule set.",
                required_inputs=["input_smiles"],
                expected_outputs=["generated_smiles"],
                parameters={
                    **_generation_parameters(parsed_task, "reinvent4_mol2mol"),
                    "input_smiles": parsed_task.input_smiles,
                },
            )
            return True

        add_step(
            "reinvent4_denovo",
            "Multi-objective screening without seed inputs falls back to de novo generation.",
            required_inputs=[],
            expected_outputs=["generated_smiles"],
            parameters=_generation_parameters(parsed_task, "reinvent4_denovo"),
        )
        return True

    def _add_evaluation_steps(self, parsed_task: ParsedTask, add_step, notes: list[str], *, candidate_source: str) -> None:
        if _needs_docking(parsed_task):
            self._add_docking_step(parsed_task, add_step, notes, strict=False, candidate_source=candidate_source)
        if _needs_synthesizability(parsed_task):
            add_step(
                "scscore",
                "Synthesizability objective requires SCScore evaluation.",
                required_inputs=[candidate_source],
                expected_outputs=["scscore", "canonical_smiles"],
                parameters={"input_source": candidate_source},
            )
        if _needs_toxicity(parsed_task):
            add_step(
                "toxicity",
                "Toxicity or safety objective requires toxicity prediction.",
                required_inputs=[candidate_source],
                expected_outputs=["toxicity_score", "toxicity_verdict"],
                parameters={"input_source": candidate_source},
            )
        if "bioactivity" in parsed_task.objectives or parsed_task.constraints.min_pmic is not None:
            add_step(
                "pmic",
                "Bioactivity objective requests antibacterial pMIC prediction.",
                required_inputs=[candidate_source],
                expected_outputs=["pmic_score", "estimated_mic_um"],
                parameters={"input_source": candidate_source},
            )

    def _add_docking_step(
        self,
        parsed_task: ParsedTask,
        add_step,
        notes: list[str],
        *,
        strict: bool,
        candidate_source: str = "input_smiles",
    ) -> None:
        missing = _missing_docking_inputs(parsed_task, strict=strict)
        if missing:
            notes.append(f"Docking requested but missing inputs: {', '.join(missing)}.")
            if not strict:
                return
        required_inputs = ["protein_path", "pocket_center", "candidate_ligand_structures"]
        if strict:
            required_inputs = ["protein_path", "ligand_path", "pocket_center", "box_size"]
        add_step(
            "vina",
            "Docking objective requires AutoDock Vina evaluation.",
            required_inputs=required_inputs,
            expected_outputs=["docking_score", "docked_pose_path", "minimized_pose_path"],
            parameters={
                "protein_path": parsed_task.protein_path,
                "ligand_path": parsed_task.ligand_path,
                "pocket_center": parsed_task.pocket_center,
                "box_size": parsed_task.box_size,
                "input_source": candidate_source,
            },
        )

    def _expected_outputs(self, parsed_task: ParsedTask, steps: list[PlannedToolCall]) -> list[str]:
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

    def _workflow(
        self,
        parsed_task: ParsedTask,
        steps: list[PlannedToolCall],
        notes: list[str],
        expected_outputs: list[str],
    ) -> PlannedWorkflow:
        return PlannedWorkflow(
            task_id=parsed_task.task_id,
            planner_type=self.planner_type,
            tool_sequence=steps,
            expected_outputs=expected_outputs,
            notes=notes,
        )


def _generation_parameters(parsed_task: ParsedTask, tool_name: str) -> dict[str, Any]:
    count = parsed_task.constraints.max_candidates
    parameters: dict[str, Any] = {}
    if count is not None:
        if tool_name == "rxnflow":
            parameters["num_samples"] = count
        elif tool_name in {"reinvent4_denovo", "reinvent4_mol2mol", "reinvent4_libinvent"}:
            parameters["num_variants"] = count
        elif tool_name == "scaffold":
            parameters["num_analogs"] = count
        elif tool_name == "libinvent":
            parameters["num_decorations"] = count
        else:
            parameters["max_candidates"] = count
    if tool_name == "rxnflow":
        run_seed = _run_seed(parsed_task)
        if run_seed is not None:
            parameters["seed"] = run_seed
    return parameters


def _run_seed(parsed_task: ParsedTask) -> int | None:
    for key in ("run_seed", "rxnflow_seed", "random_seed", "seed"):
        value = parsed_task.metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _has_pocket_without_input(parsed_task: ParsedTask) -> bool:
    return bool(parsed_task.protein_path and parsed_task.pocket_center and not parsed_task.input_smiles)


def _has_attachment_point(parsed_task: ParsedTask) -> bool:
    return any("*" in smiles for smiles in parsed_task.input_smiles)


def _scaffold_generation_tool(parsed_task: ParsedTask) -> str:
    query = parsed_task.raw_user_query.lower()
    explicit_libinvent = "libinvent" in query or "lib-invent" in query or "reinvent4_libinvent" in query
    return "reinvent4_libinvent" if explicit_libinvent else "scaffold"


def _needs_docking(parsed_task: ParsedTask) -> bool:
    return parsed_task.constraints.require_docking or "binding" in parsed_task.objectives


def _needs_synthesizability(parsed_task: ParsedTask) -> bool:
    return parsed_task.constraints.require_synthesizability or "synthesizability" in parsed_task.objectives


def _needs_toxicity(parsed_task: ParsedTask) -> bool:
    return parsed_task.constraints.require_toxicity or "toxicity" in parsed_task.objectives


def _needs_evaluation_only(parsed_task: ParsedTask) -> bool:
    return _needs_docking(parsed_task) or _needs_synthesizability(parsed_task) or _needs_toxicity(parsed_task)


def _missing_docking_inputs(parsed_task: ParsedTask, *, strict: bool) -> list[str]:
    required = ["protein_path", "pocket_center"]
    if strict:
        required.extend(["ligand_path", "box_size"])
    return [field for field in required if not getattr(parsed_task, field)]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
