from __future__ import annotations

from egvr.rule_planner import RuleBasedPlanner, plan_workflow
from egvr.task_schema import ParsedTask, TaskConstraints


def _tool_names(workflow):
    return [step.tool_name for step in workflow.tool_sequence]


def test_planner_pocket_generation_adds_fallback_and_requested_evaluators():
    task = ParsedTask(
        task_id="task-pocket",
        raw_user_query="generate for pocket",
        task_type="pocket_conditioned_generation",
        objectives=["binding", "synthesizability", "toxicity"],
        constraints=TaskConstraints(
            max_candidates=12,
            require_docking=True,
            require_synthesizability=True,
            require_toxicity=True,
        ),
        protein_path="agent_workspace/1HVR.pdb",
        pocket_center=[1, 2, 3],
    )

    workflow = plan_workflow(task)

    assert _tool_names(workflow) == ["rxnflow", "reinvent4_denovo", "vina", "scscore", "toxicity"]
    assert workflow.tool_sequence[0].parameters["num_samples"] == 12
    assert workflow.tool_sequence[1].parameters["execute_if"] == "rxnflow_failed_or_empty"
    assert "evaluated_candidates" in workflow.expected_outputs


def test_planner_passes_rxnflow_seed_when_available():
    task = ParsedTask(
        task_id="task-pocket-seed",
        raw_user_query="generate for pocket seed=2",
        task_type="pocket_conditioned_generation",
        objectives=["synthesizability"],
        constraints=TaskConstraints(max_candidates=5, require_synthesizability=True),
        protein_path="agent_workspace/1HVR.pdb",
        pocket_center=[1, 2, 3],
        metadata={"run_seed": 2},
    )

    workflow = plan_workflow(task)

    assert workflow.tool_sequence[0].tool_name == "rxnflow"
    assert workflow.tool_sequence[0].parameters["num_samples"] == 5
    assert workflow.tool_sequence[0].parameters["seed"] == 2
    assert "seed" not in workflow.tool_sequence[1].parameters


def test_planner_hit_to_lead_uses_mol2mol_then_evaluators():
    task = ParsedTask(
        task_id="task-hit",
        raw_user_query="optimize CCO",
        task_type="hit_to_lead_optimization",
        objectives=["synthesizability", "toxicity"],
        constraints=TaskConstraints(max_candidates=25, require_synthesizability=True, require_toxicity=True),
        input_smiles=["CCO"],
    )

    workflow = RuleBasedPlanner().plan(task)

    assert _tool_names(workflow) == ["reinvent4_mol2mol", "scscore", "toxicity"]
    assert workflow.tool_sequence[0].action == "mol2mol"
    assert workflow.tool_sequence[0].parameters["num_variants"] == 25
    assert workflow.tool_sequence[1].required_inputs == ["generated_smiles"]


def test_planner_docking_evaluation_only_selects_vina():
    task = ParsedTask(
        task_id="task-dock",
        raw_user_query="dock ligand",
        task_type="docking_evaluation",
        objectives=["binding"],
        constraints=TaskConstraints(require_docking=True),
        protein_path="agent_workspace/1HVR.pdb",
        ligand_path="agent_workspace/ligands/lig_0.pdb",
        pocket_center=[1, 2, 3],
        box_size=[20, 20, 20],
    )

    workflow = plan_workflow(task)

    assert _tool_names(workflow) == ["vina"]
    assert workflow.expected_outputs == ["docking_score"]
    assert workflow.tool_sequence[0].required_inputs == ["protein_path", "ligand_path", "pocket_center", "box_size"]


def test_planner_scaffold_generation_uses_scaffold_tool_by_default():
    task = ParsedTask(
        task_id="task-scaffold",
        raw_user_query="decorate scaffold",
        task_type="scaffold_conditioned_generation",
        objectives=["synthesizability"],
        constraints=TaskConstraints(require_synthesizability=True),
        input_smiles=["c1cc([*])ccc1"],
    )

    workflow = plan_workflow(task)

    assert _tool_names(workflow) == ["scaffold", "scscore"]
    assert workflow.tool_sequence[0].parameters["input_smiles"] == ["c1cc([*])ccc1"]


def test_planner_scaffold_generation_keeps_explicit_libinvent_request():
    task = ParsedTask(
        task_id="task-scaffold-libinvent",
        raw_user_query="use LibInvent to decorate scaffold",
        task_type="scaffold_conditioned_generation",
        objectives=["synthesizability"],
        constraints=TaskConstraints(require_synthesizability=True),
        input_smiles=["c1cc([*])ccc1"],
    )

    workflow = plan_workflow(task)

    assert _tool_names(workflow) == ["reinvent4_libinvent", "scscore"]
    assert workflow.tool_sequence[0].action == "libinvent"


def test_planner_multi_objective_adds_ranking_outputs_and_pmic():
    task = ParsedTask(
        task_id="task-multi",
        raw_user_query="rank molecules",
        task_type="multi_objective_screening",
        objectives=["synthesizability", "toxicity", "bioactivity"],
        constraints=TaskConstraints(require_synthesizability=True, require_toxicity=True, require_ranking=True),
        input_smiles=["CCO", "CCC"],
    )

    workflow = plan_workflow(task)

    assert _tool_names(workflow) == ["reinvent4_mol2mol", "scscore", "toxicity", "pmic"]
    assert "ranked_candidates" in workflow.expected_outputs
    assert "pareto_candidates" in workflow.expected_outputs


def test_planner_unknown_task_returns_empty_plan_with_note():
    task = ParsedTask(task_id="task-unknown", raw_user_query="think about this", task_type="unknown")

    workflow = plan_workflow(task)

    assert workflow.selected_tools == []
    assert workflow.tool_sequence == []
    assert workflow.notes == ["No chemistry task type could be planned conservatively."]
