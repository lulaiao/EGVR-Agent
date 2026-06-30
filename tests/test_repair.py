from __future__ import annotations

from CAi.toolkit.agent_planner.repair import suggest_repair
from CAi.toolkit.agent_planner.rule_planner import plan_workflow
from CAi.toolkit.agent_planner.task_schema import CandidateRecord, ParsedTask, TaskConstraints, ToolCallRecord, VerifierResult


def test_repair_adds_denovo_fallback_after_rxnflow_failure():
    task = ParsedTask(
        task_id="task-pocket",
        raw_user_query="generate for pocket",
        task_type="pocket_conditioned_generation",
        protein_path="agent_workspace/1HVR.pdb",
        pocket_center=[1, 2, 3],
    )
    workflow = plan_workflow(task)
    calls = [ToolCallRecord(tool_name="rxnflow", success=False, error="boom")]

    repair_plan = suggest_repair(task, workflow, calls, [], VerifierResult(success=False, failure_reason="no candidates"))

    assert repair_plan.should_retry is True
    assert repair_plan.actions[0].tool_name == "reinvent4_denovo"
    assert repair_plan.repaired_workflow is not None


def test_repair_reduces_mol2mol_count_and_adds_fallback():
    task = ParsedTask(
        task_id="task-hit",
        raw_user_query="optimize",
        task_type="hit_to_lead_optimization",
        constraints=TaskConstraints(max_candidates=20),
        input_smiles=["CCO"],
    )
    workflow = plan_workflow(task)
    calls = [ToolCallRecord(tool_name="reinvent4_mol2mol", success=False, error="oom")]

    repair_plan = suggest_repair(task, workflow, calls, [], VerifierResult(success=False, failure_reason="generation failed"))

    assert [action.action_type for action in repair_plan.actions] == [
        "retry_with_reduced_generation_count",
        "fallback_tool",
    ]
    assert repair_plan.actions[0].parameters["num_variants"] == 10


def test_repair_marks_vina_failure_without_retry():
    task = ParsedTask(task_id="task-dock", raw_user_query="dock", task_type="docking_evaluation")
    workflow = plan_workflow(task)
    calls = [ToolCallRecord(tool_name="vina", success=False, error="missing ligand")]
    candidates = [CandidateRecord(smiles="CCO")]

    repair_plan = suggest_repair(task, workflow, calls, candidates)

    assert repair_plan.should_retry is False
    assert repair_plan.actions[0].action_type == "mark_missing_docking"


def test_repair_retries_failed_denovo_generation():
    task = ParsedTask(
        task_id="task-denovo",
        raw_user_query="generate de novo",
        task_type="de_novo_generation",
        constraints=TaskConstraints(max_candidates=8),
    )
    workflow = plan_workflow(task)
    calls = [ToolCallRecord(tool_name="reinvent4_denovo", success=False, error="empty output")]

    repair_plan = suggest_repair(task, workflow, calls, [], VerifierResult(success=False, failure_reason="no candidates"))

    assert repair_plan.should_retry is True
    assert repair_plan.actions[0].tool_name == "reinvent4_denovo"
    assert repair_plan.actions[0].parameters["num_variants"] == 4


def test_repair_retries_empty_denovo_generation():
    task = ParsedTask(
        task_id="task-denovo-empty",
        raw_user_query="generate de novo",
        task_type="de_novo_generation",
        constraints=TaskConstraints(max_candidates=6),
    )
    workflow = plan_workflow(task)
    calls = [ToolCallRecord(tool_name="reinvent4_denovo", success=True, outputs={"success": True, "molecules_smiles": []})]

    repair_plan = suggest_repair(task, workflow, calls, [], VerifierResult(success=False, failure_reason="no candidates"))

    assert repair_plan.should_retry is True
    assert repair_plan.actions[0].tool_name == "reinvent4_denovo"
    assert repair_plan.actions[0].parameters["num_variants"] == 3


def test_repair_retries_missing_scscore_evidence_only_with_verifier_reason():
    task = ParsedTask(
        task_id="task-missing-scscore",
        raw_user_query="generate de novo for synthesizability",
        task_type="de_novo_generation",
        constraints=TaskConstraints(require_synthesizability=True),
    )
    workflow = plan_workflow(task)
    calls = [
        ToolCallRecord(tool_name="reinvent4_denovo", success=True, outputs={"success": True, "molecules_smiles": ["CCO"]}),
        ToolCallRecord(tool_name="scscore", success=True, outputs={"success": True, "results": []}),
    ]
    candidates = [CandidateRecord(smiles="CCO", source_tool="reinvent4_denovo", is_valid=True)]

    verifier_guided = suggest_repair(
        task,
        workflow,
        calls,
        candidates,
        VerifierResult(success=False, failure_reason="passes_synthesizability"),
    )
    scheduled_like = suggest_repair(task, workflow, calls, candidates, None)

    assert verifier_guided.should_retry is True
    assert verifier_guided.actions[0].action_type == "retry_evaluator_for_missing_evidence"
    assert verifier_guided.actions[0].tool_name == "scscore"
    assert scheduled_like.should_retry is False


def test_repair_retries_missing_toxicity_evidence_only_with_verifier_reason():
    task = ParsedTask(
        task_id="task-missing-toxicity",
        raw_user_query="generate de novo for toxicity",
        task_type="de_novo_generation",
        constraints=TaskConstraints(require_toxicity=True),
    )
    workflow = plan_workflow(task)
    calls = [
        ToolCallRecord(tool_name="reinvent4_denovo", success=True, outputs={"success": True, "molecules_smiles": ["CCO"]}),
        ToolCallRecord(tool_name="toxicity", success=True, outputs={"success": True, "smiles": "CCO"}),
    ]
    candidates = [CandidateRecord(smiles="CCO", source_tool="reinvent4_denovo", is_valid=True)]

    verifier_guided = suggest_repair(
        task,
        workflow,
        calls,
        candidates,
        VerifierResult(success=False, failure_reason="passes_toxicity"),
    )
    scheduled_like = suggest_repair(task, workflow, calls, candidates, None)

    assert verifier_guided.should_retry is True
    assert verifier_guided.actions[0].tool_name == "toxicity"
    assert scheduled_like.should_retry is False
