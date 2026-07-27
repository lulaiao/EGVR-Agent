from __future__ import annotations

from egvr.task_schema import (
    CandidateRecord,
    ParsedTask,
    PlannedToolCall,
    PlannedWorkflow,
    TaskConstraints,
    ToolCallRecord,
)
from egvr.verifier import verify_workflow


def test_verifier_success_for_scored_ranked_candidates():
    task = ParsedTask(
        task_id="task",
        raw_user_query="rank",
        task_type="multi_objective_screening",
        objectives=["synthesizability", "toxicity"],
        constraints=TaskConstraints(require_synthesizability=True, require_toxicity=True, require_ranking=True),
        input_smiles=["CCO"],
    )
    workflow = PlannedWorkflow(
        task_id="task",
        planner_type="rule_based",
        tool_sequence=[PlannedToolCall(tool_name="scscore", reason="score")],
        expected_outputs=["ranked_candidates"],
    )
    calls = [ToolCallRecord(tool_name="scscore", success=True)]
    candidates = [CandidateRecord(smiles="CCO", rank=1, scscore=1.5, toxicity_score=0.1)]

    result = verify_workflow(task, workflow, calls, candidates)

    assert result.success is True
    assert result.checks["has_valid_smiles"] is True
    assert result.metrics["candidate_count"] == 1
    assert result.metrics["best_sa_score"] is None
    assert result.metrics["sa_score_coverage"] == 0.0


def test_verifier_fails_when_required_toxicity_missing():
    task = ParsedTask(
        task_id="task",
        raw_user_query="toxicity",
        task_type="hit_to_lead_optimization",
        objectives=["toxicity"],
        constraints=TaskConstraints(require_toxicity=True),
        input_smiles=["CCO"],
    )
    workflow = PlannedWorkflow(task_id="task", planner_type="rule_based")
    calls = [ToolCallRecord(tool_name="toxicity", success=True)]
    candidates = [CandidateRecord(smiles="CCO")]

    result = verify_workflow(task, workflow, calls, candidates)

    assert result.success is False
    assert result.checks["passes_toxicity"] is False
    assert "passes_toxicity" in result.failure_reason


def test_verifier_requires_generated_candidates_for_generation_workflow():
    task = ParsedTask(
        task_id="task",
        raw_user_query="optimize",
        task_type="hit_to_lead_optimization",
        objectives=["synthesizability"],
        constraints=TaskConstraints(require_synthesizability=True),
        input_smiles=["CCO"],
    )
    workflow = PlannedWorkflow(
        task_id="task",
        planner_type="rule_based",
        tool_sequence=[PlannedToolCall(tool_name="reinvent4_mol2mol", reason="generate")],
        expected_outputs=["generated_smiles", "candidate_records"],
    )
    calls = [ToolCallRecord(tool_name="reinvent4_mol2mol", success=False, error="generation failed")]
    candidates = [CandidateRecord(smiles="CCO", source_tool="input", scscore=1.5)]

    result = verify_workflow(task, workflow, calls, candidates)

    assert result.success is False
    assert result.checks["has_valid_smiles"] is False
    assert result.metrics["candidate_count"] == 1
    assert result.metrics["valid_smiles_count"] == 0
    assert "has_tool_success" in result.failure_reason
    assert "has_valid_smiles" in result.failure_reason


def test_verifier_allows_docking_only_without_smiles():
    task = ParsedTask(
        task_id="task",
        raw_user_query="dock",
        task_type="docking_evaluation",
        objectives=["binding"],
        constraints=TaskConstraints(require_docking=True),
    )
    workflow = PlannedWorkflow(
        task_id="task",
        planner_type="rule_based",
        tool_sequence=[PlannedToolCall(tool_name="vina", reason="dock")],
        expected_outputs=["docking_score"],
    )
    calls = [
        ToolCallRecord(
            tool_name="vina",
            success=True,
            outputs={"best_docking_score_kcal_mol": -8.2},
        )
    ]

    result = verify_workflow(task, workflow, calls, [])

    assert result.success is True
    assert result.checks["has_valid_smiles"] is False
    assert result.checks["has_docking_scores"] is True


def test_verifier_reports_sa_score_and_posebusters_metrics():
    task = ParsedTask(
        task_id="task",
        raw_user_query="score",
        task_type="multi_objective_screening",
        objectives=["synthesizability"],
        constraints=TaskConstraints(require_synthesizability=True),
        input_smiles=["CCO"],
    )
    workflow = PlannedWorkflow(task_id="task", planner_type="rule_based")
    calls = [ToolCallRecord(tool_name="sa_score", success=True)]
    candidates = [
        CandidateRecord(smiles="CCO", sa_score=2.2, posebusters_pass=True),
        CandidateRecord(smiles="CCC", sa_score=3.1, posebusters_pass=False),
    ]

    result = verify_workflow(task, workflow, calls, candidates)

    assert result.success is True
    assert result.checks["has_sa_score_evidence"] is True
    assert result.checks["has_posebusters_evidence"] is True
    assert result.metrics["best_sa_score"] == 2.2
    assert result.metrics["sa_score_coverage"] == 1.0
    assert result.metrics["posebusters_pass_rate"] == 0.5
    assert result.metrics["posebusters_coverage"] == 1.0


def test_verifier_reports_rdkit_property_metrics_without_requiring_them():
    task = ParsedTask(
        task_id="task",
        raw_user_query="score",
        task_type="de_novo_generation",
        objectives=[],
    )
    workflow = PlannedWorkflow(task_id="task", planner_type="rule_based")
    calls = [ToolCallRecord(tool_name="rdkit_property_verifier", success=True)]
    candidates = [
        CandidateRecord(
            smiles="CCO",
            metadata={
                "rdkit_properties": {
                    "qed": 0.42,
                    "logp": -0.01,
                    "lipinski_pass": True,
                    "pains_flags": [],
                }
            },
        ),
        CandidateRecord(
            smiles="c1ccccc1",
            metadata={
                "rdkit_properties": {
                    "qed": 0.51,
                    "logp": 1.7,
                    "lipinski_pass": True,
                    "pains_flags": ["demo_alert"],
                }
            },
        ),
    ]

    result = verify_workflow(task, workflow, calls, candidates)

    assert result.success is True
    assert result.metrics["rdkit_property_coverage"] == 1.0
    assert result.metrics["mean_qed"] == 0.46499999999999997
    assert result.metrics["lipinski_pass_rate"] == 1.0
    assert result.metrics["pains_flag_rate"] == 0.5
