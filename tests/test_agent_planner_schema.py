from __future__ import annotations

import pytest

from CAi.toolkit.agent_planner.task_schema import (
    CandidateRecord,
    ParsedTask,
    PlannedToolCall,
    PlannedWorkflow,
    TaskConstraints,
    ToolCallRecord,
    ToolMetadata,
    VerifierResult,
)


def test_parsed_task_normalizes_pocket_center_and_lists():
    task = ParsedTask(
        task_id="task-1",
        raw_user_query="Generate molecules for this pocket.",
        task_type="pocket_conditioned_generation",
        objectives=("binding", "toxicity"),
        constraints=TaskConstraints(require_docking=True, require_toxicity=True),
        protein_path="agent_workspace/1HVR.pdb",
        pocket_center=(1, "2.5", 3),
        ligand_path="agent_workspace/ligands/lig_0.pdb",
        box_size=(20, 20, "20"),
        input_smiles=("CCO",),
        context_files=("notes.txt",),
    )

    assert task.pocket_center == [1.0, 2.5, 3.0]
    assert task.box_size == [20.0, 20.0, 20.0]
    assert task.ligand_path == "agent_workspace/ligands/lig_0.pdb"
    assert task.objectives == ["binding", "toxicity"]
    assert task.input_smiles == ["CCO"]
    assert task.to_dict()["constraints"]["require_docking"] is True


def test_parsed_task_rejects_invalid_pocket_center_length():
    with pytest.raises(ValueError, match="pocket_center"):
        ParsedTask(
            task_id="task-1",
            raw_user_query="bad center",
            pocket_center=[1.0, 2.0],
        )


def test_planned_workflow_derives_selected_tools_from_sequence():
    workflow = PlannedWorkflow(
        task_id="task-1",
        planner_type="rule_based",
        tool_sequence=[
            PlannedToolCall(
                tool_name="rxnflow",
                reason="Pocket-conditioned generation is available.",
                required_inputs=["protein_path", "pocket_center"],
                expected_outputs=["generated_smiles"],
            ),
            {
                "tool_name": "scscore",
                "reason": "Synthesizability requested.",
                "expected_outputs": ["scscore"],
                "required_inputs": ["generated_smiles"],
            },
        ],
    )

    assert workflow.selected_tools == ["rxnflow", "scscore"]
    assert workflow.tool_sequence[1].tool_name == "scscore"
    assert workflow.to_dict()["tool_sequence"][0]["tool_name"] == "rxnflow"


def test_candidate_and_call_records_have_independent_defaults():
    first = CandidateRecord(smiles="CCO", errors=["warn"])
    second = CandidateRecord(smiles="CCC")
    call = ToolCallRecord(tool_name="scscore", inputs={"smiles": "CCO"}, outputs={"scscore": 1.2}, success=True)

    first.errors.append("extra")

    assert second.errors == []
    assert call.to_dict()["outputs"]["scscore"] == 1.2


def test_tool_metadata_and_verifier_result_serialize():
    metadata = ToolMetadata(
        tool_name="example",
        description="Example tool",
        supported_task_types=["de_novo_generation"],
        required_inputs=[],
        optional_inputs=["num_variants"],
        outputs=["generated_smiles"],
        typical_failures=["empty output"],
        estimated_cost="low",
        downstream_tools=["scscore"],
        chemistry_role="generator",
        backend_tool_name="example_backend",
        tags=["generation"],
    )
    result = VerifierResult(success=False, checks={"has_valid_smiles": False}, failure_reason="No candidates")

    assert metadata.supports_task_type("de_novo_generation")
    assert metadata.to_dict()["downstream_tools"] == ["scscore"]
    assert result.to_dict()["checks"]["has_valid_smiles"] is False
