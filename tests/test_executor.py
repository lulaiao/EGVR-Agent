from __future__ import annotations

from CAi.toolkit.agent_planner.executor import WorkflowExecutor
from CAi.toolkit.agent_planner.rule_planner import plan_workflow
from CAi.toolkit.agent_planner.task_schema import ParsedTask, TaskConstraints


def test_executor_runs_mock_generation_and_batch_scscore():
    task = ParsedTask(
        task_id="task-denovo",
        raw_user_query="generate",
        task_type="de_novo_generation",
        objectives=["synthesizability"],
        constraints=TaskConstraints(max_candidates=2, require_synthesizability=True),
    )
    workflow = plan_workflow(task)

    executor = WorkflowExecutor(
        tool_functions={
            "reinvent4_denovo": lambda **kwargs: {"success": True, "molecules_smiles": ["CCO", "CCC"]},
            "scscore": lambda **kwargs: {
                "success": True,
                "results": [
                    {"input_smiles": "CCO", "canonical_smiles": "CCO", "scscore": 1.2},
                    {"input_smiles": "CCC", "canonical_smiles": "CCC", "scscore": 2.4},
                ],
            },
        }
    )

    calls, candidates = executor.execute(task, workflow)

    assert [call.tool_name for call in calls] == ["reinvent4_denovo", "scscore"]
    assert all(call.success for call in calls)
    assert [candidate.smiles for candidate in candidates] == ["CCO", "CCC"]
    assert [candidate.scscore for candidate in candidates] == [1.2, 2.4]


def test_executor_skips_conditional_fallback_after_successful_rxnflow():
    task = ParsedTask(
        task_id="task-pocket",
        raw_user_query="generate for pocket",
        task_type="pocket_conditioned_generation",
        objectives=[],
        constraints=TaskConstraints(max_candidates=2),
        protein_path="agent_workspace/1HVR.pdb",
        pocket_center=[1, 2, 3],
    )
    workflow = plan_workflow(task)
    executor = WorkflowExecutor(
        tool_functions={
            "rxnflow": lambda **kwargs: {"success": True, "molecules_smiles": ["CCO"]},
            "reinvent4_denovo": lambda **kwargs: {"success": True, "molecules_smiles": ["CCC"]},
        }
    )

    calls, candidates = executor.execute(task, workflow)

    assert [call.tool_name for call in calls] == ["rxnflow", "reinvent4_denovo"]
    assert calls[1].metadata["skipped"] is True
    assert [candidate.smiles for candidate in candidates] == ["CCO"]


def test_executor_records_tool_failure_without_swallowing_error():
    task = ParsedTask(
        task_id="task-denovo",
        raw_user_query="generate",
        task_type="de_novo_generation",
    )
    workflow = plan_workflow(task)
    executor = WorkflowExecutor(tool_functions={"reinvent4_denovo": lambda **kwargs: {"success": False, "error": "boom"}})

    calls, candidates = executor.execute(task, workflow)

    assert calls[0].success is False
    assert calls[0].error == "boom"
    assert candidates == []


def test_executor_allows_mock_vina_with_generated_candidate_smiles():
    task = ParsedTask(
        task_id="task-pocket",
        raw_user_query="generate for pocket",
        task_type="pocket_conditioned_generation",
        objectives=["binding"],
        constraints=TaskConstraints(require_docking=True),
        protein_path="agent_workspace/1HVR.pdb",
        pocket_center=[1, 2, 3],
    )
    workflow = plan_workflow(task)
    executor = WorkflowExecutor(
        tool_functions={
            "rxnflow": lambda **kwargs: {"success": True, "molecules_smiles": ["CCO"]},
            "reinvent4_denovo": lambda **kwargs: {"success": True, "molecules_smiles": ["CCC"]},
            "vina": lambda **kwargs: {"success": True, "best_docking_score_kcal_mol": -7.1},
        }
    )

    calls, candidates = executor.execute(task, workflow)

    assert [call.tool_name for call in calls] == ["rxnflow", "reinvent4_denovo", "vina"]
    assert calls[2].inputs["candidate_smiles"] == ["CCO"]
    assert candidates[0].docking_score == -7.1
