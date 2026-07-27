from __future__ import annotations

import json

from egvr.llm_router_execution_audit_runner import run_llm_router_execution_audit


def test_llm_router_execution_supports_verifier_guided_repair(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "router-repair",
                "raw_user_query": "Generate 1 molecule de novo for synthesizability.",
                "expected_task_type": "de_novo_generation",
                "expected_tools": ["reinvent4_denovo", "scscore"],
                "should_succeed": True,
                "metadata": {
                    "failure_injections": {
                        "reinvent4_denovo": [
                            {"call_index": 1, "mode": "error", "error": "transient"}
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    workflow = {
        "task_id": "router-repair",
        "planner_type": "llm_as_router",
        "selected_tools": ["reinvent4_denovo", "scscore"],
        "tool_sequence": [
            {
                "tool_name": "reinvent4_denovo",
                "reason": "generate",
                "action": "de_novo",
                "expected_outputs": ["generated_smiles"],
                "required_inputs": [],
                "optional_inputs": ["num_variants"],
                "parameters": {"num_variants": 1},
            },
            {
                "tool_name": "scscore",
                "reason": "score",
                "action": "default",
                "expected_outputs": ["scscore"],
                "required_inputs": ["generated_smiles"],
                "optional_inputs": [],
                "parameters": {"input_source": "generated_smiles"},
            },
        ],
        "expected_outputs": ["generated_smiles", "candidate_records", "evaluated_candidates"],
        "notes": [],
    }
    router = tmp_path / "router.json"
    router.write_text(
        json.dumps({"task_results": [{"task_id": "router-repair", "raw_response": json.dumps(workflow)}]}),
        encoding="utf-8",
    )

    payload = run_llm_router_execution_audit(
        benchmark_paths=[benchmark],
        router_summary_path=router,
        router_name="test-router",
        output_path=tmp_path / "result.json",
        execution_mode="mock",
        repair_mode="verifier_guided",
        project_root=tmp_path,
    )

    result = payload["results"][0]
    assert payload["repair_mode"] == "verifier_guided"
    assert result["repair_executed"] is True
    assert result["repair_success"] is True
    assert result["repair_tool_call_count"] > 0
    assert result["task_success"] is True
