from __future__ import annotations

import json

from egvr.repair_quality_builder import build_and_write_repair_quality_table


def test_repair_quality_builder_separates_recovery_false_repair_and_preservation(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "planner_baseline": "full_copilot",
                "execution_mode": "real",
                "results": [
                    {
                        "task_id": "recoverable",
                        "expected_success": True,
                        "repairability": "recoverable",
                        "initial_evidence_verified_success": False,
                        "task_success": True,
                        "agent_claimed_success": True,
                        "repair_executed": True,
                        "repair_success": True,
                        "fallback_executed": False,
                        "repair_plan": {"actions": [{"action_type": "retry", "tool_name": "scscore"}]},
                        "expected_repair_tools": ["scscore"],
                        "initial_tool_call_count": 2,
                        "repair_tool_call_count": 1,
                        "tool_call_count": 3,
                        "candidate_evaluation_call_count": 1,
                        "total_elapsed_sec": 3.0,
                    },
                    {
                        "task_id": "irrecoverable",
                        "expected_success": False,
                        "repairability": "irrecoverable",
                        "initial_evidence_verified_success": False,
                        "task_success": False,
                        "agent_claimed_success": False,
                        "repair_executed": True,
                        "repair_success": False,
                        "fallback_executed": True,
                        "repair_plan": {"actions": [{"action_type": "fallback_tool", "tool_name": "denovo"}]},
                        "initial_tool_call_count": 1,
                        "repair_tool_call_count": 1,
                        "tool_call_count": 2,
                        "candidate_evaluation_call_count": 0,
                        "total_elapsed_sec": 2.0,
                    },
                    {
                        "task_id": "healthy",
                        "expected_success": True,
                        "repairability": "healthy",
                        "initial_evidence_verified_success": True,
                        "task_success": True,
                        "agent_claimed_success": True,
                        "repair_executed": False,
                        "repair_success": False,
                        "fallback_executed": False,
                        "repair_plan": None,
                        "initial_tool_call_count": 1,
                        "repair_tool_call_count": 0,
                        "tool_call_count": 1,
                        "candidate_evaluation_call_count": 0,
                        "total_elapsed_sec": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_and_write_repair_quality_table([result], output_dir=tmp_path / "out", project_root=tmp_path)
    row = payload["rows"][0]

    assert row["repair_recovery_rate"] == 1.0
    assert row["repair_precision"] == 0.5
    assert row["false_repair_count"] == 0
    assert row["fallback_rate"] == 0.5
    assert row["irrecoverable_preservation_rate"] == 1.0
    assert row["initial_backend_call_count"] == 4
    assert row["repair_backend_call_count"] == 2
    assert (tmp_path / "out" / "repair_quality_table.tex").exists()
    assert (tmp_path / "out" / "cost_normalized_table.json").exists()
    assert (tmp_path / "out" / "failure_taxonomy_family_table.json").exists()

    cost = json.loads((tmp_path / "out" / "cost_normalized_table.json").read_text(encoding="utf-8"))["rows"][0]
    assert cost["total_backend_call_count"] == 6
    assert cost["candidate_evaluation_call_count"] == 1
    assert cost["backend_calls_per_verified_success"] == 3.0


def test_repair_quality_builder_groups_controlled_failure_families(tmp_path):
    result = tmp_path / "families.json"
    result.write_text(
        json.dumps(
            {
                "benchmark_id": "taxonomy_v3",
                "planner_baseline": "tool_status_only",
                "execution_mode": "mock",
                "results": [
                    {
                        "task_id": "generation_missing",
                        "failure_family": "generation",
                        "repairability": "recoverable",
                        "agent_claimed_success": True,
                        "evidence_verified_success": False,
                        "task_success": False,
                        "repair_executed": False,
                        "tool_call_count": 1,
                        "total_elapsed_sec": 1.0,
                    },
                    {
                        "task_id": "toxicity_healthy",
                        "failure_family": "toxicity",
                        "repairability": "healthy",
                        "agent_claimed_success": True,
                        "evidence_verified_success": True,
                        "task_success": True,
                        "repair_executed": False,
                        "tool_call_count": 2,
                        "total_elapsed_sec": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    build_and_write_repair_quality_table([result], output_dir=tmp_path / "out", project_root=tmp_path)
    rows = json.loads(
        (tmp_path / "out" / "failure_taxonomy_family_table.json").read_text(encoding="utf-8")
    )["rows"]

    assert [row["failure_family"] for row in rows] == ["generation", "toxicity"]
    assert rows[0]["false_success_count"] == 1
    assert rows[1]["evidence_verified_success_rate"] == 1.0
