from __future__ import annotations

import json

from egvr.executed_planner_comparison_builder import (
    build_executed_planner_rows,
    write_executed_planner_table,
)


def test_executed_planner_builder_unifies_router_and_structured_results(tmp_path):
    router_path = tmp_path / "router.json"
    router_path.write_text(
        json.dumps(
            {
                "router_name": "router-a",
                "repair_mode": "none",
                "execution_mode": "real",
                "summary": {
                    "router_workflow_valid_rate": 0.5,
                    "task_success_rate": 0.5,
                    "false_success_count": 0,
                    "mean_tool_call_count": 2.0,
                    "mean_total_elapsed_sec": 3.0,
                },
                "results": [{"task_id": "a", "task_success": True}, {"task_id": "b", "task_success": False}],
            }
        ),
        encoding="utf-8",
    )
    full_path = tmp_path / "full.json"
    full_path.write_text(
        json.dumps(
            {
                "planner_baseline": "full_copilot",
                "execution_mode": "real",
                "summary": {"task_success_rate": 1.0, "false_success_count": 0, "mean_tool_call_count": 3.0},
                "results": [{"task_id": "a", "task_success": True}, {"task_id": "b", "task_success": True}],
            }
        ),
        encoding="utf-8",
    )

    rows = build_executed_planner_rows([router_path, full_path], project_root=tmp_path)
    assert rows[0]["planner_family"] == "llm_router"
    assert rows[0]["workflow_valid_rate"] == 0.5
    assert rows[0]["backend_calls_per_verified_success"] == 4.0
    assert rows[0]["elapsed_sec_per_verified_success"] == 6.0
    assert rows[1]["repair_policy"] == "verifier_guided"
    assert rows[1]["verified_success_rate"] == 1.0

    payload = write_executed_planner_table(rows, output_dir=tmp_path / "out", project_root=tmp_path)
    assert payload["row_count"] == 2
    assert (tmp_path / "out" / "executed_planner_comparison_table.tex").exists()
