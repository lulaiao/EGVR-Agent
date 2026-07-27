from __future__ import annotations

import json

from egvr.tool_menu_execution_summary_builder import (
    build_tool_menu_execution_summary,
    write_tool_menu_execution_summary,
)


def test_builds_matched_planning_and_execution_table(tmp_path):
    planning = {
        "model": "gemini-test",
        "router_mode": "api",
        "task_count": 2,
        "rows": [
            {
                "menu_condition": "all_tool",
                "task_count": 2,
                "mean_exposed_tools": 13.0,
                "schema_validity": 1.0,
                "required_tool_recall": 0.75,
                "tool_precision": 0.8,
                "exact_order_match_rate": 0.5,
                "missing_required_input_count": 1,
            },
            {
                "menu_condition": "task_conditioned",
                "task_count": 2,
                "mean_exposed_tools": 3.0,
                "schema_validity": 1.0,
                "required_tool_recall": 1.0,
                "tool_precision": 1.0,
                "exact_order_match_rate": 1.0,
                "missing_required_input_count": 0,
            },
        ],
    }
    all_tool = {
        "summary": {
            "executed_task_count": 2,
            "evidence_verified_success_rate": 0.5,
            "false_success_count": 0,
            "mean_backend_call_count": 3.0,
            "mean_total_elapsed_sec": 10.0,
        },
        "results": [{}, {}],
    }
    task_conditioned = {
        "summary": {
            "executed_task_count": 2,
            "evidence_verified_success_rate": 1.0,
            "false_success_count": 0,
            "mean_backend_call_count": 4.0,
            "mean_total_elapsed_sec": 12.0,
        },
        "results": [{}, {}],
    }
    paths = {}
    for name, payload in (
        ("planning", planning),
        ("all", all_tool),
        ("conditioned", task_conditioned),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    result = build_tool_menu_execution_summary(
        planning_summary_path=paths["planning"],
        all_tool_execution_path=paths["all"],
        task_conditioned_execution_path=paths["conditioned"],
    )
    write_tool_menu_execution_summary(result, tmp_path / "table")

    assert result["rows"][0]["evidence_verified_success_rate"] == 0.5
    assert result["rows"][1]["evidence_verified_success_rate"] == 1.0
    assert (tmp_path / "table" / "strict_tool_menu_execution_table.csv").exists()
    assert "\\label{tab:gemini-tool-menu-executed}" in (
        tmp_path / "table" / "strict_tool_menu_execution_table.tex"
    ).read_text(encoding="utf-8")
