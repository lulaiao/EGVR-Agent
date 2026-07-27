from __future__ import annotations

import json

from egvr.planner_comparison_builder import build_planner_comparison


def test_planner_comparison_combines_saved_llm_and_structured_rows(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "dock_1",
                "raw_user_query": (
                    "Dock ligand_path=/tmp/lig.sdf against protein_path=/tmp/rec.pdbqt "
                    "with pocket_center=[1,2,3] and box_size=[20,20,20]."
                ),
                "expected_tools": ["vina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    llm = tmp_path / "llm.json"
    llm.write_text(
        json.dumps(
            {
                "row": {
                    "task_count": 1,
                    "valid_json_count": 1,
                    "valid_schema_count": 1,
                    "tool_recall": 1.0,
                    "tool_precision": 1.0,
                    "workflow_order_match_rate": 1.0,
                    "missing_required_input_count": 0,
                    "hallucinated_tool_count": 0,
                    "mean_selected_tool_count": 1.0,
                },
                "task_results": [{"task_id": "dock_1"}],
            }
        ),
        encoding="utf-8",
    )

    payload = build_planner_comparison(
        llm_results=[("gemini-2.5-pro", llm)],
        benchmark_paths=[benchmark],
        output_dir=tmp_path / "out",
        project_root=tmp_path,
    )

    assert [row["method"] for row in payload["rows"]] == [
        "gemini-2.5-pro",
        "structured_planner",
        "egvr_agent",
    ]
    assert payload["rows"][1]["valid_schema_rate"] == 1.0
    assert payload["rows"][1]["required_tool_recall"] == 1.0
    assert payload["rows"][2]["source"] == "deterministic_rule_planner_on_locked_65_tasks"
    assert (tmp_path / "out" / "planner_comparison_65_table.tex").exists()
