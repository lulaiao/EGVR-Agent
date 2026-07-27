from __future__ import annotations

import json

from egvr.tool_menu_comparison_runner import run_tool_menu_comparison


def test_tool_menu_comparison_changes_only_exposed_registry(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "dock_1",
                "raw_user_query": (
                    "Dock ligand /tmp/ligand.sdf against protein /tmp/protein.pdb "
                    "with pocket center [1, 2, 3] and box size [20, 20, 20]."
                ),
                "expected_task_type": "docking_evaluation",
                "expected_tools": ["vina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_tool_menu_comparison(
        benchmark_paths=[benchmark],
        output_dir=tmp_path / "out",
        router_mode="heuristic",
        project_root=tmp_path,
    )

    rows = {row["menu_condition"]: row for row in payload["rows"]}
    assert rows["all_tool"]["mean_exposed_tools"] > rows["task_conditioned"]["mean_exposed_tools"]
    assert rows["all_tool"]["schema_validity"] == 1.0
    assert rows["task_conditioned"]["required_tool_recall"] == 1.0
    assert (tmp_path / "out" / "all_tool.planning.json").exists()
    assert (tmp_path / "out" / "task_conditioned.planning.json").exists()
