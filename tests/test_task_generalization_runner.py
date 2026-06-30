from __future__ import annotations

import csv

from CAi.toolkit.agent_planner.task_generalization_runner import run_task_generalization_summary


def test_task_generalization_runner_groups_mock_tasks_by_type(tmp_path):
    benchmark = "CAi/toolkit/agent_planner/benchmarks/task_generalization_v1.jsonl"

    payload = run_task_generalization_summary(
        benchmark,
        output_dir=tmp_path / "task_generalization",
        execution_mode="mock",
        planner_baseline="rule_based_planner",
    )

    assert (tmp_path / "task_generalization" / "task_generalization_summary.json").exists()
    assert (tmp_path / "task_generalization" / "task_generalization_summary.csv").exists()
    assert (tmp_path / "task_generalization" / "task_generalization_v1.rule_based_planner.json").exists()

    rows = {row["task_type"]: row for row in payload["rows"]}
    assert set(rows) == {
        "docking_evaluation",
        "hit_to_lead_optimization",
        "pocket_conditioned_generation",
        "scaffold_conditioned_generation",
    }
    assert rows["pocket_conditioned_generation"]["task_count"] == 3
    assert rows["hit_to_lead_optimization"]["task_success_rate"] == 1.0
    assert rows["scaffold_conditioned_generation"]["valid_candidate_rate"] == 1.0
    assert "reinvent4_mol2mol" in rows["hit_to_lead_optimization"]["tools"]
    assert "scaffold" in rows["scaffold_conditioned_generation"]["tools"]

    with (tmp_path / "task_generalization" / "task_generalization_summary.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 4
