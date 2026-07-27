from __future__ import annotations

import csv
import json

from egvr.repeated_suite_runner import run_repeated_baseline_suite


def test_repeated_suite_runner_aggregates_mock_failure_taxonomy(tmp_path):
    benchmark = "egvr/benchmarks/failure_recovery_taxonomy_v2.jsonl"

    payload = run_repeated_baseline_suite(
        benchmark,
        output_dir=tmp_path / "repeated",
        repeats=3,
        planner_baselines=["rule_based_planner", "full_copilot"],
        execution_mode="mock",
    )

    assert payload["repeat_count"] == 3
    assert (tmp_path / "repeated" / "failure_recovery_taxonomy_v2.repeated_summary.json").exists()
    assert (tmp_path / "repeated" / "failure_recovery_taxonomy_v2.repeated_summary.csv").exists()
    assert (tmp_path / "repeated" / "failure_recovery_taxonomy_v2.repeated_detail.csv").exists()
    assert (tmp_path / "repeated" / "repeat_01" / "failure_recovery_taxonomy_v2.baseline_summary.json").exists()

    rows = {row["planner_baseline"]: row for row in payload["rows"]}
    assert rows["rule_based_planner"]["repeat_count"] == 3
    assert rows["full_copilot"]["repeat_count"] == 3
    assert rows["full_copilot"]["mean_task_success_rate"] > rows["rule_based_planner"]["mean_task_success_rate"]
    assert rows["full_copilot"]["false_success_count"] == 0
    assert rows["full_copilot"]["std_task_success_rate"] == 0.0

    with (tmp_path / "repeated" / "failure_recovery_taxonomy_v2.repeated_summary.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["planner_baseline"] for row in csv_rows] == ["rule_based_planner", "full_copilot"]

    saved = json.loads(
        (tmp_path / "repeated" / "failure_recovery_taxonomy_v2.repeated_summary.json").read_text(encoding="utf-8")
    )
    assert saved["artifacts"]["detail_csv"].endswith("failure_recovery_taxonomy_v2.repeated_detail.csv")
