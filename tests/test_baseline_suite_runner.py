from __future__ import annotations

import csv
import json

import pytest

from egvr.baseline_suite_runner import run_baseline_suite


def test_baseline_suite_runner_writes_table_ready_outputs(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "crossdocked-smoke",
                "raw_user_query": (
                    "Generate 5 molecules for protein_path=/tmp/target.pdb "
                    "pocket_center=[1,2,3] for synthesizability and toxicity."
                ),
                "expected_task_type": "pocket_conditioned_generation",
                "expected_tools": ["rxnflow", "reinvent4_denovo", "scscore", "toxicity"],
                "should_succeed": True,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "suite"

    payload = run_baseline_suite(
        benchmark,
        output_dir=output_dir,
        planner_baselines=[
            "all_tool_agent",
            "fixed_pipeline",
            "rule_based_planner",
            "full_copilot",
            "scheduled_fallback_no_verifier",
            "verifier_only_no_repair",
        ],
        execution_mode="mock",
    )

    summary_json = output_dir / "tasks.baseline_summary.json"
    summary_csv = output_dir / "tasks.baseline_summary.csv"
    assert summary_json.exists()
    assert summary_csv.exists()
    assert (output_dir / "tasks.all_tool_agent.json").exists()
    assert (output_dir / "tasks.fixed_pipeline.json").exists()
    assert (output_dir / "tasks.rule_based_planner.json").exists()
    assert (output_dir / "tasks.full_copilot.json").exists()
    assert (output_dir / "tasks.scheduled_fallback_no_verifier.json").exists()
    assert (output_dir / "tasks.verifier_only_no_repair.json").exists()
    assert payload["planner_baselines"] == [
        "all_tool_agent",
        "fixed_pipeline",
        "rule_based_planner",
        "full_copilot",
        "scheduled_fallback_no_verifier",
        "verifier_only_no_repair",
    ]
    assert len(payload["rows"]) == 6

    rows_by_baseline = {row["planner_baseline"]: row for row in payload["rows"]}
    assert rows_by_baseline["all_tool_agent"]["planner_tool_recall"] == 1.0
    assert rows_by_baseline["all_tool_agent"]["planner_tool_precision"] == 4 / 13
    assert rows_by_baseline["all_tool_agent"]["mean_selected_tool_count"] == 13.0
    assert rows_by_baseline["all_tool_agent"]["mean_extra_tool_count"] == 9.0
    assert rows_by_baseline["fixed_pipeline"]["planner_tool_recall"] == 0.75
    assert rows_by_baseline["rule_based_planner"]["planner_tool_recall"] == 1.0
    assert rows_by_baseline["full_copilot"]["planner_tool_recall"] == 1.0
    assert rows_by_baseline["scheduled_fallback_no_verifier"]["planner_tool_recall"] == 1.0
    assert rows_by_baseline["verifier_only_no_repair"]["planner_tool_recall"] == 1.0
    assert rows_by_baseline["rule_based_planner"]["task_success_count"] == 1
    assert rows_by_baseline["full_copilot"]["repair_attempt_count"] == 0

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["planner_baseline"] for row in csv_rows] == [
        "all_tool_agent",
        "fixed_pipeline",
        "rule_based_planner",
        "full_copilot",
        "scheduled_fallback_no_verifier",
        "verifier_only_no_repair",
    ]
    assert csv_rows[0]["benchmark_id"] == "tasks"
    assert csv_rows[0]["mean_selected_tool_count"] == "13.0"


def test_baseline_suite_runner_rejects_unknown_baseline(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps({"task_id": "case-1", "raw_user_query": "Generate molecules.", "should_succeed": True}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported planner baseline"):
        run_baseline_suite(benchmark, output_dir=tmp_path / "suite", planner_baselines=["unknown"])


def test_failure_recovery_mock_slice_shows_full_copilot_repair_benefit(tmp_path):
    benchmark = "egvr/benchmarks/failure_recovery_mock_v1.jsonl"

    payload = run_baseline_suite(
        benchmark,
        output_dir=tmp_path / "failure_suite",
        planner_baselines=["rule_based_planner", "full_copilot"],
        execution_mode="mock",
    )

    rows = {row["planner_baseline"]: row for row in payload["rows"]}
    assert rows["rule_based_planner"]["task_success_count"] == 0
    assert rows["full_copilot"]["task_success_count"] == 2
    assert rows["full_copilot"]["repair_attempt_count"] == 3
    assert rows["full_copilot"]["repair_success_count"] == 2
    assert rows["full_copilot"]["verifier_expectation_match"] == 1.0


def test_failure_recovery_real_or_injected_slice_runs_in_mock_mode(tmp_path):
    benchmark = "egvr/benchmarks/failure_recovery_real_or_injected_v1.jsonl"

    payload = run_baseline_suite(
        benchmark,
        output_dir=tmp_path / "injected_suite",
        planner_baselines=["rule_based_planner", "full_copilot"],
        execution_mode="mock",
    )

    rows = {row["planner_baseline"]: row for row in payload["rows"]}
    assert rows["rule_based_planner"]["task_success_count"] == 0
    assert rows["full_copilot"]["task_success_count"] == 2
    assert rows["full_copilot"]["repair_attempt_count"] == 3
    assert rows["full_copilot"]["repair_success_count"] == 2
    assert rows["full_copilot"]["verifier_expectation_match"] == 1.0


def test_failure_recovery_taxonomy_v2_runs_in_mock_mode(tmp_path):
    benchmark = "egvr/benchmarks/failure_recovery_taxonomy_v2.jsonl"

    payload = run_baseline_suite(
        benchmark,
        output_dir=tmp_path / "taxonomy_suite",
        planner_baselines=["rule_based_planner", "full_copilot"],
        execution_mode="mock",
    )

    rows = {row["planner_baseline"]: row for row in payload["rows"]}
    assert rows["rule_based_planner"]["total"] == 9
    assert rows["full_copilot"]["total"] == 9
    assert rows["full_copilot"]["task_success_count"] > rows["rule_based_planner"]["task_success_count"]
    assert rows["full_copilot"]["repair_attempt_count"] >= 4
    assert rows["full_copilot"]["verifier_expectation_match"] == 1.0


def test_failure_recovery_taxonomy_v2_repair_ablation_baselines_run_in_mock_mode(tmp_path):
    benchmark = "egvr/benchmarks/failure_recovery_taxonomy_v2.jsonl"

    payload = run_baseline_suite(
        benchmark,
        output_dir=tmp_path / "repair_ablation_suite",
        planner_baselines=[
            "rule_based_planner",
            "verifier_only_no_repair",
            "scheduled_fallback_no_verifier",
            "full_copilot",
        ],
        execution_mode="mock",
    )

    rows = {row["planner_baseline"]: row for row in payload["rows"]}
    assert rows["verifier_only_no_repair"]["task_success_count"] == rows["rule_based_planner"]["task_success_count"]
    assert rows["verifier_only_no_repair"]["repair_attempt_count"] == 0
    assert rows["scheduled_fallback_no_verifier"]["repair_attempt_count"] > 0
    assert rows["full_copilot"]["repair_attempt_count"] > 0
    assert rows["full_copilot"]["verifier_expectation_match"] == 1.0


def test_ambiguous_evidence_slice_separates_scheduled_fallback_from_full_copilot(tmp_path):
    benchmark = "egvr/benchmarks/failure_recovery_ambiguous_evidence_v1.jsonl"

    payload = run_baseline_suite(
        benchmark,
        output_dir=tmp_path / "ambiguous_evidence_suite",
        planner_baselines=[
            "rule_based_planner",
            "verifier_only_no_repair",
            "scheduled_fallback_no_verifier",
            "full_copilot",
        ],
        execution_mode="mock",
    )

    rows = {row["planner_baseline"]: row for row in payload["rows"]}
    assert rows["rule_based_planner"]["task_success_count"] == 0
    assert rows["verifier_only_no_repair"]["task_success_count"] == 0
    assert rows["scheduled_fallback_no_verifier"]["task_success_count"] == 0
    assert rows["scheduled_fallback_no_verifier"]["repair_attempt_count"] == 0
    assert rows["full_copilot"]["task_success_count"] == 3
    assert rows["full_copilot"]["repair_attempt_count"] == 3
    assert rows["full_copilot"]["repair_success_count"] == 3
    assert rows["full_copilot"]["verifier_expectation_match"] == 1.0
