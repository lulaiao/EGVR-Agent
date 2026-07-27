from __future__ import annotations

import json
import pytest

from egvr.benchmark_runner import BenchmarkRunner, load_benchmark_cases, summarize_results


def test_benchmark_runner_loads_jsonl_and_runs_mock_pipeline(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "case-1",
                        "raw_user_query": "Optimize hit_smiles=CCO for scscore and toxicity.",
                        "expected_task_type": "hit_to_lead_optimization",
                        "expected_tools": ["reinvent4_mol2mol", "scscore", "toxicity"],
                        "should_succeed": True,
                    }
                ),
                json.dumps(
                    {
                        "task_id": "case-2",
                        "raw_user_query": "Recover from failed molecule generation.",
                        "expected_task_type": "failure_recovery",
                        "expected_tools": ["reinvent4_denovo"],
                        "should_succeed": False,
                        "mock_outputs": {
                            "reinvent4_denovo": {"success": False, "error": "mock generation failure"}
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    payload = BenchmarkRunner(execution_mode="mock").run_file(benchmark, output_path=output)

    assert len(load_benchmark_cases(benchmark)) == 2
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["parser_accuracy"] == 1.0
    assert payload["summary"]["planner_tool_coverage_rate"] == 1.0
    assert payload["summary"]["planner_tool_recall"] == 1.0
    assert payload["summary"]["planner_tool_precision"] == 1.0
    assert payload["planner_baseline"] == "rule_based_planner"
    assert payload["summary"]["verifier_expectation_match"] == 1.0
    assert output.exists()
    assert payload["results"][1]["repair_plan"] is not None


def test_benchmark_runner_records_elapsed_rollups(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "elapsed-case",
                "raw_user_query": "Optimize hit_smiles=CCO for scscore and toxicity.",
                "expected_task_type": "hit_to_lead_optimization",
                "expected_tools": ["reinvent4_mol2mol", "scscore", "toxicity"],
                "should_succeed": True,
            }
        ),
        encoding="utf-8",
    )

    payload = BenchmarkRunner(execution_mode="mock").run_file(benchmark)
    result = payload["results"][0]

    assert result["total_elapsed_sec"] >= 0.0
    assert set(result["tool_elapsed_sec"]) == {"reinvent4_mol2mol", "scscore", "toxicity"}
    assert result["total_elapsed_sec"] == pytest.approx(sum(result["tool_elapsed_sec"].values()))
    assert payload["summary"]["mean_total_elapsed_sec"] == pytest.approx(result["total_elapsed_sec"])
    assert payload["summary"]["median_total_elapsed_sec"] == pytest.approx(result["total_elapsed_sec"])


def test_benchmark_runner_supports_fixed_pipeline_baseline(tmp_path):
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

    payload = BenchmarkRunner(execution_mode="mock", planner_baseline="fixed_pipeline").run_file(benchmark)

    assert payload["planner_baseline"] == "fixed_pipeline"
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["task_success_rate"] == 1.0
    assert payload["summary"]["planner_tool_coverage_rate"] == 0.0
    assert payload["summary"]["planner_tool_recall"] == 0.75
    assert payload["summary"]["planner_tool_precision"] == 1.0
    assert payload["results"][0]["planner_type"] == "fixed_pipeline"
    assert payload["results"][0]["selected_tools"] == ["rxnflow", "scscore", "toxicity"]
    assert payload["results"][0]["missing_tools"] == ["reinvent4_denovo"]


def test_benchmark_runner_supports_all_tool_agent_exposure_baseline(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "crossdocked-exposure",
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

    payload = BenchmarkRunner(execution_mode="mock", planner_baseline="all_tool_agent").run_file(benchmark)
    result = payload["results"][0]

    assert payload["planner_baseline"] == "all_tool_agent"
    assert result["planner_type"] == "all_tool_agent"
    assert result["selected_tool_count"] == 13
    assert result["tool_sequence_length"] == 4
    assert result["extra_tool_count"] == 9
    assert payload["summary"]["mean_selected_tool_count"] == 13.0
    assert payload["summary"]["mean_tool_sequence_length"] == 4.0
    assert payload["summary"]["mean_extra_tool_count"] == 9.0
    assert payload["summary"]["planner_tool_recall"] == 1.0
    assert payload["summary"]["planner_tool_precision"] == 4 / 13
    assert payload["summary"]["task_success_rate"] == 1.0


def test_full_copilot_executes_repair_after_verifier_failure(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "repair-denovo",
                "raw_user_query": "Generate 4 molecules de novo for synthesizability and toxicity.",
                "expected_task_type": "de_novo_generation",
                "expected_tools": ["reinvent4_denovo", "scscore", "toxicity"],
                "should_succeed": True,
                "mock_outputs": {
                    "reinvent4_denovo": [
                        {"success": False, "error": "mock denovo failure"},
                        {"success": True, "molecules_smiles": ["CCN", "CCCO"]},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    payload = BenchmarkRunner(execution_mode="mock", planner_baseline="full_copilot").run_file(benchmark)
    result = payload["results"][0]

    assert payload["planner_baseline"] == "full_copilot"
    assert payload["summary"]["repair_attempt_count"] == 1
    assert payload["summary"]["repair_success_count"] == 1
    assert payload["summary"]["repair_success_rate"] == 1.0
    assert result["planner_type"] == "full_copilot"
    assert result["repair_executed"] is True
    assert result["repair_success"] is True
    assert result["task_success"] is True
    assert "reinvent4_denovo" in result["selected_tools"]
    assert result["metrics"]["valid_smiles_count"] == 2


def test_egvr_agent_is_the_public_repair_baseline(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "egvr-repair-denovo",
                "raw_user_query": "Generate 2 molecules de novo for synthesizability.",
                "expected_task_type": "de_novo_generation",
                "expected_tools": ["reinvent4_denovo", "scscore"],
                "should_succeed": True,
                "mock_outputs": {
                    "reinvent4_denovo": [
                        {"success": False, "error": "controlled initial failure"},
                        {"success": True, "molecules_smiles": ["CCN", "CCO"]},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    payload = BenchmarkRunner(execution_mode="mock", planner_baseline="egvr_agent").run_file(benchmark)
    result = payload["results"][0]

    assert payload["planner_baseline"] == "egvr_agent"
    assert result["planner_type"] == "egvr_agent"
    assert result["repair_executed"] is True
    assert result["repair_success"] is True
    assert result["task_success"] is True


def test_full_copilot_respects_explicit_repair_budget(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "repair-budget",
                "raw_user_query": "Generate 2 molecules de novo for synthesizability.",
                "expected_task_type": "de_novo_generation",
                "expected_tools": ["reinvent4_denovo", "scscore"],
                "should_succeed": True,
                "mock_outputs": {
                    "reinvent4_denovo": [
                        {"success": False, "error": "initial failure"},
                        {"success": False, "error": "first repair failure"},
                        {"success": True, "molecules_smiles": ["CCN", "CCO"]},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    no_repair = BenchmarkRunner(
        execution_mode="mock", planner_baseline="full_copilot", repair_budget=0
    ).run_file(benchmark)["results"][0]
    two_rounds = BenchmarkRunner(
        execution_mode="mock", planner_baseline="full_copilot", repair_budget=2
    ).run_file(benchmark)["results"][0]

    assert no_repair["repair_executed"] is False
    assert no_repair["repair_rounds_executed"] == 0
    assert no_repair["task_success"] is False
    assert two_rounds["repair_executed"] is True
    assert two_rounds["repair_rounds_executed"] == 2
    assert len(two_rounds["repair_plan_history"]) == 2
    assert two_rounds["task_success"] is True


def test_failure_injection_applies_before_mock_recovery_output(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "injected-denovo",
                "raw_user_query": "Recover from failed de novo molecule generation for synthesizability and toxicity.",
                "expected_task_type": "failure_recovery",
                "expected_tools": ["reinvent4_denovo", "scscore", "toxicity"],
                "should_succeed": True,
                "mock_outputs": {
                    "reinvent4_denovo": {"success": True, "molecules_smiles": ["CCN", "CCCO"]}
                },
                "metadata": {
                    "failure_injections": {
                        "reinvent4_denovo": [
                            {"call_index": 1, "mode": "error", "error": "controlled test failure"}
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = BenchmarkRunner(execution_mode="mock", planner_baseline="full_copilot").run_file(benchmark)
    result = payload["results"][0]

    assert payload["summary"]["repair_attempt_count"] == 1
    assert payload["summary"]["repair_success_count"] == 1
    assert result["repair_executed"] is True
    assert result["task_success"] is True
    assert result["metrics"]["valid_smiles_count"] == 2


def test_benchmark_runner_rejects_unknown_baseline():
    with pytest.raises(ValueError, match="planner_baseline"):
        BenchmarkRunner(execution_mode="mock", planner_baseline="unknown")


def test_tool_status_only_separates_nominal_success_from_verified_success(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "status-only-missing-evidence",
                "raw_user_query": "Generate 1 molecule de novo for synthesizability.",
                "expected_task_type": "de_novo_generation",
                "expected_tools": ["reinvent4_denovo", "scscore"],
                "should_succeed": True,
                "metadata": {
                    "failure_injections": {
                        "scscore": [
                            {
                                "call_index": 1,
                                "output": {"success": True, "results": []},
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = BenchmarkRunner(execution_mode="mock", planner_baseline="tool_status_only").run_file(benchmark)
    result = payload["results"][0]

    assert result["agent_claimed_success"] is True
    assert result["evidence_verified_success"] is False
    assert result["task_success"] is False
    assert payload["summary"]["false_success_count"] == 1


def test_targeted_retry_no_fallback_filters_fallback_action(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "mol2mol-no-fallback",
                "raw_user_query": "Optimize hit_smiles=CCO for synthesizability and toxicity.",
                "expected_task_type": "hit_to_lead_optimization",
                "expected_tools": ["reinvent4_mol2mol", "scscore", "toxicity"],
                "should_succeed": True,
                "mock_outputs": {
                    "reinvent4_mol2mol": [
                        {"success": False, "error": "transient"},
                        {"success": True, "molecules_smiles": ["CCN"]},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    payload = BenchmarkRunner(
        execution_mode="mock",
        planner_baseline="verifier_targeted_retry_no_fallback",
    ).run_file(benchmark)
    result = payload["results"][0]

    assert result["repair_executed"] is True
    assert result["repair_success"] is True
    assert result["fallback_executed"] is False
    assert all(action["action_type"] != "fallback_tool" for action in result["repair_plan"]["actions"])
    assert result["repair_tool_call_count"] > 0


def test_summarize_results_handles_empty_input():
    assert summarize_results([]) == {
        "total": 0,
        "repair_attempt_count": 0,
        "repair_success_count": 0,
        "false_success_count": 0,
        "repair_attempt_rate": 0.0,
        "repair_success_rate": 0.0,
        "parser_accuracy": 0.0,
        "planner_tool_coverage_rate": 0.0,
        "planner_tool_precision": 0.0,
        "planner_tool_recall": 0.0,
        "planner_tool_f1": 0.0,
        "verifier_expectation_match": 0.0,
        "task_success_rate": 0.0,
        "agent_claim_success_rate": 0.0,
        "evidence_verified_success_rate": 0.0,
        "mean_selected_tool_count": 0.0,
        "mean_tool_sequence_length": 0.0,
        "mean_extra_tool_count": 0.0,
        "mean_tool_call_count": 0.0,
        "mean_backend_call_count": 0.0,
        "mean_initial_tool_call_count": 0.0,
        "mean_initial_backend_call_count": 0.0,
        "mean_repair_tool_call_count": 0.0,
        "mean_repair_backend_call_count": 0.0,
        "mean_repair_round_count": 0.0,
        "mean_candidate_evaluation_call_count": 0.0,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "failed_tool_call_count": 0,
        "tool_call_failure_rate": 0.0,
        "mean_total_elapsed_sec": 0.0,
        "median_total_elapsed_sec": 0.0,
    }
