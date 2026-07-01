from __future__ import annotations

from CAi.toolkit.agent_planner.biomedical_benchmark_runner import run_biomedical_benchmark


def test_biomedical_benchmark_runner_summarizes_clinical_v2_slice():
    summary = run_biomedical_benchmark(
        "CAi/toolkit/agent_planner/benchmarks/clinical_trial_outcome_prediction_v2_offline.jsonl"
    )

    assert summary["benchmark_id"] == "clinical_trial_outcome_prediction_v2_offline"
    assert summary["task_count"] == 20
    assert summary["false_success_count"] == 0
    assert summary["tool_selection_accuracy"] == 1.0
    assert summary["mean_provenance_coverage"] == summary["mean_evidence_coverage"]
    assert summary["verifier_expectation_match_rate"] == 1.0
    assert 0 < summary["workflow_success_rate"] < 1


def test_biomedical_benchmark_runner_summarizes_drug_target_v2_slice():
    summary = run_biomedical_benchmark(
        "CAi/toolkit/agent_planner/benchmarks/drug_target_evidence_v2_offline.jsonl"
    )

    assert summary["benchmark_id"] == "drug_target_evidence_v2_offline"
    assert summary["task_count"] == 20
    assert summary["false_success_count"] == 0
    assert summary["tool_selection_accuracy"] == 1.0
    assert summary["mean_provenance_coverage"] == summary["mean_evidence_coverage"]
    assert summary["verifier_expectation_match_rate"] == 1.0
    assert 0 < summary["workflow_success_rate"] < 1
