from __future__ import annotations

import csv
import json

import pytest

from egvr.latex_table_export import escape_latex, export_latex_tables


def test_export_latex_tables_writes_paper_table_drafts(tmp_path):
    input_dir = tmp_path / "tables"
    output_dir = tmp_path / "latex"
    input_dir.mkdir()

    _write_csv(
        input_dir / "robustness_table.csv",
        [
            "benchmark_id",
            "dataset",
            "execution_mode",
            "planner_baseline",
            "task_count",
            "task_success_count",
            "failed_task_count",
            "task_success_rate",
            "verifier_expectation_match",
            "repair_attempt_count",
            "repair_success_count",
            "repair_success_rate",
            "notes",
        ],
        [
            {
                "benchmark_id": "failure_recovery_v1",
                "dataset": "controlled_tool_failure_injection",
                "execution_mode": "real",
                "planner_baseline": "rule_based_planner",
                "task_count": "3",
                "task_success_count": "0",
                "failed_task_count": "3",
                "task_success_rate": "0.0",
                "verifier_expectation_match": "0.3333333333333333",
                "repair_attempt_count": "0",
                "repair_success_count": "0",
                "repair_success_rate": "0.0",
                "notes": "baseline",
            },
            {
                "benchmark_id": "failure_recovery_v1",
                "dataset": "controlled_tool_failure_injection",
                "execution_mode": "real",
                "planner_baseline": "full_copilot",
                "task_count": "3",
                "task_success_count": "2",
                "failed_task_count": "1",
                "task_success_rate": "0.6666666666666666",
                "verifier_expectation_match": "1.0",
                "repair_attempt_count": "3",
                "repair_success_count": "2",
                "repair_success_rate": "0.6666666666666666",
                "notes": "recovered",
            },
        ],
    )
    _write_csv(
        input_dir / "robustness_repeated_table.csv",
        [
            "benchmark_id",
            "source_benchmark_id",
            "execution_mode",
            "planner_baseline",
            "repeat_count",
            "task_count",
            "mean_task_success_rate",
            "std_task_success_rate",
            "mean_repair_success_rate",
            "std_repair_success_rate",
            "false_success_count",
            "mean_verifier_expectation_match",
            "std_verifier_expectation_match",
            "mean_tool_call_count",
            "std_tool_call_count",
            "notes",
        ],
        [
            {
                "benchmark_id": "failure_recovery_taxonomy_v2_repeated",
                "source_benchmark_id": "failure_recovery_taxonomy_v2",
                "execution_mode": "mock",
                "planner_baseline": "full_copilot",
                "repeat_count": "3",
                "task_count": "9",
                "mean_task_success_rate": "0.4444444444444444",
                "std_task_success_rate": "0.0",
                "mean_repair_success_rate": "0.6666666666666666",
                "std_repair_success_rate": "0.0",
                "false_success_count": "0",
                "mean_verifier_expectation_match": "1.0",
                "std_verifier_expectation_match": "0.0",
                "mean_tool_call_count": "27.666666666666668",
                "std_tool_call_count": "0.0",
                "notes": "repeated",
            }
        ],
    )
    _write_csv(
        input_dir / "throughput_table.csv",
        [
            "benchmark_family",
            "dataset",
            "benchmark_id",
            "execution_mode",
            "planner_baseline",
            "task_count",
            "generated_candidate_count",
            "valid_candidate_count",
            "docking_success_count",
            "mean_total_elapsed_sec",
            "seconds_per_task",
            "valid_candidates_per_sec",
            "notes",
        ],
        [
            {
                "benchmark_family": "crossdocked_generation",
                "dataset": "CrossDocked2020",
                "benchmark_id": "crossdocked",
                "execution_mode": "real",
                "planner_baseline": "rule_based_planner",
                "task_count": "3",
                "generated_candidate_count": "15",
                "valid_candidate_count": "15",
                "docking_success_count": "",
                "mean_total_elapsed_sec": "196.393",
                "seconds_per_task": "65.464",
                "valid_candidates_per_sec": "0.076377",
                "notes": "generation",
            }
        ],
    )
    _write_csv(
        input_dir / "generation_quality_table.csv",
        [
            "benchmark_family",
            "dataset",
            "benchmark_id",
            "execution_mode",
            "planner_baseline",
            "task_count",
            "generated_candidate_count",
            "valid_candidate_count",
            "valid_candidate_rate",
            "unique_smiles_count",
            "unique_smiles_rate",
            "best_scscore",
            "max_toxicity_score",
            "task_success_rate",
            "verifier_expectation_match",
            "notes",
        ],
        [
            {
                "benchmark_family": "crossdocked_generation",
                "dataset": "CrossDocked2020",
                "benchmark_id": "crossdocked",
                "execution_mode": "real",
                "planner_baseline": "rule_based_planner",
                "task_count": "3",
                "generated_candidate_count": "15",
                "valid_candidate_count": "15",
                "valid_candidate_rate": "1.0",
                "unique_smiles_count": "13",
                "unique_smiles_rate": "0.8666666666666667",
                "best_scscore": "3.5927797830672286",
                "max_toxicity_score": "0.1571",
                "task_success_rate": "1.0",
                "verifier_expectation_match": "1.0",
                "notes": "quality",
            }
        ],
    )
    _write_csv(
        input_dir / "generation_scale_table.csv",
        [
            "benchmark_id",
            "dataset",
            "execution_mode",
            "planner_baseline",
            "task_count",
            "generated_candidate_count",
            "valid_candidate_count",
            "valid_candidate_rate",
            "unique_smiles_count",
            "unique_smiles_rate",
            "mean_total_elapsed_sec",
            "seconds_per_task",
            "valid_candidates_per_sec",
            "notes",
        ],
        [
            {
                "benchmark_id": "crossdocked_rxnflow_candidates5_targets15",
                "dataset": "CrossDocked2020",
                "execution_mode": "real",
                "planner_baseline": "rule_based_planner",
                "task_count": "15",
                "generated_candidate_count": "75",
                "valid_candidate_count": "75",
                "valid_candidate_rate": "1.0",
                "unique_smiles_count": "21",
                "unique_smiles_rate": "0.28",
                "mean_total_elapsed_sec": "91.5",
                "seconds_per_task": "91.5",
                "valid_candidates_per_sec": "0.0546",
                "notes": "scale",
            }
        ],
    )
    _write_csv(
        input_dir / "crossdocked_multiseed_table.csv",
        [
            "benchmark_id",
            "dataset",
            "seed_count",
            "seeds",
            "total_target_runs",
            "total_candidates",
            "mean_task_success_rate",
            "std_task_success_rate",
            "mean_valid_candidate_rate",
            "std_valid_candidate_rate",
            "mean_unique_smiles_rate",
            "std_unique_smiles_rate",
            "mean_sa_score_coverage",
            "mean_sa_score_pass_rate",
            "mean_rdkit_property_coverage",
            "mean_qed",
            "mean_seconds_per_task",
            "std_seconds_per_task",
            "false_success_count",
            "notes",
        ],
        [
            {
                "benchmark_id": "crossdocked_rxnflow_candidates5_targets30_multiseed_v1",
                "dataset": "CrossDocked2020",
                "seed_count": "3",
                "seeds": "1,2,3",
                "total_target_runs": "90",
                "total_candidates": "450",
                "mean_task_success_rate": "1.0",
                "std_task_success_rate": "0.0",
                "mean_valid_candidate_rate": "1.0",
                "std_valid_candidate_rate": "0.0",
                "mean_unique_smiles_rate": "0.2",
                "std_unique_smiles_rate": "0.03",
                "mean_sa_score_coverage": "1.0",
                "mean_sa_score_pass_rate": "0.95",
                "mean_rdkit_property_coverage": "1.0",
                "mean_qed": "0.73",
                "mean_seconds_per_task": "90.0",
                "std_seconds_per_task": "10.0",
                "false_success_count": "0",
                "notes": "multiseed",
            }
        ],
    )
    _write_csv(
        input_dir / "tool_exposure_table.csv",
        [
            "benchmark_id",
            "dataset",
            "execution_mode",
            "planner_baseline",
            "task_count",
            "task_success_rate",
            "planner_tool_precision",
            "planner_tool_recall",
            "planner_tool_f1",
            "mean_selected_tool_count",
            "mean_extra_tool_count",
            "mean_tool_sequence_length",
            "mean_tool_call_count",
            "tool_call_failure_rate",
            "notes",
        ],
        [
            {
                "benchmark_id": "crossdocked_rxnflow_candidates5_targets15",
                "dataset": "CrossDocked2020_mocked_execution",
                "execution_mode": "mock",
                "planner_baseline": "all_tool_agent",
                "task_count": "15",
                "task_success_rate": "1.0",
                "planner_tool_precision": "0.3076923076923076",
                "planner_tool_recall": "1.0",
                "planner_tool_f1": "0.4705882352941177",
                "mean_selected_tool_count": "13.0",
                "mean_extra_tool_count": "9.0",
                "mean_tool_sequence_length": "4.0",
                "mean_tool_call_count": "5.0",
                "tool_call_failure_rate": "0.0",
                "notes": "all tools",
            }
        ],
    )
    _write_csv(
        input_dir / "failure_taxonomy_table.csv",
        [
            "failure_scenario",
            "task_type",
            "injected_tools",
            "expected_success",
            "rule_success",
            "full_success",
            "full_repair_executed",
            "full_repair_success",
            "repair_actions",
            "full_extra_tools",
            "rule_verifier_match",
            "full_verifier_match",
            "interpretation",
        ],
        [
            {
                "failure_scenario": "denovo_error_then_retry",
                "task_type": "failure_recovery",
                "injected_tools": "reinvent4_denovo",
                "expected_success": "True",
                "rule_success": "False",
                "full_success": "True",
                "full_repair_executed": "True",
                "full_repair_success": "True",
                "repair_actions": "retry:reinvent4_denovo",
                "full_extra_tools": "",
                "rule_verifier_match": "False",
                "full_verifier_match": "True",
                "interpretation": "Recovered by verifier-triggered retry without exposing extra tools.",
            }
        ],
    )
    _write_csv(
        input_dir / "ablation_table.csv",
        [
            "planner_baseline",
            "tool_exposure_benchmark_id",
            "robustness_benchmark_id",
            "mean_selected_tool_count",
            "mean_extra_tool_count",
            "planner_tool_precision",
            "planner_tool_recall",
            "robust_task_count",
            "robust_task_success_count",
            "robust_task_success_rate",
            "repair_attempt_count",
            "repair_success_count",
            "repair_success_rate",
            "verifier_expectation_match",
            "interpretation",
        ],
        [
            {
                "planner_baseline": "rule_based_planner",
                "tool_exposure_benchmark_id": "crossdocked_rxnflow_candidates5_targets15",
                "robustness_benchmark_id": "failure_recovery_taxonomy_v2",
                "mean_selected_tool_count": "4.0",
                "mean_extra_tool_count": "0.0",
                "planner_tool_precision": "1.0",
                "planner_tool_recall": "1.0",
                "robust_task_count": "9",
                "robust_task_success_count": "1",
                "robust_task_success_rate": "0.1111111111111111",
                "repair_attempt_count": "0",
                "repair_success_count": "0",
                "repair_success_rate": "0.0",
                "verifier_expectation_match": "0.6666666666666666",
                "interpretation": "same tool budget",
            },
            {
                "planner_baseline": "full_copilot",
                "tool_exposure_benchmark_id": "crossdocked_rxnflow_candidates5_targets15",
                "robustness_benchmark_id": "failure_recovery_taxonomy_v2",
                "mean_selected_tool_count": "4.0",
                "mean_extra_tool_count": "0.0",
                "planner_tool_precision": "1.0",
                "planner_tool_recall": "1.0",
                "robust_task_count": "9",
                "robust_task_success_count": "4",
                "robust_task_success_rate": "0.4444444444444444",
                "repair_attempt_count": "6",
                "repair_success_count": "4",
                "repair_success_rate": "0.6666666666666666",
                "verifier_expectation_match": "1.0",
                "interpretation": "verifier repair",
            },
        ],
    )
    _write_csv(
        input_dir / "repair_ablation_table.csv",
        [
            "benchmark_id",
            "execution_mode",
            "planner_baseline",
            "task_count",
            "task_success_count",
            "task_success_rate",
            "verifier_expectation_match",
            "initial_selected_tool_count",
            "mean_tool_call_count",
            "repair_attempt_count",
            "repair_success_count",
            "repair_success_rate",
            "false_success_count",
            "interpretation",
        ],
        [
            {
                "benchmark_id": "failure_recovery_taxonomy_v2",
                "execution_mode": "mock",
                "planner_baseline": "scheduled_fallback_no_verifier",
                "task_count": "9",
                "task_success_count": "3",
                "task_success_rate": "0.3333333333333333",
                "verifier_expectation_match": "0.8888888888888888",
                "initial_selected_tool_count": "3.0",
                "mean_tool_call_count": "12.0",
                "repair_attempt_count": "4",
                "repair_success_count": "3",
                "repair_success_rate": "0.75",
                "false_success_count": "0",
                "interpretation": "scheduled",
            }
            ],
        )
    _write_csv(
        input_dir / "repair_ablation_repeated_table.csv",
        [
            "benchmark_id",
            "source_benchmark_id",
            "execution_mode",
            "planner_baseline",
            "repeat_count",
            "task_count",
            "mean_task_success_rate",
            "std_task_success_rate",
            "mean_repair_success_rate",
            "std_repair_success_rate",
            "false_success_count",
            "mean_verifier_expectation_match",
            "std_verifier_expectation_match",
            "mean_tool_call_count",
            "std_tool_call_count",
            "notes",
        ],
        [
            {
                "benchmark_id": "failure_recovery_ambiguous_evidence_real_or_injected_v2_repeated",
                "source_benchmark_id": "failure_recovery_ambiguous_evidence_real_or_injected_v2",
                "execution_mode": "real",
                "planner_baseline": "full_copilot",
                "repeat_count": "3",
                "task_count": "12",
                "mean_task_success_rate": "1.0",
                "std_task_success_rate": "0.0",
                "mean_repair_success_rate": "1.0",
                "std_repair_success_rate": "0.0",
                "false_success_count": "0",
                "mean_verifier_expectation_match": "1.0",
                "std_verifier_expectation_match": "0.0",
                "mean_tool_call_count": "8.0",
                "std_tool_call_count": "0.0",
                "notes": "repeated ambiguous evidence",
            }
        ],
    )
    _write_csv(
        input_dir / "ambiguous_failure_modes_table.csv",
        [
            "evidence_family",
            "failure_scenario",
            "ambiguity_type",
            "missing_evidence_mode",
            "verifier_check",
            "repair_tool",
            "task_count",
            "repair_attempt_count",
            "real_retry_success_count",
            "real_retry_success_rate",
            "task_success_count",
            "task_success_rate",
            "false_success_count",
            "real_evidence",
            "example_task_id",
        ],
        [
            {
                "evidence_family": "synthesizability",
                "failure_scenario": "scscore_empty_results_then_real_retry",
                "ambiguity_type": "nominal_tool_success_missing_evaluator_evidence",
                "missing_evidence_mode": "empty_results",
                "verifier_check": "passes_synthesizability",
                "repair_tool": "scscore",
                "task_count": "1",
                "repair_attempt_count": "1",
                "real_retry_success_count": "1",
                "real_retry_success_rate": "1.0",
                "task_success_count": "1",
                "task_success_rate": "1.0",
                "false_success_count": "0",
                "real_evidence": "SCScore=1.000",
                "example_task_id": "ambiguous_v2_scscore_empty_results_then_real_retry",
            }
        ],
    )
    _write_csv(
        input_dir / "ambiguous_failure_modes_repeated_table.csv",
        [
            "evidence_family",
            "failure_scenario",
            "ambiguity_type",
            "missing_evidence_mode",
            "verifier_check",
            "repair_tool",
            "repeat_count",
            "task_count",
            "repair_attempt_count",
            "real_retry_success_count",
            "real_retry_success_rate",
            "task_success_count",
            "task_success_rate",
            "false_success_count",
            "real_evidence",
            "example_task_id",
        ],
        [
            {
                "evidence_family": "synthesizability",
                "failure_scenario": "scscore_empty_results_then_real_retry",
                "ambiguity_type": "nominal_tool_success_missing_evaluator_evidence",
                "missing_evidence_mode": "empty_results",
                "verifier_check": "passes_synthesizability",
                "repair_tool": "scscore",
                "repeat_count": "3",
                "task_count": "3",
                "repair_attempt_count": "3",
                "real_retry_success_count": "3",
                "real_retry_success_rate": "1.0",
                "task_success_count": "3",
                "task_success_rate": "1.0",
                "false_success_count": "0",
                "real_evidence": "SCScore=1.000",
                "example_task_id": "ambiguous_v2_scscore_empty_results_then_real_retry",
            }
        ],
    )
    _write_csv(
        input_dir / "task_generalization_table.csv",
        [
            "task_type",
            "benchmark_id",
            "dataset",
            "execution_mode",
            "planner_baseline",
            "task_count",
            "task_success_rate",
            "valid_candidate_rate",
            "verifier_expectation_match",
            "mean_elapsed_sec",
            "tools",
            "notes",
        ],
        [
            {
                "task_type": "hit_to_lead_optimization",
                "benchmark_id": "task_generalization_v1",
                "dataset": "task_generalization_mock_v1",
                "execution_mode": "mock",
                "planner_baseline": "rule_based_planner",
                "task_count": "3",
                "task_success_rate": "1.0",
                "valid_candidate_rate": "1.0",
                "verifier_expectation_match": "1.0",
                "mean_elapsed_sec": "0.01",
                "tools": "reinvent4_mol2mol, scscore, toxicity",
                "notes": "task generalization",
            }
        ],
    )
    _write_csv(
        input_dir / "tool_admission_table.csv",
        [
            "tool_name",
            "tool_role",
            "independent_evidence",
            "failure_modes_structured",
            "runtime_cost",
            "environment_risk",
            "paper_claim_supported",
            "decision",
        ],
        [
            {
                "tool_name": "rdkit_property_verifier",
                "tool_role": "verifier",
                "independent_evidence": "QED and PAINS",
                "failure_modes_structured": "True",
                "runtime_cost": "low",
                "environment_risk": "low",
                "paper_claim_supported": "verifier evidence",
                "decision": "p0_admitted_evaluated",
            }
        ],
    )
    _write_csv(
        input_dir / "verifier_evidence_table.csv",
        [
            "evidence_family",
            "dataset",
            "evidence_type",
            "task_count",
            "candidate_count",
            "evaluable_candidate_count",
            "evidence_count",
            "coverage",
            "pass_count",
            "pass_rate",
            "best_sa_score",
            "mean_sa_score",
            "pose_artifact_count",
            "status",
            "notes",
        ],
        [
            {
                "evidence_family": "crossdocked_generation",
                "dataset": "CrossDocked2020",
                "evidence_type": "sa_score",
                "task_count": "15",
                "candidate_count": "75",
                "evaluable_candidate_count": "75",
                "evidence_count": "75",
                "coverage": "1.0",
                "pass_count": "75",
                "pass_rate": "1.0",
                "best_sa_score": "2.12345",
                "mean_sa_score": "3.45678",
                "pose_artifact_count": "",
                "status": "available",
                "notes": "sa evidence",
            },
            {
                "evidence_family": "litpcba_docking",
                "dataset": "LIT-PCBA",
                "evidence_type": "posebusters",
                "task_count": "15",
                "candidate_count": "15",
                "evaluable_candidate_count": "15",
                "evidence_count": "15",
                "coverage": "1.0",
                "pass_count": "0",
                "pass_rate": "0.0",
                "best_sa_score": "",
                "mean_sa_score": "",
                "pose_artifact_count": "15",
                "status": "available",
                "notes": "posebusters evaluated",
            },
            {
                "evidence_family": "crossdocked_generation",
                "dataset": "CrossDocked2020",
                "evidence_type": "rdkit_property_verifier",
                "task_count": "15",
                "candidate_count": "75",
                "evaluable_candidate_count": "75",
                "evidence_count": "75",
                "coverage": "1.0",
                "pass_count": "70",
                "pass_rate": "0.9333333333333333",
                "best_sa_score": "",
                "mean_sa_score": "",
                "pose_artifact_count": "",
                "status": "available",
                "notes": "rdkit property evidence",
            },
        ],
    )
    _write_csv(
        input_dir / "property_verifier_table.csv",
        [
            "evidence_family",
            "dataset",
            "task_count",
            "candidate_count",
            "valid_smiles_count",
            "property_coverage",
            "mean_qed",
            "mean_logp",
            "mean_molwt",
            "lipinski_pass_count",
            "lipinski_pass_rate",
            "pains_flag_count",
            "pains_flag_rate",
            "brenk_flag_count",
            "brenk_flag_rate",
            "status",
            "notes",
        ],
        [
            {
                "evidence_family": "crossdocked_generation",
                "dataset": "CrossDocked2020",
                "task_count": "15",
                "candidate_count": "75",
                "valid_smiles_count": "75",
                "property_coverage": "1.0",
                "mean_qed": "0.54321",
                "mean_logp": "1.234",
                "mean_molwt": "280.0",
                "lipinski_pass_count": "70",
                "lipinski_pass_rate": "0.9333333333333333",
                "pains_flag_count": "2",
                "pains_flag_rate": "0.02666666666666667",
                "brenk_flag_count": "5",
                "brenk_flag_rate": "0.06666666666666667",
                "status": "available",
                "notes": "property evidence",
            }
        ],
    )
    _write_csv(
        input_dir / "posebusters_top_failures_table.csv",
        [
            "dataset",
            "evidence_family",
            "check_name",
            "category",
            "evaluated_count",
            "fail_count",
            "fail_rate",
            "example_task_ids",
        ],
        [
            {
                "dataset": "LIT-PCBA",
                "evidence_family": "litpcba_docking",
                "check_name": "sanitization",
                "category": "molecule_validity",
                "evaluated_count": "15",
                "fail_count": "15",
                "fail_rate": "1.0",
                "example_task_ids": "litpcba_docking_000_ADRB2, litpcba_docking_001_ALDH1",
            }
        ],
    )
    _write_csv(
        input_dir / "posebusters_failure_modes_table.csv",
        [
            "dataset",
            "evidence_family",
            "check_name",
            "category",
            "pose_count",
            "evaluated_count",
            "pass_count",
            "fail_count",
            "missing_count",
            "fail_rate",
            "example_task_ids",
            "interpretation",
        ],
        [
            {
                "dataset": "LIT-PCBA",
                "evidence_family": "litpcba_docking",
                "check_name": "sanitization",
                "category": "molecule_validity",
                "pose_count": "15",
                "evaluated_count": "15",
                "pass_count": "0",
                "fail_count": "15",
                "missing_count": "0",
                "fail_rate": "1.0",
                "example_task_ids": "litpcba_docking_000_ADRB2, litpcba_docking_001_ALDH1",
                "interpretation": "The predicted ligand cannot be sanitized by RDKit.",
            }
        ],
    )
    _write_csv(
        input_dir / "pdbbind_prep_gate_table.csv",
        [
            "dataset",
            "readiness_status",
            "ready_target_count",
            "index_file_count",
            "receptor_prep_target_count",
            "prep_success_count",
            "prep_success_rate",
            "template_required_count",
            "runtime_error_count",
            "timeout_count",
            "prepared_pilot_task_count",
            "real_pilot_task_count",
            "real_pilot_success_rate",
            "best_docking_score",
            "mean_elapsed_sec",
            "false_success_count",
            "gate_status",
            "evidence_role",
            "notes",
        ],
        [
            {
                "dataset": "PDBbind v2020 refined",
                "readiness_status": "ready",
                "ready_target_count": "10",
                "index_file_count": "12",
                "receptor_prep_target_count": "10",
                "prep_success_count": "0",
                "prep_success_rate": "0.0",
                "template_required_count": "9",
                "runtime_error_count": "0",
                "timeout_count": "1",
                "prepared_pilot_task_count": "0",
                "real_pilot_task_count": "",
                "real_pilot_success_rate": "",
                "best_docking_score": "",
                "mean_elapsed_sec": "",
                "false_success_count": "",
                "gate_status": "execution_blocked_no_prepared_receptors",
                "evidence_role": "appendix_gate_not_main_claim",
                "notes": "PDBbind local data readiness is confirmed, but receptor preparation gate prevents false success.",
            }
        ],
    )
    _write_csv(
        input_dir / "llm_router_baseline_table.csv",
        [
            "benchmark_id",
            "dataset",
            "router_mode",
            "task_count",
            "valid_json_count",
            "valid_schema_count",
            "invalid_json_rate",
            "invalid_schema_rate",
            "hallucinated_tool_count",
            "hallucinated_tool_rate",
            "missing_required_input_count",
            "missing_required_input_rate",
            "tool_precision",
            "tool_recall",
            "tool_f1",
            "workflow_order_match_rate",
            "mean_selected_tool_count",
            "mean_extra_tool_count",
            "notes",
        ],
        [
            {
                "benchmark_id": "llm_as_router_planning_v1",
                "dataset": "mixed_molecular_planning_tasks",
                "router_mode": "heuristic",
                "task_count": "3",
                "valid_json_count": "3",
                "valid_schema_count": "3",
                "invalid_json_rate": "0.0",
                "invalid_schema_rate": "0.0",
                "hallucinated_tool_count": "0",
                "hallucinated_tool_rate": "0.0",
                "missing_required_input_count": "0",
                "missing_required_input_rate": "0.0",
                "tool_precision": "1.0",
                "tool_recall": "1.0",
                "tool_f1": "1.0",
                "workflow_order_match_rate": "1.0",
                "mean_selected_tool_count": "4.0",
                "mean_extra_tool_count": "0.0",
                "notes": "planning only",
            }
        ],
    )
    _write_csv(
        input_dir / "natural_failure_audit_table.csv",
        [
            "dataset",
            "benchmark_id",
            "task_type",
            "tool_name",
            "failure_family",
            "failure_count",
            "affected_task_count",
            "example_task_ids",
            "source_kind",
            "source_path",
            "notes",
        ],
        [
            {
                "dataset": "PDBbind+",
                "benchmark_id": "pdbbindplus_v2020r1_prepared_pilot_v3",
                "task_type": "docking_evaluation",
                "tool_name": "vina",
                "failure_family": "docking_runtime_failure",
                "failure_count": "2",
                "affected_task_count": "2",
                "example_task_ids": "017_1a3e,025_1a94",
                "source_kind": "trace_tool_call",
                "source_path": "logs/example.jsonl",
                "notes": "vina failed",
            }
        ],
    )
    _write_csv(
        input_dir / "evidence_audit_table.csv",
        [
            "claim_id",
            "claim",
            "claim_strength",
            "evidence_role",
            "table_name",
            "table_label",
            "benchmark_id",
            "dataset",
            "execution_mode",
            "evidence_type",
            "is_real_result",
            "is_controlled",
            "is_supporting",
            "row_count",
            "result_path",
            "notes",
        ],
        [
            {
                "claim_id": "C1",
                "claim": "Verifier-guided repair improves robustness.",
                "claim_strength": "strong",
                "evidence_role": "main",
                "table_name": "robustness_repeated",
                "table_label": "tab:robustness-repeated",
                "benchmark_id": "failure_recovery_taxonomy_v2_repeated",
                "dataset": "controlled_tool_failure_injection",
                "execution_mode": "real",
                "evidence_type": "repeated_real_controlled",
                "is_real_result": "true",
                "is_controlled": "true",
                "is_supporting": "false",
                "row_count": "2",
                "result_path": "logs/robustness_repeated_table.json",
                "notes": "audit",
            }
        ],
    )
    _write_csv(
        input_dir / "statistical_summary_table.csv",
        [
            "claim_id",
            "benchmark_id",
            "dataset",
            "planner_baseline",
            "metric",
            "n",
            "estimate",
            "std",
            "ci95_low",
            "ci95_high",
            "false_success_count",
            "source_table",
            "interpretation",
        ],
        [
            {
                "claim_id": "C1",
                "benchmark_id": "failure_recovery_taxonomy_v2_repeated",
                "dataset": "controlled_tool_failure_injection",
                "planner_baseline": "full_copilot",
                "metric": "task_success_rate",
                "n": "27",
                "estimate": "0.4444444444",
                "std": "0.0",
                "ci95_low": "0.276",
                "ci95_high": "0.628",
                "false_success_count": "0",
                "source_table": "robustness_repeated",
                "interpretation": "summary",
            }
        ],
    )

    payload = export_latex_tables(input_dir=input_dir, output_dir=output_dir)

    assert payload["table_count"] == 24
    assert (output_dir / "robustness_table.tex").exists()
    assert (output_dir / "robustness_repeated_table.tex").exists()
    assert (output_dir / "throughput_table.tex").exists()
    assert (output_dir / "generation_quality_table.tex").exists()
    assert (output_dir / "generation_scale_table.tex").exists()
    assert (output_dir / "crossdocked_multiseed_table.tex").exists()
    assert (output_dir / "tool_exposure_table.tex").exists()
    assert (output_dir / "failure_taxonomy_table.tex").exists()
    assert (output_dir / "ablation_table.tex").exists()
    assert (output_dir / "repair_ablation_table.tex").exists()
    assert (output_dir / "repair_ablation_repeated_table.tex").exists()
    assert (output_dir / "ambiguous_failure_modes_table.tex").exists()
    assert (output_dir / "ambiguous_failure_modes_repeated_table.tex").exists()
    assert (output_dir / "task_generalization_table.tex").exists()
    assert (output_dir / "tool_admission_table.tex").exists()
    assert (output_dir / "verifier_evidence_table.tex").exists()
    assert (output_dir / "property_verifier_table.tex").exists()
    assert (output_dir / "posebusters_top_failures_table.tex").exists()
    assert (output_dir / "posebusters_failure_modes_table.tex").exists()
    assert (output_dir / "pdbbind_prep_gate_table.tex").exists()
    assert (output_dir / "llm_router_baseline_table.tex").exists()
    assert (output_dir / "natural_failure_audit_table.tex").exists()
    assert (output_dir / "evidence_audit_table.tex").exists()
    assert (output_dir / "statistical_summary_table.tex").exists()
    assert (output_dir / "latex_tables.summary.json").exists()

    robustness_tex = (output_dir / "robustness_table.tex").read_text(encoding="utf-8")
    assert r"\begin{table}[t]" in robustness_tex
    assert "rule-based planner" in robustness_tex
    assert r"rule-based planner & 3 & 0/3 & 0.0\% & 33.3\% & -- & --" in robustness_tex
    assert "2/3" in robustness_tex
    assert r"66.7\%" in robustness_tex
    assert r"\label{tab:robustness}" in robustness_tex

    robustness_repeated_tex = (output_dir / "robustness_repeated_table.tex").read_text(encoding="utf-8")
    assert "EGVR-Agent" in robustness_repeated_tex
    assert "44.4\\% +/- 0.0\\%" in robustness_repeated_tex
    assert r"\label{tab:robustness-repeated}" in robustness_repeated_tex

    throughput_tex = (output_dir / "throughput_table.tex").read_text(encoding="utf-8")
    assert "CrossDocked gen." in throughput_tex
    assert "65.5" in throughput_tex
    assert "0.076" in throughput_tex

    generation_tex = (output_dir / "generation_quality_table.tex").read_text(encoding="utf-8")
    assert "86.7" in generation_tex
    assert "3.593" in generation_tex

    generation_scale_tex = (output_dir / "generation_scale_table.tex").read_text(encoding="utf-8")
    assert "crossdocked\\_rxnflow\\_candidates5\\_targets15" in generation_scale_tex
    assert "28.0\\%" in generation_scale_tex
    assert r"\label{tab:generation-scale}" in generation_scale_tex

    multiseed_tex = (output_dir / "crossdocked_multiseed_table.tex").read_text(encoding="utf-8")
    assert "crossdocked\\_rxnflow\\_candidates5\\_targets30\\_multiseed\\_v1" in multiseed_tex
    assert "3 & 90 & 450" in multiseed_tex
    assert "20.0\\% +/- 3.0\\%" in multiseed_tex
    assert r"\label{tab:crossdocked-multiseed}" in multiseed_tex

    exposure_tex = (output_dir / "tool_exposure_table.tex").read_text(encoding="utf-8")
    assert "all-tool agent" in exposure_tex
    assert r"30.8\%" in exposure_tex
    assert r"\label{tab:tool-exposure}" in exposure_tex

    taxonomy_tex = (output_dir / "failure_taxonomy_table.tex").read_text(encoding="utf-8")
    assert "denovo error then retry" in taxonomy_tex
    assert "reinvent4\\_denovo" in taxonomy_tex
    assert "Recovered by verifier-triggered retry" in taxonomy_tex
    assert r"\label{tab:failure-taxonomy}" in taxonomy_tex

    ablation_tex = (output_dir / "ablation_table.tex").read_text(encoding="utf-8")
    assert "rule-based planner & 4.0 & 0.0 & 100.0\\% & 100.0\\% & 1/9" in ablation_tex
    assert "EGVR-Agent & 4.0 & 0.0 & 100.0\\% & 100.0\\% & 4/9" in ablation_tex
    assert "4/6" in ablation_tex
    assert r"\label{tab:ablation-tool-repair}" in ablation_tex

    repair_ablation_tex = (output_dir / "repair_ablation_table.tex").read_text(encoding="utf-8")
    assert "scheduled fallback" in repair_ablation_tex
    assert "3/9" in repair_ablation_tex
    assert r"\label{tab:repair-ablation}" in repair_ablation_tex

    repair_ablation_repeated_tex = (output_dir / "repair_ablation_repeated_table.tex").read_text(encoding="utf-8")
    assert "EGVR-Agent" in repair_ablation_repeated_tex
    assert "100.0\\% +/- 0.0\\%" in repair_ablation_repeated_tex
    assert r"\label{tab:repair-ablation-repeated}" in repair_ablation_repeated_tex

    ambiguous_tex = (output_dir / "ambiguous_failure_modes_table.tex").read_text(encoding="utf-8")
    assert "SCScore" in ambiguous_tex
    assert "empty results" in ambiguous_tex
    assert "passes synthesizability" in ambiguous_tex
    assert "1/1" in ambiguous_tex
    assert r"\label{tab:ambiguous-failure-modes}" in ambiguous_tex

    ambiguous_repeated_tex = (output_dir / "ambiguous_failure_modes_repeated_table.tex").read_text(encoding="utf-8")
    assert "SCScore" in ambiguous_repeated_tex
    assert "3/3" in ambiguous_repeated_tex
    assert r"\label{tab:ambiguous-failure-modes-repeated}" in ambiguous_repeated_tex

    task_generalization_tex = (output_dir / "task_generalization_table.tex").read_text(encoding="utf-8")
    assert "hit to lead optimization" in task_generalization_tex
    assert "reinvent4\\_mol2mol" in task_generalization_tex
    assert r"\label{tab:task-generalization}" in task_generalization_tex

    tool_admission_tex = (output_dir / "tool_admission_table.tex").read_text(encoding="utf-8")
    assert "rdkit\\_property\\_verifier" in tool_admission_tex
    assert "p0\\_admitted\\_evaluated" in tool_admission_tex
    assert r"\label{tab:tool-admission}" in tool_admission_tex

    verifier_tex = (output_dir / "verifier_evidence_table.tex").read_text(encoding="utf-8")
    assert "SA\\_Score" in verifier_tex
    assert "PoseBusters" in verifier_tex
    assert "RDKit properties" in verifier_tex
    assert "100.0\\%" in verifier_tex
    assert "0.0\\%" in verifier_tex
    assert "available" in verifier_tex
    assert r"\label{tab:verifier-evidence}" in verifier_tex

    property_tex = (output_dir / "property_verifier_table.tex").read_text(encoding="utf-8")
    assert "Mean QED" in property_tex
    assert "93.3\\%" in property_tex
    assert r"\label{tab:property-verifier}" in property_tex

    top_failures_tex = (output_dir / "posebusters_top_failures_table.tex").read_text(encoding="utf-8")
    assert "LIT-PCBA" in top_failures_tex
    assert "sanitization" in top_failures_tex
    assert "molecule validity" in top_failures_tex
    assert "15 & 15 & 100.0\\%" in top_failures_tex
    assert "000\\_ADRB2" in top_failures_tex
    assert r"\label{tab:posebusters-top-failures}" in top_failures_tex

    posebusters_tex = (output_dir / "posebusters_failure_modes_table.tex").read_text(encoding="utf-8")
    assert "LIT-PCBA" in posebusters_tex
    assert "sanitization" in posebusters_tex
    assert "molecule validity" in posebusters_tex
    assert "15 & 15 & 100.0\\%" in posebusters_tex
    assert "000\\_ADRB2" in posebusters_tex
    assert r"\label{tab:posebusters-failure-modes}" in posebusters_tex

    pdbbind_tex = (output_dir / "pdbbind_prep_gate_table.tex").read_text(encoding="utf-8")
    assert "PDBbind v2020 refined" in pdbbind_tex
    assert "ready & 10 & 12 & 0/10 & 9 & 0 & 1" in pdbbind_tex
    assert "-- & -- & -- & --" in pdbbind_tex
    assert "execution blocked no prepared receptors" in pdbbind_tex
    assert r"\label{tab:pdbbind-prep-gate}" in pdbbind_tex

    llm_router_tex = (output_dir / "llm_router_baseline_table.tex").read_text(encoding="utf-8")
    assert "heuristic" in llm_router_tex
    assert r"\label{tab:llm-router-baseline}" in llm_router_tex

    natural_failure_tex = (output_dir / "natural_failure_audit_table.tex").read_text(encoding="utf-8")
    assert "docking runtime failure" in natural_failure_tex
    assert r"\label{tab:natural-failure-audit}" in natural_failure_tex

    evidence_audit_tex = (output_dir / "evidence_audit_table.tex").read_text(encoding="utf-8")
    assert "tab:robustness-repeated" in evidence_audit_tex
    assert r"\label{tab:evidence-audit}" in evidence_audit_tex

    statistical_tex = (output_dir / "statistical_summary_table.tex").read_text(encoding="utf-8")
    assert "44.4\\%" in statistical_tex
    assert r"\label{tab:statistical-summary}" in statistical_tex

    manifest = json.loads((output_dir / "latex_tables.summary.json").read_text(encoding="utf-8"))
    assert manifest["tables"]["robustness"]["row_count"] == 2
    assert manifest["tables"]["robustness_repeated"]["row_count"] == 1
    assert manifest["tables"]["generation_scale"]["row_count"] == 1
    assert manifest["tables"]["crossdocked_multiseed"]["row_count"] == 1
    assert manifest["tables"]["tool_exposure"]["row_count"] == 1
    assert manifest["tables"]["failure_taxonomy"]["row_count"] == 1
    assert manifest["tables"]["ablation"]["row_count"] == 2
    assert manifest["tables"]["repair_ablation"]["row_count"] == 1
    assert manifest["tables"]["repair_ablation_repeated"]["row_count"] == 1
    assert manifest["tables"]["ambiguous_failure_modes"]["row_count"] == 1
    assert manifest["tables"]["ambiguous_failure_modes_repeated"]["row_count"] == 1
    assert manifest["tables"]["task_generalization"]["row_count"] == 1
    assert manifest["tables"]["tool_admission"]["row_count"] == 1
    assert manifest["tables"]["verifier_evidence"]["row_count"] == 3
    assert manifest["tables"]["property_verifier"]["row_count"] == 1
    assert manifest["tables"]["posebusters_top_failures"]["row_count"] == 1
    assert manifest["tables"]["posebusters_failure_modes"]["row_count"] == 1
    assert manifest["tables"]["pdbbind_prep_gate"]["row_count"] == 1
    assert manifest["tables"]["llm_router_baseline"]["row_count"] == 1
    assert manifest["tables"]["natural_failure_audit"]["row_count"] == 1
    assert manifest["tables"]["evidence_audit"]["row_count"] == 1
    assert manifest["tables"]["statistical_summary"]["row_count"] == 1


def test_escape_latex_handles_common_special_characters():
    assert escape_latex("full_copilot & 100%") == r"full\_copilot \& 100\%"


def test_export_latex_tables_rejects_unknown_view(tmp_path):
    with pytest.raises(ValueError, match="Unsupported LaTeX table view"):
        export_latex_tables(input_dir=tmp_path, output_dir=tmp_path / "latex", table_names=["unknown"])


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
