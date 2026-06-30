from __future__ import annotations

import csv
import json

from CAi.toolkit.agent_planner.master_table_builder import (
    ABLATION_TABLE_COLUMNS,
    AMBIGUOUS_FAILURE_MODE_TABLE_COLUMNS,
    AMBIGUOUS_FAILURE_MODE_REPEATED_TABLE_COLUMNS,
    CROSSDOCKED_MULTISEED_TABLE_COLUMNS,
    GENERATION_QUALITY_TABLE_COLUMNS,
    GENERATION_SCALE_TABLE_COLUMNS,
    FAILURE_TAXONOMY_TABLE_COLUMNS,
    MASTER_TABLE_COLUMNS,
    PDBBIND_PREP_GATE_TABLE_COLUMNS,
    POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS,
    POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS,
    PROPERTY_VERIFIER_TABLE_COLUMNS,
    REPAIR_ABLATION_TABLE_COLUMNS,
    REPAIR_ABLATION_REPEATED_TABLE_COLUMNS,
    ROBUSTNESS_TABLE_COLUMNS,
    ROBUSTNESS_REPEATED_TABLE_COLUMNS,
    TASK_GENERALIZATION_TABLE_COLUMNS,
    THROUGHPUT_TABLE_COLUMNS,
    TOOL_EXPOSURE_TABLE_COLUMNS,
    TOOL_ADMISSION_TABLE_COLUMNS,
    VERIFIER_EVIDENCE_TABLE_COLUMNS,
    _llm_router_baseline_rows,
    build_master_baseline_table,
)


def test_master_table_builder_writes_rows_for_three_benchmark_families(tmp_path):
    crossdocked_record = tmp_path / "crossdocked.summary.json"
    crossdocked_record.write_text(
        json.dumps(
            {
                "benchmark_id": "crossdocked_rxnflow_candidates5_smoke_3",
                "dataset": "CrossDocked2020",
                "execution_mode": "real",
                "planner_baseline": "rule_based_planner",
                "result_path": "logs/crossdocked_result.json",
                "task_count": 2,
                "summary": {
                    "parser_accuracy": 1.0,
                    "planner_tool_coverage_rate": 1.0,
                    "planner_tool_precision": 1.0,
                    "planner_tool_recall": 1.0,
                    "planner_tool_f1": 1.0,
                    "task_success_rate": 1.0,
                    "total": 2,
                    "verifier_expectation_match": 1.0,
                },
                "global_candidate_summary": {
                    "generated_candidate_count": 10,
                    "valid_candidate_count": 10,
                    "unique_smiles_count_across_tasks": 8,
                },
                "per_target": [
                    {
                        "best_scscore": 3.5,
                        "max_toxicity_score": 0.11,
                        "tool_elapsed_sec": {"total": 100.0},
                    },
                    {
                        "best_scscore": 2.9,
                        "max_toxicity_score": 0.13,
                        "tool_elapsed_sec": {"total": 140.0},
                    },
                ],
                "notes": ["Smoke generation succeeded."],
            }
        ),
        encoding="utf-8",
    )

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    litpcba_result = logs_dir / "litpcba_vina_prepared_15_result.json"
    litpcba_result.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "expected_tools": ["vina"],
                        "selected_tools": ["vina"],
                        "task_success": True,
                        "metrics": {"best_docking_score": -7.5},
                    },
                    {
                        "expected_tools": ["vina"],
                        "selected_tools": ["vina"],
                        "task_success": True,
                        "metrics": {"best_docking_score": -10.5},
                    },
                ],
                "summary": {
                    "parser_accuracy": 1.0,
                    "planner_tool_recall": 1.0,
                    "task_success_rate": 1.0,
                    "total": 2,
                    "verifier_expectation_match": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    litpcba_elapsed_result = logs_dir / "litpcba_vina_prepared_15_elapsed_result.json"
    litpcba_elapsed_result.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "expected_tools": ["vina"],
                        "selected_tools": ["vina"],
                        "task_success": True,
                        "total_elapsed_sec": 11.0,
                        "tool_elapsed_sec": {"vina": 11.0},
                        "metrics": {"best_docking_score": -7.5},
                    },
                    {
                        "expected_tools": ["vina"],
                        "selected_tools": ["vina"],
                        "task_success": True,
                        "total_elapsed_sec": 14.0,
                        "tool_elapsed_sec": {"vina": 14.0},
                        "metrics": {"best_docking_score": -10.5},
                    },
                ],
                "summary": {
                    "mean_total_elapsed_sec": 12.5,
                    "median_total_elapsed_sec": 12.5,
                    "parser_accuracy": 1.0,
                    "planner_tool_recall": 1.0,
                    "task_success_rate": 1.0,
                    "total": 2,
                    "verifier_expectation_match": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    failure_suite = tmp_path / "failure.baseline_summary.json"
    failure_suite.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "planner_baseline": "rule_based_planner",
                        "result_path": "logs/failure.rule.json",
                        "task_success_count": 0,
                        "failed_task_count": 3,
                    },
                    {
                        "planner_baseline": "full_copilot",
                        "result_path": "logs/failure.full.json",
                        "task_success_count": 2,
                        "failed_task_count": 1,
                    },
                ],
                "results": {
                    "rule_based_planner": {
                        "summary": {
                            "parser_accuracy": 1.0,
                            "planner_tool_coverage_rate": 1.0,
                            "planner_tool_precision": 1.0,
                            "planner_tool_recall": 1.0,
                            "planner_tool_f1": 1.0,
                            "repair_attempt_count": 0,
                            "repair_success_count": 0,
                            "repair_success_rate": 0.0,
                            "task_success_rate": 0.0,
                            "total": 3,
                            "verifier_expectation_match": 1 / 3,
                        }
                    },
                    "full_copilot": {
                        "summary": {
                            "parser_accuracy": 1.0,
                            "planner_tool_coverage_rate": 1.0,
                            "planner_tool_precision": 1.0,
                            "planner_tool_recall": 1.0,
                            "planner_tool_f1": 1.0,
                            "repair_attempt_count": 3,
                            "repair_success_count": 2,
                            "repair_success_rate": 2 / 3,
                            "task_success_rate": 2 / 3,
                            "total": 3,
                            "verifier_expectation_match": 1.0,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    failure_record = tmp_path / "failure.real.summary.json"
    failure_record.write_text(
        json.dumps(
            {
                "benchmark_id": "failure_recovery_real_or_injected_v1",
                "execution_mode": "real",
                "summary_json": str(failure_suite),
                "task_count": 3,
                "baseline_results": [
                    {
                        "planner_baseline": "rule_based_planner",
                        "task_success_count": 0,
                        "failed_task_count": 3,
                        "task_success_rate": 0.0,
                        "repair_attempt_count": 0,
                        "repair_success_count": 0,
                        "repair_success_rate": 0.0,
                        "verifier_expectation_match": 1 / 3,
                    },
                    {
                        "planner_baseline": "full_copilot",
                        "task_success_count": 2,
                        "failed_task_count": 1,
                        "task_success_rate": 2 / 3,
                        "repair_attempt_count": 3,
                        "repair_success_count": 2,
                        "repair_success_rate": 2 / 3,
                        "verifier_expectation_match": 1.0,
                    },
                ],
                "full_copilot_case_results": [
                    {
                        "candidate_count": 10,
                        "valid_smiles_count": 10,
                        "unique_smiles_count": 10,
                        "best_scscore": 2.2,
                        "max_toxicity_score": 0.16,
                    },
                    {
                        "candidate_count": 2,
                        "valid_smiles_count": 2,
                        "unique_smiles_count": 2,
                        "best_scscore": 3.7,
                        "max_toxicity_score": 0.12,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    tool_exposure_summary = tmp_path / "tool_exposure.baseline_summary.json"
    tool_exposure_summary.write_text(
        json.dumps(
            {
                "benchmark_id": "crossdocked_rxnflow_candidates5_targets15",
                "execution_mode": "mock",
                "rows": [
                    {
                        "planner_baseline": "all_tool_agent",
                        "total": 15,
                        "task_success_count": 15,
                        "failed_task_count": 0,
                        "task_success_rate": 1.0,
                        "parser_accuracy": 1.0,
                        "planner_tool_coverage_rate": 1.0,
                        "planner_tool_precision": 0.4,
                        "planner_tool_recall": 1.0,
                        "planner_tool_f1": 4 / 7,
                        "mean_selected_tool_count": 10.0,
                        "mean_tool_sequence_length": 4.0,
                        "mean_extra_tool_count": 6.0,
                        "mean_tool_call_count": 5.0,
                        "failed_tool_call_count": 0,
                        "tool_call_failure_rate": 0.0,
                        "verifier_expectation_match": 1.0,
                        "result_path": "logs/tool_exposure.all.json",
                    },
                    {
                        "planner_baseline": "fixed_pipeline",
                        "total": 15,
                        "task_success_count": 15,
                        "failed_task_count": 0,
                        "task_success_rate": 1.0,
                        "parser_accuracy": 1.0,
                        "planner_tool_coverage_rate": 0.0,
                        "planner_tool_precision": 1.0,
                        "planner_tool_recall": 0.75,
                        "planner_tool_f1": 6 / 7,
                        "mean_selected_tool_count": 3.0,
                        "mean_tool_sequence_length": 3.0,
                        "mean_extra_tool_count": 0.0,
                        "mean_tool_call_count": 4.0,
                        "failed_tool_call_count": 0,
                        "tool_call_failure_rate": 0.0,
                        "verifier_expectation_match": 1.0,
                        "result_path": "logs/tool_exposure.fixed.json",
                    },
                    {
                        "planner_baseline": "rule_based_planner",
                        "total": 15,
                        "task_success_count": 15,
                        "failed_task_count": 0,
                        "task_success_rate": 1.0,
                        "parser_accuracy": 1.0,
                        "planner_tool_coverage_rate": 1.0,
                        "planner_tool_precision": 1.0,
                        "planner_tool_recall": 1.0,
                        "planner_tool_f1": 1.0,
                        "mean_selected_tool_count": 4.0,
                        "mean_tool_sequence_length": 4.0,
                        "mean_extra_tool_count": 0.0,
                        "mean_tool_call_count": 5.0,
                        "failed_tool_call_count": 0,
                        "tool_call_failure_rate": 0.0,
                        "verifier_expectation_match": 1.0,
                        "result_path": "logs/tool_exposure.rule.json",
                    },
                    {
                        "planner_baseline": "full_copilot",
                        "total": 15,
                        "task_success_count": 15,
                        "failed_task_count": 0,
                        "task_success_rate": 1.0,
                        "parser_accuracy": 1.0,
                        "planner_tool_coverage_rate": 1.0,
                        "planner_tool_precision": 1.0,
                        "planner_tool_recall": 1.0,
                        "planner_tool_f1": 1.0,
                        "mean_selected_tool_count": 4.0,
                        "mean_tool_sequence_length": 4.0,
                        "mean_extra_tool_count": 0.0,
                        "mean_tool_call_count": 5.0,
                        "failed_tool_call_count": 0,
                        "tool_call_failure_rate": 0.0,
                        "verifier_expectation_match": 1.0,
                        "result_path": "logs/tool_exposure.full.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_master_baseline_table(
        crossdocked_record_path=crossdocked_record,
        litpcba_result_path="logs/litpcba_vina_prepared_15_result.json",
        failure_record_path=failure_record,
        tool_exposure_summary_path=tool_exposure_summary,
        output_dir=tmp_path / "master",
        record_output_path=tmp_path / "master.record.json",
        project_root=tmp_path,
    )

    assert payload["row_count"] == 8
    assert (tmp_path / "master" / "master_baseline_table.json").exists()
    assert (tmp_path / "master" / "master_baseline_table.csv").exists()
    assert (tmp_path / "master" / "robustness_table.json").exists()
    assert (tmp_path / "master" / "robustness_table.csv").exists()
    assert (tmp_path / "master" / "robustness_repeated_table.json").exists()
    assert (tmp_path / "master" / "robustness_repeated_table.csv").exists()
    assert (tmp_path / "master" / "throughput_table.json").exists()
    assert (tmp_path / "master" / "throughput_table.csv").exists()
    assert (tmp_path / "master" / "generation_quality_table.json").exists()
    assert (tmp_path / "master" / "generation_quality_table.csv").exists()
    assert (tmp_path / "master" / "generation_scale_table.json").exists()
    assert (tmp_path / "master" / "generation_scale_table.csv").exists()
    assert (tmp_path / "master" / "crossdocked_multiseed_table.json").exists()
    assert (tmp_path / "master" / "crossdocked_multiseed_table.csv").exists()
    assert (tmp_path / "master" / "tool_exposure_table.json").exists()
    assert (tmp_path / "master" / "tool_exposure_table.csv").exists()
    assert (tmp_path / "master" / "failure_taxonomy_table.json").exists()
    assert (tmp_path / "master" / "failure_taxonomy_table.csv").exists()
    assert (tmp_path / "master" / "ablation_table.json").exists()
    assert (tmp_path / "master" / "ablation_table.csv").exists()
    assert (tmp_path / "master" / "repair_ablation_table.json").exists()
    assert (tmp_path / "master" / "repair_ablation_table.csv").exists()
    assert (tmp_path / "master" / "repair_ablation_repeated_table.json").exists()
    assert (tmp_path / "master" / "repair_ablation_repeated_table.csv").exists()
    assert (tmp_path / "master" / "ambiguous_failure_modes_table.json").exists()
    assert (tmp_path / "master" / "ambiguous_failure_modes_table.csv").exists()
    assert (tmp_path / "master" / "ambiguous_failure_modes_repeated_table.json").exists()
    assert (tmp_path / "master" / "ambiguous_failure_modes_repeated_table.csv").exists()
    assert (tmp_path / "master" / "task_generalization_table.json").exists()
    assert (tmp_path / "master" / "task_generalization_table.csv").exists()
    assert (tmp_path / "master" / "tool_admission_table.json").exists()
    assert (tmp_path / "master" / "tool_admission_table.csv").exists()
    assert (tmp_path / "master" / "verifier_evidence_table.json").exists()
    assert (tmp_path / "master" / "verifier_evidence_table.csv").exists()
    assert (tmp_path / "master" / "property_verifier_table.json").exists()
    assert (tmp_path / "master" / "property_verifier_table.csv").exists()
    assert (tmp_path / "master" / "posebusters_top_failures_table.json").exists()
    assert (tmp_path / "master" / "posebusters_top_failures_table.csv").exists()
    assert (tmp_path / "master" / "posebusters_failure_modes_table.json").exists()
    assert (tmp_path / "master" / "posebusters_failure_modes_table.csv").exists()
    assert (tmp_path / "master" / "pdbbind_prep_gate_table.json").exists()
    assert (tmp_path / "master" / "pdbbind_prep_gate_table.csv").exists()
    assert (tmp_path / "master" / "llm_router_baseline_table.json").exists()
    assert (tmp_path / "master" / "llm_router_baseline_table.csv").exists()
    assert (tmp_path / "master" / "natural_failure_audit_table.json").exists()
    assert (tmp_path / "master" / "natural_failure_audit_table.csv").exists()
    assert (tmp_path / "master" / "evidence_audit_table.json").exists()
    assert (tmp_path / "master" / "evidence_audit_table.csv").exists()
    assert (tmp_path / "master" / "statistical_summary_table.json").exists()
    assert (tmp_path / "master" / "statistical_summary_table.csv").exists()
    assert (tmp_path / "master.record.json").exists()

    rows = payload["rows"]
    families = {row["benchmark_family"] for row in rows}
    assert families == {"crossdocked_generation", "litpcba_docking", "failure_recovery", "tool_exposure_budget"}

    crossdocked = next(row for row in rows if row["benchmark_family"] == "crossdocked_generation")
    assert crossdocked["generated_candidate_count"] == 10
    assert crossdocked["unique_smiles_count"] == 8
    assert crossdocked["best_scscore"] == 2.9
    assert crossdocked["mean_total_elapsed_sec"] == 120.0

    litpcba = next(row for row in rows if row["benchmark_family"] == "litpcba_docking")
    assert litpcba["benchmark_id"] == "litpcba_vina_prepared_15"
    assert litpcba["docking_success_count"] == 2
    assert litpcba["best_docking_score"] == -10.5
    assert litpcba["mean_docking_score"] == -9.0
    assert litpcba["mean_total_elapsed_sec"] == 12.5
    assert litpcba["result_path"] == "logs/litpcba_vina_prepared_15_elapsed_result.json"

    failure_rows = {row["planner_baseline"]: row for row in rows if row["benchmark_family"] == "failure_recovery"}
    assert failure_rows["rule_based_planner"]["task_success_count"] == 0
    assert failure_rows["full_copilot"]["task_success_count"] == 2
    assert failure_rows["full_copilot"]["repair_success_count"] == 2
    assert failure_rows["full_copilot"]["generated_candidate_count"] == 12

    exposure_rows = {row["planner_baseline"]: row for row in rows if row["benchmark_family"] == "tool_exposure_budget"}
    assert exposure_rows["all_tool_agent"]["mean_selected_tool_count"] == 10.0
    assert exposure_rows["all_tool_agent"]["mean_extra_tool_count"] == 6.0
    assert exposure_rows["all_tool_agent"]["planner_tool_precision"] == 0.4
    assert exposure_rows["rule_based_planner"]["mean_selected_tool_count"] == 4.0

    with (tmp_path / "master" / "master_baseline_table.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert list(csv_rows[0].keys()) == MASTER_TABLE_COLUMNS
    assert csv_rows[0]["benchmark_family"] == "crossdocked_generation"

    views = payload["views"]
    assert set(views) == {
        "robustness",
        "robustness_repeated",
        "throughput",
        "generation_quality",
        "generation_scale",
        "crossdocked_multiseed",
        "tool_exposure",
        "failure_taxonomy",
        "ablation",
        "repair_ablation",
        "repair_ablation_repeated",
        "ambiguous_failure_modes",
        "ambiguous_failure_modes_repeated",
        "task_generalization",
        "tool_admission",
        "verifier_evidence",
        "property_verifier",
            "posebusters_top_failures",
            "posebusters_failure_modes",
            "pdbbind_prep_gate",
            "llm_router_baseline",
            "natural_failure_audit",
            "evidence_audit",
            "statistical_summary",
        }
    assert payload["artifacts"]["views"]["robustness"]["csv"] == "master/robustness_table.csv"

    robustness = views["robustness"]
    assert robustness["columns"] == ROBUSTNESS_TABLE_COLUMNS
    assert robustness["row_count"] == 2
    robustness_rows = {row["planner_baseline"]: row for row in robustness["rows"]}
    assert robustness_rows["rule_based_planner"]["task_success_rate"] == 0.0
    assert robustness_rows["full_copilot"]["repair_success_rate"] == 2 / 3

    robustness_repeated = views["robustness_repeated"]
    assert robustness_repeated["columns"] == ROBUSTNESS_REPEATED_TABLE_COLUMNS
    assert robustness_repeated["row_count"] == 0

    throughput = views["throughput"]
    assert throughput["columns"] == THROUGHPUT_TABLE_COLUMNS
    assert throughput["row_count"] == 4
    crossdocked_throughput = next(row for row in throughput["rows"] if row["benchmark_family"] == "crossdocked_generation")
    assert crossdocked_throughput["seconds_per_task"] == 120.0
    assert crossdocked_throughput["valid_candidates_per_sec"] == 10 / (120.0 * 2)

    generation_quality = views["generation_quality"]
    assert generation_quality["columns"] == GENERATION_QUALITY_TABLE_COLUMNS
    assert generation_quality["row_count"] == 3
    crossdocked_quality = next(
        row for row in generation_quality["rows"] if row["benchmark_family"] == "crossdocked_generation"
    )
    assert crossdocked_quality["valid_candidate_rate"] == 1.0
    assert crossdocked_quality["unique_smiles_rate"] == 0.8

    generation_scale = views["generation_scale"]
    assert generation_scale["columns"] == GENERATION_SCALE_TABLE_COLUMNS
    assert generation_scale["row_count"] == 0
    crossdocked_multiseed = views["crossdocked_multiseed"]
    assert crossdocked_multiseed["columns"] == CROSSDOCKED_MULTISEED_TABLE_COLUMNS
    assert crossdocked_multiseed["row_count"] == 0

    tool_exposure = views["tool_exposure"]
    assert tool_exposure["columns"] == TOOL_EXPOSURE_TABLE_COLUMNS
    assert tool_exposure["row_count"] == 4
    exposure_view_rows = {row["planner_baseline"]: row for row in tool_exposure["rows"]}
    assert exposure_view_rows["all_tool_agent"]["mean_selected_tool_count"] == 10.0
    assert exposure_view_rows["fixed_pipeline"]["planner_tool_recall"] == 0.75

    failure_taxonomy = views["failure_taxonomy"]
    assert failure_taxonomy["columns"] == FAILURE_TAXONOMY_TABLE_COLUMNS
    assert failure_taxonomy["row_count"] == 0

    ablation = views["ablation"]
    assert ablation["columns"] == ABLATION_TABLE_COLUMNS
    assert ablation["row_count"] == 2
    ablation_rows = {row["planner_baseline"]: row for row in ablation["rows"]}
    assert ablation_rows["rule_based_planner"]["mean_selected_tool_count"] == 4.0
    assert ablation_rows["rule_based_planner"]["repair_attempt_count"] == 0
    assert ablation_rows["rule_based_planner"]["robust_task_success_count"] == 0
    assert ablation_rows["full_copilot"]["mean_selected_tool_count"] == 4.0
    assert ablation_rows["full_copilot"]["mean_extra_tool_count"] == 0.0
    assert ablation_rows["full_copilot"]["repair_success_count"] == 2
    assert "verifier-triggered repair" in ablation_rows["full_copilot"]["interpretation"]

    repair_ablation = views["repair_ablation"]
    assert repair_ablation["columns"] == REPAIR_ABLATION_TABLE_COLUMNS
    assert repair_ablation["row_count"] == 0
    repair_ablation_repeated = views["repair_ablation_repeated"]
    assert repair_ablation_repeated["columns"] == REPAIR_ABLATION_REPEATED_TABLE_COLUMNS
    assert repair_ablation_repeated["row_count"] == 0

    ambiguous_failure_modes = views["ambiguous_failure_modes"]
    assert ambiguous_failure_modes["columns"] == AMBIGUOUS_FAILURE_MODE_TABLE_COLUMNS
    assert ambiguous_failure_modes["row_count"] == 0
    ambiguous_failure_modes_repeated = views["ambiguous_failure_modes_repeated"]
    assert ambiguous_failure_modes_repeated["columns"] == AMBIGUOUS_FAILURE_MODE_REPEATED_TABLE_COLUMNS
    assert ambiguous_failure_modes_repeated["row_count"] == 0

    task_generalization = views["task_generalization"]
    assert task_generalization["columns"] == TASK_GENERALIZATION_TABLE_COLUMNS
    assert task_generalization["row_count"] == 0

    tool_admission = views["tool_admission"]
    assert tool_admission["columns"] == TOOL_ADMISSION_TABLE_COLUMNS
    assert tool_admission["row_count"] == 0

    verifier_evidence = views["verifier_evidence"]
    assert verifier_evidence["columns"] == VERIFIER_EVIDENCE_TABLE_COLUMNS
    assert verifier_evidence["row_count"] == 0
    property_verifier = views["property_verifier"]
    assert property_verifier["columns"] == PROPERTY_VERIFIER_TABLE_COLUMNS
    assert property_verifier["row_count"] == 0
    top_failures = views["posebusters_top_failures"]
    assert top_failures["columns"] == POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS
    assert top_failures["row_count"] == 0
    failure_modes = views["posebusters_failure_modes"]
    assert failure_modes["columns"] == POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS
    assert failure_modes["row_count"] == 0
    pdbbind_gate = views["pdbbind_prep_gate"]
    assert pdbbind_gate["columns"] == PDBBIND_PREP_GATE_TABLE_COLUMNS
    assert pdbbind_gate["row_count"] == 0

    with (tmp_path / "master" / "robustness_table.csv").open("r", encoding="utf-8", newline="") as handle:
        robustness_csv_rows = list(csv.DictReader(handle))
    assert list(robustness_csv_rows[0].keys()) == ROBUSTNESS_TABLE_COLUMNS


def test_master_table_builder_includes_verifier_evidence_view(tmp_path):
    payload = build_master_baseline_table(
        crossdocked_record_path=_minimal_crossdocked_record(tmp_path),
        litpcba_result_path=_minimal_litpcba_result(tmp_path),
        failure_record_path=_minimal_failure_summary(tmp_path),
        tool_exposure_summary_path=None,
        verifier_evidence_summary_path=_verifier_evidence_summary(tmp_path),
        pdbbindplus_pose_sanity_summary_path=_pdbbindplus_pose_sanity_summary(tmp_path),
        output_dir=tmp_path / "master",
        record_output_path=tmp_path / "master.record.json",
        project_root=tmp_path,
    )

    verifier_view = payload["views"]["verifier_evidence"]
    assert verifier_view["columns"] == VERIFIER_EVIDENCE_TABLE_COLUMNS
    assert verifier_view["row_count"] == 3
    rows = {(row["dataset"], row["evidence_type"]): row for row in verifier_view["rows"]}
    assert rows[("CrossDocked2020", "sa_score")]["coverage"] == 1.0
    assert rows[("LIT-PCBA", "posebusters")]["status"] == "available"
    assert rows[("LIT-PCBA", "posebusters")]["coverage"] == 1.0
    assert rows[("LIT-PCBA", "posebusters")]["pass_rate"] == 0.0
    assert rows[("PDBbind+ v2020.R1", "posebusters")]["status"] == "available"
    with (tmp_path / "master" / "verifier_evidence_table.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert list(csv_rows[0].keys()) == VERIFIER_EVIDENCE_TABLE_COLUMNS
    top_failures = payload["views"]["posebusters_top_failures"]
    assert top_failures["columns"] == POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS
    assert top_failures["row_count"] == 2
    assert top_failures["rows"][0]["check_name"] == "sanitization"
    assert top_failures["rows"][0]["dataset"] == "LIT-PCBA"
    assert top_failures["rows"][0]["fail_rate"] == 1.0
    failure_modes = payload["views"]["posebusters_failure_modes"]
    assert failure_modes["columns"] == POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS
    assert failure_modes["row_count"] == 2
    assert failure_modes["rows"][0]["check_name"] == "sanitization"
    assert failure_modes["rows"][0]["dataset"] == "LIT-PCBA"
    assert failure_modes["rows"][0]["fail_rate"] == 1.0


def test_master_table_builder_includes_crossdocked_multiseed_view(tmp_path):
    summary = tmp_path / "crossdocked_multiseed_summary.json"
    summary.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "benchmark_id": "crossdocked_multiseed",
                        "dataset": "CrossDocked2020",
                        "seed_count": 2,
                        "seeds": "1,2",
                        "total_target_runs": 60,
                        "total_candidates": 300,
                        "mean_task_success_rate": 1.0,
                        "std_task_success_rate": 0.0,
                        "mean_valid_candidate_rate": 1.0,
                        "std_valid_candidate_rate": 0.0,
                        "mean_unique_smiles_rate": 0.2,
                        "std_unique_smiles_rate": 0.03,
                        "mean_sa_score_coverage": 1.0,
                        "mean_sa_score_pass_rate": 0.95,
                        "mean_rdkit_property_coverage": 1.0,
                        "mean_qed": 0.75,
                        "mean_seconds_per_task": 90.0,
                        "std_seconds_per_task": 10.0,
                        "false_success_count": 0,
                        "notes": "repeatability evidence",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = build_master_baseline_table(
        crossdocked_record_path=_minimal_crossdocked_record(tmp_path),
        litpcba_result_path=_minimal_litpcba_result(tmp_path),
        failure_record_path=_minimal_failure_summary(tmp_path),
        tool_exposure_summary_path=None,
        crossdocked_multiseed_summary_path=summary,
        output_dir=tmp_path / "master",
        record_output_path=tmp_path / "master.record.json",
        project_root=tmp_path,
    )

    view = payload["views"]["crossdocked_multiseed"]
    assert view["columns"] == CROSSDOCKED_MULTISEED_TABLE_COLUMNS
    assert view["row_count"] == 1
    assert view["rows"][0]["total_candidates"] == 300
    assert view["rows"][0]["mean_unique_smiles_rate"] == 0.2


def test_master_table_builder_includes_property_verifier_view(tmp_path):
    payload = build_master_baseline_table(
        crossdocked_record_path=_minimal_crossdocked_record(tmp_path),
        litpcba_result_path=_minimal_litpcba_result(tmp_path),
        failure_record_path=_minimal_failure_summary(tmp_path),
        tool_exposure_summary_path=None,
        property_verifier_summary_path=_property_verifier_summary(tmp_path),
        output_dir=tmp_path / "master",
        record_output_path=tmp_path / "master.record.json",
        project_root=tmp_path,
    )

    property_view = payload["views"]["property_verifier"]
    assert property_view["columns"] == PROPERTY_VERIFIER_TABLE_COLUMNS
    assert property_view["row_count"] == 1
    row = property_view["rows"][0]
    assert row["candidate_count"] == 2
    assert row["property_coverage"] == 1.0
    assert row["lipinski_pass_rate"] == 1.0
    verifier_rows = {
        item["evidence_type"]: item
        for item in payload["views"]["verifier_evidence"]["rows"]
    }
    assert verifier_rows["rdkit_property_verifier"]["pass_rate"] == 1.0
    assert (tmp_path / "master" / "property_verifier_table.csv").exists()


def test_master_table_builder_includes_pdbbind_prep_gate_view(tmp_path):
    readiness = tmp_path / "pdbbind_readiness_summary.json"
    readiness.write_text(
        json.dumps(
            {
                "status": "ready",
                "best_candidate": {
                    "ready": True,
                    "ready_target_count": 10,
                    "index_file_count": 12,
                },
            }
        ),
        encoding="utf-8",
    )
    receptor_prep = tmp_path / "pdbbind_receptor_prep_summary.json"
    receptor_prep.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 10,
                    "prep_success_count": 0,
                    "prep_failure_count": 10,
                    "prep_success_rate": 0.0,
                    "template_required_count": 9,
                    "failure_counts": {
                        "histidine_template_ambiguity": 9,
                        "runtime_error": 0,
                        "timeout": 1,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    prepared_pilot = tmp_path / "pdbbind_refined_prepared_pilot.jsonl"
    prepared_pilot.write_text("", encoding="utf-8")

    payload = build_master_baseline_table(
        crossdocked_record_path=_minimal_crossdocked_record(tmp_path),
        litpcba_result_path=_minimal_litpcba_result(tmp_path),
        failure_record_path=_minimal_failure_summary(tmp_path),
        tool_exposure_summary_path=None,
        pdbbind_readiness_summary_path=readiness,
        pdbbind_receptor_prep_summary_path=receptor_prep,
        pdbbind_prepared_pilot_path=prepared_pilot,
        output_dir=tmp_path / "master",
        record_output_path=tmp_path / "master.record.json",
        project_root=tmp_path,
    )

    view = payload["views"]["pdbbind_prep_gate"]
    assert view["columns"] == PDBBIND_PREP_GATE_TABLE_COLUMNS
    assert view["row_count"] == 1
    row = view["rows"][0]
    assert row["readiness_status"] == "ready"
    assert row["ready_target_count"] == 10
    assert row["index_file_count"] == 12
    assert row["prep_success_count"] == 0
    assert row["prep_success_rate"] == 0.0
    assert row["template_required_count"] == 9
    assert row["runtime_error_count"] == 0
    assert row["timeout_count"] == 1
    assert row["prepared_pilot_task_count"] == 0
    assert row["real_pilot_task_count"] is None
    assert row["real_pilot_success_rate"] is None
    assert row["best_docking_score"] is None
    assert row["false_success_count"] is None
    assert row["gate_status"] == "execution_blocked_no_prepared_receptors"
    assert row["evidence_role"] == "appendix_gate_not_main_claim"
    assert "prevents false docking success claims" in row["notes"]
    with (tmp_path / "master" / "pdbbind_prep_gate_table.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert list(csv_rows[0].keys()) == PDBBIND_PREP_GATE_TABLE_COLUMNS
    assert csv_rows[0]["gate_status"] == "execution_blocked_no_prepared_receptors"


def test_master_table_builder_includes_completed_pdbbindplus_prepared_pilot(tmp_path):
    readiness = tmp_path / "pdbbindplus_readiness_summary.json"
    readiness.write_text(
        json.dumps(
            {
                "status": "ready",
                "best_candidate": {
                    "root": "<external_pdbbind_plus_root>",
                    "ready": True,
                    "ready_target_count": 19037,
                    "index_file_count": 11,
                },
            }
        ),
        encoding="utf-8",
    )
    receptor_prep = tmp_path / "pdbbindplus_receptor_prep_summary.json"
    receptor_prep.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 20,
                    "prep_success_count": 17,
                    "prep_failure_count": 3,
                    "prep_success_rate": 0.85,
                    "failure_counts": {
                        "command_failed": 1,
                        "runtime_error": 1,
                        "timeout": 1,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    prepared_pilot = tmp_path / "pdbbindplus_prepared_pilot_17.jsonl"
    prepared_pilot.write_text(
        "\n".join(json.dumps({"task_id": f"pdbbind_{idx:03d}"}) for idx in range(17)) + "\n",
        encoding="utf-8",
    )
    pilot_result = tmp_path / "pdbbindplus_prepared_pilot_17_result.json"
    pilot_result.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 17,
                    "task_success_rate": 1.0,
                    "mean_total_elapsed_sec": 24.0,
                    "false_success_count": 0,
                },
                "results": [
                    {"metrics": {"best_docking_score": -8.8}},
                    {"metrics": {"best_docking_score": -6.1}},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_master_baseline_table(
        crossdocked_record_path=_minimal_crossdocked_record(tmp_path),
        litpcba_result_path=_minimal_litpcba_result(tmp_path),
        failure_record_path=_minimal_failure_summary(tmp_path),
        tool_exposure_summary_path=None,
        pdbbind_readiness_summary_path=readiness,
        pdbbind_receptor_prep_summary_path=receptor_prep,
        pdbbind_prepared_pilot_path=prepared_pilot,
        pdbbind_prepared_pilot_result_path=pilot_result,
        output_dir=tmp_path / "master",
        record_output_path=tmp_path / "master.record.json",
        project_root=tmp_path,
    )

    row = payload["views"]["pdbbind_prep_gate"]["rows"][0]
    assert row["dataset"] == "PDBbind+ v2020.R1"
    assert row["ready_target_count"] == 19037
    assert row["prep_success_count"] == 17
    assert row["prepared_pilot_task_count"] == 17
    assert row["real_pilot_task_count"] == 17
    assert row["real_pilot_success_rate"] == 1.0
    assert row["best_docking_score"] == -8.8
    assert row["mean_elapsed_sec"] == 24.0
    assert row["false_success_count"] == 0
    assert row["gate_status"] == "prepared_pilot_completed"
    assert "not an affinity benchmark" in row["notes"]


def test_master_table_builder_accepts_failure_taxonomy_suite_summary(tmp_path):
    crossdocked_record = tmp_path / "crossdocked.summary.json"
    crossdocked_record.write_text(
        json.dumps(
            {
                "benchmark_id": "crossdocked_rxnflow_candidates5_targets15",
                "dataset": "CrossDocked2020",
                "execution_mode": "real",
                "planner_baseline": "rule_based_planner",
                "task_count": 1,
                "summary": {"total": 1, "task_success_rate": 1.0},
                "global_candidate_summary": {
                    "generated_candidate_count": 5,
                    "valid_candidate_count": 5,
                    "unique_smiles_count_across_tasks": 5,
                },
                "per_target": [],
            }
        ),
        encoding="utf-8",
    )

    litpcba_result = tmp_path / "litpcba_vina_prepared_15_result.json"
    litpcba_result.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "expected_tools": ["vina"],
                        "selected_tools": ["vina"],
                        "task_success": True,
                        "metrics": {"best_docking_score": -8.0},
                    }
                ],
                "summary": {"total": 1, "task_success_rate": 1.0},
            }
        ),
        encoding="utf-8",
    )

    failure_summary = tmp_path / "failure_recovery_taxonomy_v2.baseline_summary.json"
    failure_summary.write_text(
        json.dumps(
            {
                "benchmark_id": "failure_recovery_taxonomy_v2",
                "execution_mode": "real",
                "results": {
                    "rule_based_planner": {
                        "result_path": "logs/failure_v2.rule.json",
                        "summary": {
                            "mean_total_elapsed_sec": 12.9,
                            "task_success_rate": 1 / 9,
                            "total": 9,
                        },
                    },
                    "full_copilot": {
                        "result_path": "logs/failure_v2.full.json",
                        "summary": {
                            "mean_total_elapsed_sec": 274.7,
                            "task_success_rate": 4 / 9,
                            "total": 9,
                        },
                    },
                },
                "rows": [
                    {
                        "benchmark_id": "failure_recovery_taxonomy_v2",
                        "execution_mode": "real",
                        "failed_task_count": 8,
                        "failed_tool_call_count": 14,
                        "mean_tool_call_count": 2.8888888889,
                        "planner_baseline": "rule_based_planner",
                        "repair_attempt_count": 0,
                        "repair_success_count": 0,
                        "repair_success_rate": 0.0,
                        "result_path": "logs/failure_v2.rule.json",
                        "task_success_count": 1,
                        "task_success_rate": 1 / 9,
                        "tool_call_failure_rate": 0.5384615385,
                        "total": 9,
                        "verifier_expectation_match": 2 / 3,
                    },
                    {
                        "benchmark_id": "failure_recovery_taxonomy_v2",
                        "execution_mode": "real",
                        "failed_task_count": 5,
                        "failed_tool_call_count": 19,
                        "mean_tool_call_count": 27.6666666667,
                        "planner_baseline": "full_copilot",
                        "repair_attempt_count": 6,
                        "repair_success_count": 4,
                        "repair_success_rate": 2 / 3,
                        "result_path": "logs/failure_v2.full.json",
                        "task_success_count": 4,
                        "task_success_rate": 4 / 9,
                        "tool_call_failure_rate": 0.0763052209,
                        "total": 9,
                        "verifier_expectation_match": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    taxonomy_benchmark = tmp_path / "CAi" / "toolkit" / "agent_planner" / "benchmarks" / "failure_recovery_taxonomy_v2.jsonl"
    taxonomy_benchmark.parent.mkdir(parents=True)
    taxonomy_tasks = [
        {
            "task_id": "case_retry",
            "expected_task_type": "failure_recovery",
            "should_succeed": True,
            "metadata": {
                "failure_scenario": "denovo_error_then_retry",
                "failure_injections": {"reinvent4_denovo": [{"mode": "error"}]},
            },
        },
        {
            "task_id": "case_mark_incomplete",
            "expected_task_type": "de_novo_generation",
            "should_succeed": False,
            "metadata": {
                "failure_scenario": "scscore_failure_marks_incomplete",
                "failure_injections": {"scscore": [{"mode": "error"}]},
            },
        },
    ]
    taxonomy_benchmark.write_text(
        "\n".join(json.dumps(item) for item in taxonomy_tasks) + "\n",
        encoding="utf-8",
    )

    taxonomy_result_dir = tmp_path / "logs" / "baseline_runs" / "failure_recovery_taxonomy_v2_real"
    taxonomy_result_dir.mkdir(parents=True)
    (taxonomy_result_dir / "failure_recovery_taxonomy_v2.rule_based_planner.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": "case_retry",
                        "task_success": False,
                        "verifier_matched_expectation": False,
                    },
                    {
                        "task_id": "case_mark_incomplete",
                        "task_success": False,
                        "verifier_matched_expectation": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (taxonomy_result_dir / "failure_recovery_taxonomy_v2.full_copilot.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": "case_retry",
                        "task_success": True,
                        "repair_executed": True,
                        "repair_success": True,
                        "extra_tools": [],
                        "verifier_matched_expectation": True,
                        "repair_plan": {
                            "actions": [
                                {
                                    "action_type": "retry_with_reduced_generation_count",
                                    "tool_name": "reinvent4_denovo",
                                }
                            ]
                        },
                    },
                    {
                        "task_id": "case_mark_incomplete",
                        "task_success": False,
                        "repair_executed": False,
                        "repair_success": False,
                        "extra_tools": [],
                        "verifier_matched_expectation": True,
                        "repair_plan": {
                            "actions": [
                                {
                                    "action_type": "mark_incomplete_evaluation",
                                    "tool_name": "scscore",
                                }
                            ]
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = build_master_baseline_table(
        crossdocked_record_path=crossdocked_record,
        litpcba_result_path=litpcba_result,
        failure_record_path=failure_summary,
        tool_exposure_summary_path=None,
        output_dir=tmp_path / "master",
        record_output_path=None,
        project_root=tmp_path,
    )

    failure_rows = {row["planner_baseline"]: row for row in payload["rows"] if row["benchmark_family"] == "failure_recovery"}
    assert set(failure_rows) == {"rule_based_planner", "full_copilot"}
    assert failure_rows["rule_based_planner"]["benchmark_id"] == "failure_recovery_taxonomy_v2"
    assert failure_rows["rule_based_planner"]["task_success_count"] == 1
    assert failure_rows["rule_based_planner"]["repair_attempt_count"] == 0
    assert failure_rows["rule_based_planner"]["mean_total_elapsed_sec"] == 12.9
    assert failure_rows["full_copilot"]["task_success_count"] == 4
    assert failure_rows["full_copilot"]["repair_success_count"] == 4
    assert failure_rows["full_copilot"]["mean_tool_call_count"] == 27.6666666667
    assert failure_rows["full_copilot"]["result_path"] == "logs/failure_v2.full.json"
    assert "4 of 6" in failure_rows["full_copilot"]["notes"]

    robustness = payload["views"]["robustness"]
    assert robustness["row_count"] == 2
    robustness_rows = {row["planner_baseline"]: row for row in robustness["rows"]}
    assert robustness_rows["full_copilot"]["repair_success_rate"] == 2 / 3

    taxonomy = payload["views"]["failure_taxonomy"]
    assert taxonomy["columns"] == FAILURE_TAXONOMY_TABLE_COLUMNS
    assert taxonomy["row_count"] == 2
    taxonomy_rows = {row["failure_scenario"]: row for row in taxonomy["rows"]}
    assert taxonomy_rows["denovo_error_then_retry"]["injected_tools"] == "reinvent4_denovo"
    assert taxonomy_rows["denovo_error_then_retry"]["full_success"] is True
    assert taxonomy_rows["denovo_error_then_retry"]["repair_actions"] == "retry:reinvent4_denovo"
    assert "without exposing extra tools" in taxonomy_rows["denovo_error_then_retry"]["interpretation"]
    assert taxonomy_rows["scscore_failure_marks_incomplete"]["full_success"] is False
    assert "prevents a false success" in taxonomy_rows["scscore_failure_marks_incomplete"]["interpretation"]


def _minimal_crossdocked_record(tmp_path):
    path = tmp_path / "crossdocked.summary.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "crossdocked",
                "dataset": "CrossDocked2020",
                "execution_mode": "real",
                "planner_baseline": "rule_based_planner",
                "task_count": 1,
                "summary": {"total": 1, "task_success_rate": 1.0},
                "global_candidate_summary": {
                    "generated_candidate_count": 1,
                    "valid_candidate_count": 1,
                    "unique_smiles_count_across_tasks": 1,
                },
                "per_target": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _minimal_litpcba_result(tmp_path):
    path = tmp_path / "litpcba_vina_prepared_15_result.json"
    path.write_text(
        json.dumps(
            {
                "planner_baseline": "rule_based_planner",
                "results": [{"expected_tools": ["vina"], "selected_tools": ["vina"], "task_success": True}],
                "summary": {"total": 1, "task_success_rate": 1.0},
            }
        ),
        encoding="utf-8",
    )
    return path


def _minimal_failure_summary(tmp_path):
    path = tmp_path / "failure_recovery_taxonomy_v2.baseline_summary.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "failure_recovery_taxonomy_v2",
                "execution_mode": "real",
                "rows": [
                    {
                        "planner_baseline": "rule_based_planner",
                        "task_success_count": 0,
                        "failed_task_count": 1,
                        "task_success_rate": 0.0,
                    },
                    {
                        "planner_baseline": "full_copilot",
                        "task_success_count": 1,
                        "failed_task_count": 0,
                        "task_success_rate": 1.0,
                        "repair_attempt_count": 1,
                        "repair_success_count": 1,
                        "repair_success_rate": 1.0,
                    },
                ],
                "results": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def _verifier_evidence_summary(tmp_path):
    path = tmp_path / "verifier_evidence_summary.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "verifier_enhancement_v1",
                "posebusters_failure_modes": {
                    "columns": POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS,
                    "row_count": 1,
                    "rows": [
                        {
                            "check_name": "sanitization",
                            "category": "molecule_validity",
                            "pose_count": 1,
                            "evaluated_count": 1,
                            "pass_count": 0,
                            "fail_count": 1,
                            "missing_count": 0,
                            "fail_rate": 1.0,
                            "example_task_ids": "litpcba_docking_000_ADRB2",
                            "interpretation": "The predicted ligand cannot be sanitized by RDKit.",
                        }
                    ],
                },
                "rows": [
                    {
                        "evidence_family": "crossdocked_generation",
                        "dataset": "CrossDocked2020",
                        "evidence_type": "sa_score",
                        "task_count": 1,
                        "candidate_count": 2,
                        "evaluable_candidate_count": 2,
                        "evidence_count": 2,
                        "coverage": 1.0,
                        "pass_count": 2,
                        "pass_rate": 1.0,
                        "best_sa_score": 2.1,
                        "mean_sa_score": 2.3,
                        "pose_artifact_count": None,
                        "status": "available",
                        "notes": "SA score evidence.",
                    },
                    {
                        "evidence_family": "litpcba_docking",
                        "dataset": "LIT-PCBA",
                        "evidence_type": "posebusters",
                        "task_count": 1,
                        "candidate_count": 1,
                        "evaluable_candidate_count": 1,
                        "evidence_count": 1,
                        "coverage": 1.0,
                        "pass_count": 0,
                        "pass_rate": 0.0,
                        "best_sa_score": None,
                        "mean_sa_score": None,
                        "pose_artifact_count": 1,
                        "status": "available",
                        "notes": "PoseBusters evaluated.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _property_verifier_summary(tmp_path):
    path = tmp_path / "property_verifier_summary.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "rdkit_property_verifier_v1",
                "property_rows": [
                    {
                        "evidence_family": "crossdocked_generation",
                        "dataset": "CrossDocked2020",
                        "task_count": 1,
                        "candidate_count": 2,
                        "valid_smiles_count": 2,
                        "property_coverage": 1.0,
                        "mean_qed": 0.55,
                        "mean_logp": 1.2,
                        "mean_molwt": 220.0,
                        "lipinski_pass_count": 2,
                        "lipinski_pass_rate": 1.0,
                        "pains_flag_count": 0,
                        "pains_flag_rate": 0.0,
                        "brenk_flag_count": 1,
                        "brenk_flag_rate": 0.5,
                        "status": "available",
                        "notes": "RDKit property evidence.",
                    }
                ],
                "verifier_evidence_rows": [
                    {
                        "evidence_family": "crossdocked_generation",
                        "dataset": "CrossDocked2020",
                        "evidence_type": "rdkit_property_verifier",
                        "task_count": 1,
                        "candidate_count": 2,
                        "evaluable_candidate_count": 2,
                        "evidence_count": 2,
                        "coverage": 1.0,
                        "pass_count": 2,
                        "pass_rate": 1.0,
                        "best_sa_score": None,
                        "mean_sa_score": None,
                        "pose_artifact_count": None,
                        "status": "available",
                        "notes": "Lipinski pass-rate evidence.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _pdbbindplus_pose_sanity_summary(tmp_path):
    path = tmp_path / "pdbbindplus_pose_sanity_summary.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "pdbbindplus_pose_sanity_v1",
                "posebusters_failure_modes": {
                    "columns": POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS,
                    "row_count": 1,
                    "rows": [
                        {
                            "check_name": "volume_overlap_with_protein",
                            "category": "protein_ligand_geometry",
                            "pose_count": 1,
                            "evaluated_count": 1,
                            "pass_count": 0,
                            "fail_count": 1,
                            "missing_count": 0,
                            "fail_rate": 1.0,
                            "example_task_ids": "pdbbind_refined_prepared_pilot_000_10gs",
                            "interpretation": "The pose overlaps protein volume under PoseBusters thresholds.",
                        }
                    ],
                },
                "rows": [
                    {
                        "evidence_family": "pdbbindplus_docking",
                        "dataset": "PDBbind+ v2020.R1",
                        "evidence_type": "posebusters",
                        "task_count": 1,
                        "candidate_count": 1,
                        "evaluable_candidate_count": 1,
                        "evidence_count": 1,
                        "coverage": 1.0,
                        "pass_count": 0,
                        "pass_rate": 0.0,
                        "best_sa_score": None,
                        "mean_sa_score": None,
                        "pose_artifact_count": 1,
                        "status": "available",
                        "notes": "PoseBusters evaluated PDBbind+ pose.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_llm_router_baseline_rows_support_multiple_models(tmp_path):
    deepseek_summary = tmp_path / "llm_router_baseline_deepseek_v4_pro_full_replayed.json"
    deepseek_summary.write_text(
        json.dumps(
            {
                "row": {
                    "benchmark_id": "llm_as_router_planning_v1",
                    "dataset": "mixed_molecular_planning_tasks",
                    "router_mode": "api_replay",
                    "task_count": 65,
                    "valid_json_count": 64,
                    "valid_schema_count": 63,
                },
                "task_results": [
                    {"api_metadata": {"api_model": "deepseek-v4-pro"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    qwen_summary = tmp_path / "llm_router_baseline_qwen3_7_plus_nothink_full_replayed.json"
    qwen_summary.write_text(
        json.dumps(
            {
                "row": {
                    "benchmark_id": "llm_as_router_planning_v1",
                    "dataset": "mixed_molecular_planning_tasks",
                    "router_mode": "api_replay",
                    "task_count": 65,
                    "valid_json_count": 65,
                    "valid_schema_count": 65,
                },
                "task_results": [
                    {"api_metadata": {"api_model": "qwen3.7-plus"}},
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = _llm_router_baseline_rows([deepseek_summary, qwen_summary])

    assert [row["model"] for row in rows] == ["deepseek-v4-pro", "qwen3.7-plus"]
    assert [row["task_count"] for row in rows] == [65, 65]
