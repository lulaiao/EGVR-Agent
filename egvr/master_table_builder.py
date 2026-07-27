"""Build a paper-ready master table from completed benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .evidence_audit_builder import EVIDENCE_AUDIT_COLUMNS, build_evidence_audit_rows
from .llm_router_baseline_runner import LLM_ROUTER_BASELINE_COLUMNS
from .natural_failure_audit_runner import NATURAL_FAILURE_AUDIT_COLUMNS
from .statistical_summary_builder import STATISTICAL_SUMMARY_COLUMNS, build_statistical_summary_rows


def _is_repair_baseline(value: Any) -> bool:
    return value in {"egvr_agent", "full_copilot"}


MASTER_TABLE_COLUMNS = [
    "benchmark_family",
    "dataset",
    "benchmark_id",
    "execution_mode",
    "planner_baseline",
    "task_count",
    "task_success_count",
    "failed_task_count",
    "task_success_rate",
    "parser_accuracy",
    "planner_tool_coverage_rate",
    "planner_tool_precision",
    "planner_tool_recall",
    "planner_tool_f1",
    "mean_selected_tool_count",
    "mean_tool_sequence_length",
    "mean_extra_tool_count",
    "mean_tool_call_count",
    "failed_tool_call_count",
    "tool_call_failure_rate",
    "verifier_expectation_match",
    "repair_attempt_count",
    "repair_success_count",
    "repair_success_rate",
    "generated_candidate_count",
    "valid_candidate_count",
    "unique_smiles_count",
    "docking_success_count",
    "best_docking_score",
    "mean_docking_score",
    "best_scscore",
    "max_toxicity_score",
    "mean_total_elapsed_sec",
    "result_path",
    "record_path",
    "notes",
]


ROBUSTNESS_TABLE_COLUMNS = [
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
]

ROBUSTNESS_REPEATED_TABLE_COLUMNS = [
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
]

REPAIR_ABLATION_REPEATED_TABLE_COLUMNS = list(ROBUSTNESS_REPEATED_TABLE_COLUMNS)


THROUGHPUT_TABLE_COLUMNS = [
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
]


GENERATION_QUALITY_TABLE_COLUMNS = [
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
]

GENERATION_SCALE_TABLE_COLUMNS = [
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
]

CROSSDOCKED_MULTISEED_TABLE_COLUMNS = [
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
]


TOOL_EXPOSURE_TABLE_COLUMNS = [
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
]


FAILURE_TAXONOMY_TABLE_COLUMNS = [
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
]


ABLATION_TABLE_COLUMNS = [
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
]

REPAIR_ABLATION_TABLE_COLUMNS = [
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
]

AMBIGUOUS_FAILURE_MODE_TABLE_COLUMNS = [
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
]

AMBIGUOUS_FAILURE_MODE_REPEATED_TABLE_COLUMNS = [
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
]

TASK_GENERALIZATION_TABLE_COLUMNS = [
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
]

TOOL_ADMISSION_TABLE_COLUMNS = [
    "tool_name",
    "tool_role",
    "independent_evidence",
    "failure_modes_structured",
    "runtime_cost",
    "environment_risk",
    "paper_claim_supported",
    "decision",
]

VERIFIER_EVIDENCE_TABLE_COLUMNS = [
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
]

PROPERTY_VERIFIER_TABLE_COLUMNS = [
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
]

POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS = [
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
]

POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS = [
    "dataset",
    "evidence_family",
    "check_name",
    "category",
    "evaluated_count",
    "fail_count",
    "fail_rate",
    "example_task_ids",
]

PDBBIND_PREP_GATE_TABLE_COLUMNS = [
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
]


LLM_ROUTER_BASELINE_TABLE_COLUMNS = ["model"] + list(LLM_ROUTER_BASELINE_COLUMNS)
NATURAL_FAILURE_AUDIT_TABLE_COLUMNS = list(NATURAL_FAILURE_AUDIT_COLUMNS)
EVIDENCE_AUDIT_TABLE_COLUMNS = list(EVIDENCE_AUDIT_COLUMNS)
STATISTICAL_SUMMARY_TABLE_COLUMNS = list(STATISTICAL_SUMMARY_COLUMNS)


PAPER_VIEW_COLUMNS = {
    "robustness": ROBUSTNESS_TABLE_COLUMNS,
    "robustness_repeated": ROBUSTNESS_REPEATED_TABLE_COLUMNS,
    "throughput": THROUGHPUT_TABLE_COLUMNS,
    "generation_quality": GENERATION_QUALITY_TABLE_COLUMNS,
    "generation_scale": GENERATION_SCALE_TABLE_COLUMNS,
    "crossdocked_multiseed": CROSSDOCKED_MULTISEED_TABLE_COLUMNS,
    "tool_exposure": TOOL_EXPOSURE_TABLE_COLUMNS,
    "failure_taxonomy": FAILURE_TAXONOMY_TABLE_COLUMNS,
    "ablation": ABLATION_TABLE_COLUMNS,
    "repair_ablation": REPAIR_ABLATION_TABLE_COLUMNS,
    "repair_ablation_repeated": REPAIR_ABLATION_REPEATED_TABLE_COLUMNS,
    "ambiguous_failure_modes": AMBIGUOUS_FAILURE_MODE_TABLE_COLUMNS,
    "ambiguous_failure_modes_repeated": AMBIGUOUS_FAILURE_MODE_REPEATED_TABLE_COLUMNS,
    "task_generalization": TASK_GENERALIZATION_TABLE_COLUMNS,
    "tool_admission": TOOL_ADMISSION_TABLE_COLUMNS,
    "verifier_evidence": VERIFIER_EVIDENCE_TABLE_COLUMNS,
    "property_verifier": PROPERTY_VERIFIER_TABLE_COLUMNS,
    "posebusters_top_failures": POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS,
    "posebusters_failure_modes": POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS,
    "pdbbind_prep_gate": PDBBIND_PREP_GATE_TABLE_COLUMNS,
    "llm_router_baseline": LLM_ROUTER_BASELINE_TABLE_COLUMNS,
    "natural_failure_audit": NATURAL_FAILURE_AUDIT_TABLE_COLUMNS,
    "evidence_audit": EVIDENCE_AUDIT_TABLE_COLUMNS,
    "statistical_summary": STATISTICAL_SUMMARY_TABLE_COLUMNS,
}


DEFAULT_CROSSDOCKED_RECORD = (
    "egvr/benchmarks/records/crossdocked_rxnflow_candidates5_targets15.summary.json"
)
DEFAULT_CROSSDOCKED_TARGETS30_RECORD = (
    "egvr/benchmarks/records/crossdocked_rxnflow_candidates5_targets30.summary.json"
)
DEFAULT_GENERATION_SCALE_RECORDS = (DEFAULT_CROSSDOCKED_RECORD, DEFAULT_CROSSDOCKED_TARGETS30_RECORD)
DEFAULT_CROSSDOCKED_MULTISEED_SUMMARY = (
    "logs/baseline_runs/crossdocked_multiseed_v1/crossdocked_multiseed_summary.json"
)
DEFAULT_LITPCBA_RESULT = "logs/litpcba_vina_prepared_15_result.json"
DEFAULT_LITPCBA_ELAPSED_RESULT = "logs/litpcba_vina_prepared_15_elapsed_result.json"
DEFAULT_FAILURE_RECORD = (
    "logs/baseline_runs/failure_recovery_taxonomy_v2_real/"
    "failure_recovery_taxonomy_v2.baseline_summary.json"
)
DEFAULT_TOOL_EXPOSURE_SUMMARY = (
    "logs/baseline_runs/tool_exposure_budget_v1_mock/"
    "crossdocked_rxnflow_candidates5_targets15.baseline_summary.json"
)
DEFAULT_FAILURE_TAXONOMY_BENCHMARK = "egvr/benchmarks/failure_recovery_taxonomy_v2.jsonl"
DEFAULT_FAILURE_TAXONOMY_RULE_RESULT = (
    "logs/baseline_runs/failure_recovery_taxonomy_v2_real/"
    "failure_recovery_taxonomy_v2.rule_based_planner.json"
)
DEFAULT_FAILURE_TAXONOMY_FULL_RESULT = (
    "logs/baseline_runs/failure_recovery_taxonomy_v2_real/"
    "failure_recovery_taxonomy_v2.full_copilot.json"
)
DEFAULT_OUTPUT_DIR = "logs/baseline_runs/master_baseline_table"
DEFAULT_RECORD_OUTPUT = "egvr/benchmarks/records/master_baseline_table.summary.json"
DEFAULT_VERIFIER_EVIDENCE_SUMMARY = (
    "logs/baseline_runs/verifier_enhancement_crossdocked30_v1/verifier_evidence_summary.json"
)
DEFAULT_VERIFIER_EVIDENCE_FALLBACK_SUMMARY = (
    "logs/baseline_runs/verifier_enhancement_v1/verifier_evidence_summary.json"
)
DEFAULT_ROBUSTNESS_REPEATED_SUMMARY = (
    "logs/baseline_runs/failure_recovery_taxonomy_v2_repeated_real/"
    "failure_recovery_taxonomy_v2.repeated_summary.json"
)
DEFAULT_REPAIR_ABLATION_SUMMARY = (
    "logs/baseline_runs/failure_recovery_ambiguous_evidence_real_or_injected_v2_real/"
    "failure_recovery_ambiguous_evidence_real_or_injected_v2.baseline_summary.json"
)
DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY = (
    "logs/baseline_runs/failure_recovery_ambiguous_evidence_v2_repeated_real/"
    "failure_recovery_ambiguous_evidence_real_or_injected_v2.repeated_summary.json"
)
DEFAULT_AMBIGUOUS_FAILURE_MODE_BENCHMARK = (
    "egvr/benchmarks/failure_recovery_ambiguous_evidence_real_or_injected_v2.jsonl"
)
DEFAULT_AMBIGUOUS_FAILURE_MODE_FULL_RESULT = (
    "logs/baseline_runs/failure_recovery_ambiguous_evidence_real_or_injected_v2_real/"
    "failure_recovery_ambiguous_evidence_real_or_injected_v2.full_copilot.json"
)
DEFAULT_TASK_GENERALIZATION_SUMMARY = (
    "logs/baseline_runs/task_generalization_real_v1_real_scaffold_default/task_generalization_summary.json"
)
DEFAULT_TASK_GENERALIZATION_REAL_SUMMARY = "logs/baseline_runs/task_generalization_real_v1_real/task_generalization_summary.json"
DEFAULT_TASK_GENERALIZATION_FALLBACK_SUMMARIES = (
    DEFAULT_TASK_GENERALIZATION_REAL_SUMMARY,
    "logs/baseline_runs/task_generalization_v1_mock/task_generalization_summary.json",
)
DEFAULT_TOOL_ADMISSION_CARDS = "egvr/configs/tool_admission_cards.json"
DEFAULT_PROPERTY_VERIFIER_SUMMARY = (
    "logs/baseline_runs/verifier_enhancement_crossdocked30_v1/property_verifier_summary.json"
)
DEFAULT_PROPERTY_VERIFIER_FALLBACK_SUMMARY = (
    "logs/baseline_runs/verifier_enhancement_v1/property_verifier_summary.json"
)
DEFAULT_PDBBINDPLUS_POSE_SANITY_SUMMARY = (
    "logs/baseline_runs/pdbbindplus_pose_sanity_v2/pdbbindplus_pose_sanity_summary.json"
)
DEFAULT_PDBBINDPLUS_POSE_SANITY_FALLBACK_SUMMARIES = (
    "logs/baseline_runs/pdbbindplus_pose_sanity_v1/pdbbindplus_pose_sanity_summary.json",
)
DEFAULT_PDBBIND_READINESS_SUMMARY = (
    "logs/baseline_runs/pdbbind_readiness_probe_v1/pdbbindplus_v2020r1_readiness_summary.json"
)
DEFAULT_PDBBIND_RECEPTOR_PREP_SUMMARY = (
    "logs/baseline_runs/pdbbind_receptor_prep_probe_v1/pdbbindplus_v2020r1_prep_summary_50_fixed.json"
)
DEFAULT_PDBBIND_PREPARED_PILOT = (
    "egvr/benchmarks/pdbbindplus_v2020r1_prepared_pilot_30.jsonl"
)
DEFAULT_PDBBIND_PREPARED_PILOT_FALLBACKS = (
    "egvr/benchmarks/pdbbindplus_v2020r1_prepared_pilot_17.jsonl",
    "egvr/benchmarks/pdbbindplus_v2020r1_prepared_pilot_15.jsonl",
    "egvr/benchmarks/pdbbindplus_v2020r1_prepared_pilot_5.jsonl",
)
DEFAULT_PDBBIND_PREPARED_PILOT_RESULT = (
    "logs/baseline_runs/pdbbindplus_v2020r1_prepared_pilot_v3/"
    "pdbbindplus_v2020r1_prepared_pilot_30_result.json"
)
DEFAULT_PDBBIND_PREPARED_PILOT_RESULT_FALLBACKS = (
    "logs/baseline_runs/pdbbindplus_v2020r1_prepared_pilot_v2/"
    "pdbbindplus_v2020r1_prepared_pilot_17_result.json",
    "logs/baseline_runs/pdbbindplus_v2020r1_prepared_pilot_v2/"
    "pdbbindplus_v2020r1_prepared_pilot_15_result.json",
    "logs/baseline_runs/pdbbindplus_v2020r1_prepared_pilot_v1/"
    "pdbbindplus_v2020r1_prepared_pilot_5_result.json",
)
DEFAULT_LLM_ROUTER_BASELINE_SUMMARY = (
    "logs/baseline_runs/llm_as_router_planning_v1/llm_router_baseline_summary.json"
)
DEFAULT_NATURAL_FAILURE_AUDIT_SUMMARY = (
    "logs/baseline_runs/natural_failure_audit_v1/natural_failure_audit_summary.json"
)


def build_master_baseline_rows(
    *,
    crossdocked_record_path: str | Path,
    litpcba_result_path: str | Path,
    failure_record_path: str | Path,
    tool_exposure_summary_path: str | Path | None = DEFAULT_TOOL_EXPOSURE_SUMMARY,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read completed benchmark artifacts and return one normalized row per comparison line."""

    root = _project_root(project_root)
    crossdocked_path = _resolve_path(crossdocked_record_path, root)
    litpcba_path = _resolve_litpcba_result_path(litpcba_result_path, root)
    failure_path = _resolve_path(failure_record_path, root)
    tool_exposure_path = _resolve_path(tool_exposure_summary_path, root) if tool_exposure_summary_path else None

    rows = [
        _crossdocked_generation_row(_load_json(crossdocked_path), crossdocked_path, root),
        _litpcba_docking_row(_load_json(litpcba_path), litpcba_path, root),
    ]
    rows.extend(_failure_recovery_rows(_load_json(failure_path), failure_path, root))
    if tool_exposure_path:
        rows.extend(_tool_exposure_rows(_load_json(tool_exposure_path), tool_exposure_path, root))
    return rows


def build_master_baseline_table(
    *,
    output_dir: str | Path | None = None,
    record_output_path: str | Path | None = None,
    crossdocked_record_path: str | Path | None = None,
    litpcba_result_path: str | Path | None = None,
    failure_record_path: str | Path | None = None,
    tool_exposure_summary_path: str | Path | None = DEFAULT_TOOL_EXPOSURE_SUMMARY,
    verifier_evidence_summary_path: str | Path | None = DEFAULT_VERIFIER_EVIDENCE_SUMMARY,
    generation_scale_record_paths: list[str | Path] | tuple[str | Path, ...] | None = DEFAULT_GENERATION_SCALE_RECORDS,
    crossdocked_multiseed_summary_path: str | Path | None = DEFAULT_CROSSDOCKED_MULTISEED_SUMMARY,
    robustness_repeated_summary_path: str | Path | None = DEFAULT_ROBUSTNESS_REPEATED_SUMMARY,
    repair_ablation_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_SUMMARY,
    repair_ablation_repeated_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY,
    ambiguous_failure_mode_benchmark_path: str | Path | None = DEFAULT_AMBIGUOUS_FAILURE_MODE_BENCHMARK,
    ambiguous_failure_mode_full_result_path: str | Path | None = DEFAULT_AMBIGUOUS_FAILURE_MODE_FULL_RESULT,
    ambiguous_failure_mode_repeated_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY,
    task_generalization_summary_path: str | Path | None = DEFAULT_TASK_GENERALIZATION_SUMMARY,
    tool_admission_cards_path: str | Path | None = DEFAULT_TOOL_ADMISSION_CARDS,
    property_verifier_summary_path: str | Path | None = DEFAULT_PROPERTY_VERIFIER_SUMMARY,
    pdbbindplus_pose_sanity_summary_path: str | Path | None = DEFAULT_PDBBINDPLUS_POSE_SANITY_SUMMARY,
    pdbbind_readiness_summary_path: str | Path | None = DEFAULT_PDBBIND_READINESS_SUMMARY,
    pdbbind_receptor_prep_summary_path: str | Path | None = DEFAULT_PDBBIND_RECEPTOR_PREP_SUMMARY,
    pdbbind_prepared_pilot_path: str | Path | None = DEFAULT_PDBBIND_PREPARED_PILOT,
    pdbbind_prepared_pilot_result_path: str | Path | None = DEFAULT_PDBBIND_PREPARED_PILOT_RESULT,
    llm_router_baseline_summary_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = DEFAULT_LLM_ROUTER_BASELINE_SUMMARY,
    natural_failure_audit_summary_path: str | Path | None = DEFAULT_NATURAL_FAILURE_AUDIT_SUMMARY,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build rows from default artifacts and write CSV/JSON outputs."""

    root = _project_root(project_root)
    rows = build_master_baseline_rows(
        crossdocked_record_path=crossdocked_record_path or DEFAULT_CROSSDOCKED_RECORD,
        litpcba_result_path=litpcba_result_path or DEFAULT_LITPCBA_RESULT,
        failure_record_path=failure_record_path or DEFAULT_FAILURE_RECORD,
        tool_exposure_summary_path=tool_exposure_summary_path,
        project_root=root,
    )
    return write_master_table(
        rows,
        output_dir=output_dir or _resolve_path(DEFAULT_OUTPUT_DIR, root),
        record_output_path=record_output_path or _resolve_path(DEFAULT_RECORD_OUTPUT, root),
        verifier_evidence_summary_path=verifier_evidence_summary_path,
        generation_scale_record_paths=generation_scale_record_paths,
        crossdocked_multiseed_summary_path=crossdocked_multiseed_summary_path,
        robustness_repeated_summary_path=robustness_repeated_summary_path,
        repair_ablation_summary_path=repair_ablation_summary_path,
        repair_ablation_repeated_summary_path=repair_ablation_repeated_summary_path,
        ambiguous_failure_mode_benchmark_path=ambiguous_failure_mode_benchmark_path,
        ambiguous_failure_mode_full_result_path=ambiguous_failure_mode_full_result_path,
        ambiguous_failure_mode_repeated_summary_path=ambiguous_failure_mode_repeated_summary_path,
        task_generalization_summary_path=task_generalization_summary_path,
        tool_admission_cards_path=tool_admission_cards_path,
        property_verifier_summary_path=property_verifier_summary_path,
        pdbbindplus_pose_sanity_summary_path=pdbbindplus_pose_sanity_summary_path,
        pdbbind_readiness_summary_path=pdbbind_readiness_summary_path,
        pdbbind_receptor_prep_summary_path=pdbbind_receptor_prep_summary_path,
        pdbbind_prepared_pilot_path=pdbbind_prepared_pilot_path,
        pdbbind_prepared_pilot_result_path=pdbbind_prepared_pilot_result_path,
        llm_router_baseline_summary_path=llm_router_baseline_summary_path,
        natural_failure_audit_summary_path=natural_failure_audit_summary_path,
        project_root=root,
    )


def write_master_table(
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    record_output_path: str | Path | None = None,
    verifier_evidence_summary_path: str | Path | None = DEFAULT_VERIFIER_EVIDENCE_SUMMARY,
    generation_scale_record_paths: list[str | Path] | tuple[str | Path, ...] | None = DEFAULT_GENERATION_SCALE_RECORDS,
    crossdocked_multiseed_summary_path: str | Path | None = DEFAULT_CROSSDOCKED_MULTISEED_SUMMARY,
    robustness_repeated_summary_path: str | Path | None = DEFAULT_ROBUSTNESS_REPEATED_SUMMARY,
    repair_ablation_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_SUMMARY,
    repair_ablation_repeated_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY,
    ambiguous_failure_mode_benchmark_path: str | Path | None = DEFAULT_AMBIGUOUS_FAILURE_MODE_BENCHMARK,
    ambiguous_failure_mode_full_result_path: str | Path | None = DEFAULT_AMBIGUOUS_FAILURE_MODE_FULL_RESULT,
    ambiguous_failure_mode_repeated_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY,
    task_generalization_summary_path: str | Path | None = DEFAULT_TASK_GENERALIZATION_SUMMARY,
    tool_admission_cards_path: str | Path | None = DEFAULT_TOOL_ADMISSION_CARDS,
    property_verifier_summary_path: str | Path | None = DEFAULT_PROPERTY_VERIFIER_SUMMARY,
    pdbbindplus_pose_sanity_summary_path: str | Path | None = DEFAULT_PDBBINDPLUS_POSE_SANITY_SUMMARY,
    pdbbind_readiness_summary_path: str | Path | None = DEFAULT_PDBBIND_READINESS_SUMMARY,
    pdbbind_receptor_prep_summary_path: str | Path | None = DEFAULT_PDBBIND_RECEPTOR_PREP_SUMMARY,
    pdbbind_prepared_pilot_path: str | Path | None = DEFAULT_PDBBIND_PREPARED_PILOT,
    pdbbind_prepared_pilot_result_path: str | Path | None = DEFAULT_PDBBIND_PREPARED_PILOT_RESULT,
    llm_router_baseline_summary_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = DEFAULT_LLM_ROUTER_BASELINE_SUMMARY,
    natural_failure_audit_summary_path: str | Path | None = DEFAULT_NATURAL_FAILURE_AUDIT_SUMMARY,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write master rows as CSV plus JSON, optionally mirrored as a benchmark record."""

    root = _project_root(project_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "master_baseline_table.csv"
    json_path = output_path / "master_baseline_table.json"
    record_path = Path(record_output_path) if record_output_path else None
    paper_views = build_paper_table_views(
        rows,
        verifier_evidence_summary_path=verifier_evidence_summary_path,
        generation_scale_record_paths=generation_scale_record_paths,
        crossdocked_multiseed_summary_path=crossdocked_multiseed_summary_path,
        robustness_repeated_summary_path=robustness_repeated_summary_path,
        repair_ablation_summary_path=repair_ablation_summary_path,
        repair_ablation_repeated_summary_path=repair_ablation_repeated_summary_path,
        ambiguous_failure_mode_benchmark_path=ambiguous_failure_mode_benchmark_path,
        ambiguous_failure_mode_full_result_path=ambiguous_failure_mode_full_result_path,
        ambiguous_failure_mode_repeated_summary_path=ambiguous_failure_mode_repeated_summary_path,
        task_generalization_summary_path=task_generalization_summary_path,
        tool_admission_cards_path=tool_admission_cards_path,
        property_verifier_summary_path=property_verifier_summary_path,
        pdbbindplus_pose_sanity_summary_path=pdbbindplus_pose_sanity_summary_path,
        pdbbind_readiness_summary_path=pdbbind_readiness_summary_path,
        pdbbind_receptor_prep_summary_path=pdbbind_receptor_prep_summary_path,
        pdbbind_prepared_pilot_path=pdbbind_prepared_pilot_path,
        pdbbind_prepared_pilot_result_path=pdbbind_prepared_pilot_result_path,
        llm_router_baseline_summary_path=llm_router_baseline_summary_path,
        natural_failure_audit_summary_path=natural_failure_audit_summary_path,
        project_root=root,
    )
    view_artifacts = write_paper_table_views(paper_views, output_path, project_root=root)
    payload = {
        "table_id": "master_baseline_table",
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "columns": MASTER_TABLE_COLUMNS,
        "row_count": len(rows),
        "artifacts": {
            "csv": _display_path(csv_path, root),
            "json": _display_path(json_path, root),
            "record": _display_path(record_path, root) if record_path else None,
            "views": view_artifacts,
        },
        "rows": rows,
        "views": paper_views,
        "notes": [
            "Rows are aggregated from completed benchmark artifacts; this command does not rerun tool execution.",
            "Blank metric cells mean the metric is not applicable to that benchmark family.",
            "Paper views include aggregate projections plus failure-taxonomy and ablation views for rebuttal support.",
        ],
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(rows, csv_path)
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_paper_table_views(
    rows: list[dict[str, Any]],
    *,
    project_root: str | Path | None = None,
    failure_taxonomy_benchmark_path: str | Path = DEFAULT_FAILURE_TAXONOMY_BENCHMARK,
    failure_taxonomy_rule_result_path: str | Path = DEFAULT_FAILURE_TAXONOMY_RULE_RESULT,
    failure_taxonomy_full_result_path: str | Path = DEFAULT_FAILURE_TAXONOMY_FULL_RESULT,
    verifier_evidence_summary_path: str | Path | None = DEFAULT_VERIFIER_EVIDENCE_SUMMARY,
    generation_scale_record_paths: list[str | Path] | tuple[str | Path, ...] | None = DEFAULT_GENERATION_SCALE_RECORDS,
    crossdocked_multiseed_summary_path: str | Path | None = DEFAULT_CROSSDOCKED_MULTISEED_SUMMARY,
    robustness_repeated_summary_path: str | Path | None = DEFAULT_ROBUSTNESS_REPEATED_SUMMARY,
    repair_ablation_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_SUMMARY,
    repair_ablation_repeated_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY,
    ambiguous_failure_mode_benchmark_path: str | Path | None = DEFAULT_AMBIGUOUS_FAILURE_MODE_BENCHMARK,
    ambiguous_failure_mode_full_result_path: str | Path | None = DEFAULT_AMBIGUOUS_FAILURE_MODE_FULL_RESULT,
    ambiguous_failure_mode_repeated_summary_path: str | Path | None = DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY,
    task_generalization_summary_path: str | Path | None = DEFAULT_TASK_GENERALIZATION_SUMMARY,
    tool_admission_cards_path: str | Path | None = DEFAULT_TOOL_ADMISSION_CARDS,
    property_verifier_summary_path: str | Path | None = DEFAULT_PROPERTY_VERIFIER_SUMMARY,
    pdbbindplus_pose_sanity_summary_path: str | Path | None = DEFAULT_PDBBINDPLUS_POSE_SANITY_SUMMARY,
    pdbbind_readiness_summary_path: str | Path | None = DEFAULT_PDBBIND_READINESS_SUMMARY,
    pdbbind_receptor_prep_summary_path: str | Path | None = DEFAULT_PDBBIND_RECEPTOR_PREP_SUMMARY,
    pdbbind_prepared_pilot_path: str | Path | None = DEFAULT_PDBBIND_PREPARED_PILOT,
    pdbbind_prepared_pilot_result_path: str | Path | None = DEFAULT_PDBBIND_PREPARED_PILOT_RESULT,
    llm_router_baseline_summary_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = DEFAULT_LLM_ROUTER_BASELINE_SUMMARY,
    natural_failure_audit_summary_path: str | Path | None = DEFAULT_NATURAL_FAILURE_AUDIT_SUMMARY,
) -> dict[str, dict[str, Any]]:
    """Create focused paper-table views from master rows."""

    root = _project_root(project_root)
    robustness_rows = [
        _ordered_view_row(_robustness_view_row(row), ROBUSTNESS_TABLE_COLUMNS)
        for row in rows
        if row.get("benchmark_family") == "failure_recovery"
    ]
    throughput_rows = [
        _ordered_view_row(_throughput_view_row(row), THROUGHPUT_TABLE_COLUMNS)
        for row in rows
        if _has_any_metric(row, ["generated_candidate_count", "valid_candidate_count", "docking_success_count"])
    ]
    generation_quality_rows = [
        _ordered_view_row(_generation_quality_view_row(row), GENERATION_QUALITY_TABLE_COLUMNS)
        for row in rows
        if row.get("generated_candidate_count") is not None
    ]
    generation_scale_rows = _generation_scale_rows(generation_scale_record_paths, root)
    crossdocked_multiseed_rows = _crossdocked_multiseed_rows(
        _resolve_path(crossdocked_multiseed_summary_path, root) if crossdocked_multiseed_summary_path else None
    )
    tool_exposure_rows = [
        _ordered_view_row(_tool_exposure_view_row(row), TOOL_EXPOSURE_TABLE_COLUMNS)
        for row in rows
        if row.get("benchmark_family") == "tool_exposure_budget"
    ]
    failure_taxonomy_rows = _failure_taxonomy_rows(
        benchmark_path=_resolve_path(failure_taxonomy_benchmark_path, root),
        rule_result_path=_resolve_path(failure_taxonomy_rule_result_path, root),
        full_result_path=_resolve_path(failure_taxonomy_full_result_path, root),
    )
    ablation_rows = _ablation_rows(tool_exposure_rows, robustness_rows)
    robustness_repeated_rows = _robustness_repeated_rows(
        _resolve_path(robustness_repeated_summary_path, root) if robustness_repeated_summary_path else None
    )
    repair_ablation_rows = _repair_ablation_rows(
        _resolve_path(repair_ablation_summary_path, root) if repair_ablation_summary_path else None
    )
    repair_ablation_repeated_rows = _repair_ablation_repeated_rows(
        _resolve_path(repair_ablation_repeated_summary_path, root) if repair_ablation_repeated_summary_path else None
    )
    ambiguous_failure_mode_rows = _ambiguous_failure_mode_rows(
        benchmark_path=_resolve_path(ambiguous_failure_mode_benchmark_path, root)
        if ambiguous_failure_mode_benchmark_path
        else None,
        full_result_path=_resolve_path(ambiguous_failure_mode_full_result_path, root)
        if ambiguous_failure_mode_full_result_path
        else None,
    )
    ambiguous_failure_mode_repeated_rows = _ambiguous_failure_mode_repeated_rows(
        benchmark_path=_resolve_path(ambiguous_failure_mode_benchmark_path, root)
        if ambiguous_failure_mode_benchmark_path
        else None,
        repeated_summary_path=_resolve_path(ambiguous_failure_mode_repeated_summary_path, root)
        if ambiguous_failure_mode_repeated_summary_path
        else None,
        project_root=root,
    )
    task_generalization_rows = _task_generalization_rows(
        _resolve_task_generalization_summary_path(task_generalization_summary_path, root)
    )
    tool_admission_rows = _tool_admission_rows(
        _resolve_path(tool_admission_cards_path, root) if tool_admission_cards_path else None
    )
    verifier_evidence_summary = _resolve_summary_path_with_default_fallback(
        verifier_evidence_summary_path,
        root,
        default_path=DEFAULT_VERIFIER_EVIDENCE_SUMMARY,
        fallback_path=DEFAULT_VERIFIER_EVIDENCE_FALLBACK_SUMMARY,
    )
    property_verifier_summary = _resolve_summary_path_with_default_fallback(
        property_verifier_summary_path,
        root,
        default_path=DEFAULT_PROPERTY_VERIFIER_SUMMARY,
        fallback_path=DEFAULT_PROPERTY_VERIFIER_FALLBACK_SUMMARY,
    )
    verifier_evidence_rows = _verifier_evidence_rows(verifier_evidence_summary)
    property_verifier_rows = _property_verifier_rows(property_verifier_summary)
    verifier_evidence_rows.extend(
        _property_verifier_evidence_rows(property_verifier_summary)
    )
    pdbbindplus_pose_sanity_summary = _resolve_path_with_default_fallbacks(
        pdbbindplus_pose_sanity_summary_path,
        root,
        default_path=DEFAULT_PDBBINDPLUS_POSE_SANITY_SUMMARY,
        fallback_paths=DEFAULT_PDBBINDPLUS_POSE_SANITY_FALLBACK_SUMMARIES,
    )
    verifier_evidence_rows.extend(_verifier_evidence_rows(pdbbindplus_pose_sanity_summary))
    posebusters_failure_mode_rows = _posebusters_failure_mode_rows(verifier_evidence_summary)
    posebusters_failure_mode_rows.extend(_posebusters_failure_mode_rows(pdbbindplus_pose_sanity_summary))
    posebusters_top_failure_rows = _posebusters_top_failure_rows(posebusters_failure_mode_rows)
    resolved_pdbbind_prepared_pilot = _resolve_path_with_default_fallbacks(
        pdbbind_prepared_pilot_path,
        root,
        default_path=DEFAULT_PDBBIND_PREPARED_PILOT,
        fallback_paths=DEFAULT_PDBBIND_PREPARED_PILOT_FALLBACKS,
    )
    resolved_pdbbind_prepared_pilot_result = _resolve_path_with_default_fallbacks(
        pdbbind_prepared_pilot_result_path,
        root,
        default_path=DEFAULT_PDBBIND_PREPARED_PILOT_RESULT,
        fallback_paths=DEFAULT_PDBBIND_PREPARED_PILOT_RESULT_FALLBACKS,
    )
    pdbbind_prep_gate_rows = _pdbbind_prep_gate_rows(
        readiness_summary_path=_resolve_path(pdbbind_readiness_summary_path, root)
        if pdbbind_readiness_summary_path
        else None,
        receptor_prep_summary_path=_resolve_path(pdbbind_receptor_prep_summary_path, root)
        if pdbbind_receptor_prep_summary_path
        else None,
        prepared_pilot_path=resolved_pdbbind_prepared_pilot,
        prepared_pilot_result_path=resolved_pdbbind_prepared_pilot_result,
    )
    base_views = {
        "robustness": {
            "description": "Failure-recovery robustness under controlled tool failure injection.",
            "columns": ROBUSTNESS_TABLE_COLUMNS,
            "row_count": len(robustness_rows),
            "rows": robustness_rows,
        },
        "robustness_repeated": {
            "description": "Repeated robustness aggregate for controlled failure injection.",
            "columns": ROBUSTNESS_REPEATED_TABLE_COLUMNS,
            "row_count": len(robustness_repeated_rows),
            "rows": robustness_repeated_rows,
        },
        "throughput": {
            "description": "Execution throughput and count metrics from completed real benchmark artifacts.",
            "columns": THROUGHPUT_TABLE_COLUMNS,
            "row_count": len(throughput_rows),
            "rows": throughput_rows,
        },
        "generation_quality": {
            "description": "Candidate validity, uniqueness, synthesizability, and toxicity-oriented generation metrics.",
            "columns": GENERATION_QUALITY_TABLE_COLUMNS,
            "row_count": len(generation_quality_rows),
            "rows": generation_quality_rows,
        },
        "generation_scale": {
            "description": "Generation scale-up rows for CrossDocked target slices.",
            "columns": GENERATION_SCALE_TABLE_COLUMNS,
            "row_count": len(generation_scale_rows),
            "rows": generation_scale_rows,
        },
        "crossdocked_multiseed": {
            "description": "CrossDocked30 repeatability across RxnFlow seeds.",
            "columns": CROSSDOCKED_MULTISEED_TABLE_COLUMNS,
            "row_count": len(crossdocked_multiseed_rows),
            "rows": crossdocked_multiseed_rows,
        },
        "tool_exposure": {
            "description": "Tool exposure, over-selection, and planning precision across planner baselines.",
            "columns": TOOL_EXPOSURE_TABLE_COLUMNS,
            "row_count": len(tool_exposure_rows),
            "rows": tool_exposure_rows,
        },
        "failure_taxonomy": {
            "description": "Per-scenario taxonomy for the failure-recovery robustness slice.",
            "columns": FAILURE_TAXONOMY_TABLE_COLUMNS,
            "row_count": len(failure_taxonomy_rows),
            "rows": failure_taxonomy_rows,
        },
        "ablation": {
            "description": "Rebuttal-oriented ablation combining initial tool exposure with robustness repair metrics.",
            "columns": ABLATION_TABLE_COLUMNS,
            "row_count": len(ablation_rows),
            "rows": ablation_rows,
        },
        "repair_ablation": {
            "description": "Ablation separating verifier-guided repair from scheduled fallback and verifier-only baselines.",
            "columns": REPAIR_ABLATION_TABLE_COLUMNS,
            "row_count": len(repair_ablation_rows),
            "rows": repair_ablation_rows,
        },
        "repair_ablation_repeated": {
            "description": "Repeated repair-ablation aggregate for ambiguous-evidence wrapper injection.",
            "columns": REPAIR_ABLATION_REPEATED_TABLE_COLUMNS,
            "row_count": len(repair_ablation_repeated_rows),
            "rows": repair_ablation_repeated_rows,
        },
        "ambiguous_failure_modes": {
            "description": "Appendix breakdown of controlled ambiguous-evidence modes and verifier-triggered real retries.",
            "columns": AMBIGUOUS_FAILURE_MODE_TABLE_COLUMNS,
            "row_count": len(ambiguous_failure_mode_rows),
            "rows": ambiguous_failure_mode_rows,
        },
        "ambiguous_failure_modes_repeated": {
            "description": "Repeated appendix breakdown of ambiguous-evidence modes and verifier-triggered real retries.",
            "columns": AMBIGUOUS_FAILURE_MODE_REPEATED_TABLE_COLUMNS,
            "row_count": len(ambiguous_failure_mode_repeated_rows),
            "rows": ambiguous_failure_mode_repeated_rows,
        },
        "task_generalization": {
            "description": "Task-family coverage for non-pocket generation workflows.",
            "columns": TASK_GENERALIZATION_TABLE_COLUMNS,
            "row_count": len(task_generalization_rows),
            "rows": task_generalization_rows,
        },
        "tool_admission": {
            "description": "Tool admission cards for deciding which new tools support the paper claim.",
            "columns": TOOL_ADMISSION_TABLE_COLUMNS,
            "row_count": len(tool_admission_rows),
            "rows": tool_admission_rows,
        },
        "verifier_evidence": {
            "description": "Chemistry-grounded verifier evidence beyond tool-call success.",
            "columns": VERIFIER_EVIDENCE_TABLE_COLUMNS,
            "row_count": len(verifier_evidence_rows),
            "rows": verifier_evidence_rows,
        },
        "property_verifier": {
            "description": "RDKit property-verifier evidence for generated candidates.",
            "columns": PROPERTY_VERIFIER_TABLE_COLUMNS,
            "row_count": len(property_verifier_rows),
            "rows": property_verifier_rows,
        },
        "posebusters_top_failures": {
            "description": "Compact main-text view of the highest-frequency PoseBusters failures among evaluated docking poses.",
            "columns": POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS,
            "row_count": len(posebusters_top_failure_rows),
            "rows": posebusters_top_failure_rows,
        },
        "posebusters_failure_modes": {
            "description": "Per-check PoseBusters failure modes for evaluated docking-pose appendices.",
            "columns": POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS,
            "row_count": len(posebusters_failure_mode_rows),
            "rows": posebusters_failure_mode_rows,
        },
        "pdbbind_prep_gate": {
            "description": "PDBbind local-data readiness and receptor-preparation gate for deferred docking pilot evidence.",
            "columns": PDBBIND_PREP_GATE_TABLE_COLUMNS,
            "row_count": len(pdbbind_prep_gate_rows),
            "rows": pdbbind_prep_gate_rows,
        },
    }
    llm_router_baseline_rows = _llm_router_baseline_rows(
        _resolve_llm_router_summary_paths(llm_router_baseline_summary_path, root)
    )
    natural_failure_audit_rows = _natural_failure_audit_rows(
        _resolve_path(natural_failure_audit_summary_path, root) if natural_failure_audit_summary_path else None
    )
    extended_views = {
        **base_views,
        "llm_router_baseline": {
            "description": "Planning-only LLM-as-router baseline validation.",
            "columns": LLM_ROUTER_BASELINE_TABLE_COLUMNS,
            "row_count": len(llm_router_baseline_rows),
            "rows": llm_router_baseline_rows,
        },
        "natural_failure_audit": {
            "description": "Observed natural failures from existing real-run traces, excluding controlled injection.",
            "columns": NATURAL_FAILURE_AUDIT_TABLE_COLUMNS,
            "row_count": len(natural_failure_audit_rows),
            "rows": natural_failure_audit_rows,
        },
    }
    evidence_audit_rows = build_evidence_audit_rows(master_payload={"views": extended_views}, project_root=root)
    extended_views["evidence_audit"] = {
        "description": "Claim-to-evidence audit for paper claims.",
        "columns": EVIDENCE_AUDIT_TABLE_COLUMNS,
        "row_count": len(evidence_audit_rows),
        "rows": evidence_audit_rows,
    }
    statistical_summary_rows = build_statistical_summary_rows(
        master_payload={"views": extended_views},
        project_root=root,
    )
    extended_views["statistical_summary"] = {
        "description": "Statistical summary of key rate metrics with Wilson confidence intervals.",
        "columns": STATISTICAL_SUMMARY_TABLE_COLUMNS,
        "row_count": len(statistical_summary_rows),
        "rows": statistical_summary_rows,
    }
    return extended_views


def write_paper_table_views(
    views: dict[str, dict[str, Any]],
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """Write derived paper-table views as paired CSV/JSON files."""

    root = _project_root(project_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, str]] = {}
    for view_name, view_payload in views.items():
        columns = PAPER_VIEW_COLUMNS[view_name]
        csv_path = output_path / f"{view_name}_table.csv"
        json_path = output_path / f"{view_name}_table.json"
        json_path.write_text(
            json.dumps(view_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_csv(view_payload["rows"], csv_path, columns=columns)
        artifacts[view_name] = {
            "csv": _display_path(csv_path, root) or str(csv_path),
            "json": _display_path(json_path, root) or str(json_path),
        }
    return artifacts


def _crossdocked_generation_row(data: dict[str, Any], record_path: Path, project_root: Path) -> dict[str, Any]:
    summary = data.get("summary", {})
    global_candidates = data.get("global_candidate_summary", {})
    per_target = data.get("per_target", [])
    task_count = _as_int(data.get("task_count", summary.get("total", len(per_target))))
    task_success_rate = _as_float(summary.get("task_success_rate"))
    best_scscores = [_as_float(item.get("best_scscore")) for item in per_target]
    max_toxicity_scores = [_as_float(item.get("max_toxicity_score")) for item in per_target]
    elapsed_values = [
        _as_float(item.get("tool_elapsed_sec", {}).get("total"))
        for item in per_target
        if isinstance(item.get("tool_elapsed_sec"), dict)
    ]
    return _ordered_row(
        {
            "benchmark_family": "crossdocked_generation",
            "dataset": data.get("dataset", "CrossDocked2020"),
            "benchmark_id": data.get("benchmark_id", record_path.stem.removesuffix(".summary")),
            "execution_mode": data.get("execution_mode", "real"),
            "planner_baseline": data.get("planner_baseline", "rule_based_planner"),
            "task_count": task_count,
            "task_success_count": _success_count(task_count, task_success_rate),
            "failed_task_count": task_count - _success_count(task_count, task_success_rate),
            "task_success_rate": task_success_rate,
            "parser_accuracy": _as_float(summary.get("parser_accuracy")),
            "planner_tool_coverage_rate": _as_float(summary.get("planner_tool_coverage_rate")),
            "planner_tool_precision": _as_float(summary.get("planner_tool_precision")),
            "planner_tool_recall": _as_float(summary.get("planner_tool_recall")),
            "planner_tool_f1": _as_float(summary.get("planner_tool_f1")),
            "mean_selected_tool_count": _as_float(summary.get("mean_selected_tool_count")),
            "mean_tool_sequence_length": _as_float(summary.get("mean_tool_sequence_length")),
            "mean_extra_tool_count": _as_float(summary.get("mean_extra_tool_count")),
            "mean_tool_call_count": _as_float(summary.get("mean_tool_call_count")),
            "failed_tool_call_count": _as_int(summary.get("failed_tool_call_count")),
            "tool_call_failure_rate": _as_float(summary.get("tool_call_failure_rate")),
            "verifier_expectation_match": _as_float(summary.get("verifier_expectation_match")),
            "generated_candidate_count": _as_int(global_candidates.get("generated_candidate_count")),
            "valid_candidate_count": _as_int(global_candidates.get("valid_candidate_count")),
            "unique_smiles_count": _as_int(global_candidates.get("unique_smiles_count_across_tasks")),
            "best_scscore": _min_non_null(best_scscores),
            "max_toxicity_score": _max_non_null(max_toxicity_scores),
            "mean_total_elapsed_sec": _mean_non_null(elapsed_values),
            "result_path": data.get("result_path"),
            "record_path": _display_path(record_path, project_root),
            "notes": " ".join(data.get("notes", [])),
        }
    )


def _litpcba_docking_row(data: dict[str, Any], result_path: Path, project_root: Path) -> dict[str, Any]:
    summary = data.get("summary", {})
    results = data.get("results", [])
    task_count = _as_int(summary.get("total", len(results)))
    task_success_count = sum(1 for item in results if item.get("task_success"))
    docking_scores = [
        _as_float(item.get("metrics", {}).get("best_docking_score"))
        for item in results
        if isinstance(item.get("metrics"), dict)
    ]
    tool_metrics = _tool_metrics_from_results(results)
    benchmark_id = result_path.stem.removesuffix("_elapsed_result").removesuffix("_result")
    elapsed_values = [
        _as_float(item.get("total_elapsed_sec"))
        for item in results
        if isinstance(item, dict)
    ]
    return _ordered_row(
        {
            "benchmark_family": "litpcba_docking",
            "dataset": "LIT-PCBA",
            "benchmark_id": benchmark_id,
            "execution_mode": "real",
            "planner_baseline": data.get("planner_baseline", "rule_based_planner"),
            "task_count": task_count,
            "task_success_count": task_success_count,
            "failed_task_count": task_count - task_success_count,
            "task_success_rate": _as_float(summary.get("task_success_rate")),
            "parser_accuracy": _as_float(summary.get("parser_accuracy")),
            "planner_tool_coverage_rate": tool_metrics["coverage_rate"],
            "planner_tool_precision": tool_metrics["precision"],
            "planner_tool_recall": _as_float(summary.get("planner_tool_recall", tool_metrics["recall"])),
            "planner_tool_f1": tool_metrics["f1"],
            "mean_selected_tool_count": _as_float(summary.get("mean_selected_tool_count")),
            "mean_tool_sequence_length": _as_float(summary.get("mean_tool_sequence_length")),
            "mean_extra_tool_count": _as_float(summary.get("mean_extra_tool_count")),
            "mean_tool_call_count": _as_float(summary.get("mean_tool_call_count")),
            "failed_tool_call_count": _as_int(summary.get("failed_tool_call_count")),
            "tool_call_failure_rate": _as_float(summary.get("tool_call_failure_rate")),
            "verifier_expectation_match": _as_float(summary.get("verifier_expectation_match")),
            "docking_success_count": len([score for score in docking_scores if score is not None]),
            "best_docking_score": _min_non_null(docking_scores),
            "mean_docking_score": _mean_non_null(docking_scores),
            "mean_total_elapsed_sec": _as_float(summary.get("mean_total_elapsed_sec"))
            or _mean_non_null(elapsed_values),
            "result_path": _display_path(result_path, project_root),
            "record_path": None,
            "notes": "Prepared 15-target LIT-PCBA Vina docking slice; lower docking score is better.",
        }
    )


def _failure_recovery_rows(data: dict[str, Any], record_path: Path, project_root: Path) -> list[dict[str, Any]]:
    if isinstance(data.get("rows"), list) and "baseline_results" not in data:
        return _failure_recovery_rows_from_suite_summary(data, record_path, project_root)

    summary_json_path = data.get("summary_json")
    suite_payload = _load_optional_json(_resolve_path(summary_json_path, project_root)) if summary_json_path else {}
    suite_rows = {
        item.get("planner_baseline"): item
        for item in suite_payload.get("rows", [])
        if isinstance(item, dict) and item.get("planner_baseline")
    }
    suite_results = suite_payload.get("results", {}) if isinstance(suite_payload.get("results"), dict) else {}
    full_copilot_cases = data.get("full_copilot_case_results", [])
    rows: list[dict[str, Any]] = []
    for baseline_result in data.get("baseline_results", []):
        baseline = baseline_result.get("planner_baseline")
        suite_row = suite_rows.get(baseline, {})
        suite_summary = suite_results.get(baseline, {}).get("summary", {})
        task_count = _as_int(
            suite_row.get("total", suite_summary.get("total", data.get("task_count", len(full_copilot_cases))))
        )
        task_success_count = _as_int(baseline_result.get("task_success_count", suite_row.get("task_success_count")))
        failed_task_count = _as_int(baseline_result.get("failed_task_count", suite_row.get("failed_task_count")))
        candidate_metrics = _failure_candidate_metrics(full_copilot_cases if _is_repair_baseline(baseline) else [])
        rows.append(
            _ordered_row(
                {
                    "benchmark_family": "failure_recovery",
                    "dataset": "controlled_tool_failure_injection",
                    "benchmark_id": data.get("benchmark_id", record_path.stem.removesuffix(".real.summary")),
                    "execution_mode": data.get("execution_mode", "real"),
                    "planner_baseline": baseline,
                    "task_count": task_count,
                    "task_success_count": task_success_count,
                    "failed_task_count": failed_task_count,
                    "task_success_rate": _as_float(
                        baseline_result.get("task_success_rate", suite_summary.get("task_success_rate"))
                    ),
                    "parser_accuracy": _as_float(suite_summary.get("parser_accuracy")),
                    "planner_tool_coverage_rate": _as_float(suite_summary.get("planner_tool_coverage_rate")),
                    "planner_tool_precision": _as_float(suite_summary.get("planner_tool_precision")),
                    "planner_tool_recall": _as_float(suite_summary.get("planner_tool_recall")),
                    "planner_tool_f1": _as_float(suite_summary.get("planner_tool_f1")),
                    "mean_selected_tool_count": _as_float(suite_summary.get("mean_selected_tool_count")),
                    "mean_tool_sequence_length": _as_float(suite_summary.get("mean_tool_sequence_length")),
                    "mean_extra_tool_count": _as_float(suite_summary.get("mean_extra_tool_count")),
                    "mean_tool_call_count": _as_float(suite_summary.get("mean_tool_call_count")),
                    "failed_tool_call_count": _as_int(suite_summary.get("failed_tool_call_count")),
                    "tool_call_failure_rate": _as_float(suite_summary.get("tool_call_failure_rate")),
                    "verifier_expectation_match": _as_float(
                        baseline_result.get(
                            "verifier_expectation_match", suite_summary.get("verifier_expectation_match")
                        )
                    ),
                    "repair_attempt_count": _as_int(
                        baseline_result.get("repair_attempt_count", suite_summary.get("repair_attempt_count"))
                    ),
                    "repair_success_count": _as_int(
                        baseline_result.get("repair_success_count", suite_summary.get("repair_success_count"))
                    ),
                    "repair_success_rate": _as_float(
                        baseline_result.get("repair_success_rate", suite_summary.get("repair_success_rate"))
                    ),
                    "generated_candidate_count": candidate_metrics["candidate_count"],
                    "valid_candidate_count": candidate_metrics["valid_smiles_count"],
                    "unique_smiles_count": candidate_metrics["unique_smiles_count"],
                    "best_scscore": candidate_metrics["best_scscore"],
                    "max_toxicity_score": candidate_metrics["max_toxicity_score"],
                    "result_path": suite_row.get("result_path") or _failure_result_path(data, baseline),
                    "record_path": _display_path(record_path, project_root),
                    "notes": _failure_notes(data, baseline),
                }
            )
        )
    return rows


def _failure_recovery_rows_from_suite_summary(
    data: dict[str, Any],
    record_path: Path,
    project_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmark_id = data.get("benchmark_id", record_path.stem.removesuffix(".baseline_summary"))
    execution_mode = data.get("execution_mode", "real")
    dataset = data.get("dataset", "controlled_tool_failure_injection")
    suite_results = data.get("results", {}) if isinstance(data.get("results"), dict) else {}
    for suite_row in data.get("rows", []):
        if not isinstance(suite_row, dict):
            continue
        baseline = suite_row.get("planner_baseline")
        result_payload = suite_results.get(baseline, {}) if isinstance(suite_results.get(baseline), dict) else {}
        result_summary = result_payload.get("summary", {}) if isinstance(result_payload.get("summary"), dict) else {}
        task_count = _as_int(suite_row.get("total", result_summary.get("total")))
        task_success_count = _as_int(suite_row.get("task_success_count"))
        failed_task_count = _as_int(suite_row.get("failed_task_count"))
        if suite_row.get("failed_task_count") is None and task_count:
            failed_task_count = max(task_count - task_success_count, 0)
        rows.append(
            _ordered_row(
                {
                    "benchmark_family": "failure_recovery",
                    "dataset": dataset,
                    "benchmark_id": benchmark_id,
                    "execution_mode": execution_mode,
                    "planner_baseline": baseline,
                    "task_count": task_count,
                    "task_success_count": task_success_count,
                    "failed_task_count": failed_task_count,
                    "task_success_rate": _as_float(
                        suite_row.get("task_success_rate", result_summary.get("task_success_rate"))
                    ),
                    "parser_accuracy": _as_float(
                        suite_row.get("parser_accuracy", result_summary.get("parser_accuracy"))
                    ),
                    "planner_tool_coverage_rate": _as_float(
                        suite_row.get(
                            "planner_tool_coverage_rate",
                            result_summary.get("planner_tool_coverage_rate"),
                        )
                    ),
                    "planner_tool_precision": _as_float(
                        suite_row.get("planner_tool_precision", result_summary.get("planner_tool_precision"))
                    ),
                    "planner_tool_recall": _as_float(
                        suite_row.get("planner_tool_recall", result_summary.get("planner_tool_recall"))
                    ),
                    "planner_tool_f1": _as_float(
                        suite_row.get("planner_tool_f1", result_summary.get("planner_tool_f1"))
                    ),
                    "mean_selected_tool_count": _as_float(
                        suite_row.get("mean_selected_tool_count", result_summary.get("mean_selected_tool_count"))
                    ),
                    "mean_tool_sequence_length": _as_float(
                        suite_row.get(
                            "mean_tool_sequence_length",
                            result_summary.get("mean_tool_sequence_length"),
                        )
                    ),
                    "mean_extra_tool_count": _as_float(
                        suite_row.get("mean_extra_tool_count", result_summary.get("mean_extra_tool_count"))
                    ),
                    "mean_tool_call_count": _as_float(
                        suite_row.get("mean_tool_call_count", result_summary.get("mean_tool_call_count"))
                    ),
                    "failed_tool_call_count": _as_int(
                        suite_row.get("failed_tool_call_count", result_summary.get("failed_tool_call_count"))
                    ),
                    "tool_call_failure_rate": _as_float(
                        suite_row.get("tool_call_failure_rate", result_summary.get("tool_call_failure_rate"))
                    ),
                    "verifier_expectation_match": _as_float(
                        suite_row.get(
                            "verifier_expectation_match",
                            result_summary.get("verifier_expectation_match"),
                        )
                    ),
                    "repair_attempt_count": _as_int(
                        suite_row.get("repair_attempt_count", result_summary.get("repair_attempt_count"))
                    ),
                    "repair_success_count": _as_int(
                        suite_row.get("repair_success_count", result_summary.get("repair_success_count"))
                    ),
                    "repair_success_rate": _as_float(
                        suite_row.get("repair_success_rate", result_summary.get("repair_success_rate"))
                    ),
                    "mean_total_elapsed_sec": _as_float(
                        suite_row.get("mean_total_elapsed_sec", result_summary.get("mean_total_elapsed_sec"))
                    ),
                    "result_path": suite_row.get("result_path") or result_payload.get("result_path"),
                    "record_path": _display_path(record_path, project_root),
                    "notes": _failure_suite_notes(benchmark_id, suite_row),
                }
            )
        )
    return rows


def _failure_suite_notes(benchmark_id: str, suite_row: dict[str, Any]) -> str:
    baseline = suite_row.get("planner_baseline")
    benchmark_label = "taxonomy v2" if "taxonomy_v2" in benchmark_id else benchmark_id
    if _is_repair_baseline(baseline):
        attempts = _as_int(suite_row.get("repair_attempt_count"))
        successes = _as_int(suite_row.get("repair_success_count"))
        return (
            f"Verifier-triggered repair/fallback recovers {successes} of "
            f"{attempts} attempted repairs in {benchmark_label}."
        )
    if baseline == "rule_based_planner":
        return f"No execution-guided repair is attempted in {benchmark_label}."
    return ""


def _tool_exposure_rows(data: dict[str, Any], summary_path: Path, project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmark_id = data.get("benchmark_id", summary_path.stem.removesuffix(".baseline_summary"))
    execution_mode = data.get("execution_mode", "mock")
    dataset = data.get("dataset", "CrossDocked2020_mocked_execution")
    for suite_row in data.get("rows", []):
        if not isinstance(suite_row, dict):
            continue
        baseline = suite_row.get("planner_baseline")
        task_count = _as_int(suite_row.get("total"))
        task_success_count = _as_int(suite_row.get("task_success_count"))
        rows.append(
            _ordered_row(
                {
                    "benchmark_family": "tool_exposure_budget",
                    "dataset": dataset,
                    "benchmark_id": benchmark_id,
                    "execution_mode": execution_mode,
                    "planner_baseline": baseline,
                    "task_count": task_count,
                    "task_success_count": task_success_count,
                    "failed_task_count": _as_int(suite_row.get("failed_task_count")),
                    "task_success_rate": _as_float(suite_row.get("task_success_rate")),
                    "parser_accuracy": _as_float(suite_row.get("parser_accuracy")),
                    "planner_tool_coverage_rate": _as_float(suite_row.get("planner_tool_coverage_rate")),
                    "planner_tool_precision": _as_float(suite_row.get("planner_tool_precision")),
                    "planner_tool_recall": _as_float(suite_row.get("planner_tool_recall")),
                    "planner_tool_f1": _as_float(suite_row.get("planner_tool_f1")),
                    "mean_selected_tool_count": _as_float(suite_row.get("mean_selected_tool_count")),
                    "mean_tool_sequence_length": _as_float(suite_row.get("mean_tool_sequence_length")),
                    "mean_extra_tool_count": _as_float(suite_row.get("mean_extra_tool_count")),
                    "mean_tool_call_count": _as_float(suite_row.get("mean_tool_call_count")),
                    "failed_tool_call_count": _as_int(suite_row.get("failed_tool_call_count")),
                    "tool_call_failure_rate": _as_float(suite_row.get("tool_call_failure_rate")),
                    "verifier_expectation_match": _as_float(suite_row.get("verifier_expectation_match")),
                    "result_path": suite_row.get("result_path"),
                    "record_path": _display_path(summary_path, project_root),
                    "notes": _tool_exposure_notes(baseline),
                }
            )
        )
    return rows


def _tool_exposure_notes(baseline: str | None) -> str:
    if baseline == "all_tool_agent":
        return "Static all-tools exposure baseline; extra tools quantify context/tool over-exposure."
    if baseline == "fixed_pipeline":
        return "Fixed pipeline uses fewer tools but misses the planned fallback generator."
    if baseline == "rule_based_planner":
        return "Rule-based planner exposes the minimal complete chemistry-aware tool set."
    if _is_repair_baseline(baseline):
        return "Full copilot keeps rule-based exposure and adds verifier-triggered repair behavior."
    return ""


def _failure_candidate_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scscores = [_as_float(item.get("best_scscore")) for item in cases]
    toxicity_scores = [_as_float(item.get("max_toxicity_score")) for item in cases]
    return {
        "candidate_count": sum(_as_int(item.get("candidate_count")) for item in cases),
        "valid_smiles_count": sum(_as_int(item.get("valid_smiles_count")) for item in cases),
        "unique_smiles_count": sum(_as_int(item.get("unique_smiles_count")) for item in cases),
        "best_scscore": _min_non_null(scscores),
        "max_toxicity_score": _max_non_null(toxicity_scores),
    }


def _failure_notes(data: dict[str, Any], baseline: str | None) -> str:
    interpretation = data.get("interpretation", {})
    if _is_repair_baseline(baseline):
        return str(
            interpretation.get(
                "primary_effect",
                "Execution-guided repair/fallback is enabled for EGVR-Agent.",
            )
        )
    if baseline == "rule_based_planner":
        return "No repair is attempted after controlled wrapper failure injection."
    return ""


def _failure_result_path(data: dict[str, Any], baseline: str | None) -> str | None:
    if baseline == "rule_based_planner":
        return data.get("rule_based_result_json") or data.get("result_dir")
    if _is_repair_baseline(baseline):
        return data.get("full_copilot_result_json") or data.get("result_dir")
    return data.get("result_dir")


def _tool_metrics_from_results(results: list[dict[str, Any]]) -> dict[str, float | None]:
    if not results:
        return {"coverage_rate": None, "precision": None, "recall": None, "f1": None}

    coverage_hits = 0
    selected_total = 0
    expected_total = 0
    overlap_total = 0
    for item in results:
        expected = set(item.get("expected_tools", []))
        selected = set(item.get("selected_tools", []))
        coverage_hits += int(expected.issubset(selected)) if expected else 0
        selected_total += len(selected)
        expected_total += len(expected)
        overlap_total += len(expected & selected)

    precision = overlap_total / selected_total if selected_total else None
    recall = overlap_total / expected_total if expected_total else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "coverage_rate": coverage_hits / len(results),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _robustness_view_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": row.get("benchmark_id"),
        "dataset": row.get("dataset"),
        "execution_mode": row.get("execution_mode"),
        "planner_baseline": row.get("planner_baseline"),
        "task_count": row.get("task_count"),
        "task_success_count": row.get("task_success_count"),
        "failed_task_count": row.get("failed_task_count"),
        "task_success_rate": row.get("task_success_rate"),
        "verifier_expectation_match": row.get("verifier_expectation_match"),
        "repair_attempt_count": row.get("repair_attempt_count"),
        "repair_success_count": row.get("repair_success_count"),
        "repair_success_rate": row.get("repair_success_rate"),
        "notes": row.get("notes"),
    }


def _throughput_view_row(row: dict[str, Any]) -> dict[str, Any]:
    mean_elapsed = _as_float(row.get("mean_total_elapsed_sec"))
    task_count = _as_int(row.get("task_count"))
    valid_count = _as_int(row.get("valid_candidate_count"))
    estimated_total_elapsed = mean_elapsed * task_count if mean_elapsed is not None and task_count else None
    return {
        "benchmark_family": row.get("benchmark_family"),
        "dataset": row.get("dataset"),
        "benchmark_id": row.get("benchmark_id"),
        "execution_mode": row.get("execution_mode"),
        "planner_baseline": row.get("planner_baseline"),
        "task_count": row.get("task_count"),
        "generated_candidate_count": row.get("generated_candidate_count"),
        "valid_candidate_count": row.get("valid_candidate_count"),
        "docking_success_count": row.get("docking_success_count"),
        "mean_total_elapsed_sec": mean_elapsed,
        "seconds_per_task": mean_elapsed,
        "valid_candidates_per_sec": valid_count / estimated_total_elapsed if estimated_total_elapsed and valid_count else None,
        "notes": row.get("notes"),
    }


def _generation_quality_view_row(row: dict[str, Any]) -> dict[str, Any]:
    generated_count = _as_int(row.get("generated_candidate_count"))
    valid_count = _as_int(row.get("valid_candidate_count"))
    unique_count = _as_int(row.get("unique_smiles_count"))
    return {
        "benchmark_family": row.get("benchmark_family"),
        "dataset": row.get("dataset"),
        "benchmark_id": row.get("benchmark_id"),
        "execution_mode": row.get("execution_mode"),
        "planner_baseline": row.get("planner_baseline"),
        "task_count": row.get("task_count"),
        "generated_candidate_count": row.get("generated_candidate_count"),
        "valid_candidate_count": row.get("valid_candidate_count"),
        "valid_candidate_rate": valid_count / generated_count if generated_count else None,
        "unique_smiles_count": row.get("unique_smiles_count"),
        "unique_smiles_rate": unique_count / generated_count if generated_count else None,
        "best_scscore": row.get("best_scscore"),
        "max_toxicity_score": row.get("max_toxicity_score"),
        "task_success_rate": row.get("task_success_rate"),
        "verifier_expectation_match": row.get("verifier_expectation_match"),
        "notes": row.get("notes"),
    }


def _generation_scale_rows(
    record_paths: list[str | Path] | tuple[str | Path, ...] | None,
    project_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_path in record_paths or ():
        resolved = _resolve_path(record_path, project_root)
        if not resolved.exists():
            continue
        master_row = _crossdocked_generation_row(_load_json(resolved), resolved, project_root)
        rows.append(_ordered_view_row(_generation_scale_view_row(master_row), GENERATION_SCALE_TABLE_COLUMNS))
    return rows


def _generation_scale_view_row(row: dict[str, Any]) -> dict[str, Any]:
    generated_count = _as_int(row.get("generated_candidate_count"))
    valid_count = _as_int(row.get("valid_candidate_count"))
    unique_count = _as_int(row.get("unique_smiles_count"))
    mean_elapsed = _as_float(row.get("mean_total_elapsed_sec"))
    task_count = _as_int(row.get("task_count"))
    estimated_total_elapsed = mean_elapsed * task_count if mean_elapsed is not None and task_count else None
    return {
        "benchmark_id": row.get("benchmark_id"),
        "dataset": row.get("dataset"),
        "execution_mode": row.get("execution_mode"),
        "planner_baseline": row.get("planner_baseline"),
        "task_count": row.get("task_count"),
        "generated_candidate_count": row.get("generated_candidate_count"),
        "valid_candidate_count": row.get("valid_candidate_count"),
        "valid_candidate_rate": valid_count / generated_count if generated_count else None,
        "unique_smiles_count": row.get("unique_smiles_count"),
        "unique_smiles_rate": unique_count / generated_count if generated_count else None,
        "mean_total_elapsed_sec": mean_elapsed,
        "seconds_per_task": mean_elapsed,
        "valid_candidates_per_sec": valid_count / estimated_total_elapsed if estimated_total_elapsed and valid_count else None,
        "notes": row.get("notes"),
    }


def _crossdocked_multiseed_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if isinstance(row, dict):
            rows.append(_ordered_view_row(row, CROSSDOCKED_MULTISEED_TABLE_COLUMNS))
    return rows


def _tool_exposure_view_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": row.get("benchmark_id"),
        "dataset": row.get("dataset"),
        "execution_mode": row.get("execution_mode"),
        "planner_baseline": row.get("planner_baseline"),
        "task_count": row.get("task_count"),
        "task_success_rate": row.get("task_success_rate"),
        "planner_tool_precision": row.get("planner_tool_precision"),
        "planner_tool_recall": row.get("planner_tool_recall"),
        "planner_tool_f1": row.get("planner_tool_f1"),
        "mean_selected_tool_count": row.get("mean_selected_tool_count"),
        "mean_extra_tool_count": row.get("mean_extra_tool_count"),
        "mean_tool_sequence_length": row.get("mean_tool_sequence_length"),
        "mean_tool_call_count": row.get("mean_tool_call_count"),
        "tool_call_failure_rate": row.get("tool_call_failure_rate"),
        "notes": row.get("notes"),
    }


def _failure_taxonomy_rows(
    *,
    benchmark_path: Path,
    rule_result_path: Path,
    full_result_path: Path,
) -> list[dict[str, Any]]:
    if not (benchmark_path.exists() and rule_result_path.exists() and full_result_path.exists()):
        return []

    tasks = _load_jsonl(benchmark_path)
    rule_results = _results_by_task(_load_json(rule_result_path))
    full_results = _results_by_task(_load_json(full_result_path))
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task.get("task_id")
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        rule_result = rule_results.get(task_id, {})
        full_result = full_results.get(task_id, {})
        injected_tools = _join_names((metadata.get("failure_injections") or {}).keys())
        full_extra_tools = _join_names(full_result.get("extra_tools", []))
        row = {
            "failure_scenario": metadata.get("failure_scenario", task_id),
            "task_type": task.get("expected_task_type"),
            "injected_tools": injected_tools,
            "expected_success": task.get("should_succeed"),
            "rule_success": rule_result.get("task_success"),
            "full_success": full_result.get("task_success"),
            "full_repair_executed": full_result.get("repair_executed"),
            "full_repair_success": full_result.get("repair_success"),
            "repair_actions": _repair_actions(full_result),
            "full_extra_tools": full_extra_tools,
            "rule_verifier_match": rule_result.get("verifier_matched_expectation"),
            "full_verifier_match": full_result.get("verifier_matched_expectation"),
            "interpretation": _failure_taxonomy_interpretation(
                expected_success=task.get("should_succeed"),
                rule_success=rule_result.get("task_success"),
                full_success=full_result.get("task_success"),
                full_repair_executed=full_result.get("repair_executed"),
                full_verifier_match=full_result.get("verifier_matched_expectation"),
                full_extra_tools=full_extra_tools,
            ),
        }
        rows.append(_ordered_view_row(row, FAILURE_TAXONOMY_TABLE_COLUMNS))
    return rows


def _ablation_rows(
    tool_exposure_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exposure_by_baseline = {
        row.get("planner_baseline"): row
        for row in tool_exposure_rows
        if row.get("planner_baseline")
    }
    rows: list[dict[str, Any]] = []
    for robustness_row in robustness_rows:
        baseline = robustness_row.get("planner_baseline")
        exposure_row = exposure_by_baseline.get(baseline)
        if not exposure_row:
            continue
        row = {
            "planner_baseline": baseline,
            "tool_exposure_benchmark_id": exposure_row.get("benchmark_id"),
            "robustness_benchmark_id": robustness_row.get("benchmark_id"),
            "mean_selected_tool_count": exposure_row.get("mean_selected_tool_count"),
            "mean_extra_tool_count": exposure_row.get("mean_extra_tool_count"),
            "planner_tool_precision": exposure_row.get("planner_tool_precision"),
            "planner_tool_recall": exposure_row.get("planner_tool_recall"),
            "robust_task_count": robustness_row.get("task_count"),
            "robust_task_success_count": robustness_row.get("task_success_count"),
            "robust_task_success_rate": robustness_row.get("task_success_rate"),
            "repair_attempt_count": robustness_row.get("repair_attempt_count"),
            "repair_success_count": robustness_row.get("repair_success_count"),
            "repair_success_rate": robustness_row.get("repair_success_rate"),
            "verifier_expectation_match": robustness_row.get("verifier_expectation_match"),
            "interpretation": _ablation_interpretation(baseline, exposure_row, robustness_row),
        }
        rows.append(_ordered_view_row(row, ABLATION_TABLE_COLUMNS))
    return rows


def _robustness_repeated_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if isinstance(row, dict):
            rows.append(_ordered_view_row(row, ROBUSTNESS_REPEATED_TABLE_COLUMNS))
    return rows


def _repair_ablation_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        baseline = row.get("planner_baseline")
        values = {
            "benchmark_id": payload.get("benchmark_id", row.get("benchmark_id")),
            "execution_mode": row.get("execution_mode", payload.get("execution_mode")),
            "planner_baseline": baseline,
            "task_count": row.get("total"),
            "task_success_count": row.get("task_success_count"),
            "task_success_rate": row.get("task_success_rate"),
            "verifier_expectation_match": row.get("verifier_expectation_match"),
            "initial_selected_tool_count": row.get("mean_selected_tool_count"),
            "mean_tool_call_count": row.get("mean_tool_call_count"),
            "repair_attempt_count": row.get("repair_attempt_count"),
            "repair_success_count": row.get("repair_success_count"),
            "repair_success_rate": row.get("repair_success_rate"),
            "false_success_count": row.get("false_success_count", 0),
            "interpretation": _repair_ablation_interpretation(row),
        }
        rows.append(_ordered_view_row(values, REPAIR_ABLATION_TABLE_COLUMNS))
    return rows


def _repair_ablation_repeated_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if isinstance(row, dict):
            rows.append(_ordered_view_row(row, REPAIR_ABLATION_REPEATED_TABLE_COLUMNS))
    return rows


def _ambiguous_failure_mode_rows(
    *,
    benchmark_path: Path | None,
    full_result_path: Path | None,
) -> list[dict[str, Any]]:
    if benchmark_path is None or full_result_path is None:
        return []
    if not benchmark_path.exists() or not full_result_path.exists():
        return []

    tasks = _load_jsonl(benchmark_path)
    full_results = _results_by_task(_load_json(full_result_path))
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task.get("task_id")
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        result = full_results.get(task_id, {})
        repair_plan = result.get("repair_plan", {}) if isinstance(result.get("repair_plan"), dict) else {}
        actions = repair_plan.get("actions", []) if isinstance(repair_plan.get("actions"), list) else []
        repair_tool = _first_action_tool(actions)
        repair_attempt = bool(result.get("repair_executed"))
        repair_success = bool(result.get("repair_success"))
        task_success = bool(result.get("task_success"))
        expected_success = bool(task.get("should_succeed", True))
        evidence_family = metadata.get("evidence_family") or _evidence_family_from_tool(repair_tool)
        row = {
            "evidence_family": evidence_family,
            "failure_scenario": metadata.get("failure_scenario", task_id),
            "ambiguity_type": metadata.get("ambiguity_type"),
            "missing_evidence_mode": _missing_evidence_mode(metadata.get("failure_scenario"), metadata.get("ambiguity_type")),
            "verifier_check": _verifier_check_for_evidence_family(evidence_family),
            "repair_tool": repair_tool,
            "task_count": 1,
            "repair_attempt_count": 1 if repair_attempt else 0,
            "real_retry_success_count": 1 if repair_success else 0,
            "real_retry_success_rate": 1.0 if repair_success else 0.0,
            "task_success_count": 1 if task_success else 0,
            "task_success_rate": 1.0 if task_success else 0.0,
            "false_success_count": 1 if task_success and not expected_success else 0,
            "real_evidence": _real_evidence_cell(result.get("metrics", {})),
            "example_task_id": task_id,
        }
        rows.append(_ordered_view_row(row, AMBIGUOUS_FAILURE_MODE_TABLE_COLUMNS))
    return sorted(rows, key=_ambiguous_failure_mode_sort_key)


def _ambiguous_failure_mode_repeated_rows(
    *,
    benchmark_path: Path | None,
    repeated_summary_path: Path | None,
    project_root: Path,
) -> list[dict[str, Any]]:
    if benchmark_path is None or repeated_summary_path is None:
        return []
    if not benchmark_path.exists() or not repeated_summary_path.exists():
        return []

    tasks = _load_jsonl(benchmark_path)
    task_by_id = {task.get("task_id"): task for task in tasks if task.get("task_id")}
    payload = _load_json(repeated_summary_path)
    repeated_results: list[dict[str, dict[str, Any]]] = []
    for detail in payload.get("detail_rows", []):
        if not isinstance(detail, dict) or not _is_repair_baseline(detail.get("planner_baseline")):
            continue
        result_path = detail.get("result_path")
        if not result_path:
            continue
        resolved = _resolve_path(result_path, project_root)
        if resolved.exists():
            repeated_results.append(_results_by_task(_load_json(resolved)))

    rows: list[dict[str, Any]] = []
    for task_id, task in task_by_id.items():
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        evidence_family = metadata.get("evidence_family")
        repair_tool = None
        repair_attempt_count = 0
        repair_success_count = 0
        task_success_count = 0
        false_success_count = 0
        evidence_cells: list[str] = []
        expected_success = bool(task.get("should_succeed", True))
        for result_by_task in repeated_results:
            result = result_by_task.get(task_id, {})
            repair_plan = result.get("repair_plan", {}) if isinstance(result.get("repair_plan"), dict) else {}
            actions = repair_plan.get("actions", []) if isinstance(repair_plan.get("actions"), list) else []
            repair_tool = repair_tool or _first_action_tool(actions)
            evidence_family = evidence_family or _evidence_family_from_tool(repair_tool)
            repair_attempt = bool(result.get("repair_executed"))
            repair_success = bool(result.get("repair_success"))
            task_success = bool(result.get("task_success"))
            repair_attempt_count += 1 if repair_attempt else 0
            repair_success_count += 1 if repair_success else 0
            task_success_count += 1 if task_success else 0
            false_success_count += 1 if task_success and not expected_success else 0
            evidence = _real_evidence_cell(result.get("metrics", {}))
            if evidence and evidence not in evidence_cells:
                evidence_cells.append(evidence)
        repeat_count = len(repeated_results)
        row = {
            "evidence_family": evidence_family,
            "failure_scenario": metadata.get("failure_scenario", task_id),
            "ambiguity_type": metadata.get("ambiguity_type"),
            "missing_evidence_mode": _missing_evidence_mode(metadata.get("failure_scenario"), metadata.get("ambiguity_type")),
            "verifier_check": _verifier_check_for_evidence_family(evidence_family),
            "repair_tool": repair_tool,
            "repeat_count": repeat_count,
            "task_count": repeat_count,
            "repair_attempt_count": repair_attempt_count,
            "real_retry_success_count": repair_success_count,
            "real_retry_success_rate": repair_success_count / repair_attempt_count if repair_attempt_count else 0.0,
            "task_success_count": task_success_count,
            "task_success_rate": task_success_count / repeat_count if repeat_count else 0.0,
            "false_success_count": false_success_count,
            "real_evidence": "; ".join(evidence_cells[:3]),
            "example_task_id": task_id,
        }
        rows.append(_ordered_view_row(row, AMBIGUOUS_FAILURE_MODE_REPEATED_TABLE_COLUMNS))
    return sorted(rows, key=_ambiguous_failure_mode_sort_key)


def _task_generalization_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if isinstance(row, dict):
            rows.append(_ordered_view_row(row, TASK_GENERALIZATION_TABLE_COLUMNS))
    return rows


def _resolve_task_generalization_summary_path(summary_path: str | Path | None, root: Path) -> Path | None:
    if summary_path is None:
        return None
    resolved = _resolve_path(summary_path, root)
    if resolved.exists() or str(summary_path) != DEFAULT_TASK_GENERALIZATION_SUMMARY:
        return resolved
    for fallback in DEFAULT_TASK_GENERALIZATION_FALLBACK_SUMMARIES:
        candidate = _resolve_path(fallback, root)
        if candidate.exists():
            return candidate
    return resolved


def _resolve_summary_path_with_default_fallback(
    summary_path: str | Path | None,
    root: Path,
    *,
    default_path: str,
    fallback_path: str,
) -> Path | None:
    if summary_path is None:
        return None
    resolved = _resolve_path(summary_path, root)
    if resolved.exists() or str(summary_path) != default_path:
        return resolved
    fallback = _resolve_path(fallback_path, root)
    if fallback.exists():
        return fallback
    return resolved


def _resolve_path_with_default_fallbacks(
    path: str | Path | None,
    root: Path,
    *,
    default_path: str,
    fallback_paths: tuple[str, ...],
) -> Path | None:
    if path is None:
        return None
    resolved = _resolve_path(path, root)
    if resolved.exists() or str(path) != default_path:
        return resolved
    for fallback_path in fallback_paths:
        fallback = _resolve_path(fallback_path, root)
        if fallback.exists():
            return fallback
    return resolved


def _tool_admission_rows(cards_path: Path | None) -> list[dict[str, Any]]:
    if cards_path is None or not cards_path.exists():
        return []
    payload = _load_json(cards_path)
    cards = payload.get("tool_admission_cards", [])
    rows: list[dict[str, Any]] = []
    for card in cards:
        if isinstance(card, dict):
            rows.append(_ordered_view_row(card, TOOL_ADMISSION_TABLE_COLUMNS))
    return rows


def _verifier_evidence_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if isinstance(row, dict):
            rows.append(_ordered_view_row(row, VERIFIER_EVIDENCE_TABLE_COLUMNS))
    return rows


def _property_verifier_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("property_rows", payload.get("rows", [])):
        if isinstance(row, dict):
            rows.append(_ordered_view_row(row, PROPERTY_VERIFIER_TABLE_COLUMNS))
    return rows


def _property_verifier_evidence_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("verifier_evidence_rows", []):
        if isinstance(row, dict):
            rows.append(_ordered_view_row(row, VERIFIER_EVIDENCE_TABLE_COLUMNS))
    return rows


def _posebusters_failure_mode_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    context = _posebusters_summary_context(payload)
    failure_modes = payload.get("posebusters_failure_modes", {})
    rows: list[dict[str, Any]] = []
    if isinstance(failure_modes, dict):
        source_rows = failure_modes.get("rows", [])
    else:
        source_rows = []
    for row in source_rows:
        if isinstance(row, dict):
            enriched = {
                **row,
                "dataset": row.get("dataset") or context.get("dataset"),
                "evidence_family": row.get("evidence_family") or context.get("evidence_family"),
            }
            rows.append(_ordered_view_row(enriched, POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS))
    return rows


def _posebusters_summary_context(payload: dict[str, Any]) -> dict[str, Any]:
    for row in payload.get("rows", []):
        if isinstance(row, dict) and row.get("evidence_type") == "posebusters":
            return {
                "dataset": row.get("dataset"),
                "evidence_family": row.get("evidence_family"),
            }
    return {"dataset": None, "evidence_family": None}


def _posebusters_top_failure_rows(rows: list[dict[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    failing_rows = [
        row
        for row in rows
        if (_as_int(row.get("fail_count")) or 0) > 0
    ]
    ranked = sorted(
        failing_rows,
        key=lambda row: (
            -(_as_int(row.get("fail_count")) or 0),
            -(_as_float(row.get("fail_rate")) or 0.0),
            str(row.get("check_name") or ""),
        ),
    )
    return [
        _ordered_view_row(row, POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS)
        for row in ranked[:limit]
    ]


def _pdbbind_prep_gate_rows(
    *,
    readiness_summary_path: Path | None,
    receptor_prep_summary_path: Path | None,
    prepared_pilot_path: Path | None,
    prepared_pilot_result_path: Path | None,
) -> list[dict[str, Any]]:
    readiness = _load_optional_json(readiness_summary_path) if readiness_summary_path else {}
    receptor_prep = _load_optional_json(receptor_prep_summary_path) if receptor_prep_summary_path else {}
    pilot_result = _load_optional_json(prepared_pilot_result_path) if prepared_pilot_result_path else {}
    if not readiness and not receptor_prep:
        return []

    best_candidate = readiness.get("best_candidate", {}) if isinstance(readiness.get("best_candidate"), dict) else {}
    prep_summary = receptor_prep.get("summary", {}) if isinstance(receptor_prep.get("summary"), dict) else {}
    failure_counts = (
        prep_summary.get("failure_counts", {})
        if isinstance(prep_summary.get("failure_counts"), dict)
        else {}
    )
    prep_success_count = _as_int(prep_summary.get("prep_success_count"))
    template_required_count = _as_int(
        prep_summary.get("template_required_count", failure_counts.get("histidine_template_ambiguity"))
    )
    timeout_count = _as_int(failure_counts.get("timeout"))
    runtime_error_count = _as_int(failure_counts.get("runtime_error"))
    prepared_pilot_task_count = _jsonl_row_count(prepared_pilot_path)
    pilot_summary = pilot_result.get("summary", {}) if isinstance(pilot_result.get("summary"), dict) else {}
    pilot_results = pilot_result.get("results", []) if isinstance(pilot_result.get("results"), list) else []
    docking_scores = [
        _as_float(item.get("metrics", {}).get("best_docking_score"))
        for item in pilot_results
        if isinstance(item, dict) and isinstance(item.get("metrics"), dict)
    ]
    readiness_status = readiness.get("status") or ("ready" if best_candidate.get("ready") else "unknown")

    row = {
        "dataset": _pdbbind_dataset_name(best_candidate),
        "readiness_status": readiness_status,
        "ready_target_count": best_candidate.get("ready_target_count"),
        "index_file_count": best_candidate.get("index_file_count"),
        "receptor_prep_target_count": prep_summary.get("total"),
        "prep_success_count": prep_success_count,
        "prep_success_rate": prep_summary.get("prep_success_rate"),
        "template_required_count": template_required_count,
        "runtime_error_count": runtime_error_count,
        "timeout_count": timeout_count,
        "prepared_pilot_task_count": prepared_pilot_task_count,
        "real_pilot_task_count": pilot_summary.get("total"),
        "real_pilot_success_rate": pilot_summary.get("task_success_rate"),
        "best_docking_score": _min_non_null(docking_scores),
        "mean_elapsed_sec": pilot_summary.get("mean_total_elapsed_sec"),
        "false_success_count": pilot_summary.get("false_success_count"),
        "gate_status": _pdbbind_gate_status(
            readiness_status=readiness_status,
            prep_success_count=prep_success_count,
            prepared_pilot_task_count=prepared_pilot_task_count,
            real_pilot_task_count=_as_int(pilot_summary.get("total")),
            real_pilot_success_rate=_as_float(pilot_summary.get("task_success_rate")),
        ),
        "evidence_role": "appendix_gate_not_main_claim",
        "notes": _pdbbind_gate_notes(
            readiness_status=readiness_status,
            prep_success_count=prep_success_count,
            prepared_pilot_task_count=prepared_pilot_task_count,
            real_pilot_task_count=_as_int(pilot_summary.get("total")),
            real_pilot_success_rate=_as_float(pilot_summary.get("task_success_rate")),
            template_required_count=template_required_count,
            runtime_error_count=runtime_error_count,
            timeout_count=timeout_count,
        ),
    }
    return [_ordered_view_row(row, PDBBIND_PREP_GATE_TABLE_COLUMNS)]


def _resolve_llm_router_summary_paths(
    summary_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    root: Path,
) -> list[Path]:
    if summary_paths is None:
        return []
    if isinstance(summary_paths, (str, Path)):
        values = [summary_paths]
    else:
        values = list(summary_paths)
    return [_resolve_path(value, root) for value in values if value]


def _llm_router_baseline_rows(summary_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        if not summary_path.exists():
            continue
        data = _load_optional_json(summary_path)
        row = data.get("row")
        if not isinstance(row, dict):
            nested_rows = data.get("rows", [])
            row = nested_rows[0] if nested_rows and isinstance(nested_rows[0], dict) else {}
        if not row:
            continue
        enriched = dict(row)
        enriched["model"] = _llm_router_model_name(data, summary_path)
        rows.append(_ordered_view_row(enriched, LLM_ROUTER_BASELINE_TABLE_COLUMNS))
    return rows


def _llm_router_model_name(summary: dict[str, Any], summary_path: Path) -> str:
    row = summary.get("row") if isinstance(summary.get("row"), dict) else {}
    for key in ("model", "api_model", "llm_model"):
        value = row.get(key)
        if value:
            return str(value)
    for result in summary.get("task_results") or []:
        if not isinstance(result, dict):
            continue
        api_metadata = result.get("api_metadata") or {}
        if isinstance(api_metadata, dict) and api_metadata.get("api_model"):
            return str(api_metadata["api_model"])
    stem = summary_path.stem
    if "deepseek_v4_pro" in stem:
        return "deepseek-v4-pro"
    if "qwen3_7_plus" in stem:
        return "qwen3.7-plus"
    if "qwen" in stem:
        return "qwen"
    return "--"


def _natural_failure_audit_rows(summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or not summary_path.exists():
        return []
    data = _load_optional_json(summary_path)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [
        _ordered_view_row(row, NATURAL_FAILURE_AUDIT_TABLE_COLUMNS)
        for row in rows
        if isinstance(row, dict)
    ]


def _pdbbind_gate_status(
    *,
    readiness_status: Any,
    prep_success_count: int,
    prepared_pilot_task_count: int,
    real_pilot_task_count: int,
    real_pilot_success_rate: float | None,
) -> str:
    if str(readiness_status) != "ready":
        return "data_readiness_not_confirmed"
    if real_pilot_task_count > 0 and real_pilot_success_rate == 1.0:
        return "prepared_pilot_completed"
    if real_pilot_task_count > 0:
        return "prepared_pilot_attempted"
    if prepared_pilot_task_count > 0:
        return "prepared_pilot_ready"
    if prep_success_count >= 5:
        return "ready_to_generate_prepared_pilot"
    if prep_success_count > 0:
        return "insufficient_prepared_receptors"
    return "execution_blocked_no_prepared_receptors"


def _pdbbind_gate_notes(
    *,
    readiness_status: Any,
    prep_success_count: int,
    prepared_pilot_task_count: int,
    real_pilot_task_count: int,
    real_pilot_success_rate: float | None,
    template_required_count: int,
    runtime_error_count: int,
    timeout_count: int,
) -> str:
    if str(readiness_status) != "ready":
        return "PDBbind local data readiness is not confirmed, so no docking pilot is claimed."
    if real_pilot_task_count:
        outcome = "completed" if real_pilot_success_rate == 1.0 else "attempted"
        return (
            "PDBbind+ local data readiness is confirmed; receptor preparation produced "
            f"{prep_success_count} stable receptors, and the prepared real pilot {outcome} "
            f"{real_pilot_task_count} tasks with {real_pilot_success_rate:.1%} success. "
            "This is a small docking-infrastructure pilot, not an affinity benchmark."
        )
    if prepared_pilot_task_count:
        return (
            "PDBbind local data readiness is confirmed; receptor preparation produced "
            f"{prep_success_count} stable receptors and {prepared_pilot_task_count} prepared pilot tasks. "
            "Run a real docking smoke before claiming completed pilot evidence."
        )
    if prep_success_count:
        return (
            "PDBbind local data readiness is confirmed, but the prepared-receptor gate "
            "requires a stable prepared pilot before main-result claims."
        )
    return (
        "PDBbind local data readiness is confirmed, but receptor preparation produced "
        f"0 stable targets ({template_required_count} histidine-template ambiguities, "
        f"{runtime_error_count} runtime errors, {timeout_count} timeouts); "
        "the gate prevents false docking success claims."
    )


def _pdbbind_dataset_name(best_candidate: dict[str, Any]) -> str:
    root = str(best_candidate.get("root") or "")
    if "PDBbindPlus" in root or "P-L" in root:
        return "PDBbind+ v2020.R1"
    if (_as_int(best_candidate.get("ready_target_count")) or 0) >= 10_000:
        return "PDBbind+ v2020.R1"
    return "PDBbind v2020 refined"


def _ablation_interpretation(
    baseline: str | None,
    exposure_row: dict[str, Any],
    robustness_row: dict[str, Any],
) -> str:
    selected_tools = _as_float(exposure_row.get("mean_selected_tool_count")) or 0.0
    extra_tools = _as_float(exposure_row.get("mean_extra_tool_count")) or 0.0
    repair_attempts = _as_int(robustness_row.get("repair_attempt_count"))
    repair_success = _as_int(robustness_row.get("repair_success_count"))
    if _is_repair_baseline(baseline):
        return (
            f"Same initial tool budget ({selected_tools:.1f} tools, {extra_tools:.1f} extra) "
            f"with verifier-triggered repair recovering {repair_success}/{repair_attempts} attempts."
        )
    if baseline == "rule_based_planner":
        return (
            f"Same initial chemistry-aware tool budget ({selected_tools:.1f} tools, {extra_tools:.1f} extra) "
            "but no execution-guided repair is attempted."
        )
    return "Included for context; robustness repair metrics are only defined for the failure-recovery baselines."


def _repair_ablation_interpretation(row: dict[str, Any]) -> str:
    baseline = row.get("planner_baseline")
    false_success = _as_int(row.get("false_success_count"))
    if _is_repair_baseline(baseline):
        return "Verifier-triggered repair uses execution evidence before retry/fallback."
    if baseline == "scheduled_fallback_no_verifier":
        return "Scheduled retry/fallback ignores verifier failure reasons; compare tool calls and false success."
    if baseline == "verifier_only_no_repair":
        return "Verifier records missing evidence but intentionally performs no repair."
    if baseline == "rule_based_planner":
        return "No repair baseline for the same task-conditioned initial plan."
    if false_success:
        return "Inspect false-success cases before using this baseline as evidence."
    return "Repair ablation baseline."


def _first_action_tool(actions: list[Any]) -> str | None:
    for action in actions:
        if isinstance(action, dict) and action.get("tool_name"):
            return str(action["tool_name"])
    return None


def _evidence_family_from_tool(tool_name: str | None) -> str | None:
    return {
        "scscore": "synthesizability",
        "toxicity": "toxicity",
        "vina": "docking",
    }.get(tool_name or "")


def _verifier_check_for_evidence_family(evidence_family: Any) -> str:
    return {
        "synthesizability": "passes_synthesizability",
        "toxicity": "passes_toxicity",
        "docking": "has_docking_scores",
    }.get(str(evidence_family or ""), "")


def _missing_evidence_mode(failure_scenario: Any, ambiguity_type: Any) -> str:
    scenario = str(failure_scenario or "")
    for prefix in ("scscore_", "toxicity_", "vina_"):
        if scenario.startswith(prefix):
            scenario = scenario[len(prefix) :]
    for suffix in ("_then_real_retry",):
        if scenario.endswith(suffix):
            scenario = scenario[: -len(suffix)]
    if scenario:
        return scenario
    return str(ambiguity_type or "")


def _real_evidence_cell(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return ""
    if metrics.get("best_scscore") is not None:
        return f"SCScore={_as_float(metrics.get('best_scscore')):.3f}"
    if metrics.get("max_toxicity_score") is not None:
        return f"Tox={_as_float(metrics.get('max_toxicity_score')):.3f}"
    if metrics.get("best_docking_score") is not None:
        return f"Vina={_as_float(metrics.get('best_docking_score')):.3f}"
    return ""


def _ambiguous_failure_mode_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    family_order = {
        "synthesizability": 0,
        "toxicity": 1,
        "docking": 2,
    }
    family = str(row.get("evidence_family") or "")
    return (family_order.get(family, 99), str(row.get("failure_scenario") or ""))


def _results_by_task(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item.get("task_id"): item
        for item in payload.get("results", [])
        if isinstance(item, dict) and item.get("task_id")
    }


def _repair_actions(result: dict[str, Any]) -> str:
    repair_plan = result.get("repair_plan", {}) if isinstance(result.get("repair_plan"), dict) else {}
    actions = repair_plan.get("actions", [])
    if not isinstance(actions, list):
        return ""
    rendered = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = _repair_action_label(action.get("action_type"))
        tool_name = action.get("tool_name")
        rendered.append(f"{action_type}:{tool_name}" if tool_name else action_type)
    return "; ".join(rendered)


def _repair_action_label(action_type: str | None) -> str:
    mapping = {
        "retry_with_reduced_generation_count": "retry",
        "fallback_tool": "fallback",
        "mark_incomplete_evaluation": "mark_incomplete",
        "mark_missing_docking": "mark_missing_docking",
    }
    return mapping.get(action_type or "", action_type or "")


def _failure_taxonomy_interpretation(
    *,
    expected_success: Any,
    rule_success: Any,
    full_success: Any,
    full_repair_executed: Any,
    full_verifier_match: Any,
    full_extra_tools: str,
) -> str:
    if full_success and not rule_success:
        if full_extra_tools:
            return "Recovered by verifier-triggered fallback, with the added tool recorded in trace."
        return "Recovered by verifier-triggered retry without exposing extra tools."
    if full_success and rule_success:
        if full_repair_executed:
            return "Both succeed; EGVR-Agent additionally records a verifier-triggered repair trace."
        return "Both baselines succeed."
    if expected_success is False and full_verifier_match:
        return "Correctly remains failed; verifier prevents a false success claim."
    if expected_success and not full_success:
        return "Still unresolved; trace exposes the unrecovered tool failure."
    return "Failure is surfaced with explicit verifier evidence."


def _join_names(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, dict):
        values = values.keys()
    if isinstance(values, str):
        return values
    try:
        return ", ".join(str(value) for value in values)
    except TypeError:
        return str(values)


def _has_any_metric(row: dict[str, Any], keys: list[str]) -> bool:
    return any(row.get(key) is not None for key in keys)


def _ordered_view_row(values: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {column: values.get(column) for column in columns}


def _write_csv(rows: list[dict[str, Any]], output_path: Path, *, columns: list[str] | None = None) -> None:
    fieldnames = columns or MASTER_TABLE_COLUMNS
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in fieldnames})


def _ordered_row(values: dict[str, Any]) -> dict[str, Any]:
    return {column: values.get(column) for column in MASTER_TABLE_COLUMNS}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _jsonl_row_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not str(path) or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    resolved = Path(path)
    if not path:
        return resolved
    if resolved.is_absolute():
        return resolved
    return project_root / resolved


def _resolve_litpcba_result_path(path: str | Path, project_root: Path) -> Path:
    requested = _resolve_path(path, project_root)
    if Path(path).as_posix() == DEFAULT_LITPCBA_RESULT:
        elapsed_result = _resolve_path(DEFAULT_LITPCBA_ELAPSED_RESULT, project_root)
        if elapsed_result.exists():
            return elapsed_result
    return requested


def _display_path(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _success_count(total: int, rate: float | None) -> int:
    if rate is None:
        return 0
    return int(round(total * rate))


def _min_non_null(values: list[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    return min(numbers) if numbers else None


def _max_non_null(values: list[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    return max(numbers) if numbers else None


def _mean_non_null(values: list[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    return mean(numbers) if numbers else None


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the EGVR master baseline table from completed artifacts.")
    parser.add_argument("--project-root", help="Project root; defaults to the installed package root.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for master CSV/JSON outputs.")
    parser.add_argument(
        "--record-output",
        default=DEFAULT_RECORD_OUTPUT,
        help="Optional benchmark record JSON path. Use an empty string to disable.",
    )
    parser.add_argument("--crossdocked-record", default=DEFAULT_CROSSDOCKED_RECORD)
    parser.add_argument("--litpcba-result", default=DEFAULT_LITPCBA_RESULT)
    parser.add_argument("--failure-record", default=DEFAULT_FAILURE_RECORD)
    parser.add_argument(
        "--tool-exposure-summary",
        default=DEFAULT_TOOL_EXPOSURE_SUMMARY,
        help="Optional baseline-suite summary JSON for the tool-exposure paper view. Use an empty string to disable.",
    )
    parser.add_argument(
        "--robustness-repeated-summary",
        default=DEFAULT_ROBUSTNESS_REPEATED_SUMMARY,
        help="Optional repeated-suite summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--repair-ablation-summary",
        default=DEFAULT_REPAIR_ABLATION_SUMMARY,
        help="Optional repair-ablation baseline-suite summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--repair-ablation-repeated-summary",
        default=DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY,
        help="Optional repeated repair-ablation summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--ambiguous-failure-mode-benchmark",
        default=DEFAULT_AMBIGUOUS_FAILURE_MODE_BENCHMARK,
        help="Optional ambiguous-evidence benchmark JSONL. Use an empty string to disable.",
    )
    parser.add_argument(
        "--ambiguous-failure-mode-full-result",
        default=DEFAULT_AMBIGUOUS_FAILURE_MODE_FULL_RESULT,
        help="Optional EGVR-Agent (or legacy full_copilot) result JSON for ambiguous-evidence failure modes.",
    )
    parser.add_argument(
        "--ambiguous-failure-mode-repeated-summary",
        default=DEFAULT_REPAIR_ABLATION_REPEATED_SUMMARY,
        help="Optional repeated ambiguous-evidence summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--task-generalization-summary",
        default=DEFAULT_TASK_GENERALIZATION_SUMMARY,
        help="Optional task-generalization summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--tool-admission-cards",
        default=DEFAULT_TOOL_ADMISSION_CARDS,
        help="Optional tool-admission cards JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--verifier-evidence-summary",
        default=DEFAULT_VERIFIER_EVIDENCE_SUMMARY,
        help="Optional verifier-evidence summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--property-verifier-summary",
        default=DEFAULT_PROPERTY_VERIFIER_SUMMARY,
        help="Optional RDKit property-verifier summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--pdbbindplus-pose-sanity-summary",
        default=DEFAULT_PDBBINDPLUS_POSE_SANITY_SUMMARY,
        help="Optional PDBbind+ PoseBusters pose-sanity summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--pdbbind-readiness-summary",
        default=DEFAULT_PDBBIND_READINESS_SUMMARY,
        help="Optional PDBbind local-data readiness summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--pdbbind-receptor-prep-summary",
        default=DEFAULT_PDBBIND_RECEPTOR_PREP_SUMMARY,
        help="Optional PDBbind receptor-preparation probe summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--pdbbind-prepared-pilot",
        default=DEFAULT_PDBBIND_PREPARED_PILOT,
        help="Optional prepared PDBbind pilot JSONL. Use an empty string to disable.",
    )
    parser.add_argument(
        "--pdbbind-prepared-pilot-result",
        default=DEFAULT_PDBBIND_PREPARED_PILOT_RESULT,
        help="Optional prepared PDBbind pilot result JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--generation-scale-records",
        nargs="*",
        default=list(DEFAULT_GENERATION_SCALE_RECORDS),
        help="Optional CrossDocked summary records for the generation-scale table.",
    )
    parser.add_argument(
        "--crossdocked-multiseed-summary",
        default=DEFAULT_CROSSDOCKED_MULTISEED_SUMMARY,
        help="Optional CrossDocked multi-seed summary JSON. Use an empty string to disable.",
    )
    parser.add_argument(
        "--llm-router-baseline-summary",
        nargs="*",
        default=[DEFAULT_LLM_ROUTER_BASELINE_SUMMARY],
        help="Optional planning-only LLM-router baseline summary JSON(s). Use an empty string to disable.",
    )
    parser.add_argument(
        "--natural-failure-audit-summary",
        default=DEFAULT_NATURAL_FAILURE_AUDIT_SUMMARY,
        help="Optional natural failure audit summary JSON. Use an empty string to disable.",
    )
    args = parser.parse_args()

    root = _project_root(args.project_root)
    record_output = args.record_output or None
    tool_exposure_summary = args.tool_exposure_summary or None
    robustness_repeated_summary = args.robustness_repeated_summary or None
    repair_ablation_summary = args.repair_ablation_summary or None
    repair_ablation_repeated_summary = args.repair_ablation_repeated_summary or None
    ambiguous_failure_mode_benchmark = args.ambiguous_failure_mode_benchmark or None
    ambiguous_failure_mode_full_result = args.ambiguous_failure_mode_full_result or None
    ambiguous_failure_mode_repeated_summary = args.ambiguous_failure_mode_repeated_summary or None
    task_generalization_summary = args.task_generalization_summary or None
    tool_admission_cards = args.tool_admission_cards or None
    verifier_evidence_summary = args.verifier_evidence_summary or None
    property_verifier_summary = args.property_verifier_summary or None
    pdbbindplus_pose_sanity_summary = args.pdbbindplus_pose_sanity_summary or None
    pdbbind_readiness_summary = args.pdbbind_readiness_summary or None
    pdbbind_receptor_prep_summary = args.pdbbind_receptor_prep_summary or None
    pdbbind_prepared_pilot = args.pdbbind_prepared_pilot or None
    pdbbind_prepared_pilot_result = args.pdbbind_prepared_pilot_result or None
    generation_scale_records = args.generation_scale_records or []
    crossdocked_multiseed_summary = args.crossdocked_multiseed_summary or None
    llm_router_baseline_summary = [path for path in (args.llm_router_baseline_summary or []) if path] or None
    natural_failure_audit_summary = args.natural_failure_audit_summary or None
    payload = build_master_baseline_table(
        output_dir=_resolve_path(args.output_dir, root),
        record_output_path=_resolve_path(record_output, root) if record_output else None,
        crossdocked_record_path=args.crossdocked_record,
        litpcba_result_path=args.litpcba_result,
        failure_record_path=args.failure_record,
        tool_exposure_summary_path=tool_exposure_summary,
        generation_scale_record_paths=generation_scale_records,
        crossdocked_multiseed_summary_path=crossdocked_multiseed_summary,
        robustness_repeated_summary_path=robustness_repeated_summary,
        repair_ablation_summary_path=repair_ablation_summary,
        repair_ablation_repeated_summary_path=repair_ablation_repeated_summary,
        ambiguous_failure_mode_benchmark_path=ambiguous_failure_mode_benchmark,
        ambiguous_failure_mode_full_result_path=ambiguous_failure_mode_full_result,
        ambiguous_failure_mode_repeated_summary_path=ambiguous_failure_mode_repeated_summary,
        task_generalization_summary_path=task_generalization_summary,
        tool_admission_cards_path=tool_admission_cards,
        verifier_evidence_summary_path=verifier_evidence_summary,
        property_verifier_summary_path=property_verifier_summary,
        pdbbindplus_pose_sanity_summary_path=pdbbindplus_pose_sanity_summary,
        pdbbind_readiness_summary_path=pdbbind_readiness_summary,
        pdbbind_receptor_prep_summary_path=pdbbind_receptor_prep_summary,
        pdbbind_prepared_pilot_path=pdbbind_prepared_pilot,
        pdbbind_prepared_pilot_result_path=pdbbind_prepared_pilot_result,
        llm_router_baseline_summary_path=llm_router_baseline_summary,
        natural_failure_audit_summary_path=natural_failure_audit_summary,
        project_root=root,
    )
    print(
        json.dumps(
            {
                "row_count": payload["row_count"],
                "artifacts": payload["artifacts"],
                "rows": payload["rows"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
