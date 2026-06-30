"""Export paper-table CSV views to lightweight LaTeX table drafts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_INPUT_DIR = "logs/baseline_runs/master_baseline_table"
DEFAULT_OUTPUT_DIR = "logs/baseline_runs/master_baseline_table/latex"


@dataclass(frozen=True)
class LatexColumn:
    header: str
    value: Callable[[dict[str, str]], str]


@dataclass(frozen=True)
class LatexTableSpec:
    name: str
    input_csv: str
    output_tex: str
    caption: str
    label: str
    columns: tuple[LatexColumn, ...]


ROBUSTNESS_SPEC = LatexTableSpec(
    name="robustness",
    input_csv="robustness_table.csv",
    output_tex="robustness_table.tex",
    caption="Robustness under controlled tool-failure injection.",
    label="tab:robustness",
    columns=(
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Success", lambda row: _count_pair(row.get("task_success_count"), row.get("task_count"))),
        LatexColumn("Success Rate", lambda row: _rate_cell(row.get("task_success_rate"))),
        LatexColumn("Verifier Match", lambda row: _rate_cell(row.get("verifier_expectation_match"))),
        LatexColumn("Repair", lambda row: _repair_pair(row)),
        LatexColumn("Repair Rate", lambda row: _repair_rate_cell(row)),
    ),
)

ROBUSTNESS_REPEATED_SPEC = LatexTableSpec(
    name="robustness_repeated",
    input_csv="robustness_repeated_table.csv",
    output_tex="robustness_repeated_table.tex",
    caption="Repeated robustness under controlled tool-failure injection.",
    label="tab:robustness-repeated",
    columns=(
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Repeats", lambda row: _int_cell(row.get("repeat_count"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Success", lambda row: _pm_rate_cell(row.get("mean_task_success_rate"), row.get("std_task_success_rate"))),
        LatexColumn("Repair", lambda row: _pm_rate_cell(row.get("mean_repair_success_rate"), row.get("std_repair_success_rate"))),
        LatexColumn("Verifier", lambda row: _pm_rate_cell(row.get("mean_verifier_expectation_match"), row.get("std_verifier_expectation_match"))),
        LatexColumn("False Success", lambda row: _int_cell(row.get("false_success_count"))),
        LatexColumn("Calls", lambda row: _pm_float_cell(row.get("mean_tool_call_count"), row.get("std_tool_call_count"), digits=1)),
    ),
)


THROUGHPUT_SPEC = LatexTableSpec(
    name="throughput",
    input_csv="throughput_table.csv",
    output_tex="throughput_table.tex",
    caption="Execution throughput summary for completed real benchmark artifacts.",
    label="tab:throughput",
    columns=(
        LatexColumn("Benchmark", lambda row: _benchmark_name(row.get("benchmark_family"), row.get("dataset"))),
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Generated", lambda row: _int_cell(row.get("generated_candidate_count"))),
        LatexColumn("Valid", lambda row: _int_cell(row.get("valid_candidate_count"))),
        LatexColumn("Docked", lambda row: _int_cell(row.get("docking_success_count"))),
        LatexColumn("Mean sec/task", lambda row: _float_cell(row.get("seconds_per_task"), digits=1)),
        LatexColumn("Valid/sec", lambda row: _float_cell(row.get("valid_candidates_per_sec"), digits=3)),
    ),
)


GENERATION_QUALITY_SPEC = LatexTableSpec(
    name="generation_quality",
    input_csv="generation_quality_table.csv",
    output_tex="generation_quality_table.tex",
    caption="Candidate quality metrics for generation-oriented benchmark slices.",
    label="tab:generation-quality",
    columns=(
        LatexColumn("Benchmark", lambda row: _benchmark_name(row.get("benchmark_family"), row.get("dataset"))),
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Generated", lambda row: _int_cell(row.get("generated_candidate_count"))),
        LatexColumn("Valid Rate", lambda row: _rate_cell(row.get("valid_candidate_rate"))),
        LatexColumn("Unique Rate", lambda row: _rate_cell(row.get("unique_smiles_rate"))),
        LatexColumn("Best SCScore", lambda row: _float_cell(row.get("best_scscore"), digits=3)),
        LatexColumn("Max Tox.", lambda row: _float_cell(row.get("max_toxicity_score"), digits=3)),
        LatexColumn("Task Success", lambda row: _rate_cell(row.get("task_success_rate"))),
    ),
)

GENERATION_SCALE_SPEC = LatexTableSpec(
    name="generation_scale",
    input_csv="generation_scale_table.csv",
    output_tex="generation_scale_table.tex",
    caption="Generation scale-up across CrossDocked target slices.",
    label="tab:generation-scale",
    columns=(
        LatexColumn("Benchmark", lambda row: _text_or_dash(row.get("benchmark_id"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Generated", lambda row: _int_cell(row.get("generated_candidate_count"))),
        LatexColumn("Valid Rate", lambda row: _rate_cell(row.get("valid_candidate_rate"))),
        LatexColumn("Unique Rate", lambda row: _rate_cell(row.get("unique_smiles_rate"))),
        LatexColumn("Mean sec/task", lambda row: _float_cell(row.get("seconds_per_task"), digits=1)),
        LatexColumn("Valid/sec", lambda row: _float_cell(row.get("valid_candidates_per_sec"), digits=3)),
    ),
)

CROSSDOCKED_MULTISEED_SPEC = LatexTableSpec(
    name="crossdocked_multiseed",
    input_csv="crossdocked_multiseed_table.csv",
    output_tex="crossdocked_multiseed_table.tex",
    caption="CrossDocked30 repeatability across RxnFlow seeds.",
    label="tab:crossdocked-multiseed",
    columns=(
        LatexColumn("Benchmark", lambda row: _text_or_dash(row.get("benchmark_id"))),
        LatexColumn("Seeds", lambda row: _int_cell(row.get("seed_count"))),
        LatexColumn("Target-runs", lambda row: _int_cell(row.get("total_target_runs"))),
        LatexColumn("Candidates", lambda row: _int_cell(row.get("total_candidates"))),
        LatexColumn("Success", lambda row: _pm_rate_cell(row.get("mean_task_success_rate"), row.get("std_task_success_rate"))),
        LatexColumn("Valid", lambda row: _pm_rate_cell(row.get("mean_valid_candidate_rate"), row.get("std_valid_candidate_rate"))),
        LatexColumn("Unique", lambda row: _pm_rate_cell(row.get("mean_unique_smiles_rate"), row.get("std_unique_smiles_rate"))),
        LatexColumn("SA cov.", lambda row: _rate_cell(row.get("mean_sa_score_coverage"))),
        LatexColumn("SA pass", lambda row: _rate_cell(row.get("mean_sa_score_pass_rate"))),
        LatexColumn("RDKit cov.", lambda row: _rate_cell(row.get("mean_rdkit_property_coverage"))),
        LatexColumn("Mean sec/task", lambda row: _pm_float_cell(row.get("mean_seconds_per_task"), row.get("std_seconds_per_task"), digits=1)),
        LatexColumn("False Success", lambda row: _int_cell(row.get("false_success_count"))),
    ),
)


TOOL_EXPOSURE_SPEC = LatexTableSpec(
    name="tool_exposure",
    input_csv="tool_exposure_table.csv",
    output_tex="tool_exposure_table.tex",
    caption="Tool exposure and planning precision across planner baselines.",
    label="tab:tool-exposure",
    columns=(
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Tools", lambda row: _float_cell(row.get("mean_selected_tool_count"), digits=1)),
        LatexColumn("Extra", lambda row: _float_cell(row.get("mean_extra_tool_count"), digits=1)),
        LatexColumn("Seq.", lambda row: _float_cell(row.get("mean_tool_sequence_length"), digits=1)),
        LatexColumn("Precision", lambda row: _rate_cell(row.get("planner_tool_precision"))),
        LatexColumn("Recall", lambda row: _rate_cell(row.get("planner_tool_recall"))),
        LatexColumn("Success", lambda row: _rate_cell(row.get("task_success_rate"))),
    ),
)


FAILURE_TAXONOMY_SPEC = LatexTableSpec(
    name="failure_taxonomy",
    input_csv="failure_taxonomy_table.csv",
    output_tex="failure_taxonomy_table.tex",
    caption="Failure taxonomy for the controlled robustness benchmark.",
    label="tab:failure-taxonomy",
    columns=(
        LatexColumn("Scenario", lambda row: _scenario_name(row.get("failure_scenario"))),
        LatexColumn("Injected", lambda row: _text_or_dash(row.get("injected_tools"))),
        LatexColumn("Expected", lambda row: _bool_cell(row.get("expected_success"))),
        LatexColumn("Rule", lambda row: _bool_cell(row.get("rule_success"))),
        LatexColumn("Full", lambda row: _bool_cell(row.get("full_success"))),
        LatexColumn("Repair", lambda row: _repair_status_cell(row)),
        LatexColumn("Extra", lambda row: _text_or_dash(row.get("full_extra_tools"))),
        LatexColumn("Mechanism", lambda row: _mechanism_cell(row)),
    ),
)


ABLATION_SPEC = LatexTableSpec(
    name="ablation",
    input_csv="ablation_table.csv",
    output_tex="ablation_table.tex",
    caption="Ablation separating initial tool exposure from execution-guided repair.",
    label="tab:ablation-tool-repair",
    columns=(
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Init. Tools", lambda row: _float_cell(row.get("mean_selected_tool_count"), digits=1)),
        LatexColumn("Extra", lambda row: _float_cell(row.get("mean_extra_tool_count"), digits=1)),
        LatexColumn("Precision", lambda row: _rate_cell(row.get("planner_tool_precision"))),
        LatexColumn("Recall", lambda row: _rate_cell(row.get("planner_tool_recall"))),
        LatexColumn("Robust", lambda row: _count_pair(row.get("robust_task_success_count"), row.get("robust_task_count"))),
        LatexColumn("Robust Rate", lambda row: _rate_cell(row.get("robust_task_success_rate"))),
        LatexColumn("Repair", lambda row: _repair_pair(row)),
        LatexColumn("Repair Rate", lambda row: _repair_rate_cell(row)),
        LatexColumn("Verifier", lambda row: _rate_cell(row.get("verifier_expectation_match"))),
    ),
)

REPAIR_ABLATION_SPEC = LatexTableSpec(
    name="repair_ablation",
    input_csv="repair_ablation_table.csv",
    output_tex="repair_ablation_table.tex",
    caption="Repair ablation separating verifier-guided repair from scheduled fallback.",
    label="tab:repair-ablation",
    columns=(
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Success", lambda row: _count_pair(row.get("task_success_count"), row.get("task_count"))),
        LatexColumn("Success Rate", lambda row: _rate_cell(row.get("task_success_rate"))),
        LatexColumn("Verifier", lambda row: _rate_cell(row.get("verifier_expectation_match"))),
        LatexColumn("Init. Tools", lambda row: _float_cell(row.get("initial_selected_tool_count"), digits=1)),
        LatexColumn("Calls", lambda row: _float_cell(row.get("mean_tool_call_count"), digits=1)),
        LatexColumn("Repair", lambda row: _repair_pair(row)),
        LatexColumn("False Success", lambda row: _int_cell(row.get("false_success_count"))),
    ),
)

REPAIR_ABLATION_REPEATED_SPEC = LatexTableSpec(
    name="repair_ablation_repeated",
    input_csv="repair_ablation_repeated_table.csv",
    output_tex="repair_ablation_repeated_table.tex",
    caption="Repeated repair ablation under ambiguous-evidence wrapper injection.",
    label="tab:repair-ablation-repeated",
    columns=(
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Repeats", lambda row: _int_cell(row.get("repeat_count"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Success", lambda row: _pm_rate_cell(row.get("mean_task_success_rate"), row.get("std_task_success_rate"))),
        LatexColumn("Repair", lambda row: _pm_rate_cell(row.get("mean_repair_success_rate"), row.get("std_repair_success_rate"))),
        LatexColumn("Verifier", lambda row: _pm_rate_cell(row.get("mean_verifier_expectation_match"), row.get("std_verifier_expectation_match"))),
        LatexColumn("False Success", lambda row: _int_cell(row.get("false_success_count"))),
        LatexColumn("Calls", lambda row: _pm_float_cell(row.get("mean_tool_call_count"), row.get("std_tool_call_count"), digits=1)),
    ),
)

AMBIGUOUS_FAILURE_MODES_SPEC = LatexTableSpec(
    name="ambiguous_failure_modes",
    input_csv="ambiguous_failure_modes_table.csv",
    output_tex="ambiguous_failure_modes_table.tex",
    caption="Ambiguous-evidence failure modes and verifier-triggered real retries.",
    label="tab:ambiguous-failure-modes",
    columns=(
        LatexColumn("Family", lambda row: _evidence_family_cell(row.get("evidence_family"))),
        LatexColumn("Missing Evidence", lambda row: _scenario_name(row.get("missing_evidence_mode"))),
        LatexColumn("Ambiguity", lambda row: _scenario_name(row.get("ambiguity_type"))),
        LatexColumn("Verifier", lambda row: _scenario_name(row.get("verifier_check"))),
        LatexColumn("Repair Tool", lambda row: _tool_name(row.get("repair_tool"))),
        LatexColumn("Retry", lambda row: _count_pair(row.get("real_retry_success_count"), row.get("repair_attempt_count"))),
        LatexColumn("Retry Rate", lambda row: _rate_cell(row.get("real_retry_success_rate"))),
        LatexColumn("Evidence", lambda row: _text_or_dash(row.get("real_evidence"))),
        LatexColumn("Example", lambda row: _short_task_id(row.get("example_task_id"))),
    ),
)

AMBIGUOUS_FAILURE_MODES_REPEATED_SPEC = LatexTableSpec(
    name="ambiguous_failure_modes_repeated",
    input_csv="ambiguous_failure_modes_repeated_table.csv",
    output_tex="ambiguous_failure_modes_repeated_table.tex",
    caption="Repeated ambiguous-evidence modes and verifier-triggered real retries.",
    label="tab:ambiguous-failure-modes-repeated",
    columns=(
        LatexColumn("Family", lambda row: _evidence_family_cell(row.get("evidence_family"))),
        LatexColumn("Missing Evidence", lambda row: _scenario_name(row.get("missing_evidence_mode"))),
        LatexColumn("Verifier", lambda row: _scenario_name(row.get("verifier_check"))),
        LatexColumn("Repair Tool", lambda row: _tool_name(row.get("repair_tool"))),
        LatexColumn("Repeats", lambda row: _int_cell(row.get("repeat_count"))),
        LatexColumn("Retry", lambda row: _count_pair(row.get("real_retry_success_count"), row.get("repair_attempt_count"))),
        LatexColumn("Retry Rate", lambda row: _rate_cell(row.get("real_retry_success_rate"))),
        LatexColumn("False Success", lambda row: _int_cell(row.get("false_success_count"))),
        LatexColumn("Evidence", lambda row: _text_or_dash(row.get("real_evidence"))),
    ),
)

TASK_GENERALIZATION_SPEC = LatexTableSpec(
    name="task_generalization",
    input_csv="task_generalization_table.csv",
    output_tex="task_generalization_table.tex",
    caption="Task generalization across molecular-design workflow families.",
    label="tab:task-generalization",
    columns=(
        LatexColumn("Task", lambda row: _scenario_name(row.get("task_type"))),
        LatexColumn("Benchmark", lambda row: _text_or_dash(row.get("benchmark_id"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Success", lambda row: _rate_cell(row.get("task_success_rate"))),
        LatexColumn("Valid", lambda row: _rate_cell(row.get("valid_candidate_rate"))),
        LatexColumn("Verifier", lambda row: _rate_cell(row.get("verifier_expectation_match"))),
        LatexColumn("Mean sec", lambda row: _float_cell(row.get("mean_elapsed_sec"), digits=1)),
        LatexColumn("Tools", lambda row: _text_or_dash(row.get("tools"))),
    ),
)

TOOL_ADMISSION_SPEC = LatexTableSpec(
    name="tool_admission",
    input_csv="tool_admission_table.csv",
    output_tex="tool_admission_table.tex",
    caption="Tool admission cards for verifier-centric system expansion.",
    label="tab:tool-admission",
    columns=(
        LatexColumn("Tool", lambda row: _text_or_dash(row.get("tool_name"))),
        LatexColumn("Role", lambda row: _text_or_dash(row.get("tool_role"))),
        LatexColumn("Evidence", lambda row: _text_or_dash(row.get("independent_evidence"))),
        LatexColumn("Failures", lambda row: _bool_cell(row.get("failure_modes_structured"))),
        LatexColumn("Cost", lambda row: _text_or_dash(row.get("runtime_cost"))),
        LatexColumn("Risk", lambda row: _text_or_dash(row.get("environment_risk"))),
        LatexColumn("Decision", lambda row: _text_or_dash(row.get("decision"))),
    ),
)


VERIFIER_EVIDENCE_SPEC = LatexTableSpec(
    name="verifier_evidence",
    input_csv="verifier_evidence_table.csv",
    output_tex="verifier_evidence_table.tex",
    caption="Additional chemistry-grounded verifier evidence beyond tool-call success.",
    label="tab:verifier-evidence",
    columns=(
        LatexColumn("Evidence", lambda row: _evidence_name(row.get("evidence_type"))),
        LatexColumn("Dataset", lambda row: _dataset_name(row.get("dataset"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Candidates", lambda row: _int_cell(row.get("candidate_count"))),
        LatexColumn("Coverage", lambda row: _rate_cell(row.get("coverage"))),
        LatexColumn("Pass Rate", lambda row: _rate_cell(row.get("pass_rate"))),
        LatexColumn("Best SA", lambda row: _float_cell(row.get("best_sa_score"), digits=3)),
        LatexColumn("Mean SA", lambda row: _float_cell(row.get("mean_sa_score"), digits=3)),
        LatexColumn("Status", lambda row: _status_cell(row.get("status"))),
    ),
)

PROPERTY_VERIFIER_SPEC = LatexTableSpec(
    name="property_verifier",
    input_csv="property_verifier_table.csv",
    output_tex="property_verifier_table.tex",
    caption="RDKit property-verifier evidence for generated candidates.",
    label="tab:property-verifier",
    columns=(
        LatexColumn("Dataset", lambda row: _dataset_name(row.get("dataset"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Candidates", lambda row: _int_cell(row.get("candidate_count"))),
        LatexColumn("Coverage", lambda row: _rate_cell(row.get("property_coverage"))),
        LatexColumn("Mean QED", lambda row: _float_cell(row.get("mean_qed"), digits=3)),
        LatexColumn("Mean LogP", lambda row: _float_cell(row.get("mean_logp"), digits=2)),
        LatexColumn("Lipinski", lambda row: _rate_cell(row.get("lipinski_pass_rate"))),
        LatexColumn("PAINS", lambda row: _rate_cell(row.get("pains_flag_rate"))),
        LatexColumn("Brenk", lambda row: _rate_cell(row.get("brenk_flag_rate"))),
        LatexColumn("Status", lambda row: _status_cell(row.get("status"))),
    ),
)

POSEBUSTERS_TOP_FAILURES_SPEC = LatexTableSpec(
    name="posebusters_top_failures",
    input_csv="posebusters_top_failures_table.csv",
    output_tex="posebusters_top_failures_table.tex",
    caption="Top PoseBusters failure modes among evaluated docking poses.",
    label="tab:posebusters-top-failures",
    columns=(
        LatexColumn("Dataset", lambda row: _dataset_name(row.get("dataset"))),
        LatexColumn("Check", lambda row: _check_name(row.get("check_name"))),
        LatexColumn("Category", lambda row: _category_name(row.get("category"))),
        LatexColumn("Eval.", lambda row: _int_cell(row.get("evaluated_count"))),
        LatexColumn("Failed", lambda row: _int_cell(row.get("fail_count"))),
        LatexColumn("Fail Rate", lambda row: _rate_cell(row.get("fail_rate"))),
        LatexColumn("Examples", lambda row: _example_tasks_cell(row.get("example_task_ids"))),
    ),
)

POSEBUSTERS_FAILURE_MODES_SPEC = LatexTableSpec(
    name="posebusters_failure_modes",
    input_csv="posebusters_failure_modes_table.csv",
    output_tex="posebusters_failure_modes_table.tex",
    caption="PoseBusters failure-mode appendix for evaluated docking poses.",
    label="tab:posebusters-failure-modes",
    columns=(
        LatexColumn("Dataset", lambda row: _dataset_name(row.get("dataset"))),
        LatexColumn("Check", lambda row: _check_name(row.get("check_name"))),
        LatexColumn("Category", lambda row: _category_name(row.get("category"))),
        LatexColumn("Eval.", lambda row: _int_cell(row.get("evaluated_count"))),
        LatexColumn("Failed", lambda row: _int_cell(row.get("fail_count"))),
        LatexColumn("Fail Rate", lambda row: _rate_cell(row.get("fail_rate"))),
        LatexColumn("Missing", lambda row: _int_cell(row.get("missing_count"))),
        LatexColumn("Examples", lambda row: _example_tasks_cell(row.get("example_task_ids"))),
    ),
)

PDBBIND_PREP_GATE_SPEC = LatexTableSpec(
    name="pdbbind_prep_gate",
    input_csv="pdbbind_prep_gate_table.csv",
    output_tex="pdbbind_prep_gate_table.tex",
    caption="PDBbind local-data readiness and receptor-preparation gate.",
    label="tab:pdbbind-prep-gate",
    columns=(
        LatexColumn("Dataset", lambda row: _dataset_name(row.get("dataset"))),
        LatexColumn("Ready", lambda row: _status_cell(row.get("readiness_status"))),
        LatexColumn("Ready Targets", lambda row: _int_cell(row.get("ready_target_count"))),
        LatexColumn("Index Files", lambda row: _int_cell(row.get("index_file_count"))),
        LatexColumn("Prep", lambda row: _count_pair(row.get("prep_success_count"), row.get("receptor_prep_target_count"))),
        LatexColumn("Template", lambda row: _int_cell(row.get("template_required_count"))),
        LatexColumn("Runtime", lambda row: _int_cell(row.get("runtime_error_count"))),
        LatexColumn("Timeout", lambda row: _int_cell(row.get("timeout_count"))),
        LatexColumn("Pilot Tasks", lambda row: _int_cell(row.get("prepared_pilot_task_count"))),
        LatexColumn("Real Tasks", lambda row: _int_cell(row.get("real_pilot_task_count"))),
        LatexColumn("Real Success", lambda row: _rate_cell(row.get("real_pilot_success_rate"))),
        LatexColumn("Best Vina", lambda row: _float_cell(row.get("best_docking_score"), digits=3)),
        LatexColumn("False Success", lambda row: _int_cell(row.get("false_success_count"))),
        LatexColumn("Gate", lambda row: _status_cell(row.get("gate_status"))),
    ),
)

LLM_ROUTER_BASELINE_SPEC = LatexTableSpec(
    name="llm_router_baseline",
    input_csv="llm_router_baseline_table.csv",
    output_tex="llm_router_baseline_table.tex",
    caption="Planning-only LLM-as-router baseline validation.",
    label="tab:llm-router-baseline",
    columns=(
        LatexColumn("Model", lambda row: _text_or_dash(row.get("model"))),
        LatexColumn("Mode", lambda row: _text_or_dash(row.get("router_mode"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("task_count"))),
        LatexColumn("Valid JSON", lambda row: _count_pair(row.get("valid_json_count"), row.get("task_count"))),
        LatexColumn("Valid Schema", lambda row: _count_pair(row.get("valid_schema_count"), row.get("task_count"))),
        LatexColumn("Hallucinated", lambda row: _int_cell(row.get("hallucinated_tool_count"))),
        LatexColumn("Precision", lambda row: _rate_cell(row.get("tool_precision"))),
        LatexColumn("Recall", lambda row: _rate_cell(row.get("tool_recall"))),
        LatexColumn("Order", lambda row: _rate_cell(row.get("workflow_order_match_rate"))),
    ),
)

NATURAL_FAILURE_AUDIT_SPEC = LatexTableSpec(
    name="natural_failure_audit",
    input_csv="natural_failure_audit_table.csv",
    output_tex="natural_failure_audit_table.tex",
    caption="Natural failure audit from existing real-run traces.",
    label="tab:natural-failure-audit",
    columns=(
        LatexColumn("Dataset", lambda row: _dataset_name(row.get("dataset"))),
        LatexColumn("Benchmark", lambda row: _short_benchmark_id(row.get("benchmark_id"))),
        LatexColumn("Tool", lambda row: _tool_name(row.get("tool_name"))),
        LatexColumn("Failure", lambda row: _scenario_name(row.get("failure_family"))),
        LatexColumn("Count", lambda row: _int_cell(row.get("failure_count"))),
        LatexColumn("Tasks", lambda row: _int_cell(row.get("affected_task_count"))),
        LatexColumn("Examples", lambda row: _example_tasks_cell(row.get("example_task_ids"))),
    ),
)

EVIDENCE_AUDIT_SPEC = LatexTableSpec(
    name="evidence_audit",
    input_csv="evidence_audit_table.csv",
    output_tex="evidence_audit_table.tex",
    caption="Claim-to-evidence audit for paper claims.",
    label="tab:evidence-audit",
    columns=(
        LatexColumn("Claim", lambda row: _text_or_dash(row.get("claim_id"))),
        LatexColumn("Strength", lambda row: _text_or_dash(row.get("claim_strength"))),
        LatexColumn("Evidence", lambda row: _text_or_dash(row.get("evidence_type"))),
        LatexColumn("Table", lambda row: _text_or_dash(row.get("table_label"))),
        LatexColumn("Real", lambda row: _bool_cell(row.get("is_real_result"))),
        LatexColumn("Controlled", lambda row: _bool_cell(row.get("is_controlled"))),
        LatexColumn("Rows", lambda row: _int_cell(row.get("row_count"))),
    ),
)

STATISTICAL_SUMMARY_SPEC = LatexTableSpec(
    name="statistical_summary",
    input_csv="statistical_summary_table.csv",
    output_tex="statistical_summary_table.tex",
    caption="Statistical summary for key rate metrics.",
    label="tab:statistical-summary",
    columns=(
        LatexColumn("Claim", lambda row: _text_or_dash(row.get("claim_id"))),
        LatexColumn("Benchmark", lambda row: _short_benchmark_id(row.get("benchmark_id"))),
        LatexColumn("Method", lambda row: _method_name(row.get("planner_baseline"))),
        LatexColumn("Metric", lambda row: _scenario_name(row.get("metric"))),
        LatexColumn("n", lambda row: _int_cell(row.get("n"))),
        LatexColumn("Estimate", lambda row: _rate_cell(row.get("estimate"))),
        LatexColumn("95\\% CI", lambda row: _ci_cell(row.get("ci95_low"), row.get("ci95_high"))),
    ),
)


TABLE_SPECS = {
    spec.name: spec
    for spec in (
        ROBUSTNESS_SPEC,
        ROBUSTNESS_REPEATED_SPEC,
        THROUGHPUT_SPEC,
        GENERATION_QUALITY_SPEC,
        GENERATION_SCALE_SPEC,
        CROSSDOCKED_MULTISEED_SPEC,
        TOOL_EXPOSURE_SPEC,
        FAILURE_TAXONOMY_SPEC,
        ABLATION_SPEC,
        REPAIR_ABLATION_SPEC,
        REPAIR_ABLATION_REPEATED_SPEC,
        AMBIGUOUS_FAILURE_MODES_SPEC,
        AMBIGUOUS_FAILURE_MODES_REPEATED_SPEC,
        TASK_GENERALIZATION_SPEC,
        TOOL_ADMISSION_SPEC,
        VERIFIER_EVIDENCE_SPEC,
        PROPERTY_VERIFIER_SPEC,
        POSEBUSTERS_TOP_FAILURES_SPEC,
        POSEBUSTERS_FAILURE_MODES_SPEC,
        PDBBIND_PREP_GATE_SPEC,
        LLM_ROUTER_BASELINE_SPEC,
        NATURAL_FAILURE_AUDIT_SPEC,
        EVIDENCE_AUDIT_SPEC,
        STATISTICAL_SUMMARY_SPEC,
    )
}


def export_latex_tables(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    table_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Convert selected paper-view CSV files into LaTeX table drafts."""

    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = table_names or tuple(TABLE_SPECS)

    tables: dict[str, dict[str, Any]] = {}
    for name in selected:
        if name not in TABLE_SPECS:
            raise ValueError(f"Unsupported LaTeX table view: {name}")
        spec = TABLE_SPECS[name]
        input_path = source_dir / spec.input_csv
        output_path = target_dir / spec.output_tex
        rows = _read_csv_rows(input_path)
        latex = render_latex_table(spec, rows)
        output_path.write_text(latex, encoding="utf-8")
        tables[name] = {
            "input_csv": str(input_path),
            "output_tex": str(output_path),
            "row_count": len(rows),
        }

    manifest_path = target_dir / "latex_tables.summary.json"
    payload = {
        "table_count": len(tables),
        "tables": tables,
        "notes": [
            "Generated LaTeX uses plain tabular and hline for portability.",
            "Empty cells are rendered as -- because some benchmark views do not expose every metric yet.",
        ],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    payload["manifest_path"] = str(manifest_path)
    return payload


def render_latex_table(spec: LatexTableSpec, rows: list[dict[str, str]]) -> str:
    """Render one CSV view as a self-contained LaTeX table draft."""

    alignment = "l" * len(spec.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{escape_latex(spec.caption)}}}",
        f"\\label{{{escape_latex(spec.label)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(escape_latex(column.header) for column in spec.columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        rendered = [escape_latex(column.value(row)) for column in spec.columns]
        lines.append(" & ".join(rendered) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def escape_latex(value: Any) -> str:
    """Escape common LaTeX special characters in plain table text."""

    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV view not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _method_name(value: str | None) -> str:
    mapping = {
        "all_tool_agent": "all-tool agent",
        "fixed_pipeline": "fixed pipeline",
        "rule_based_planner": "rule-based planner",
        "full_copilot": "full copilot",
        "llm_as_router": "LLM router",
        "scheduled_fallback_no_verifier": "scheduled fallback",
        "verifier_only_no_repair": "verifier only",
    }
    return mapping.get(value or "", value or "--")


def _benchmark_name(family: str | None, dataset: str | None) -> str:
    mapping = {
        "crossdocked_generation": "CrossDocked gen.",
        "litpcba_docking": "LIT-PCBA docking",
        "failure_recovery": "failure recovery",
        "tool_exposure_budget": "tool exposure",
    }
    return mapping.get(family or "", dataset or family or "--")


def _dataset_name(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    mapping = {
        "CrossDocked2020": "CrossDocked",
        "LIT-PCBA": "LIT-PCBA",
    }
    return mapping.get(str(value), str(value))


def _evidence_name(value: str | None) -> str:
    mapping = {
        "sa_score": "SA_Score",
        "posebusters": "PoseBusters",
        "rdkit_property_verifier": "RDKit properties",
    }
    return mapping.get(value or "", value or "--")


def _evidence_family_cell(value: str | None) -> str:
    mapping = {
        "synthesizability": "SCScore",
        "toxicity": "toxicity",
        "docking": "docking",
    }
    return mapping.get(value or "", value or "--")


def _tool_name(value: str | None) -> str:
    mapping = {
        "scscore": "SCScore",
        "toxicity": "toxicity",
        "vina": "Vina",
    }
    return mapping.get(value or "", value or "--")


def _status_cell(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    return str(value).replace("_", " ")


def _check_name(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    return str(value).replace("_", " ")


def _category_name(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    return str(value).replace("_", " ")


def _example_tasks_cell(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    tasks = [item.strip() for item in str(value).split(",") if item.strip()]
    shortened = [task.replace("litpcba_docking_", "") for task in tasks[:3]]
    return ", ".join(shortened) if shortened else "--"


def _short_benchmark_id(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    text = str(value)
    replacements = {
        "crossdocked_rxnflow_candidates5_targets30_multiseed_v1": "CrossDocked30 x3",
        "failure_recovery_taxonomy_v2_repeated": "failure taxonomy v2 x3",
        "llm_as_router_planning_v1": "LLM router planning",
        "natural_failure_audit_v1": "natural failure audit",
        "pdbbind_receptor_prep_probe_v1": "PDBbind prep gate",
    }
    if text in replacements:
        return replacements[text]
    if len(text) <= 32:
        return text
    return text[:29] + "..."


def _ci_cell(low: str | None, high: str | None) -> str:
    low_f = _float_or_none(low)
    high_f = _float_or_none(high)
    if low_f is None or high_f is None:
        return "--"
    return f"[{low_f * 100:.1f}, {high_f * 100:.1f}]"


def _short_task_id(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    text = str(value)
    text = text.replace("ambiguous_v2_", "")
    text = text.replace("_then_real_retry", "")
    return text.replace("_", " ")


def _scenario_name(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    return str(value).replace("_", " ")


def _text_or_dash(value: str | None) -> str:
    return "--" if _is_blank(value) else str(value)


def _bool_cell(value: str | None) -> str:
    if _is_blank(value):
        return "--"
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return "yes"
    if normalized in {"false", "0", "no"}:
        return "no"
    return str(value)


def _repair_status_cell(row: dict[str, str]) -> str:
    executed = _bool_cell(row.get("full_repair_executed"))
    if executed == "no":
        return "--"
    success = _bool_cell(row.get("full_repair_success"))
    if success == "yes":
        return "success"
    if success == "no":
        return "failed"
    return executed


def _mechanism_cell(row: dict[str, str]) -> str:
    interpretation = row.get("interpretation")
    if not _is_blank(interpretation):
        return str(interpretation)
    actions = row.get("repair_actions")
    return _text_or_dash(actions)


def _count_pair(success: str | None, total: str | None) -> str:
    if _is_blank(success) or _is_blank(total):
        return "--"
    return f"{_int_cell(success)}/{_int_cell(total)}"


def _repair_pair(row: dict[str, str]) -> str:
    attempts = _float_or_none(row.get("repair_attempt_count"))
    if attempts is None or attempts == 0:
        return "--"
    return _count_pair(row.get("repair_success_count"), row.get("repair_attempt_count"))


def _repair_rate_cell(row: dict[str, str]) -> str:
    attempts = _float_or_none(row.get("repair_attempt_count"))
    if attempts is None or attempts == 0:
        return "--"
    return _rate_cell(row.get("repair_success_rate"))


def _rate_cell(value: str | None) -> str:
    number = _float_or_none(value)
    if number is None:
        return "--"
    return f"{number * 100:.1f}%"


def _pm_rate_cell(mean_value: str | None, std_value: str | None) -> str:
    mean_number = _float_or_none(mean_value)
    if mean_number is None:
        return "--"
    std_number = _float_or_none(std_value) or 0.0
    return f"{mean_number * 100:.1f}% +/- {std_number * 100:.1f}%"


def _int_cell(value: str | None) -> str:
    number = _float_or_none(value)
    if number is None:
        return "--"
    return str(int(round(number)))


def _float_cell(value: str | None, *, digits: int) -> str:
    number = _float_or_none(value)
    if number is None:
        return "--"
    return f"{number:.{digits}f}"


def _pm_float_cell(mean_value: str | None, std_value: str | None, *, digits: int) -> str:
    mean_number = _float_or_none(mean_value)
    if mean_number is None:
        return "--"
    std_number = _float_or_none(std_value) or 0.0
    return f"{mean_number:.{digits}f} +/- {std_number:.{digits}f}"


def _float_or_none(value: str | None) -> float | None:
    if _is_blank(value):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _is_blank(value: str | None) -> bool:
    return value is None or str(value).strip() == ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CAi paper-view CSV tables to LaTeX drafts.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing paper-view CSV files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated .tex files.")
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=tuple(TABLE_SPECS),
        default=tuple(TABLE_SPECS),
        help="Paper-view tables to export.",
    )
    args = parser.parse_args()
    payload = export_latex_tables(input_dir=args.input_dir, output_dir=args.output_dir, table_names=args.tables)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
