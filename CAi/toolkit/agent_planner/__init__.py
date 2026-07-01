"""Trustworthy planning primitives for FullCopilot.

This package is intentionally independent from the main agent loop. It exposes
schemas, task-conditioned planning, execution records, verifier results, and
offline biomedical evidence slices that can be unit-tested without starting the
tool server or loading model dependencies.
"""

from .task_schema import (
    CandidateRecord,
    ParsedTask,
    PlannedToolCall,
    PlannedWorkflow,
    TaskConstraints,
    ToolCallRecord,
    ToolMetadata,
    VerifierResult,
)
from .baseline_planners import (
    FULL_COPILOT,
    SCHEDULED_FALLBACK_NO_VERIFIER,
    VERIFIER_ONLY_NO_REPAIR,
    FixedPipelinePlanner,
    SUPPORTED_BASELINES,
    plan_for_baseline,
)
from .rule_planner import RuleBasedPlanner, plan_workflow
from .executor import WorkflowExecutor
from .repair import RepairAction, RepairPlan, SimpleRepairPlanner, suggest_repair
from .result_normalizer import normalize_tool_output, rank_candidates
from .task_parser import RuleBasedTaskParser, parse_task
from .tool_registry import ChemistryToolRegistry, build_default_tool_registry
from .trace_logger import JSONLTraceLogger, build_trace_payload
from .verifier import WorkflowVerifier, verify_workflow
from .biomedical_schema import EvidenceRecord
from .domain_router import execute_and_verify_domain, parse_domain_task, plan_domain_workflow, route_domain

__all__ = [
    "CandidateRecord",
    "ChemistryToolRegistry",
    "EvidenceRecord",
    "JSONLTraceLogger",
    "BenchmarkCase",
    "BenchmarkRunner",
    "BIOMEDICAL_GENERALIZATION_COLUMNS",
    "DataSourceManifest",
    "SUMMARY_COLUMNS",
    "ParsedTask",
    "PlannedToolCall",
    "PlannedWorkflow",
    "TaskConstraints",
    "ToolCallRecord",
    "ToolMetadata",
    "VinaPreparationConfig",
    "VerifierResult",
    "WorkflowExecutor",
    "WorkflowVerifier",
    "RepairAction",
    "RepairPlan",
    "REPEATED_DETAIL_COLUMNS",
    "REPEATED_SUMMARY_COLUMNS",
    "TASK_GENERALIZATION_SUMMARY_COLUMNS",
    "SimpleRepairPlanner",
    "FixedPipelinePlanner",
    "FULL_COPILOT",
    "SCHEDULED_FALLBACK_NO_VERIFIER",
    "VERIFIER_ONLY_NO_REPAIR",
    "GENERATION_QUALITY_TABLE_COLUMNS",
    "MASTER_TABLE_COLUMNS",
    "PAPER_VIEW_COLUMNS",
    "PDBBIND_PREP_GATE_TABLE_COLUMNS",
    "POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS",
    "POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS",
    "REPAIR_ABLATION_TABLE_COLUMNS",
    "ROBUSTNESS_TABLE_COLUMNS",
    "ROBUSTNESS_REPEATED_TABLE_COLUMNS",
    "SUPPORTED_BASELINES",
    "TABLE_SPECS",
    "TASK_GENERALIZATION_TABLE_COLUMNS",
    "THROUGHPUT_TABLE_COLUMNS",
    "TOOL_ADMISSION_TABLE_COLUMNS",
    "VERIFIER_EVIDENCE_TABLE_COLUMNS",
    "build_trace_payload",
    "build_default_tool_registry",
    "build_master_baseline_rows",
    "build_master_baseline_table",
    "build_biomedical_generalization_table",
    "build_paper_table_views",
    "escape_latex",
    "export_latex_tables",
    "flatten_baseline_summary",
    "generate_crossdocked_pocket_tasks",
    "generate_lit_pcba_docking_tasks",
    "generate_real_benchmark_tasks",
    "load_benchmark_cases",
    "load_data_source_manifest",
    "mol2_center",
    "normalize_tool_output",
    "parse_task",
    "plan_for_baseline",
    "plan_workflow",
    "prepare_lit_pcba_vina_inputs",
    "rank_candidates",
    "run_baseline_suite",
    "run_repeated_baseline_suite",
    "run_task_generalization_summary",
    "summarize_results",
    "suggest_repair",
    "verify_workflow",
    "execute_and_verify_domain",
    "parse_domain_task",
    "plan_domain_workflow",
    "route_domain",
    "write_jsonl",
    "write_master_table",
    "write_biomedical_generalization_table",
    "write_paper_table_views",
    "write_summary_csv",
    "render_latex_table",
    "run_verifier_evidence_summary",
    "RuleBasedPlanner",
    "RuleBasedTaskParser",
]


def __getattr__(name: str):
    if name in {"SUMMARY_COLUMNS", "flatten_baseline_summary", "run_baseline_suite", "write_summary_csv"}:
        from importlib import import_module

        baseline_suite_runner = import_module(f"{__name__}.baseline_suite_runner")
        value = getattr(baseline_suite_runner, name)
        globals()[name] = value
        return value
    if name in {"BenchmarkCase", "BenchmarkRunner", "load_benchmark_cases", "summarize_results"}:
        from importlib import import_module

        benchmark_runner = import_module(f"{__name__}.benchmark_runner")
        value = getattr(benchmark_runner, name)
        globals()[name] = value
        return value
    if name in {
        "MASTER_TABLE_COLUMNS",
        "GENERATION_QUALITY_TABLE_COLUMNS",
        "PAPER_VIEW_COLUMNS",
        "PDBBIND_PREP_GATE_TABLE_COLUMNS",
        "POSEBUSTERS_FAILURE_MODE_TABLE_COLUMNS",
        "POSEBUSTERS_TOP_FAILURE_TABLE_COLUMNS",
        "REPAIR_ABLATION_TABLE_COLUMNS",
        "ROBUSTNESS_TABLE_COLUMNS",
        "ROBUSTNESS_REPEATED_TABLE_COLUMNS",
        "TASK_GENERALIZATION_TABLE_COLUMNS",
        "THROUGHPUT_TABLE_COLUMNS",
        "TOOL_ADMISSION_TABLE_COLUMNS",
        "VERIFIER_EVIDENCE_TABLE_COLUMNS",
        "build_master_baseline_rows",
        "build_master_baseline_table",
        "build_paper_table_views",
        "write_master_table",
        "write_paper_table_views",
    }:
        from importlib import import_module

        master_table_builder = import_module(f"{__name__}.master_table_builder")
        value = getattr(master_table_builder, name)
        globals()[name] = value
        return value
    if name in {
        "REPEATED_DETAIL_COLUMNS",
        "REPEATED_SUMMARY_COLUMNS",
        "run_repeated_baseline_suite",
    }:
        from importlib import import_module

        repeated_suite_runner = import_module(f"{__name__}.repeated_suite_runner")
        value = getattr(repeated_suite_runner, name)
        globals()[name] = value
        return value
    if name in {
        "TASK_GENERALIZATION_SUMMARY_COLUMNS",
        "run_task_generalization_summary",
    }:
        from importlib import import_module

        task_generalization_runner = import_module(f"{__name__}.task_generalization_runner")
        value = getattr(task_generalization_runner, name)
        globals()[name] = value
        return value
    if name in {
        "BIOMEDICAL_GENERALIZATION_COLUMNS",
        "build_biomedical_generalization_table",
        "write_biomedical_generalization_table",
    }:
        from importlib import import_module

        biomedical_table = import_module(f"{__name__}.biomedical_generalization_table")
        value = getattr(biomedical_table, name)
        globals()[name] = value
        return value
    if name in {
        "run_verifier_evidence_summary",
    }:
        from importlib import import_module

        verifier_evidence_runner = import_module(f"{__name__}.verifier_evidence_runner")
        value = getattr(verifier_evidence_runner, name)
        globals()[name] = value
        return value
    if name in {
        "TABLE_SPECS",
        "escape_latex",
        "export_latex_tables",
        "render_latex_table",
    }:
        from importlib import import_module

        latex_table_export = import_module(f"{__name__}.latex_table_export")
        value = getattr(latex_table_export, name)
        globals()[name] = value
        return value
    if name in {
        "DataSourceManifest",
        "VinaPreparationConfig",
        "generate_crossdocked_pocket_tasks",
        "generate_lit_pcba_docking_tasks",
        "generate_real_benchmark_tasks",
        "load_data_source_manifest",
        "mol2_center",
        "prepare_lit_pcba_vina_inputs",
        "write_jsonl",
    }:
        from importlib import import_module

        task_generator = import_module(f"{__name__}.benchmark_task_generator")
        value = getattr(task_generator, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'CAi.toolkit.agent_planner' has no attribute {name!r}")
