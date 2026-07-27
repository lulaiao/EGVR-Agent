# Paper-to-Code Artifact Map

This document maps the paper's mechanism claims to public implementation
modules. It is an implementation index, not a table of unpublished results.

| Mechanism or evidence | Public implementation |
|---|---|
| Structured task and workflow representation | `egvr/task_schema.py`, `egvr/task_parser.py` |
| Task-conditioned tool exposure | `egvr/tool_registry.py`, `egvr/rule_planner.py`, `egvr/tool_menu_comparison_runner.py` |
| Structured execution records | `egvr/executor.py`, `egvr/result_normalizer.py` |
| Execution-grounded verification | `egvr/verifier.py`, `egvr/evidence_corruption_runner.py` |
| Verifier-guided retry and fallback | `egvr/repair.py`, `egvr/benchmark_runner.py`, `egvr/repair_budget_runner.py` |
| Repair quality and cost accounting | `egvr/repair_quality_builder.py`, `egvr/tool_menu_execution_summary_builder.py` |
| Controlled failure taxonomy | `egvr/failure_taxonomy_v3_generator.py` |
| Scenario-clustered uncertainty | `egvr/statistical_summary_builder.py` |
| Trace consistency and duplicate handling | `egvr/trace_logger.py`, `egvr/trace_consistency_audit_runner.py` |
| LLM-router planning audit | `egvr/llm_router_baseline_runner.py`, `egvr/planner_comparison_builder.py` |
| Executed planner comparison | `egvr/llm_router_execution_audit_runner.py`, `egvr/executed_planner_comparison_builder.py` |
| Cross-domain evidence interfaces | `egvr/domain_router.py`, `egvr/biomedical_schema.py`, `egvr/biomedical_offline.py` |
| Optional private backend gate | `egvr/clinical_prediction_backend.py`, `egvr/clinical_prediction_runner.py` |

## Baseline Semantics

- `tool_status_only`: agent decision follows nominal tool status; verifier
  output remains available as external ground truth.
- `verifier_only_no_repair`: verifies evidence and reports incomplete results
  without repair.
- `verifier_targeted_retry_no_fallback`: retries the failed or missing-evidence
  step but cannot switch to a fallback tool.
- `scheduled_fallback_no_verifier`: retry or fallback is triggered without
  using verifier failure reasons.
- `egvr_agent`: verifier failures authorize bounded targeted repair or a
  declared fallback.

The identifier `full_copilot` is accepted only when reading or replaying legacy
experiment artifacts.

The controlled failure taxonomy is a mechanism benchmark. It does not estimate
the prevalence of failures in production biomedical workflows.

## Public and Private Boundaries

The public repository contains source code, synthetic/lightweight benchmark
examples, and tests. Real traces, raw LLM responses, private clinical backend
I/O, paper result tables, large datasets, and model weights are intentionally
excluded. The paper should link this repository as implementation support, not
claim that every third-party dataset is redistributed here.
