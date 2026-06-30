from __future__ import annotations

from CAi.toolkit.agent_planner.biomedical_benchmark_runner import run_biomedical_benchmark
from CAi.toolkit.agent_planner.biomedical_offline import (
    execute_biomedical_offline,
    parse_biomedical_task,
    plan_biomedical_workflow,
    verify_biomedical_workflow,
)


def test_drug_target_verifier_requires_literature_provenance():
    parsed = parse_biomedical_task(
        "Check DTI evidence for drug=DrugK target=TargetK.",
        metadata={
            "domain": "drug_target",
            "drug": "DrugK",
            "target": "TargetK",
            "mechanism": "pathway modulation",
            "kg_source": "synthetic_kg",
        },
    )
    workflow = plan_biomedical_workflow(parsed)
    tool_calls, evidence = execute_biomedical_offline(parsed, workflow)
    verifier = verify_biomedical_workflow(parsed, workflow, tool_calls, evidence)

    assert not verifier.success
    assert "literature_provenance" in verifier.failure_reason


def test_biomedical_benchmark_runner_summarizes_drug_target_slice():
    summary = run_biomedical_benchmark(
        "CAi/toolkit/agent_planner/benchmarks/drug_target_evidence_v1_offline.jsonl"
    )

    assert summary["benchmark_id"] == "drug_target_evidence_v1_offline"
    assert summary["task_count"] == 10
    assert 0 < summary["workflow_success_rate"] < 1
    assert summary["false_success_count"] == 0
