from __future__ import annotations

from CAi.toolkit.agent_planner.domain_router import (
    execute_and_verify_domain,
    parse_domain_task,
    plan_domain_workflow,
    route_domain,
)


def test_domain_router_keeps_molecular_default():
    assert route_domain("Generate 5 molecules for protein_path=target.pdb") == "molecular"


def test_domain_router_detects_clinical_trial():
    assert route_domain("Assess clinical trial NCT00000001 outcome evidence") == "clinical_trial"


def test_domain_router_detects_drug_target():
    assert route_domain("Check drug-target evidence for drug=imatinib target=BCR-ABL") == "drug_target"


def test_domain_execute_and_verify_clinical_success():
    parsed = parse_domain_task(
        "Assess clinical trial NCT00000001 outcome evidence.",
        metadata={
            "domain": "clinical_trial",
            "trial_id": "NCT00000001",
            "phase": "Phase II",
            "condition": "asthma",
            "intervention": "DrugA",
            "endpoint": "FEV1",
            "eligibility_criteria": "adult asthma",
            "enrollment": 120,
            "outcome_label": "met endpoint",
            "outcome_source": "synthetic",
        },
    )
    workflow = plan_domain_workflow(parsed)
    tool_calls, candidates, verifier = execute_and_verify_domain(parsed, workflow)

    assert candidates == []
    assert [record.tool_name for record in tool_calls] == [
        "clinical_trial_metadata_parser",
        "eligibility_evidence_checker",
        "trial_outcome_evidence_checker",
    ]
    assert verifier.success
    assert verifier.metrics["evidence_coverage"] == 1.0
