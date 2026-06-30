from __future__ import annotations

from CAi.toolkit.agent_planner.biomedical_offline import (
    execute_biomedical_offline,
    parse_biomedical_task,
    plan_biomedical_workflow,
    verify_biomedical_workflow,
)


def test_clinical_trial_verifier_fails_on_missing_outcome_provenance():
    parsed = parse_biomedical_task(
        "Assess clinical trial NCT00000010.",
        metadata={
            "domain": "clinical_trial",
            "trial_id": "NCT00000010",
            "phase": "Phase II",
            "condition": "COPD",
            "intervention": "DrugJ",
            "endpoint": "exacerbation rate",
            "eligibility_criteria": "moderate COPD",
            "enrollment": 180,
            "outcome_label": "endpoint not met",
        },
    )
    workflow = plan_biomedical_workflow(parsed)
    tool_calls, evidence = execute_biomedical_offline(parsed, workflow)
    verifier = verify_biomedical_workflow(parsed, workflow, tool_calls, evidence)

    assert not verifier.success
    assert verifier.metrics["missing_evidence_count"] == 1
    assert "clinical_outcome_provenance" in verifier.failure_reason
