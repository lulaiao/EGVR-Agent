from __future__ import annotations

from CAi.toolkit.agent_planner.biomedical_offline import parse_biomedical_task, plan_biomedical_workflow


def test_clinical_trial_parser_extracts_nct_and_fields():
    parsed = parse_biomedical_task(
        "Assess clinical trial NCT12345678 phase=Phase II condition=asthma "
        "intervention=DrugA endpoint=FEV1 eligibility=adult enrollment=120 "
        "outcome=met outcome_source=record."
    )

    assert parsed.metadata["domain"] == "clinical_trial"
    assert parsed.task_type == "clinical_trial_outcome_prediction"
    assert parsed.metadata["trial_id"] == "NCT12345678"
    assert parsed.metadata["phase"] == "Phase II"
    assert parsed.metadata["condition"] == "asthma"


def test_clinical_trial_planner_uses_three_offline_tools():
    parsed = parse_biomedical_task(
        "Assess clinical trial NCT12345678 outcome evidence.",
        metadata={"domain": "clinical_trial"},
    )
    workflow = plan_biomedical_workflow(parsed)

    assert workflow.planner_type == "biomedical_offline_rule_planner"
    assert workflow.selected_tools == [
        "clinical_trial_metadata_parser",
        "eligibility_evidence_checker",
        "trial_outcome_evidence_checker",
    ]
