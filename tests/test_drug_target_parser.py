from __future__ import annotations

from egvr.biomedical_offline import parse_biomedical_task, plan_biomedical_workflow


def test_drug_target_parser_extracts_entities():
    parsed = parse_biomedical_task(
        "Check drug-target evidence drug=imatinib target=BCR-ABL disease=CML "
        "mechanism=kinase inhibition kg_source=kg literature_source=pubmed rationale=approved."
    )

    assert parsed.metadata["domain"] == "drug_target"
    assert parsed.task_type == "drug_target_evidence"
    assert parsed.metadata["drug"] == "imatinib"
    assert parsed.metadata["target"] == "BCR-ABL"
    assert "repurposing_evidence" in parsed.objectives


def test_drug_target_planner_adds_repurposing_when_disease_present():
    parsed = parse_biomedical_task(
        "Check drug-target evidence.",
        metadata={
            "domain": "drug_target",
            "drug": "imatinib",
            "target": "BCR-ABL",
            "disease": "CML",
        },
    )
    workflow = plan_biomedical_workflow(parsed)

    assert workflow.selected_tools == [
        "drug_target_evidence_checker",
        "repurposing_evidence_checker",
    ]
