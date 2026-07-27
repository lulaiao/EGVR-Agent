from __future__ import annotations

import json
import sys

from egvr.biomedical_offline import parse_biomedical_task, plan_biomedical_workflow
from egvr.clinical_prediction_backend import build_clinicalagent_readiness
from egvr.clinical_prediction_runner import run_clinical_prediction_benchmark


def test_clinicalagent_readiness_reports_missing_private_backend(tmp_path):
    summary = build_clinicalagent_readiness(tmp_path / "missing_clinicalagent")

    assert not summary["repo_found"]
    assert not summary["ready_for_smoke"]
    assert "clinicalagent_repo" in summary["missing_items"]


def test_clinical_prediction_planner_selects_backend_tools():
    parsed = parse_biomedical_task(
        "Predict clinical trial outcome for NCT12345678 phase=Phase II condition=asthma "
        "intervention=DrugA endpoint=FEV1 eligibility=adult enrollment=120.",
        metadata={"domain": "clinical_trial", "use_clinicalagent_backend": True},
    )
    workflow = plan_biomedical_workflow(parsed)

    assert parsed.task_type == "clinical_trial_prediction"
    assert workflow.planner_type == "clinicalagent_backend_rule_planner"
    assert workflow.selected_tools == [
        "clinical_trial_metadata_parser",
        "clinicalagent_evidence_retriever",
        "clinicalagent_enrollment_predictor",
        "clinicalagent_drug_risk_checker",
        "clinicalagent_disease_risk_checker",
        "clinicalagent_outcome_predictor",
    ]


def test_clinical_prediction_runner_blocks_without_backend(tmp_path):
    benchmark = tmp_path / "clinical_prediction_smoke.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "clinical_prediction_smoke_001",
                "raw_user_query": "Predict clinical trial outcome for NCT12345678.",
                "should_succeed": True,
                "metadata": {
                    "domain": "clinical_trial",
                    "use_clinicalagent_backend": True,
                    "trial_id": "NCT12345678",
                    "phase": "Phase II",
                    "condition": "asthma",
                    "intervention": "DrugA",
                    "endpoint": "FEV1",
                    "eligibility_criteria": "adult asthma",
                    "enrollment": 120,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_clinical_prediction_benchmark(
        benchmark,
        backend_root=tmp_path / "missing_clinicalagent",
    )

    assert summary["benchmark_id"] == "clinical_prediction_smoke"
    assert not summary["backend_ready"]
    assert summary["blocked_reason"] == "backend_not_ready"
    assert summary["backend_call_success_rate"] == 0.0
    assert summary["prediction_output_coverage"] == 0.0
    assert summary["false_success_count"] == 0


def test_clinical_prediction_runner_reports_missing_evidence_families(tmp_path):
    backend = _ready_backend(tmp_path / "clinicalagent")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "\n".join(
            [
                "import json, sys",
                "task_json, output_json = sys.argv[1], sys.argv[2]",
                "payload = json.loads(open(task_json, encoding='utf-8').read())",
                "task_id = payload['task']['task_id']",
                "complete = task_id == 'clinical_prediction_complete'",
                "out = {",
                "  'backend': 'fake_clinicalagent',",
                "  'source': 'fake_private_backend',",
                "  'provenance': {'trial_source': 'unit_test'},",
                "  'external_knowledge_evidence': 'retrieved' if complete else 'retrieved',",
                "  'enrollment_evidence': 'feasible' if complete else 'feasible',",
                "  'drug_risk_evidence': 'low' if complete else 'low',",
                "  'disease_risk_evidence': 'moderate' if complete else 'moderate',",
                "  'prediction_label': 'success' if complete else None,",
                "  'prediction_confidence': 0.8 if complete else None,",
                "}",
                "open(output_json, 'w', encoding='utf-8').write(json.dumps(out))",
            ]
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "clinical_prediction.jsonl"
    tasks = [
        _clinical_prediction_task("clinical_prediction_complete", should_succeed=True),
        _clinical_prediction_task("clinical_prediction_incomplete", should_succeed=False),
    ]
    benchmark.write_text("\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8")

    summary = run_clinical_prediction_benchmark(
        benchmark,
        backend_root=backend,
        backend_command=f"{sys.executable} {adapter} {{task_json}} {{output_json}}",
    )

    assert summary["backend_call_success_rate"] == 1.0
    assert summary["prediction_output_coverage"] == 0.5
    assert summary["false_success_count"] == 0
    assert summary["verifier_expectation_match_rate"] == 1.0
    assert summary["missing_evidence_family_counts"] == {"prediction": 1}
    assert summary["failed_check_counts"]["has_prediction_output"] == 1
    incomplete = [row for row in summary["rows"] if row["task_id"].endswith("incomplete")][0]
    assert incomplete["missing_evidence"] == [
        "clinical_outcome_prediction",
        "clinical_prediction_confidence",
    ]
    assert incomplete["missing_evidence_families"] == ["prediction"]


def _ready_backend(root):
    for rel in (
        "algo/agents/tools/drugbank",
        "algo/agents/tools/enrollment/data",
        "algo/agents/tools/hetionet",
        "algo/agents/tools/risk_model",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text("test license\n", encoding="utf-8")
    (root / "algo" / "main.py").write_text("print('fake')\n", encoding="utf-8")
    (root / "algo" / "agents" / "tools" / "enrollment" / "data" / "model.pt").write_text("model\n", encoding="utf-8")
    (root / "algo" / "agents" / "tools" / "enrollment" / "data" / "data.csv").write_text("x\n", encoding="utf-8")
    return root


def _clinical_prediction_task(task_id: str, *, should_succeed: bool) -> dict:
    return {
        "task_id": task_id,
        "raw_user_query": f"Predict clinical trial outcome for NCT12345678 in task {task_id}.",
        "should_succeed": should_succeed,
        "metadata": {
            "domain": "clinical_trial",
            "use_clinicalagent_backend": True,
            "trial_id": "NCT12345678",
            "phase": "Phase II",
            "condition": "asthma",
            "intervention": "DrugA",
            "endpoint": "FEV1",
            "eligibility_criteria": "adult asthma",
            "enrollment": 120,
        },
    }
