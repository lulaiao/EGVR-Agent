from __future__ import annotations

import json

from egvr.trace_consistency_audit_runner import run_trace_consistency_audit


def _trace(trace_id: str, timestamp: str, *, task_success: bool = True) -> dict:
    return {
        "trace_id": trace_id,
        "task_id": "task-1",
        "timestamp": timestamp,
        "parsed_task": {
            "task_type": "de_novo_generation",
            "objectives": ["synthesizability"],
            "constraints": {"require_synthesizability": True},
        },
        "tool_sequence": [{"tool_name": "reinvent4_denovo"}, {"tool_name": "scscore"}],
        "tool_calls": [
            {"tool_name": "reinvent4_denovo", "success": True, "outputs": {}},
            {"tool_name": "scscore", "success": True, "outputs": {}},
        ],
        "final_candidates": [
            {"smiles": "CCO", "is_valid": True, "scscore": 2.0, "artifacts": {"path": "/tmp/a"}}
        ],
        "verifier_result": {
            "success": task_success,
            "checks": {
                "has_tool_success": True,
                "has_valid_smiles": True,
                "has_unique_molecules": True,
                "passes_synthesizability": True,
            },
            "failure_reason": None,
        },
        "task_success": task_success,
        "failure_reason": None,
        "metadata": {"planner_baseline": "full_copilot", "repair_executed": False},
    }


def test_trace_audit_deduplicates_latest_record_and_checks_consistency(tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    older = _trace("trace-old", "2026-01-01T00:00:00+00:00")
    newer = _trace("trace-new", "2026-01-02T00:00:00+00:00")
    trace_file.write_text("\n".join([json.dumps(older), json.dumps(newer)]), encoding="utf-8")

    payload = run_trace_consistency_audit(
        sources={"controlled": [trace_file]},
        quotas={"controlled": 1},
        output_dir=tmp_path / "out",
        project_root=tmp_path,
    )

    assert payload["audited_trace_count"] == 1
    assert payload["duplicate_group_count"] == 1
    assert payload["rows"][0]["trace_id"] == "trace-new"
    assert payload["rows"][0]["audit_pass"] is True
    duplicates = json.loads((tmp_path / "out" / "trace_duplicate_report.json").read_text())
    assert duplicates["rows"][0]["record_count"] == 2


def test_trace_audit_flags_success_without_required_evidence(tmp_path):
    trace_file = tmp_path / "bad.jsonl"
    bad = _trace("trace-bad", "2026-01-01T00:00:00+00:00")
    bad["final_candidates"][0]["scscore"] = None
    trace_file.write_text(json.dumps(bad), encoding="utf-8")

    payload = run_trace_consistency_audit(
        sources={"controlled": [trace_file]},
        quotas={"controlled": 1},
        output_dir=tmp_path / "out",
        project_root=tmp_path,
    )

    row = payload["rows"][0]
    assert row["audit_pass"] is False
    assert "passes_synthesizability" in row["violations"]


def test_trace_audit_supports_provenance_bearing_domain_evidence(tmp_path):
    trace_file = tmp_path / "clinical.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "task_id": "clinical-1",
                "parsed_task": {"task_type": "clinical_trial_prediction"},
                "planned_workflow": {"planner_type": "clinical_backend", "tool_sequence": []},
                "tool_calls": [{"tool_name": "predictor", "success": True}],
                "evidence_records": [
                    {
                        "evidence_type": "prediction",
                        "required": True,
                        "supports": True,
                        "value": 1,
                        "provenance": {"source": "backend"},
                    }
                ],
                "verifier_result": {
                    "success": True,
                    "checks": {
                        "has_required_evidence": True,
                        "has_provenance": True,
                        "no_missing_evidence": True,
                        "has_tool_success": True,
                    },
                    "failure_reason": None,
                },
                "task_success": True,
                "failure_reason": None,
            }
        ),
        encoding="utf-8",
    )

    payload = run_trace_consistency_audit(
        sources={"clinical": [trace_file]},
        quotas={"clinical": 1},
        output_dir=tmp_path / "out",
        project_root=tmp_path,
    )

    assert payload["rows"][0]["audit_pass"] is True
    assert payload["rows"][0]["artifact_reference_present"] is True
