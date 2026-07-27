from __future__ import annotations

import json

from egvr.evidence_corruption_runner import run_evidence_corruption


def test_evidence_corruption_detects_required_fields_and_preserves_clean_specificity(tmp_path):
    trace = {
        "trace_id": "trace_1",
        "task_id": "task_1",
        "task_success": True,
        "parsed_task": {
            "task_id": "task_1",
            "raw_user_query": "Generate molecules and score synthesizability.",
            "task_type": "de_novo_generation",
            "objectives": ["synthesizability"],
            "constraints": {"require_synthesizability": True},
        },
        "selected_tools": ["reinvent4_denovo", "scscore"],
        "tool_sequence": [
            {"tool_name": "reinvent4_denovo", "reason": "generate"},
            {"tool_name": "scscore", "reason": "score"},
        ],
        "tool_calls": [
            {"tool_name": "reinvent4_denovo", "success": True, "outputs": {"molecules_smiles": ["CCO"]}},
            {"tool_name": "scscore", "success": True, "outputs": {"results": [{"scscore": 2.1}]}},
        ],
        "final_candidates": [
            {
                "smiles": "CCO",
                "source_tool": "reinvent4_denovo",
                "is_valid": True,
                "rank": 1,
                "scscore": 2.1,
                "artifacts": {"result_path": "/tmp/result.csv"},
            }
        ],
        "verifier_result": {
            "success": True,
            "checks": {
                "has_tool_success": True,
                "has_valid_smiles": True,
                "has_unique_molecules": True,
                "passes_synthesizability": True,
            },
        },
        "metadata": {"planner_baseline": "full_copilot"},
    }
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    payload = run_evidence_corruption(traces_path=traces, output_dir=tmp_path / "out")

    assert payload["successful_trace_count"] == 1
    assert payload["integrity_gate_specificity"] == 1.0
    rows = {row["corruption_family"]: row for row in payload["rows"]}
    assert rows["required_evidence"]["integrity_gate_sensitivity"] == 1.0
    assert rows["required_score"]["integrity_gate_sensitivity"] == 1.0
    assert rows["artifact_reference"]["integrity_gate_sensitivity"] == 1.0
    assert rows["execution_order"]["integrity_gate_sensitivity"] == 1.0
    assert rows["tool_call_consistency"]["integrity_gate_sensitivity"] == 1.0
