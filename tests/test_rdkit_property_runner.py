from __future__ import annotations

import json

from CAi.toolkit.agent_planner.rdkit_property_runner import run_property_verifier_summary


def test_rdkit_property_runner_writes_property_and_verifier_rows(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "task_id": "crossdocked_demo",
                "final_candidates": [
                    {"smiles": "CCO", "is_valid": True},
                    {"smiles": "c1ccccc1", "is_valid": True},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "property_summary.json"

    payload = run_property_verifier_summary(trace_path=trace_path, output_path=output)

    assert output.exists()
    assert payload["property_rows"][0]["candidate_count"] == 2
    assert payload["property_rows"][0]["property_coverage"] == 1.0
    assert payload["property_rows"][0]["status"] == "available"
    assert payload["verifier_evidence_rows"][0]["evidence_type"] == "rdkit_property_verifier"
    assert len(payload["candidate_rows"]) == 2
