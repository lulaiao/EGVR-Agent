from __future__ import annotations

import json

from CAi.toolkit.agent_planner.evidence_audit_builder import (
    build_and_write_evidence_audit_table,
    build_evidence_audit_rows,
)


def test_evidence_audit_builder_maps_claims_to_existing_views(tmp_path):
    master_payload = {
        "views": {
            "robustness_repeated": {
                "rows": [
                    {
                        "benchmark_id": "failure_recovery_taxonomy_v2_repeated",
                        "dataset": "controlled_tool_failure_injection",
                        "execution_mode": "real",
                    }
                ]
            },
            "llm_router_baseline": {"rows": []},
        },
        "artifacts": {
            "views": {
                "robustness_repeated": {"json": "logs/robustness_repeated_table.json"},
            }
        },
    }

    rows = build_evidence_audit_rows(master_payload=master_payload, project_root=tmp_path)

    robustness = next(row for row in rows if row["claim_id"] == "C1")
    assert robustness["benchmark_id"] == "failure_recovery_taxonomy_v2_repeated"
    assert robustness["is_real_result"] is True
    assert robustness["is_controlled"] is True
    assert robustness["row_count"] == 1
    assert robustness["result_path"] == "logs/robustness_repeated_table.json"


def test_evidence_audit_builder_writes_csv_json_tex(tmp_path):
    master = tmp_path / "master.json"
    master.write_text(json.dumps({"views": {"robustness_repeated": {"rows": []}}}), encoding="utf-8")

    payload = build_and_write_evidence_audit_table(
        master_table_path=master,
        output_dir=tmp_path / "paper",
        project_root=tmp_path,
    )

    assert payload["row_count"] >= 8
    assert (tmp_path / "paper" / "evidence_audit_table.csv").exists()
    assert (tmp_path / "paper" / "evidence_audit_table.json").exists()
    assert (tmp_path / "paper" / "evidence_audit_table.tex").exists()
