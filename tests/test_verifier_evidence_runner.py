from __future__ import annotations

import json

from CAi.toolkit.agent_planner.verifier_evidence_runner import (
    _posebusters_failure_mode_rows,
    _parse_posebusters_csv,
    calculate_sa_scores,
    run_verifier_evidence_summary,
)


def test_calculate_sa_scores_scores_valid_smiles():
    payload = calculate_sa_scores(["CCO"])

    assert payload["status"] == "available"
    assert payload["results"][0]["smiles"] == "CCO"
    assert payload["results"][0]["sa_score"] > 0


def test_verifier_evidence_runner_writes_sa_and_posebusters_rows(tmp_path):
    crossdocked_trace = tmp_path / "crossdocked_traces.jsonl"
    crossdocked_trace.write_text(
        json.dumps(
            {
                "task_id": "crossdock_1",
                "final_candidates": [
                    {"smiles": "CCO", "is_valid": True, "source_tool": "rxnflow"},
                    {"smiles": "CCC", "is_valid": True, "source_tool": "rxnflow"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    litpcba_trace = tmp_path / "litpcba_traces.jsonl"
    litpcba_trace.write_text(
        json.dumps(
            {
                "task_id": "dock_1",
                "final_candidates": [
                    {
                        "source_tool": "vina",
                        "is_valid": False,
                        "artifacts": {
                            "docked_poses_file_path": "pose.pdbqt",
                            "minimized_pose_file_path": "minimized.pdbqt",
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "verifier_evidence_summary.json"

    payload = run_verifier_evidence_summary(
        input_path=None,
        output_path=output_path,
        crossdocked_trace_path=crossdocked_trace,
        litpcba_trace_path=litpcba_trace,
        project_root=tmp_path,
    )

    assert output_path.exists()
    rows = {row["evidence_type"]: row for row in payload["rows"]}
    assert rows["sa_score"]["candidate_count"] == 2
    assert rows["sa_score"]["evidence_count"] == 2
    assert rows["sa_score"]["coverage"] == 1.0
    assert rows["sa_score"]["best_sa_score"] > 0
    assert rows["posebusters"]["pose_artifact_count"] == 1
    assert rows["posebusters"]["status"] in {"not_available", "not_evaluated"}
    assert rows["posebusters"]["coverage"] == 0.0


def test_parse_posebusters_csv_skips_warning_lines():
    output = """[10:22:03] Explicit valence warning
file,molecule,position,mol_pred_loaded,sanitization,volume_overlap_with_protein
pose.sdf,pose,0,True,False,True
"""

    rows = _parse_posebusters_csv(output)

    assert rows == [
        {
            "file": "pose.sdf",
            "molecule": "pose",
            "position": "0",
            "checks": {
                "mol_pred_loaded": True,
                "sanitization": False,
                "volume_overlap_with_protein": True,
            },
        }
    ]


def test_posebusters_failure_mode_rows_aggregate_checks():
    rows = _posebusters_failure_mode_rows(
        [
            {
                "task_id": "litpcba_docking_000_ADRB2",
                "posebusters_checks": {"sanitization": False, "mol_pred_loaded": True},
            },
            {
                "task_id": "litpcba_docking_001_ALDH1",
                "posebusters_checks": {"sanitization": True, "mol_pred_loaded": True},
            },
        ]
    )

    by_check = {row["check_name"]: row for row in rows}
    assert by_check["sanitization"]["pose_count"] == 2
    assert by_check["sanitization"]["fail_count"] == 1
    assert by_check["sanitization"]["fail_rate"] == 0.5
    assert by_check["sanitization"]["example_task_ids"] == "litpcba_docking_000_ADRB2"
    assert by_check["mol_pred_loaded"]["fail_count"] == 0
