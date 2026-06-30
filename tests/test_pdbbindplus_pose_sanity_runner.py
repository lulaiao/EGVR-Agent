from __future__ import annotations

import json

from CAi.toolkit.agent_planner import pdbbindplus_pose_sanity_runner as runner


def test_pdbbindplus_pose_sanity_runner_gates_on_stable_pose_inputs(tmp_path, monkeypatch):
    pose = tmp_path / "pose.sdf"
    protein = tmp_path / "protein.pdb"
    pose.write_text("pose\n", encoding="utf-8")
    protein.write_text("protein\n", encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "task_id": "pdbbind_refined_prepared_pilot_000_10gs",
                "parsed_task": {"metadata": {"source_protein_path": str(protein)}},
                "final_candidates": [
                    {
                        "source_tool": "vina",
                        "is_valid": False,
                        "artifacts": {"minimized_pose_file_path": str(pose)},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runner,
        "_posebusters_runtime_available",
        lambda posebusters_conda_env: {"available": True, "status": "available", "error": None},
    )
    monkeypatch.setattr(
        runner,
        "_run_prepared_posebusters_cases",
        lambda preflight_rows, posebusters_conda_env: [
            {
                **preflight_rows[0],
                "posebusters_pass": False,
                "posebusters_checks": {"sanitization": True, "volume_overlap_with_protein": False},
                "status": "ok",
            }
        ],
    )

    output = tmp_path / "summary.json"
    payload = runner.run_pdbbindplus_pose_sanity_summary(
        trace_path=trace,
        output_path=output,
        benchmark_id="pdbbindplus_pose_sanity_v2",
        work_dir=tmp_path / "posebusters_inputs",
    )

    assert output.exists()
    assert payload["benchmark_id"] == "pdbbindplus_pose_sanity_v2"
    assert payload["preflight"]["pose_input_count"] == 1
    assert payload["preflight"]["convertible_count"] == 1
    row = payload["rows"][0]
    assert row["dataset"] == "PDBbind+ v2020.R1"
    assert row["status"] == "available"
    assert row["coverage"] == 1.0
    assert row["pass_rate"] == 0.0
    failure_modes = {item["check_name"]: item for item in payload["posebusters_failure_modes"]["rows"]}
    assert failure_modes["volume_overlap_with_protein"]["fail_rate"] == 1.0


def test_pdbbindplus_pose_sanity_runner_does_not_run_when_conversion_unstable(tmp_path, monkeypatch):
    pose = tmp_path / "pose.xyz"
    protein = tmp_path / "protein.pdb"
    pose.write_text("unsupported\n", encoding="utf-8")
    protein.write_text("protein\n", encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "task_id": "pdbbind_refined_prepared_pilot_001_11gs",
                "parsed_task": {"metadata": {"source_protein_path": str(protein)}},
                "final_candidates": [
                    {
                        "source_tool": "vina",
                        "is_valid": False,
                        "artifacts": {"minimized_pose_file_path": str(pose)},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_if_called(_posebusters_conda_env):
        raise AssertionError("PoseBusters runtime should not be checked after conversion preflight fails")

    monkeypatch.setattr(runner, "_posebusters_runtime_available", fail_if_called)

    payload = runner.run_pdbbindplus_pose_sanity_summary(
        trace_path=trace,
        output_path=tmp_path / "summary.json",
        work_dir=tmp_path / "posebusters_inputs",
    )

    row = payload["rows"][0]
    assert row["status"] == "conversion_not_stable"
    assert row["evidence_count"] == 0
    assert row["coverage"] == 0.0
    assert payload["pose_results"] == []
