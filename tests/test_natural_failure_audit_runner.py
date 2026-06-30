from __future__ import annotations

import json

from CAi.toolkit.agent_planner.natural_failure_audit_runner import run_natural_failure_audit


def test_natural_failure_audit_aggregates_trace_and_prep_failures(tmp_path):
    log_root = tmp_path / "logs" / "baseline_runs"
    trace_dir = log_root / "pdbbindplus_v2020r1_prepared_pilot_v3" / "traces_30"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "20260605_traces.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "task_id": "017_1a3e",
                "parsed_task": {"task_type": "docking_evaluation", "metadata": {"dataset": "PDBbind+"}},
                "task_success": False,
                "failure_reason": "vina docking runtime error",
                "tool_calls": [
                    {
                        "tool_name": "vina",
                        "success": False,
                        "error": "Vina docking runtime error: conversion failed",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    controlled_dir = log_root / "failure_recovery_taxonomy_v2_real" / "traces"
    controlled_dir.mkdir(parents=True)
    (controlled_dir / "20260605_traces.jsonl").write_text(
        json.dumps(
            {
                "task_id": "controlled",
                "task_success": False,
                "tool_calls": [{"tool_name": "scscore", "success": False, "error": "injected failure"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prep = tmp_path / "prep_summary.json"
    prep.write_text(
        json.dumps(
            {
                "benchmark_id": "pdbbind_receptor_prep_probe_v1",
                "ready_targets": [
                    {
                        "pdb_id": "1abc",
                        "prep_success": False,
                        "failure_type": "timeout",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = run_natural_failure_audit(
        log_root=log_root,
        prep_summary_paths=[prep],
        output_path=tmp_path / "audit.json",
        project_root=tmp_path,
    )

    assert payload["event_count"] == 3
    families = {row["failure_family"] for row in payload["rows"]}
    assert "format_or_pose_conversion_failure" in families
    assert "docking_runtime_failure" in families
    assert "timeout" in families
    assert all("controlled" not in row["example_task_ids"] for row in payload["rows"])
    assert (tmp_path / "audit.csv").exists()
    assert (tmp_path / "audit.tex").exists()
