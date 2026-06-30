from __future__ import annotations

import json

from CAi.toolkit.agent_planner.statistical_summary_builder import (
    build_and_write_statistical_summary_table,
    build_statistical_summary_rows,
)


def test_statistical_summary_builder_adds_wilson_intervals():
    master_payload = {
        "views": {
            "robustness_repeated": {
                "rows": [
                    {
                        "benchmark_id": "failure_recovery_taxonomy_v2_repeated",
                        "planner_baseline": "full_copilot",
                        "repeat_count": 3,
                        "task_count": 9,
                        "mean_task_success_rate": 4 / 9,
                        "std_task_success_rate": 0.0,
                        "mean_verifier_expectation_match": 1.0,
                        "std_verifier_expectation_match": 0.0,
                        "false_success_count": 0,
                    }
                ]
            },
            "pdbbind_prep_gate": {
                "rows": [
                    {
                        "benchmark_id": "pdbbind_receptor_prep_probe_v1",
                        "dataset": "PDBbind+ v2020.R1",
                        "prep_success_rate": 0.62,
                        "receptor_prep_target_count": 50,
                        "real_pilot_success_rate": 28 / 30,
                        "real_pilot_task_count": 30,
                        "false_success_count": 0,
                    }
                ]
            },
        }
    }

    rows = build_statistical_summary_rows(master_payload=master_payload)

    task_success = next(row for row in rows if row["metric"] == "task_success_rate")
    assert task_success["n"] == 27
    assert task_success["ci95_low"] is not None
    assert task_success["ci95_high"] is not None
    assert task_success["false_success_count"] == 0
    pilot_success = next(row for row in rows if row["metric"] == "real_pilot_success_rate")
    assert pilot_success["n"] == 30


def test_statistical_summary_builder_writes_outputs(tmp_path):
    master = tmp_path / "master.json"
    master.write_text(
        json.dumps(
            {
                "views": {
                    "crossdocked_multiseed": {
                        "rows": [
                            {
                                "benchmark_id": "crossdocked_rxnflow_candidates5_targets30_multiseed_v1",
                                "dataset": "CrossDocked2020",
                                "total_target_runs": 90,
                                "total_candidates": 450,
                                "mean_task_success_rate": 1.0,
                                "std_task_success_rate": 0.0,
                                "mean_valid_candidate_rate": 1.0,
                                "std_valid_candidate_rate": 0.0,
                                "mean_sa_score_pass_rate": 0.95,
                                "false_success_count": 0,
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    payload = build_and_write_statistical_summary_table(
        master_table_path=master,
        output_dir=tmp_path / "tables",
        project_root=tmp_path,
    )

    assert payload["row_count"] == 3
    assert (tmp_path / "tables" / "statistical_summary_table.csv").exists()
    assert (tmp_path / "tables" / "statistical_summary_table.json").exists()
    assert (tmp_path / "tables" / "statistical_summary_table.tex").exists()
