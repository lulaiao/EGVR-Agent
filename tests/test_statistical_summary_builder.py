from __future__ import annotations

import json

from egvr.statistical_summary_builder import (
    build_clustered_statistical_rows,
    build_and_write_statistical_summary_table,
    build_statistical_summary_rows,
    write_clustered_statistical_summary,
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
    assert task_success["n"] == 9
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


def test_clustered_statistics_count_unique_scenarios_not_repeats(tmp_path):
    paths = []
    for repeat in range(3):
        for baseline, successes in (("rule_based_planner", [False, False]), ("full_copilot", [True, False])):
            path = tmp_path / f"repeat{repeat}_{baseline}.json"
            path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "taxonomy",
                        "planner_baseline": baseline,
                        "results": [
                            {
                                "task_id": f"case-{index}",
                                "scenario_template_id": f"scenario-{index}",
                                "task_success": success,
                                "expected_success": success,
                                "agent_claimed_success": success,
                            }
                            for index, success in enumerate(successes)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)

    rows = build_clustered_statistical_rows(
        paths,
        project_root=tmp_path,
        bootstrap_samples=200,
        random_seed=20260707,
    )
    full = next(row for row in rows if row["row_type"] == "method" and row["planner_baseline"] == "full_copilot")
    delta = next(row for row in rows if row["row_type"] == "paired_delta")

    assert full["unique_scenario_count"] == 2
    assert full["observation_count"] == 6
    assert full["repeat_count"] == 3
    assert full["estimate"] == 0.5
    assert delta["estimate"] == 0.5
    payload = write_clustered_statistical_summary(rows, output_dir=tmp_path / "clustered", project_root=tmp_path)
    assert payload["row_count"] == 3
    assert (tmp_path / "clustered" / "statistical_summary_clustered.tex").exists()
