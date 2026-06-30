from __future__ import annotations

import json

from CAi.toolkit.agent_planner.crossdocked_multiseed_summary import build_crossdocked_multiseed_summary


def test_crossdocked_multiseed_summary_aggregates_seed_rows(tmp_path):
    record1 = _record(tmp_path, seed=1, unique_count=28, mean_elapsed=80.0)
    record2 = _record(tmp_path, seed=2, unique_count=35, mean_elapsed=100.0)
    verifier1 = _verifier(tmp_path, seed=1, pass_rate=0.9)
    verifier2 = _verifier(tmp_path, seed=2, pass_rate=1.0)
    prop1 = _property(tmp_path, seed=1, mean_qed=0.7)
    prop2 = _property(tmp_path, seed=2, mean_qed=0.8)

    payload = build_crossdocked_multiseed_summary(
        seed_runs=[
            {"seed": 1, "record_path": str(record1), "verifier_summary_path": str(verifier1), "property_summary_path": str(prop1)},
            {"seed": 2, "record_path": str(record2), "verifier_summary_path": str(verifier2), "property_summary_path": str(prop2)},
        ],
        output_path=tmp_path / "multiseed.json",
    )

    row = payload["rows"][0]
    assert row["seed_count"] == 2
    assert row["total_target_runs"] == 60
    assert row["total_candidates"] == 300
    assert row["mean_task_success_rate"] == 1.0
    assert row["mean_unique_smiles_rate"] == (28 / 150 + 35 / 150) / 2
    assert row["mean_sa_score_pass_rate"] == 0.95
    assert row["mean_rdkit_property_coverage"] == 1.0
    assert row["mean_qed"] == 0.75
    assert row["false_success_count"] == 0
    assert (tmp_path / "multiseed.json").exists()


def _record(tmp_path, *, seed: int, unique_count: int, mean_elapsed: float):
    path = tmp_path / f"record_seed{seed}.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": f"crossdocked_seed{seed}",
                "task_count": 30,
                "summary": {"task_success_rate": 1.0, "false_success_count": 0, "verifier_expectation_match": 1.0},
                "global_candidate_summary": {
                    "generated_candidate_count": 150,
                    "valid_candidate_count": 150,
                    "unique_smiles_count_across_tasks": unique_count,
                    "best_scscore_overall": 3.5,
                    "max_toxicity_score_overall": 0.1,
                },
                "elapsed_summary_sec": {"mean_total": mean_elapsed},
            }
        ),
        encoding="utf-8",
    )
    return path


def _verifier(tmp_path, *, seed: int, pass_rate: float):
    path = tmp_path / f"verifier_seed{seed}.json"
    path.write_text(
        json.dumps({"rows": [{"evidence_type": "sa_score", "coverage": 1.0, "pass_rate": pass_rate}]}),
        encoding="utf-8",
    )
    return path


def _property(tmp_path, *, seed: int, mean_qed: float):
    path = tmp_path / f"property_seed{seed}.json"
    path.write_text(
        json.dumps({"property_rows": [{"property_coverage": 1.0, "mean_qed": mean_qed}]}),
        encoding="utf-8",
    )
    return path
