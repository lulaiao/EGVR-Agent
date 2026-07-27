from __future__ import annotations

import json

from egvr.repair_budget_runner import (
    _prepare_budget_benchmark,
    run_repair_budget_experiment,
)


def test_repair_budget_runner_reports_budget_recovery_and_cost(tmp_path):
    benchmark = tmp_path / "tasks.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "budget_case",
                "raw_user_query": "Generate molecules de novo for synthesizability.",
                "expected_task_type": "de_novo_generation",
                "expected_tools": ["reinvent4_denovo", "scscore"],
                "should_succeed": True,
                "mock_outputs": {
                    "reinvent4_denovo": [
                        {"success": False, "error": "initial"},
                        {"success": False, "error": "repair one"},
                        {"success": True, "molecules_smiles": ["CCO"]},
                    ]
                },
                "metadata": {
                    "scenario_template_id": "generation:transient",
                    "variant_id": "v1",
                    "failure_family": "generation",
                    "repairability": "recoverable",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_repair_budget_experiment(
        benchmark_path=benchmark,
        output_dir=tmp_path / "out",
        execution_mode="mock",
        budgets=[0, 1, 2],
    )

    rows = {row["repair_budget"]: row for row in payload["rows"]}
    assert rows[0]["evidence_verified_success_rate"] == 0.0
    assert rows[1]["evidence_verified_success_rate"] == 0.0
    assert rows[2]["evidence_verified_success_rate"] == 1.0
    assert rows[2]["mean_calls_per_task"] > rows[0]["mean_calls_per_task"]
    assert (tmp_path / "out" / "repair_budget_summary.tex").exists()


def test_budget_benchmark_extends_only_irrecoverable_failure_injections(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "irrecoverable",
                        "raw_user_query": "x",
                        "metadata": {
                            "repairability": "irrecoverable",
                            "failure_injections": {
                                "vina": [{"call_index": 1, "mode": "error", "error": "persistent"}]
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "task_id": "recoverable",
                        "raw_user_query": "x",
                        "metadata": {
                            "repairability": "recoverable",
                            "failure_injections": {
                                "vina": [{"call_index": 1, "mode": "error", "error": "transient"}]
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target = _prepare_budget_benchmark(source, tmp_path / "target.jsonl", max_budget=2)
    rows = [json.loads(line) for line in target.read_text().splitlines()]

    assert [item["call_index"] for item in rows[0]["metadata"]["failure_injections"]["vina"]] == [1, 2, 3]
    assert len(rows[1]["metadata"]["failure_injections"]["vina"]) == 1
