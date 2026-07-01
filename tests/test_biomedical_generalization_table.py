from __future__ import annotations

import json

from CAi.toolkit.agent_planner.biomedical_generalization_table import (
    BIOMEDICAL_GENERALIZATION_COLUMNS,
    build_biomedical_generalization_table,
    write_biomedical_generalization_table,
)


def test_biomedical_generalization_table_combines_v2_slices(tmp_path):
    table = build_biomedical_generalization_table(
        [
            "CAi/toolkit/agent_planner/benchmarks/clinical_trial_outcome_prediction_v2_offline.jsonl",
            "CAi/toolkit/agent_planner/benchmarks/drug_target_evidence_v2_offline.jsonl",
        ]
    )

    assert table["columns"] == BIOMEDICAL_GENERALIZATION_COLUMNS
    assert table["row_count"] == 2
    assert {row["domain"] for row in table["rows"]} == {"clinical_trial", "drug_target"}
    assert all(row["task_count"] == 20 for row in table["rows"])
    assert all(row["false_success_count"] == 0 for row in table["rows"])
    assert all(row["verifier_expectation_match_rate"] == 1.0 for row in table["rows"])

    output = tmp_path / "biomedical_generalization_table.json"
    write_biomedical_generalization_table(table, output)

    assert output.exists()
    assert output.with_suffix(".csv").exists()
    assert json.loads(output.read_text(encoding="utf-8"))["row_count"] == 2
