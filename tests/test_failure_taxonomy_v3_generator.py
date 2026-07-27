from __future__ import annotations

import json
from collections import Counter

from egvr.failure_taxonomy_v3_generator import generate_failure_taxonomy_v3


def test_failure_taxonomy_v3_has_balanced_families_scenarios_and_variants(tmp_path):
    docking = tmp_path / "docking.jsonl"
    docking.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": f"dock-{index}",
                    "raw_user_query": (
                        f"Dock ligand_path=/tmp/ligand{index}.sdf against protein_path=/tmp/protein{index}.pdb "
                        "pocket_center=[1,2,3] box_size=[20,20,20]."
                    ),
                    "metadata": {"dataset": "test"},
                }
            )
            for index in range(3)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "taxonomy.jsonl"

    cases = generate_failure_taxonomy_v3(docking_benchmark=docking, output_path=output)

    assert len(cases) == 72
    assert len({case["task_id"] for case in cases}) == 72
    assert Counter(case["metadata"]["failure_family"] for case in cases) == {
        "generation": 18,
        "synthesizability": 18,
        "toxicity": 18,
        "docking": 18,
    }
    assert len({case["metadata"]["scenario_template_id"] for case in cases}) == 24
    assert all(case["metadata"]["repairability"] in {"healthy", "recoverable", "irrecoverable"} for case in cases)
    assert sum(case["metadata"]["repairability"] == "healthy" for case in cases) == 12
    assert sum(case["metadata"]["repairability"] == "recoverable" for case in cases) == 30
    assert output.read_text(encoding="utf-8").count("\n") == 72
