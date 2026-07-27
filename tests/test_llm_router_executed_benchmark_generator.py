from __future__ import annotations

import json
from collections import Counter

from egvr.llm_router_executed_benchmark_generator import (
    generate_llm_router_executed_benchmark,
)


def test_generator_selects_balanced_task_families(tmp_path):
    def write(path, task_type, count):
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "task_id": f"{task_type}-{index}",
                        "raw_user_query": "test",
                        "expected_task_type": task_type,
                        "metadata": {},
                    }
                )
                for index in range(count)
            ),
            encoding="utf-8",
        )

    crossdocked = tmp_path / "cross.jsonl"
    docking = tmp_path / "dock.jsonl"
    generalization = tmp_path / "generalization.jsonl"
    write(crossdocked, "pocket_conditioned_generation", 5)
    write(docking, "docking_evaluation", 5)
    generalization.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": f"{task_type}-{index}",
                    "raw_user_query": "test",
                    "expected_task_type": task_type,
                    "metadata": {},
                }
            )
            for task_type in ("hit_to_lead_optimization", "scaffold_conditioned_generation")
            for index in range(5)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"

    rows = generate_llm_router_executed_benchmark(
        crossdocked_path=crossdocked,
        docking_path=docking,
        generalization_path=generalization,
        output_path=output,
    )

    assert len(rows) == 20
    assert Counter(row["expected_task_type"] for row in rows) == {
        "pocket_conditioned_generation": 5,
        "docking_evaluation": 5,
        "hit_to_lead_optimization": 5,
        "scaffold_conditioned_generation": 5,
    }
