"""Build a balanced 20-task benchmark for executed LLM-router audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_TYPES = (
    "pocket_conditioned_generation",
    "docking_evaluation",
    "hit_to_lead_optimization",
    "scaffold_conditioned_generation",
)


def generate_llm_router_executed_benchmark(
    *,
    crossdocked_path: str | Path,
    docking_path: str | Path,
    generalization_path: str | Path,
    output_path: str | Path,
    per_family: int = 5,
) -> list[dict[str, Any]]:
    generalization = _load(generalization_path)
    groups = {
        "pocket_conditioned_generation": _load(crossdocked_path),
        "docking_evaluation": _load(docking_path),
        "hit_to_lead_optimization": [
            row for row in generalization if row.get("expected_task_type") == "hit_to_lead_optimization"
        ],
        "scaffold_conditioned_generation": [
            row for row in generalization if row.get("expected_task_type") == "scaffold_conditioned_generation"
        ],
    }
    selected: list[dict[str, Any]] = []
    for task_type in TASK_TYPES:
        rows = groups[task_type]
        if len(rows) < per_family:
            raise ValueError(f"Need {per_family} {task_type} cases, found {len(rows)}")
        for row in rows[:per_family]:
            copied = dict(row)
            copied["metadata"] = dict(row.get("metadata", {}))
            copied["metadata"].update(
                {
                    "llm_router_executed_slice": "v1",
                    "llm_router_task_family": task_type,
                }
            )
            selected.append(copied)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected) + "\n",
        encoding="utf-8",
    )
    return selected


def _load(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a balanced executed LLM-router benchmark.")
    parser.add_argument("--crossdocked", required=True)
    parser.add_argument("--docking", required=True)
    parser.add_argument("--generalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-family", type=int, default=5)
    args = parser.parse_args()
    rows = generate_llm_router_executed_benchmark(
        crossdocked_path=args.crossdocked,
        docking_path=args.docking,
        generalization_path=args.generalization,
        output_path=args.output,
        per_family=args.per_family,
    )
    print(json.dumps({"task_count": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
