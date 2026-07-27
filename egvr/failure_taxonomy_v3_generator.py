"""Generate the medium-scale controlled reliability taxonomy v3."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


FAMILIES = ("generation", "synthesizability", "toxicity", "docking")
SCENARIOS = (
    "healthy_control",
    "nominal_missing_evidence",
    "wrong_schema_evidence",
    "transient_explicit_failure",
    "persistent_failure",
    "dependency_failure",
)
MOLECULE_VARIANTS = ("CCO", "CCN", "c1ccccc1")


def generate_failure_taxonomy_v3(
    *,
    docking_benchmark: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    docking_cases = _load_jsonl(docking_benchmark)[:3]
    if len(docking_cases) < 3:
        raise ValueError("docking benchmark must contain at least three cases")
    cases: list[dict[str, Any]] = []
    for family in FAMILIES:
        for scenario in SCENARIOS:
            for variant_index in range(3):
                cases.append(
                    _build_case(
                        family,
                        scenario,
                        variant_index=variant_index,
                        docking_case=docking_cases[variant_index],
                    )
                )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False, sort_keys=True) for case in cases) + "\n",
        encoding="utf-8",
    )
    return cases


def _build_case(
    family: str,
    scenario: str,
    *,
    variant_index: int,
    docking_case: dict[str, Any],
) -> dict[str, Any]:
    variant_id = f"v{variant_index + 1:02d}"
    template_id = f"taxonomy_v3:{family}:{scenario}"
    task_id = f"taxonomy_v3_{family}_{scenario}_{variant_id}"
    smiles = MOLECULE_VARIANTS[variant_index]
    if family == "docking":
        case = {
            "task_id": task_id,
            "raw_user_query": docking_case["raw_user_query"],
            "expected_task_type": "docking_evaluation",
            "expected_tools": ["vina"],
            "metadata": deepcopy(docking_case.get("metadata", {})),
        }
    else:
        objective = {
            "generation": "",
            "synthesizability": " for synthesizability",
            "toxicity": " for toxicity",
        }[family]
        tools = ["reinvent4_mol2mol"] if family == "generation" else ["reinvent4_denovo"]
        if family == "synthesizability":
            tools.append("scscore")
        if family == "toxicity":
            tools.append("toxicity")
        case = {
            "task_id": task_id,
            "raw_user_query": (
                f"Optimize hit_smiles={smiles} by generating 1 analog."
                if family == "generation"
                else f"Generate 1 molecule de novo{objective}."
            ),
            "expected_task_type": "hit_to_lead_optimization" if family == "generation" else "de_novo_generation",
            "expected_tools": tools,
            "metadata": {},
        }

    repairability = _repairability(family, scenario)
    case["should_succeed"] = repairability != "irrecoverable"
    case["metadata"].update(
        {
            "split": "failure_taxonomy_v3",
            "task_family": "failure_recovery",
            "failure_family": family,
            "failure_scenario": scenario,
            "scenario_template_id": template_id,
            "variant_id": variant_id,
            "repairability": repairability,
            "controlled_wrapper_injection": scenario != "healthy_control",
            "expected_repair_tools": _expected_repair_tools(family, scenario),
            "expected_repair_action_types": _expected_action_types(family, scenario),
        }
    )
    injections = _injections(family, scenario, smiles)
    if injections:
        case["metadata"]["failure_injections"] = injections
    elif family != "docking":
        case["metadata"]["failure_injections"] = {
            "reinvent4_denovo": [{"call_index": 1, "output": _generation_output(smiles)}]
        }
    return case


def _repairability(family: str, scenario: str) -> str:
    if scenario == "healthy_control":
        return "healthy"
    if scenario in {"nominal_missing_evidence", "wrong_schema_evidence"}:
        return "recoverable"
    if family == "generation" and scenario in {"transient_explicit_failure", "persistent_failure"}:
        return "recoverable"
    return "irrecoverable"


def _expected_repair_tools(family: str, scenario: str) -> list[str]:
    if _repairability(family, scenario) != "recoverable":
        return []
    if family == "generation":
        return ["reinvent4_mol2mol", "reinvent4_denovo"]
    return [{
        "synthesizability": "scscore",
        "toxicity": "toxicity",
        "docking": "vina",
    }[family]]


def _expected_action_types(family: str, scenario: str) -> list[str]:
    if _repairability(family, scenario) != "recoverable":
        return []
    if family == "generation":
        return ["retry_with_reduced_generation_count", "fallback_tool"]
    return ["retry_evaluator_for_missing_evidence"]


def _injections(family: str, scenario: str, smiles: str) -> dict[str, list[dict[str, Any]]]:
    generation = {"reinvent4_denovo": [{"call_index": 1, "output": _generation_output(smiles)}]}
    tool = {
        "generation": "reinvent4_mol2mol",
        "synthesizability": "scscore",
        "toxicity": "toxicity",
        "docking": "vina",
    }[family]
    if scenario == "healthy_control":
        return {} if family == "docking" else generation
    if family == "generation":
        injections = {tool: _generation_injection(scenario, smiles)}
        if scenario == "dependency_failure":
            injections["reinvent4_denovo"] = [
                {"call_index": 1, "mode": "error", "error": "taxonomy_v3 missing fallback dependency"}
            ]
        return injections
    injections = {} if family == "docking" else generation
    injections[tool] = _evaluator_injection(family, scenario, smiles)
    return injections


def _generation_injection(scenario: str, smiles: str) -> list[dict[str, Any]]:
    if scenario == "nominal_missing_evidence":
        return [{"call_index": 1, "output": {"success": True, "molecules_smiles": []}}]
    if scenario == "wrong_schema_evidence":
        return [{"call_index": 1, "output": {"success": True, "molecules": [smiles]}}]
    if scenario == "transient_explicit_failure":
        return [{"call_index": 1, "mode": "error", "error": "taxonomy_v3 transient generation failure"}]
    if scenario == "persistent_failure":
        return [
            {"call_index": 1, "mode": "error", "error": "taxonomy_v3 persistent generation failure"},
            {"call_index": 2, "mode": "error", "error": "taxonomy_v3 persistent generation retry failure"},
        ]
    return [
        {"call_index": 1, "mode": "error", "error": "taxonomy_v3 missing generation dependency"},
        {"call_index": 2, "mode": "error", "error": "taxonomy_v3 missing generation dependency"},
    ]


def _evaluator_injection(family: str, scenario: str, smiles: str) -> list[dict[str, Any]]:
    if scenario == "nominal_missing_evidence":
        output = {"success": True, "results": []} if family == "synthesizability" else {"success": True}
        return [{"call_index": 1, "output": output}]
    if scenario == "wrong_schema_evidence":
        output = {
            "synthesizability": {"success": True, "results": [{"canonical_smiles": smiles}]},
            "toxicity": {"success": True, "toxicity_label": "unknown"},
            "docking": {"success": True, "score": -7.0},
        }[family]
        return [{"call_index": 1, "output": output}]
    error = {
        "transient_explicit_failure": "taxonomy_v3 transient evaluator failure",
        "persistent_failure": "taxonomy_v3 persistent evaluator failure",
        "dependency_failure": "taxonomy_v3 missing evaluator dependency",
    }[scenario]
    calls = [{"call_index": 1, "mode": "error", "error": error}]
    if scenario in {"persistent_failure", "dependency_failure"}:
        calls.append({"call_index": 2, "mode": "error", "error": error})
    return calls


def _generation_output(smiles: str) -> dict[str, Any]:
    return {"success": True, "molecules_smiles": [smiles], "controlled_generation": True}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate failure taxonomy v3 JSONL.")
    parser.add_argument("--docking-benchmark", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = generate_failure_taxonomy_v3(
        docking_benchmark=args.docking_benchmark,
        output_path=args.output,
    )
    print(json.dumps({"task_count": len(cases), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
