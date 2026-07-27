"""Controlled evidence-corruption evaluation over successful execution traces."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .task_schema import (
    CandidateRecord,
    ParsedTask,
    PlannedWorkflow,
    TaskConstraints,
    ToolCallRecord,
)
from .verifier import verify_workflow


CORRUPTION_FAMILIES = (
    "required_evidence",
    "provenance",
    "artifact_reference",
    "required_score",
    "execution_order",
    "tool_call_consistency",
)


def run_evidence_corruption(
    *,
    traces_path: str | Path,
    output_dir: str | Path,
    successful_only: bool = True,
) -> dict[str, Any]:
    traces = _read_jsonl(Path(traces_path))
    if successful_only:
        traces = [trace for trace in traces if trace.get("task_success") is True]

    rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for trace in traces:
        contract = _evidence_contract(trace)
        clean_gate = _integrity_gate(trace, contract)
        clean_replay = _workflow_verifier_replay(trace)
        clean_rows.append(
            {
                "task_id": trace.get("task_id"),
                "trace_id": trace.get("trace_id"),
                "domain": contract["domain"],
                "integrity_gate_success": clean_gate["success"],
                "workflow_verifier_eligible": clean_replay is not None,
                "workflow_verifier_success": clean_replay["success"] if clean_replay else None,
                "failure_reasons": clean_gate["failure_reasons"],
                "source": "new_offline_run",
            }
        )
        for family in CORRUPTION_FAMILIES:
            corrupted, applicable, detail = _corrupt_trace(trace, family, contract)
            if not applicable:
                continue
            gate = _integrity_gate(corrupted, contract)
            replay = _workflow_verifier_replay(corrupted)
            rows.append(
                {
                    "task_id": trace.get("task_id"),
                    "trace_id": trace.get("trace_id"),
                    "domain": contract["domain"],
                    "corruption_family": family,
                    "corruption_detail": detail,
                    "integrity_gate_detected": not gate["success"],
                    "integrity_gate_failure_reasons": gate["failure_reasons"],
                    "workflow_verifier_eligible": replay is not None,
                    "workflow_verifier_detected": (not replay["success"]) if replay else None,
                    "workflow_verifier_failure_reason": replay["failure_reason"] if replay else None,
                    "false_success_integrity_gate": gate["success"],
                    "false_success_workflow_verifier": replay["success"] if replay else None,
                    "source": "new_offline_run",
                }
            )

    aggregate = _aggregate(rows, clean_rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "evidence_corruption_per_case.csv", rows)
    _write_json(output / "evidence_corruption_per_case.json", {"rows": rows})
    _write_csv(output / "evidence_corruption_clean_controls.csv", clean_rows)
    _write_json(output / "evidence_corruption_summary.json", aggregate)
    _write_csv(output / "evidence_corruption_summary.csv", aggregate["rows"])
    (output / "evidence_corruption_summary.tex").write_text(
        _latex(aggregate["rows"]), encoding="utf-8"
    )
    return aggregate


def _evidence_contract(trace: dict[str, Any]) -> dict[str, Any]:
    clinical = isinstance(trace.get("evidence_records"), list)
    calls = [item for item in trace.get("tool_calls") or [] if isinstance(item, dict)]
    candidates = [item for item in trace.get("final_candidates") or [] if isinstance(item, dict)]
    sequence_payload = trace.get("tool_sequence") or (trace.get("planned_workflow") or {}).get("tool_sequence") or []
    sequence = [
        str(item.get("tool_name"))
        for item in sequence_payload
        if isinstance(item, dict) and item.get("tool_name")
    ]
    required_checks = _required_checks(trace)
    return {
        "domain": "clinical" if clinical else "molecular",
        "required_checks": required_checks,
        "require_provenance": clinical and any(
            item.get("required") and (item.get("provenance") or item.get("source"))
            for item in trace.get("evidence_records") or []
            if isinstance(item, dict)
        ),
        "require_artifact_reference": _has_artifact_reference(trace),
        "require_score": any(
            check in required_checks
            for check in ("has_docking_scores", "passes_synthesizability", "passes_toxicity")
        ),
        "expected_tool_order": sequence,
        "require_consistent_calls": bool(calls),
        "candidate_count": len(candidates),
    }


def _integrity_gate(trace: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    observed = _observed_evidence(trace)
    for check in contract["required_checks"]:
        if not observed.get(check, False):
            reasons.append(check)
    if contract["require_provenance"] and not observed.get("has_provenance"):
        reasons.append("has_provenance")
    if contract["require_artifact_reference"] and not _has_artifact_reference(trace):
        reasons.append("has_artifact_reference")
    if not _order_consistent(contract["expected_tool_order"], trace.get("tool_calls") or []):
        reasons.append("execution_order_consistent")
    if contract["require_consistent_calls"] and not _tool_calls_consistent(trace.get("tool_calls") or []):
        reasons.append("tool_call_consistent")
    return {"success": not reasons, "failure_reasons": sorted(set(reasons))}


def _corrupt_trace(
    trace: dict[str, Any],
    family: str,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], bool, str]:
    value = copy.deepcopy(trace)
    if family == "required_evidence":
        if contract["domain"] == "clinical":
            for record in value.get("evidence_records") or []:
                if isinstance(record, dict) and record.get("required") and record.get("value") is not None:
                    record["value"] = None
                    record["missing_reason"] = "controlled_corruption"
                    return value, True, "removed one required evidence value"
            return value, False, "no present required evidence"
        candidates = value.get("final_candidates") or []
        if "has_valid_smiles" in contract["required_checks"] and candidates:
            for candidate in candidates:
                if isinstance(candidate, dict):
                    candidate["smiles"] = None
                    candidate["is_valid"] = False
            return value, True, "removed candidate identity and validity"
        if any(
            check in contract["required_checks"]
            for check in ("has_docking_scores", "passes_synthesizability", "passes_toxicity")
        ):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    for key in ("docking_score", "scscore", "sa_score", "toxicity_score", "pmic_score"):
                        candidate[key] = None
            for call in value.get("tool_calls") or []:
                outputs = call.get("outputs") if isinstance(call, dict) else None
                if isinstance(outputs, dict):
                    _remove_score_values(outputs)
            return value, True, "removed task-required evaluator evidence"
        return value, False, "no candidate evidence"

    if family == "provenance":
        changed = False
        for record in value.get("evidence_records") or []:
            if not isinstance(record, dict) or not record.get("required"):
                continue
            for key in ("provenance", "source"):
                if record.get(key):
                    record[key] = None
                    changed = True
        return value, changed, "removed required evidence provenance"

    if family == "artifact_reference":
        if not contract["require_artifact_reference"]:
            return value, False, "no artifact contract"
        for candidate in value.get("final_candidates") or []:
            if isinstance(candidate, dict):
                candidate["artifacts"] = {}
        for call in value.get("tool_calls") or []:
            outputs = call.get("outputs") if isinstance(call, dict) else None
            if isinstance(outputs, dict):
                for key in list(outputs):
                    if "path" in str(key).lower() or "artifact" in str(key).lower():
                        outputs.pop(key, None)
        return value, True, "removed artifact/path references"

    if family == "required_score":
        if not contract["require_score"]:
            return value, False, "no required score"
        for candidate in value.get("final_candidates") or []:
            if isinstance(candidate, dict):
                for key in ("docking_score", "scscore", "sa_score", "toxicity_score", "pmic_score"):
                    candidate[key] = None
        for call in value.get("tool_calls") or []:
            outputs = call.get("outputs") if isinstance(call, dict) else None
            if isinstance(outputs, dict):
                _remove_score_values(outputs)
        return value, True, "removed required score fields"

    if family == "execution_order":
        calls = value.get("tool_calls") or []
        pair = _first_distinct_pair(calls)
        if pair is None:
            return value, False, "fewer than two distinct tool calls"
        left, right = pair
        calls[left], calls[right] = calls[right], calls[left]
        return value, True, "swapped first two distinct tool calls"

    if family == "tool_call_consistency":
        for call in value.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("success") and call.get("outputs") is not None:
                call["outputs"] = None
                return value, True, "kept success status but removed outputs"
        return value, False, "no successful call with outputs"

    raise ValueError(f"Unknown corruption family: {family}")


def _workflow_verifier_replay(trace: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(trace.get("evidence_records"), list):
        return None
    try:
        parsed_payload = dict(trace["parsed_task"])
        constraints = TaskConstraints(**dict(parsed_payload.pop("constraints", {})))
        parsed_task = ParsedTask(constraints=constraints, **parsed_payload)
        steps = list(trace.get("tool_sequence") or [])
        expected_outputs: list[str] = []
        if any(step.get("tool_name") in _GENERATION_TOOLS for step in steps if isinstance(step, dict)):
            expected_outputs.append("generated_smiles")
        if parsed_task.constraints.require_ranking:
            expected_outputs.append("ranked_candidates")
        workflow = PlannedWorkflow(
            task_id=parsed_task.task_id,
            planner_type=str((trace.get("metadata") or {}).get("planner_baseline") or "trace_replay"),
            selected_tools=list(trace.get("selected_tools") or []),
            tool_sequence=steps,
            expected_outputs=expected_outputs,
        )
        calls = [ToolCallRecord(**item) for item in trace.get("tool_calls") or []]
        candidates = [CandidateRecord(**item) for item in trace.get("final_candidates") or []]
        result = verify_workflow(parsed_task, workflow, calls, candidates)
        return {"success": result.success, "failure_reason": result.failure_reason, "checks": result.checks}
    except (KeyError, TypeError, ValueError):
        return None


def _required_checks(trace: dict[str, Any]) -> list[str]:
    verifier_checks = (trace.get("verifier_result") or {}).get("checks") or {}
    if isinstance(trace.get("evidence_records"), list):
        preferred = ("has_required_evidence", "has_provenance", "no_missing_evidence", "has_tool_success")
        return [name for name in preferred if verifier_checks.get(name) is True]
    parsed = trace.get("parsed_task") or {}
    constraints = parsed.get("constraints") or {}
    objectives = set(parsed.get("objectives") or [])
    tools = {item.get("tool_name") for item in trace.get("tool_sequence") or [] if isinstance(item, dict)}
    required = ["has_tool_success"]
    if parsed.get("task_type") != "docking_evaluation":
        required.extend(("has_valid_smiles", "has_unique_molecules"))
    if constraints.get("require_docking") or "binding" in objectives or "vina" in tools:
        required.append("has_docking_scores")
    if constraints.get("require_synthesizability") or "synthesizability" in objectives:
        required.append("passes_synthesizability")
    if constraints.get("require_toxicity") or "toxicity" in objectives:
        required.append("passes_toxicity")
    if constraints.get("require_ranking"):
        required.append("has_ranked_output")
    return required


def _observed_evidence(trace: dict[str, Any]) -> dict[str, bool]:
    if isinstance(trace.get("evidence_records"), list):
        records = [item for item in trace.get("evidence_records") or [] if isinstance(item, dict)]
        required = [item for item in records if item.get("required")]
        present = [
            item for item in required
            if item.get("value") is not None and not item.get("missing_reason") and item.get("supports") is not False
        ]
        calls = [item for item in trace.get("tool_calls") or [] if isinstance(item, dict)]
        return {
            "has_required_evidence": bool(required) and len(present) == len(required),
            "has_provenance": bool(present) and all(item.get("provenance") or item.get("source") for item in present),
            "no_missing_evidence": len(present) == len(required),
            "has_tool_success": any(item.get("success") and item.get("outputs") is not None for item in calls),
        }
    calls = [item for item in trace.get("tool_calls") or [] if isinstance(item, dict)]
    candidates = [item for item in trace.get("final_candidates") or [] if isinstance(item, dict)]
    generation = any(
        item.get("tool_name") in _GENERATION_TOOLS
        for item in trace.get("tool_sequence") or []
        if isinstance(item, dict)
    )
    completion = [item for item in candidates if not generation or item.get("source_tool") != "input"]
    valid = [item for item in completion if item.get("smiles") and item.get("is_valid")]
    smiles = [item.get("smiles") for item in valid]
    docking = any(item.get("docking_score") is not None for item in candidates) or any(
        item.get("tool_name") == "vina"
        and item.get("success")
        and isinstance(item.get("outputs"), dict)
        and (
            item["outputs"].get("best_docking_score_kcal_mol") is not None
            or item["outputs"].get("best_docking_score") is not None
        )
        for item in calls
    )
    return {
        "has_tool_success": any(item.get("success") and item.get("outputs") is not None for item in calls),
        "has_valid_smiles": bool(valid),
        "has_unique_molecules": bool(valid) and len(smiles) == len(set(smiles)),
        "has_docking_scores": docking,
        "passes_synthesizability": any(
            item.get("scscore") is not None or item.get("sa_score") is not None for item in candidates
        ),
        "passes_toxicity": any(item.get("toxicity_score") is not None for item in candidates),
        "has_ranked_output": bool(candidates) and all(item.get("rank") is not None for item in candidates),
    }


def _order_consistent(expected: list[str], calls: list[dict[str, Any]]) -> bool:
    if not expected:
        return True
    cursor = 0
    for call in calls:
        if cursor < len(expected) and isinstance(call, dict) and call.get("tool_name") == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def _tool_calls_consistent(calls: list[dict[str, Any]]) -> bool:
    if not calls:
        return False
    for call in calls:
        if not isinstance(call, dict):
            return False
        if call.get("success"):
            if call.get("outputs") is None or call.get("error"):
                return False
        elif not call.get("error") and call.get("outputs") is None:
            return False
    return True


def _has_artifact_reference(trace: dict[str, Any]) -> bool:
    if any(isinstance(item, dict) and bool(item.get("artifacts")) for item in trace.get("final_candidates") or []):
        return True
    for call in trace.get("tool_calls") or []:
        outputs = call.get("outputs") if isinstance(call, dict) else None
        if isinstance(outputs, dict) and any(
            ("path" in str(key).lower() or "artifact" in str(key).lower()) and bool(value)
            for key, value in outputs.items()
        ):
            return True
    return False


def _remove_score_values(value: Any) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if any(token in str(key).lower() for token in ("score", "scscore", "toxicity", "pmic")):
                value[key] = None
            else:
                _remove_score_values(value[key])
    elif isinstance(value, list):
        for item in value:
            _remove_score_values(item)


def _first_distinct_pair(calls: list[dict[str, Any]]) -> tuple[int, int] | None:
    for left in range(len(calls)):
        for right in range(left + 1, len(calls)):
            if calls[left].get("tool_name") != calls[right].get("tool_name"):
                return left, right
    return None


def _aggregate(rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["corruption_family"]].append(row)
    output_rows = []
    for family in CORRUPTION_FAMILIES:
        items = grouped.get(family, [])
        eligible = [item for item in items if item["workflow_verifier_eligible"]]
        gate_detected = sum(bool(item["integrity_gate_detected"]) for item in items)
        verifier_detected = sum(bool(item["workflow_verifier_detected"]) for item in eligible)
        output_rows.append(
            {
                "corruption_family": family,
                "applicable_count": len(items),
                "integrity_gate_detected_count": gate_detected,
                "integrity_gate_sensitivity": gate_detected / len(items) if items else None,
                "integrity_gate_false_success_count": len(items) - gate_detected,
                "workflow_verifier_eligible_count": len(eligible),
                "workflow_verifier_detected_count": verifier_detected,
                "workflow_verifier_sensitivity": verifier_detected / len(eligible) if eligible else None,
                "workflow_verifier_false_success_count": len(eligible) - verifier_detected,
                "source": "new_offline_run",
            }
        )
    clean_gate_pass = sum(bool(row["integrity_gate_success"]) for row in clean_rows)
    clean_verifier = [row for row in clean_rows if row["workflow_verifier_eligible"]]
    clean_verifier_pass = sum(bool(row["workflow_verifier_success"]) for row in clean_verifier)
    return {
        "experiment_id": "verifier_evidence_corruption_v1",
        "successful_trace_count": len(clean_rows),
        "corrupted_case_count": len(rows),
        "integrity_gate_specificity": clean_gate_pass / len(clean_rows) if clean_rows else None,
        "workflow_verifier_specificity": clean_verifier_pass / len(clean_verifier) if clean_verifier else None,
        "rows": output_rows,
        "notes": [
            "Corruptions are applied only when the clean successful trace contains the corresponding evidence contract.",
            "Workflow-verifier replay is available for molecular traces; clinical traces are evaluated by the provenance-aware integrity gate.",
            "The integrity gate validates required evidence, provenance, artifacts, call order, and status/output consistency.",
        ],
    }


def _latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated evidence-corruption summary.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Corruption & $N$ & Gate sens. & Workflow sens. & Gate false succ. \\\\",
        "\\midrule",
    ]
    for row in rows:
        workflow = "--" if row["workflow_verifier_sensitivity"] is None else _pct(row["workflow_verifier_sensitivity"])
        lines.append(
            f"{_tex(row['corruption_family'])} & {row['applicable_count']} & "
            f"{_pct(row['integrity_gate_sensitivity'])} & {workflow} & "
            f"{row['integrity_gate_false_success_count']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Sensitivity to controlled corruption of required execution evidence.}",
            "\\label{tab:evidence-corruption}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}\\%"


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


_GENERATION_TOOLS = {
    "rxnflow",
    "reinvent4_denovo",
    "reinvent4_mol2mol",
    "reinvent4_libinvent",
    "scaffold",
    "libinvent",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled verifier evidence-corruption evaluation.")
    parser.add_argument("--traces", required=True, help="JSONL containing selected execution traces.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-failed", action="store_true")
    args = parser.parse_args()
    payload = run_evidence_corruption(
        traces_path=args.traces,
        output_dir=args.output_dir,
        successful_only=not args.include_failed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
