"""Deterministic, stratified consistency audit for JSONL execution traces."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


AUDIT_COLUMNS = [
    "stratum",
    "source_path",
    "trace_id",
    "task_id",
    "planner_baseline",
    "schema_complete",
    "status_consistent",
    "required_evidence_consistent",
    "failure_reason_explicit",
    "repair_metadata_consistent",
    "artifact_reference_present",
    "audit_pass",
    "violations",
]

DUPLICATE_COLUMNS = [
    "stratum",
    "source_path",
    "task_id",
    "planner_baseline",
    "record_count",
    "kept_trace_id",
    "kept_timestamp",
]

REQUIRED_TOP_LEVEL = {
    "trace_id",
    "task_id",
    "timestamp",
    "parsed_task",
    "tool_sequence",
    "tool_calls",
    "final_candidates",
    "verifier_result",
    "task_success",
    "metadata",
}
CLINICAL_REQUIRED_TOP_LEVEL = {
    "task_id",
    "parsed_task",
    "planned_workflow",
    "tool_calls",
    "evidence_records",
    "verifier_result",
    "task_success",
}


def run_trace_consistency_audit(
    *,
    sources: dict[str, list[str | Path]],
    quotas: dict[str, int],
    output_dir: str | Path,
    project_root: str | Path = ".",
    random_seed: int = 20260707,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    audit_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    stratum_summary: list[dict[str, Any]] = []
    rng = random.Random(random_seed)

    for stratum in sorted(sources):
        records: list[tuple[Path, dict[str, Any]]] = []
        for source in sources[stratum]:
            for path in _trace_files(_resolve(source, root)):
                deduplicated, duplicates = _load_and_deduplicate(path, stratum=stratum, root=root)
                records.extend((path, row) for row in deduplicated)
                duplicate_rows.extend(duplicates)
        records.sort(key=lambda pair: (str(pair[0]), str(pair[1].get("task_id")), str(pair[1].get("trace_id"))))
        quota = max(0, int(quotas.get(stratum, len(records))))
        sampled = records if len(records) <= quota else rng.sample(records, quota)
        sampled.sort(key=lambda pair: (str(pair[0]), str(pair[1].get("task_id")), str(pair[1].get("trace_id"))))
        rows = [_audit_trace(row, stratum=stratum, source_path=_display_path(path, root)) for path, row in sampled]
        audit_rows.extend(rows)
        stratum_summary.append(
            {
                "stratum": stratum,
                "available_trace_count": len(records),
                "requested_sample_count": quota,
                "audited_trace_count": len(rows),
                "sample_shortfall": max(0, quota - len(rows)),
                "audit_pass_count": sum(1 for row in rows if row["audit_pass"]),
                "audit_pass_rate": _rate(sum(1 for row in rows if row["audit_pass"]), len(rows)),
                "artifact_reference_coverage": _rate(
                    sum(1 for row in rows if row["artifact_reference_present"]), len(rows)
                ),
            }
        )

    target_dir = _resolve(output_dir, root)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit_id": "trace_consistency_audit_v1",
        "random_seed": random_seed,
        "audited_trace_count": len(audit_rows),
        "duplicate_group_count": len(duplicate_rows),
        "audit_pass_count": sum(1 for row in audit_rows if row["audit_pass"]),
        "audit_pass_rate": _rate(sum(1 for row in audit_rows if row["audit_pass"]), len(audit_rows)),
        "strata": stratum_summary,
        "rows": audit_rows,
        "notes": [
            "Duplicates are defined within one JSONL file by (task_id, planner_baseline).",
            "The latest timestamp is audited; duplicate groups remain visible in a separate report.",
            "Artifact-reference coverage records traceability, not artifact quality or current path availability.",
        ],
    }
    (target_dir / "trace_consistency_audit_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(target_dir / "trace_consistency_audit_table.csv", audit_rows, AUDIT_COLUMNS)
    (target_dir / "trace_consistency_audit_table.tex").write_text(
        _summary_latex(stratum_summary), encoding="utf-8"
    )
    duplicate_payload = {"duplicate_group_count": len(duplicate_rows), "rows": duplicate_rows}
    (target_dir / "trace_duplicate_report.json").write_text(
        json.dumps(duplicate_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(target_dir / "trace_duplicate_report.csv", duplicate_rows, DUPLICATE_COLUMNS)
    return payload


def _load_and_deduplicate(
    path: Path,
    *,
    stratum: str,
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            row = {
                "trace_id": f"invalid_json_line_{line_no}",
                "task_id": f"invalid_json_line_{line_no}",
                "timestamp": "",
                "_json_error": str(exc),
            }
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        planned = row.get("planned_workflow") if isinstance(row.get("planned_workflow"), dict) else {}
        key = (
            str(row.get("task_id")),
            str(metadata.get("planner_baseline") or planned.get("planner_type") or "unknown"),
        )
        grouped[key].append(row)

    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for (task_id, planner), rows in grouped.items():
        rows.sort(key=lambda row: _timestamp_sort_key(row.get("timestamp")))
        latest = rows[-1]
        kept.append(latest)
        if len(rows) > 1:
            duplicates.append(
                {
                    "stratum": stratum,
                    "source_path": _display_path(path, root),
                    "task_id": task_id,
                    "planner_baseline": planner,
                    "record_count": len(rows),
                    "kept_trace_id": latest.get("trace_id"),
                    "kept_timestamp": latest.get("timestamp"),
                }
            )
    return kept, duplicates


def _audit_trace(trace: dict[str, Any], *, stratum: str, source_path: str) -> dict[str, Any]:
    violations: list[str] = []
    is_evidence_trace = isinstance(trace.get("evidence_records"), list)
    required_schema = CLINICAL_REQUIRED_TOP_LEVEL if is_evidence_trace else REQUIRED_TOP_LEVEL
    schema_complete = required_schema.issubset(trace) and not trace.get("_json_error")
    if not schema_complete:
        violations.append("missing_or_invalid_top_level_schema")

    verifier = trace.get("verifier_result") if isinstance(trace.get("verifier_result"), dict) else {}
    verifier_success = verifier.get("success")
    status_consistent = isinstance(verifier_success, bool) and trace.get("task_success") is verifier_success
    if trace.get("failure_reason") != verifier.get("failure_reason"):
        status_consistent = False
    if not status_consistent:
        violations.append("task_verifier_status_mismatch")

    required = _required_evidence_checks(trace) if is_evidence_trace else _required_checks(trace)
    observed = _observed_domain_evidence(trace) if is_evidence_trace else _observed_evidence(trace)
    verifier_checks = verifier.get("checks") if isinstance(verifier.get("checks"), dict) else {}
    evidence_mismatches = [
        check
        for check in required
        if bool(verifier_checks.get(check)) != bool(observed.get(check))
    ]
    success_with_missing = bool(trace.get("task_success")) and any(not observed.get(check) for check in required)
    required_evidence_consistent = not evidence_mismatches and not success_with_missing
    if evidence_mismatches:
        violations.append("evidence_mismatch:" + ";".join(sorted(evidence_mismatches)))
    if success_with_missing:
        violations.append("success_with_missing_required_evidence")

    failure_reason_explicit = bool(trace.get("task_success")) or bool(trace.get("failure_reason"))
    if not failure_reason_explicit:
        violations.append("failed_without_explicit_reason")

    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    planned = trace.get("planned_workflow") if isinstance(trace.get("planned_workflow"), dict) else {}
    repair_executed = bool(metadata.get("repair_executed"))
    repair_plan_present = isinstance(metadata.get("repair_plan"), dict)
    initial_count = metadata.get("initial_tool_call_count")
    repair_count = metadata.get("repair_tool_call_count")
    repair_metadata_consistent = True
    if repair_executed and not repair_plan_present:
        repair_metadata_consistent = False
    if repair_executed and isinstance(repair_count, int) and repair_count <= 0:
        repair_metadata_consistent = False
    if repair_executed and isinstance(initial_count, int) and len(trace.get("tool_calls") or []) <= initial_count:
        repair_metadata_consistent = False
    if not repair_metadata_consistent:
        violations.append("repair_metadata_inconsistent")

    artifact_reference_present = any(
        isinstance(candidate, dict) and bool(candidate.get("artifacts"))
        for candidate in trace.get("final_candidates") or []
    ) or any(
        isinstance(call, dict)
        and isinstance(call.get("outputs"), dict)
        and any("path" in str(key).lower() or "artifact" in str(key).lower() for key in call["outputs"])
        for call in trace.get("tool_calls") or []
    ) or any(
        isinstance(record, dict) and bool(record.get("provenance"))
        for record in trace.get("evidence_records") or []
    )
    return {
        "stratum": stratum,
        "source_path": source_path,
        "trace_id": trace.get("trace_id") or f"domain_trace:{trace.get('task_id')}",
        "task_id": trace.get("task_id"),
        "planner_baseline": metadata.get("planner_baseline") or planned.get("planner_type"),
        "schema_complete": schema_complete,
        "status_consistent": status_consistent,
        "required_evidence_consistent": required_evidence_consistent,
        "failure_reason_explicit": failure_reason_explicit,
        "repair_metadata_consistent": repair_metadata_consistent,
        "artifact_reference_present": artifact_reference_present,
        "audit_pass": all(
            [
                schema_complete,
                status_consistent,
                required_evidence_consistent,
                failure_reason_explicit,
                repair_metadata_consistent,
            ]
        ),
        "violations": " | ".join(violations),
    }


def _required_checks(trace: dict[str, Any]) -> list[str]:
    parsed = trace.get("parsed_task") if isinstance(trace.get("parsed_task"), dict) else {}
    constraints = parsed.get("constraints") if isinstance(parsed.get("constraints"), dict) else {}
    objectives = set(parsed.get("objectives") or [])
    task_type = parsed.get("task_type")
    steps = trace.get("tool_sequence") or []
    tool_names = {step.get("tool_name") for step in steps if isinstance(step, dict)}
    required = ["has_tool_success"]
    if task_type != "docking_evaluation":
        required.extend(["has_valid_smiles", "has_unique_molecules"])
    if constraints.get("require_docking") or "binding" in objectives or "vina" in tool_names:
        required.append("has_docking_scores")
    if constraints.get("require_synthesizability") or "synthesizability" in objectives:
        required.append("passes_synthesizability")
    if constraints.get("require_toxicity") or "toxicity" in objectives:
        required.append("passes_toxicity")
    if constraints.get("require_ranking"):
        required.append("has_ranked_output")
    return required


def _observed_evidence(trace: dict[str, Any]) -> dict[str, bool]:
    calls = [call for call in trace.get("tool_calls") or [] if isinstance(call, dict)]
    candidates = [item for item in trace.get("final_candidates") or [] if isinstance(item, dict)]
    generation_tools = {
        "rxnflow",
        "reinvent4_denovo",
        "reinvent4_mol2mol",
        "reinvent4_libinvent",
        "scaffold",
        "libinvent",
    }
    has_generation_step = any(
        isinstance(step, dict) and step.get("tool_name") in generation_tools
        for step in trace.get("tool_sequence") or []
    )
    completion_candidates = [
        item for item in candidates if not has_generation_step or item.get("source_tool") != "input"
    ]
    valid = [item for item in completion_candidates if item.get("smiles") and item.get("is_valid")]
    smiles = [item.get("smiles") for item in valid]
    docking = any(item.get("docking_score") is not None for item in candidates) or any(
        call.get("tool_name") == "vina"
        and call.get("success")
        and isinstance(call.get("outputs"), dict)
        and (
            call["outputs"].get("best_docking_score_kcal_mol") is not None
            or call["outputs"].get("best_docking_score") is not None
        )
        for call in calls
    )
    return {
        "has_tool_success": any(call.get("success") and not (call.get("metadata") or {}).get("skipped") for call in calls),
        "has_valid_smiles": bool(valid),
        "has_unique_molecules": bool(valid) and len(smiles) == len(set(smiles)),
        "has_docking_scores": docking,
        "passes_synthesizability": any(
            item.get("scscore") is not None or item.get("sa_score") is not None for item in candidates
        ),
        "passes_toxicity": any(item.get("toxicity_score") is not None for item in candidates),
        "has_ranked_output": bool(candidates) and all(item.get("rank") is not None for item in candidates),
    }


def _required_evidence_checks(trace: dict[str, Any]) -> list[str]:
    checks = trace.get("verifier_result", {}).get("checks", {})
    preferred = ["has_required_evidence", "has_provenance", "no_missing_evidence", "has_tool_success"]
    return [name for name in preferred if name in checks]


def _observed_domain_evidence(trace: dict[str, Any]) -> dict[str, bool]:
    records = [item for item in trace.get("evidence_records") or [] if isinstance(item, dict)]
    required = [item for item in records if item.get("required")]
    missing = [
        item
        for item in required
        if item.get("missing_reason") or item.get("value") is None or item.get("supports") is False
    ]
    present_required = [item for item in required if item not in missing]
    provenance_complete = bool(present_required) and all(
        bool(item.get("provenance") or item.get("source")) for item in present_required
    )
    calls = [item for item in trace.get("tool_calls") or [] if isinstance(item, dict)]
    return {
        "has_required_evidence": bool(required) and not missing,
        "has_provenance": provenance_complete,
        "no_missing_evidence": not missing,
        "has_tool_success": any(item.get("success") for item in calls),
    }


def _timestamp_sort_key(value: Any) -> tuple[int, str]:
    if not value:
        return (0, "")
    try:
        return (1, datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat())
    except ValueError:
        return (0, str(value))


def _trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jsonl")) if path.is_dir() else []


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated trace consistency audit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Stratum & Available & Audited & Pass rate & Artifact refs \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['stratum'])} & {row['available_trace_count']} & {row['audited_trace_count']} & "
            f"{_pct(row['audit_pass_rate'])} & {_pct(row['artifact_reference_coverage'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Stratified consistency audit of structured execution traces.}",
            "\\label{tab:trace-consistency-audit}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}\\%"


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def _parse_assignments(values: Iterable[str], *, integer: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = defaultdict(list) if not integer else {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE, got: {value}")
        name, raw = value.split("=", 1)
        if integer:
            result[name] = int(raw)
        else:
            result[name].append(raw)
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a stratified consistency audit over JSONL traces.")
    parser.add_argument("--source", action="append", required=True, help="STRATUM=PATH; repeatable.")
    parser.add_argument("--quota", action="append", required=True, help="STRATUM=COUNT; repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--random-seed", type=int, default=20260707)
    args = parser.parse_args()
    payload = run_trace_consistency_audit(
        sources=_parse_assignments(args.source),
        quotas=_parse_assignments(args.quota, integer=True),
        output_dir=args.output_dir,
        project_root=args.project_root,
        random_seed=args.random_seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
