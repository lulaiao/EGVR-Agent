"""Private clinical prediction wrapper runner.

The runner keeps ClinicalAgent as an external backend. It can write a readiness
or blocked summary without executing any model. Real execution requires a
private adapter command supplied through ``--backend-command``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .biomedical_offline import (
    CLINICAL_PREDICTION_TASK_TYPE,
    EVIDENCE_TYPE_TO_FAMILY,
    _run_offline_tool,
    parse_biomedical_task,
    plan_biomedical_workflow,
    verify_biomedical_workflow,
)
from .biomedical_schema import EvidenceRecord, evidence_records_to_dicts
from .clinical_prediction_backend import build_clinicalagent_readiness, default_backend_root
from .task_schema import PlannedWorkflow, ToolCallRecord


CLINICALAGENT_TOOL_TO_EVIDENCE = {
    "clinicalagent_evidence_retriever": ("external_knowledge_evidence", ("external_knowledge_evidence", "used_evidence")),
    "clinicalagent_enrollment_predictor": ("enrollment_evidence", ("enrollment_evidence", "enrollment_prediction")),
    "clinicalagent_drug_risk_checker": ("drug_risk_evidence", ("drug_risk_evidence", "drug_risk")),
    "clinicalagent_disease_risk_checker": ("disease_risk_evidence", ("disease_risk_evidence", "disease_risk")),
    "clinicalagent_outcome_predictor": ("clinical_outcome_prediction", ("prediction_label", "predicted_label", "outcome_prediction")),
}


def run_clinical_prediction_benchmark(
    benchmark_path: str | Path,
    *,
    backend_root: str | Path | None = None,
    backend_command: str | None = None,
    output_dir: str | Path | None = None,
    trace_log: str | Path | None = None,
) -> dict[str, Any]:
    readiness = build_clinicalagent_readiness(backend_root)
    tasks = _read_jsonl(benchmark_path)
    if not readiness["ready_for_smoke"] or not backend_command:
        return _blocked_summary(
            benchmark_path,
            tasks,
            readiness=readiness,
            blocked_reason="backend_not_ready" if not readiness["ready_for_smoke"] else "missing_backend_command",
        )

    output_root = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="egvr_clinical_"))
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path = Path(trace_log) if trace_log else None
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for task in tasks:
        started = datetime.now(timezone.utc)
        parsed = parse_biomedical_task(
            str(task.get("raw_user_query") or ""),
            task_id=task.get("task_id"),
            metadata={
                **(task.get("metadata") or {}),
                "domain": "clinical_trial",
                "task_type": CLINICAL_PREDICTION_TASK_TYPE,
                "clinical_backend": "clinicalagent",
                "use_clinicalagent_backend": True,
            },
        )
        workflow = plan_biomedical_workflow(parsed)
        backend_result, backend_error = _invoke_backend_command(
            backend_command,
            backend_root=Path(readiness["backend_root"]),
            task=task,
            parsed_task=parsed.to_dict(),
            output_root=output_root,
        )
        tool_calls, evidence = _execute_clinicalagent_wrapper(parsed.metadata, workflow, backend_result, backend_error)
        verifier = verify_biomedical_workflow(parsed, workflow, tool_calls, evidence)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        expected_success = bool(task.get("should_succeed", True))
        missing_evidence = list(verifier.metadata.get("missing_evidence") or [])
        missing_evidence_families = list(verifier.metadata.get("missing_evidence_families") or [])
        failed_checks = list(verifier.metadata.get("failed_checks") or [])
        row = {
            "task_id": parsed.task_id,
            "task_success": verifier.success,
            "expected_success": expected_success,
            "verifier_expectation_match": verifier.success == expected_success,
            "false_success": verifier.success and not expected_success,
            "backend_call_success": backend_error is None,
            "prediction_output_coverage": _has_evidence(evidence, "clinical_outcome_prediction"),
            "evidence_coverage": verifier.metrics.get("evidence_coverage"),
            "provenance_coverage": _provenance_coverage(evidence_records_to_dicts(evidence)),
            "missing_evidence_count": verifier.metrics.get("missing_evidence_count"),
            "missing_evidence": missing_evidence,
            "missing_evidence_families": missing_evidence_families,
            "failed_checks": failed_checks,
            "elapsed_time_sec": elapsed,
            "failure_reason": verifier.failure_reason or backend_error,
        }
        rows.append(row)
        traces.append(
            {
                "task_id": parsed.task_id,
                "raw_user_query": parsed.raw_user_query,
                "parsed_task": parsed.to_dict(),
                "planned_workflow": workflow.to_dict(),
                "tool_calls": [record.to_dict() for record in tool_calls],
                "evidence_records": evidence_records_to_dicts(evidence),
                "verifier_result": verifier.to_dict(),
                "task_success": verifier.success,
                "failure_reason": verifier.failure_reason or backend_error,
            }
        )

    if trace_path:
        with trace_path.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace, sort_keys=True) + "\n")

    return _summarize_rows(benchmark_path, rows, readiness=readiness, backend_command_provided=True)


def write_clinical_prediction_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(summary.get("rows") or [])
    if rows:
        with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _invoke_backend_command(
    backend_command: str,
    *,
    backend_root: Path,
    task: dict[str, Any],
    parsed_task: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    task_id = str(task.get("task_id") or parsed_task.get("task_id"))
    task_json = output_root / f"{task_id}.input.json"
    output_json = output_root / f"{task_id}.output.json"
    task_json.write_text(
        json.dumps({"task": task, "parsed_task": parsed_task}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = backend_command.format(
        backend_root=str(backend_root),
        task_json=str(task_json),
        output_json=str(output_json),
        task_id=task_id,
    )
    try:
        completed = subprocess.run(
            shlex.split(command),
            cwd=str(backend_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as exc:  # pragma: no cover - exercised by integration runs
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return None, (completed.stderr or completed.stdout or f"backend exited {completed.returncode}").strip()
    if not output_json.exists():
        return None, f"backend output missing: {output_json}"
    try:
        return json.loads(output_json.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"backend output invalid JSON: {exc}"


def _execute_clinicalagent_wrapper(
    metadata: dict[str, Any],
    workflow: PlannedWorkflow,
    backend_result: dict[str, Any] | None,
    backend_error: str | None,
) -> tuple[list[ToolCallRecord], list[EvidenceRecord]]:
    records: list[ToolCallRecord] = []
    evidence: list[EvidenceRecord] = []
    for step in workflow.tool_sequence:
        started = _now_iso()
        if step.tool_name == "clinical_trial_metadata_parser":
            produced, missing = _run_offline_tool(
                type("_Task", (), {"metadata": metadata})(),
                step.tool_name,
            )
            success = True
            error = None
        elif backend_error:
            produced, missing, success, error = [], [step.tool_name], False, backend_error
        else:
            produced, missing = _records_for_backend_step(backend_result or {}, step.tool_name)
            success, error = True, None
        evidence.extend(produced)
        records.append(
            ToolCallRecord(
                tool_name=step.tool_name,
                action=step.action,
                inputs={"metadata_digest": _digest(metadata), "backend_output_digest": _digest(backend_result or {})},
                outputs={
                    "evidence_records": evidence_records_to_dicts(produced),
                    "missing_fields": missing,
                    "backend": "clinicalagent",
                },
                success=success,
                error=error,
                started_at=started,
                finished_at=_now_iso(),
                elapsed_time_sec=0.0,
                metadata={"missing_fields": missing, "backend": "clinicalagent"},
            )
        )
    return records, evidence


def _records_for_backend_step(output: dict[str, Any], tool_name: str) -> tuple[list[EvidenceRecord], list[str]]:
    if tool_name == "clinicalagent_outcome_predictor":
        records, missing = _one_record(output, "clinical_outcome_prediction", ("prediction_label", "predicted_label", "outcome_prediction"))
        confidence_record, confidence_missing = _one_record(
            output,
            "clinical_prediction_confidence",
            ("prediction_confidence", "confidence", "score", "prediction_score"),
        )
        return [*records, *confidence_record], [*missing, *confidence_missing]
    evidence_type, keys = CLINICALAGENT_TOOL_TO_EVIDENCE.get(tool_name, (tool_name, (tool_name,)))
    return _one_record(output, evidence_type, keys)


def _one_record(output: dict[str, Any], evidence_type: str, keys: tuple[str, ...]) -> tuple[list[EvidenceRecord], list[str]]:
    value = None
    used_key = None
    for key in keys:
        if output.get(key) not in (None, ""):
            value = output[key]
            used_key = key
            break
    provenance = output.get("provenance") or output.get("sources") or {}
    source = output.get("source") or output.get("backend") or "clinicalagent_backend"
    missing = [] if value not in (None, "") else [evidence_type]
    is_missing = value in (None, "")
    return [
        EvidenceRecord(
            evidence_type=evidence_type,
            value=value,
            evidence_family=EVIDENCE_TYPE_TO_FAMILY.get(evidence_type, "clinical_backend"),
            source=str(source) if value not in (None, "") else None,
            supports=not is_missing,
            required=True,
            confidence=_coerce_float(output.get("confidence") or output.get("prediction_confidence")),
            missing_reason=f"missing_{evidence_type}" if is_missing else None,
            provenance={"backend": "clinicalagent", "field": used_key, "sources": provenance}
            if value not in (None, "")
            else {},
            metadata={"field": used_key, "missing": is_missing},
        )
    ], missing


def _blocked_summary(
    benchmark_path: str | Path,
    tasks: list[dict[str, Any]],
    *,
    readiness: dict[str, Any],
    blocked_reason: str,
) -> dict[str, Any]:
    rows = [
        {
            "task_id": task.get("task_id"),
            "task_success": False,
            "expected_success": bool(task.get("should_succeed", True)),
            "verifier_expectation_match": not bool(task.get("should_succeed", True)),
            "false_success": False,
            "backend_call_success": False,
            "prediction_output_coverage": False,
            "evidence_coverage": None,
            "provenance_coverage": None,
            "missing_evidence_count": None,
            "missing_evidence": [],
            "missing_evidence_families": [],
            "failed_checks": ["backend_ready"],
            "elapsed_time_sec": 0.0,
            "failure_reason": blocked_reason,
        }
        for task in tasks
    ]
    return _summarize_rows(benchmark_path, rows, readiness=readiness, backend_command_provided=False, blocked_reason=blocked_reason)


def _summarize_rows(
    benchmark_path: str | Path,
    rows: list[dict[str, Any]],
    *,
    readiness: dict[str, Any],
    backend_command_provided: bool,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "benchmark_id": Path(benchmark_path).stem,
        "task_count": len(rows),
        "backend_ready": readiness["ready_for_smoke"],
        "backend_command_provided": backend_command_provided,
        "blocked_reason": blocked_reason,
        "missing_backend_items": readiness.get("missing_items", []),
        "backend_call_success_rate": _rate(row["backend_call_success"] for row in rows),
        "prediction_output_coverage": _rate(row["prediction_output_coverage"] for row in rows),
        "mean_evidence_coverage": _mean(row["evidence_coverage"] for row in rows),
        "mean_provenance_coverage": _mean(row["provenance_coverage"] for row in rows),
        "mean_missing_evidence_count": _mean(row["missing_evidence_count"] for row in rows),
        "missing_evidence_family_counts": _count_values(row.get("missing_evidence_families") for row in rows),
        "failed_check_counts": _count_values(row.get("failed_checks") for row in rows),
        "verifier_expectation_match_rate": _rate(row["verifier_expectation_match"] for row in rows),
        "false_success_count": sum(1 for row in rows if row["false_success"]),
        "mean_elapsed_sec": _mean(row["elapsed_time_sec"] for row in rows),
        "rows": rows,
        "readiness": readiness,
        "notes": "Private clinical prediction backend pilot; not a clinical prediction SOTA claim.",
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _has_evidence(records: list[EvidenceRecord], evidence_type: str) -> bool:
    return any(record.evidence_type == evidence_type and record.has_value() and record.has_provenance() for record in records)


def _rate(values) -> float | None:
    flags = list(values)
    if not flags:
        return None
    return sum(1 for value in flags if value) / len(flags)


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in values:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            iterable = item
        else:
            iterable = [item]
        for value in iterable:
            if value in (None, ""):
                continue
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _provenance_coverage(evidence_records: list[dict[str, Any]]) -> float | None:
    if not evidence_records:
        return None
    covered = 0
    for record in evidence_records:
        has_value = record.get("value") not in (None, "")
        has_provenance = bool(record.get("source") or record.get("provenance"))
        if has_value and has_provenance:
            covered += 1
    return covered / len(evidence_records)


def _digest(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def _coerce_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a private ClinicalAgent backend prediction benchmark.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend-root", default=str(default_backend_root()))
    parser.add_argument("--backend-command", default=None)
    parser.add_argument("--trace-log", default=None)
    args = parser.parse_args()
    summary = run_clinical_prediction_benchmark(
        args.benchmark,
        backend_root=args.backend_root,
        backend_command=args.backend_command,
        output_dir=Path(args.output).with_suffix("").parent / "backend_io",
        trace_log=args.trace_log,
    )
    write_clinical_prediction_summary(summary, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"rows", "readiness"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
