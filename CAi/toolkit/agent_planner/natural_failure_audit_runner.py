"""Aggregate naturally observed failures from existing real-run artifacts.

Controlled wrapper-injection benchmarks are excluded by default. The goal is to
show that real molecular toolchains produce auditable failures and evidence
gaps, without treating the audit as a complete production failure distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_ROOT = "logs/baseline_runs"
DEFAULT_OUTPUT = "logs/baseline_runs/natural_failure_audit_v1/natural_failure_audit_summary.json"
DEFAULT_PREP_SUMMARIES = (
    "logs/baseline_runs/pdbbind_receptor_prep_probe_v1/pdbbind_receptor_prep_summary_50.json",
    "logs/baseline_runs/pdbbind_receptor_prep_probe_v1/pdbbindplus_v2020r1_prep_summary_50_fixed.json",
)

NATURAL_FAILURE_AUDIT_COLUMNS = [
    "dataset",
    "benchmark_id",
    "task_type",
    "tool_name",
    "failure_family",
    "failure_count",
    "affected_task_count",
    "example_task_ids",
    "source_kind",
    "source_path",
    "notes",
]


def run_natural_failure_audit(
    *,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    prep_summary_paths: tuple[str | Path, ...] | list[str | Path] = DEFAULT_PREP_SUMMARIES,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    log_path = _resolve_path(log_root, root)
    events: list[dict[str, Any]] = []
    for trace_path in sorted(log_path.glob("**/*traces*.jsonl")):
        if _is_controlled_trace(trace_path):
            continue
        events.extend(_events_from_trace_file(trace_path, root))
    for summary_path in prep_summary_paths:
        path = _resolve_path(summary_path, root)
        if path.exists():
            events.extend(_events_from_receptor_prep_summary(path, root))

    rows = _aggregate_events(events)
    out_path = _resolve_path(output_path, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    tex_path = out_path.with_suffix(".tex")
    _write_csv(rows, csv_path, NATURAL_FAILURE_AUDIT_COLUMNS)
    payload = {
        "benchmark_id": "natural_failure_audit_v1",
        "execution_mode": "real_trace_audit",
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "columns": NATURAL_FAILURE_AUDIT_COLUMNS,
        "event_count": len(events),
        "row_count": len(rows),
        "rows": rows,
        "artifacts": {
            "json": _display_path(out_path, root),
            "csv": _display_path(csv_path, root),
            "tex": _display_path(tex_path, root),
        },
        "notes": [
            "Controlled wrapper-injection trace directories are excluded by default.",
            "The audit summarizes observed failures in existing artifacts; it is not a complete natural failure distribution.",
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tex_path.write_text(render_latex(rows), encoding="utf-8")
    return payload


def _events_from_trace_file(trace_path: Path, project_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for trace in _read_jsonl(trace_path):
        task_id = str(trace.get("task_id") or trace.get("parsed_task", {}).get("task_id") or "--")
        task_type = str(trace.get("parsed_task", {}).get("task_type") or trace.get("task_type") or "--")
        dataset = _dataset_from_trace(trace, trace_path)
        benchmark_id = _benchmark_from_path(trace_path)
        tool_calls = _tool_calls_from_trace(trace)
        for call in tool_calls:
            success = call.get("success")
            errors = _call_errors(call)
            if success is False or errors:
                events.append(
                    {
                        "dataset": dataset,
                        "benchmark_id": benchmark_id,
                        "task_type": task_type,
                        "tool_name": call.get("tool_name") or "--",
                        "failure_family": _classify_failure(" ".join(errors) or json.dumps(call, ensure_ascii=False)),
                        "task_id": task_id,
                        "source_kind": "trace_tool_call",
                        "source_path": _display_path(trace_path, project_root),
                        "notes": _short_text(" ".join(errors)),
                    }
                )
        if trace.get("task_success") is False or trace.get("task_success") == "false":
            verifier = trace.get("verifier_result", {}) if isinstance(trace.get("verifier_result"), dict) else {}
            reason = trace.get("failure_reason") or verifier.get("failure_reason") or verifier.get("status")
            if reason:
                events.append(
                    {
                        "dataset": dataset,
                        "benchmark_id": benchmark_id,
                        "task_type": task_type,
                        "tool_name": "--",
                        "failure_family": _classify_failure(str(reason)),
                        "task_id": task_id,
                        "source_kind": "trace_task_failure",
                        "source_path": _display_path(trace_path, project_root),
                        "notes": _short_text(str(reason)),
                    }
                )
    return events


def _events_from_receptor_prep_summary(path: Path, project_root: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    targets = data.get("ready_targets", data.get("targets", []))
    if not isinstance(targets, list):
        return events
    for item in targets:
        if not isinstance(item, dict) or item.get("prep_success") is True:
            continue
        failure_type = item.get("failure_type") or item.get("failure_family") or item.get("stderr_excerpt") or "receptor_prep_failed"
        events.append(
            {
                "dataset": "PDBbind+" if "pdbbindplus" in path.name.lower() else "PDBbind",
                "benchmark_id": data.get("benchmark_id", path.stem),
                "task_type": "receptor_preparation",
                "tool_name": "mk_prepare_receptor",
                "failure_family": _classify_failure(str(failure_type)),
                "task_id": item.get("pdb_id") or "--",
                "source_kind": "receptor_prep_probe",
                "source_path": _display_path(path, project_root),
                "notes": _short_text(str(failure_type)),
            }
        )
    return events


def _aggregate_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = (
            str(event.get("dataset") or "--"),
            str(event.get("benchmark_id") or "--"),
            str(event.get("task_type") or "--"),
            str(event.get("tool_name") or "--"),
            str(event.get("failure_family") or "--"),
            str(event.get("source_kind") or "--"),
            str(event.get("source_path") or "--"),
        )
        groups[key].append(event)
    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        dataset, benchmark_id, task_type, tool_name, failure_family, source_kind, source_path = key
        task_ids = sorted({str(item.get("task_id") or "--") for item in group})
        notes = Counter(str(item.get("notes") or "") for item in group).most_common(1)
        rows.append(
            {
                "dataset": dataset,
                "benchmark_id": benchmark_id,
                "task_type": task_type,
                "tool_name": tool_name,
                "failure_family": failure_family,
                "failure_count": len(group),
                "affected_task_count": len(task_ids),
                "example_task_ids": ",".join(task_ids[:5]),
                "source_kind": source_kind,
                "source_path": source_path,
                "notes": notes[0][0] if notes else "",
            }
        )
    return rows


def render_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Natural failure audit from existing real-run traces.}",
        r"\label{tab:natural-failure-audit}",
        r"\begin{tabular}{llllr}",
        r"\hline",
        r"Dataset & Benchmark & Tool & Failure & Count \\",
        r"\hline",
    ]
    for row in rows[:12]:
        lines.append(
            " & ".join(
                _escape_latex(value)
                for value in (
                    row.get("dataset"),
                    row.get("benchmark_id"),
                    row.get("tool_name"),
                    row.get("failure_family"),
                    row.get("failure_count"),
                )
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _tool_calls_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("tool_calls", "tool_call_records"):
        value = trace.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _call_errors(call: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("error", "failure_reason", "stderr"):
        value = call.get(key)
        if value:
            errors.append(str(value))
    value = call.get("errors")
    if isinstance(value, list):
        errors.extend(str(item) for item in value if item)
    outputs = call.get("outputs")
    if isinstance(outputs, dict):
        for key in ("error", "failure_reason", "stderr"):
            if outputs.get(key):
                errors.append(str(outputs[key]))
    return errors


def _classify_failure(text: str) -> str:
    lower = text.lower()
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    if "cannot reach tool server" in lower or "connection refused" in lower or "max retries exceeded" in lower:
        return "tool_server_unreachable"
    if "template" in lower or "histidine" in lower or "residue" in lower:
        return "receptor_template_or_residue_issue"
    if "attachment point" in lower or "invalid smiles" in lower or "complete molecule smiles" in lower:
        return "invalid_or_unsupported_molecule_input"
    if "convert" in lower or "conversion" in lower or "pdbqt" in lower or "sdf" in lower or ".mol2" in lower:
        return "format_or_pose_conversion_failure"
    if "missing" in lower or "not found" in lower or "no such file" in lower:
        return "missing_artifact_or_input"
    if "vina" in lower or "dock" in lower:
        return "docking_runtime_failure"
    if "verifier" in lower or "evidence" in lower or "score" in lower:
        return "missing_or_failed_evidence"
    if "error" in lower or "exception" in lower or "traceback" in lower:
        return "runtime_error"
    return "other_failure"


def _dataset_from_trace(trace: dict[str, Any], trace_path: Path) -> str:
    parsed = trace.get("parsed_task", {}) if isinstance(trace.get("parsed_task"), dict) else {}
    metadata = parsed.get("metadata", {}) if isinstance(parsed.get("metadata"), dict) else {}
    if metadata.get("dataset"):
        return str(metadata["dataset"])
    lower_path = str(trace_path).lower()
    if "pdbbindplus" in lower_path:
        return "PDBbind+"
    if "pdbbind" in lower_path:
        return "PDBbind"
    if "litpcba" in lower_path:
        return "LIT-PCBA"
    if "crossdocked" in lower_path:
        return "CrossDocked2020"
    return "--"


def _benchmark_from_path(path: Path) -> str:
    parts = path.parts
    if "baseline_runs" in parts:
        idx = parts.index("baseline_runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def _is_controlled_trace(path: Path) -> bool:
    lower = str(path).lower()
    controlled_tokens = (
        "failure_recovery",
        "ambiguous_evidence",
        "repair_ablation",
        "mock",
        "injected",
        "injection",
    )
    return any(token in lower for token in controlled_tokens)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _short_text(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text[:limit]


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _escape_latex(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate natural failures from existing real traces.")
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    payload = run_natural_failure_audit(
        log_root=args.log_root,
        output_path=args.output,
        project_root=args.project_root,
    )
    print(json.dumps({"row_count": payload["row_count"], "event_count": payload["event_count"]}, indent=2))


if __name__ == "__main__":
    main()
