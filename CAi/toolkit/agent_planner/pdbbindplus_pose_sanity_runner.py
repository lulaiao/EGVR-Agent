"""Run gated PoseBusters sanity checks for completed PDBbind+ docking traces."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .verifier_evidence_runner import (
    DEFAULT_OBABEL_CONDA_ENV,
    POSEBUSTERS_FAILURE_MODE_COLUMNS,
    VERIFIER_EVIDENCE_COLUMNS,
    _load_trace_records,
    _ordered_verifier_row,
    _parse_posebusters_csv,
    _posebusters_command,
    _posebusters_failure_mode_rows,
    _prepare_posebusters_pose,
    _trace_candidates,
    _trace_posebusters_inputs,
)


BENCHMARK_ID = "pdbbindplus_pose_sanity_v1"
DATASET_NAME = "PDBbind+ v2020.R1"
DEFAULT_TRACE = (
    "logs/baseline_runs/pdbbindplus_v2020r1_prepared_pilot_v2/"
    "traces_17/20260604_traces.jsonl"
)
DEFAULT_OUTPUT = "logs/baseline_runs/pdbbindplus_pose_sanity_v1/pdbbindplus_pose_sanity_summary.json"
DEFAULT_WORK_DIR = "logs/baseline_runs/pdbbindplus_pose_sanity_v1/posebusters_inputs"


def run_pdbbindplus_pose_sanity_summary(
    *,
    trace_path: str | Path = DEFAULT_TRACE,
    output_path: str | Path = DEFAULT_OUTPUT,
    project_root: str | Path | None = None,
    benchmark_id: str = BENCHMARK_ID,
    posebusters_conda_env: str | None = None,
    obabel_conda_env: str = DEFAULT_OBABEL_CONDA_ENV,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    min_convertible_fraction: float = 1.0,
) -> dict[str, Any]:
    """Build a conservative PDBbind+ pose-sanity summary from completed traces.

    PoseBusters is executed only after every discovered pose/protein artifact is
    present and the conversion preflight reaches ``min_convertible_fraction``.
    """

    root = Path(project_root or Path.cwd())
    trace_file = _resolve_path(trace_path, root)
    output_file = _resolve_path(output_path, root)
    work_path = _resolve_path(work_dir, root)
    trace_records = _load_trace_records(trace_file)
    candidates = _trace_candidates(trace_records)
    pose_inputs = _trace_posebusters_inputs(trace_records)
    preflight = _preflight_pose_inputs(
        pose_inputs,
        work_dir=work_path,
        obabel_conda_env=obabel_conda_env,
    )
    runtime = {"available": False, "status": "not_evaluated", "error": None}
    pose_results: list[dict[str, Any]] = []
    status = _preflight_status(preflight, min_convertible_fraction=min_convertible_fraction)
    if status == "ready":
        runtime = _posebusters_runtime_available(posebusters_conda_env)
        if runtime["available"]:
            pose_results = _run_prepared_posebusters_cases(
                preflight["rows"],
                posebusters_conda_env=posebusters_conda_env,
            )
            status = _execution_status(pose_results, expected_count=preflight["pose_input_count"])
        else:
            status = "not_available"
    evidence_row = _pdbbindplus_pose_evidence_row(
        candidates=candidates,
        task_count=len(trace_records),
        preflight=preflight,
        pose_results=pose_results,
        status=status,
        runtime=runtime,
    )
    failure_modes = _posebusters_failure_mode_rows(pose_results)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_id": benchmark_id,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "artifacts": {
            "trace": str(trace_file),
            "summary_json": str(output_file),
            "work_dir": str(work_path),
        },
        "environment": {
            "posebusters_available_current_env": importlib.util.find_spec("posebusters") is not None,
            "posebusters_cli_on_path": shutil.which("bust") is not None,
            "posebusters_conda_env": posebusters_conda_env,
            "posebusters_runtime": runtime,
            "obabel_conda_env": obabel_conda_env,
        },
        "preflight": preflight,
        "columns": VERIFIER_EVIDENCE_COLUMNS,
        "rows": [evidence_row],
        "posebusters_failure_modes": {
            "columns": POSEBUSTERS_FAILURE_MODE_COLUMNS,
            "row_count": len(failure_modes),
            "rows": failure_modes,
        },
        "pose_results": pose_results,
        "notes": [
            "PoseBusters is gated by artifact existence and pose conversion preflight.",
            "Unavailable runtime, conversion failures, and failed checks are recorded explicitly.",
            "This summary evaluates docking-pose sanity only; it is not an affinity or activity benchmark.",
        ],
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _preflight_pose_inputs(
    pose_inputs: list[dict[str, Any]],
    *,
    work_dir: Path,
    obabel_conda_env: str,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in pose_inputs:
        pose_path = Path(case["pose_path"])
        protein_path = Path(case["protein_path"])
        row = {
            **case,
            "pose_exists": pose_path.exists(),
            "protein_exists": protein_path.exists(),
            "conversion_status": None,
            "posebusters_input_path": None,
            "error": None,
        }
        if not row["pose_exists"]:
            row["conversion_status"] = "missing_pose"
            row["error"] = f"Pose artifact not found: {pose_path}"
        elif not row["protein_exists"]:
            row["conversion_status"] = "missing_protein"
            row["error"] = f"Protein artifact not found: {protein_path}"
        else:
            prepared = _prepare_posebusters_pose(case, work_dir=work_dir, obabel_conda_env=obabel_conda_env)
            row["conversion_status"] = prepared.get("status")
            row["posebusters_input_path"] = prepared.get("posebusters_input_path")
            row["error"] = prepared.get("error")
        rows.append(row)
    convertible_count = sum(1 for row in rows if _is_convertible(row))
    pose_input_count = len(pose_inputs)
    return {
        "pose_input_count": pose_input_count,
        "existing_pose_count": sum(1 for row in rows if row["pose_exists"]),
        "existing_protein_count": sum(1 for row in rows if row["protein_exists"]),
        "convertible_count": convertible_count,
        "conversion_success_rate": convertible_count / pose_input_count if pose_input_count else None,
        "stable_conversion": pose_input_count > 0 and convertible_count == pose_input_count,
        "rows": rows,
    }


def _is_convertible(row: dict[str, Any]) -> bool:
    return (
        row.get("pose_exists") is True
        and row.get("protein_exists") is True
        and row.get("conversion_status") in {"prepared", "converted"}
        and row.get("posebusters_input_path")
        and not row.get("error")
    )


def _preflight_status(preflight: dict[str, Any], *, min_convertible_fraction: float) -> str:
    pose_input_count = int(preflight.get("pose_input_count") or 0)
    if pose_input_count == 0:
        return "not_evaluated"
    if preflight.get("existing_pose_count") != pose_input_count or preflight.get("existing_protein_count") != pose_input_count:
        return "artifact_missing"
    conversion_rate = preflight.get("conversion_success_rate")
    if conversion_rate is None or float(conversion_rate) < min_convertible_fraction:
        return "conversion_not_stable"
    return "ready"


def _posebusters_runtime_available(posebusters_conda_env: str | None) -> dict[str, Any]:
    if posebusters_conda_env:
        command = [
            os.environ.get("CONDA_EXE", "conda"),
            "run",
            "-n",
            posebusters_conda_env,
            "python",
            "-c",
            "import posebusters",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        except Exception as exc:
            return {"available": False, "status": "not_available", "error": str(exc)}
        if completed.returncode == 0:
            return {"available": True, "status": "available", "error": None}
        return {
            "available": False,
            "status": "not_available",
            "error": (completed.stderr or completed.stdout or "PoseBusters conda env check failed.").strip(),
        }
    if importlib.util.find_spec("posebusters") is not None or shutil.which("bust"):
        return {"available": True, "status": "available", "error": None}
    return {"available": False, "status": "not_available", "error": "PoseBusters runtime is unavailable."}


def _run_prepared_posebusters_cases(
    preflight_rows: list[dict[str, Any]],
    *,
    posebusters_conda_env: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in preflight_rows:
        if not _is_convertible(row):
            rows.append({**row, "posebusters_pass": None, "posebusters_checks": {}, "status": "preflight_failed"})
            continue
        command = _posebusters_command(
            str(row["posebusters_input_path"]),
            str(row["protein_path"]),
            posebusters_conda_env=posebusters_conda_env,
        )
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
        except Exception as exc:
            rows.append(
                {
                    **row,
                    "posebusters_pass": None,
                    "posebusters_checks": {},
                    "status": "error",
                    "error": f"PoseBusters execution failed: {exc}",
                }
            )
            continue
        parsed_rows = _parse_posebusters_csv(completed.stdout)
        if not parsed_rows:
            rows.append(
                {
                    **row,
                    "posebusters_pass": None,
                    "posebusters_checks": {},
                    "status": "error",
                    "error": (completed.stderr or completed.stdout or "PoseBusters produced no CSV rows.").strip(),
                }
            )
            continue
        checks = parsed_rows[0]["checks"]
        rows.append(
            {
                **row,
                "posebusters_pass": bool(checks) and all(value is True for value in checks.values()),
                "posebusters_checks": checks,
                "status": "ok",
                "stderr": completed.stderr.strip() or None,
            }
        )
    return rows


def _execution_status(pose_results: list[dict[str, Any]], *, expected_count: int) -> str:
    evidence_count = sum(1 for row in pose_results if row.get("posebusters_pass") is not None)
    if evidence_count == expected_count and expected_count > 0:
        return "available"
    if evidence_count > 0:
        return "partial"
    return "error"


def _pdbbindplus_pose_evidence_row(
    *,
    candidates: list[Any],
    task_count: int,
    preflight: dict[str, Any],
    pose_results: list[dict[str, Any]],
    status: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    pass_values = [row.get("posebusters_pass") for row in pose_results if row.get("posebusters_pass") is not None]
    pass_count = sum(1 for value in pass_values if bool(value))
    pose_artifact_count = int(preflight.get("pose_input_count") or 0)
    if status == "ready":
        status = "not_evaluated"
    return _ordered_verifier_row(
        {
            "evidence_family": "pdbbindplus_docking",
            "dataset": DATASET_NAME,
            "evidence_type": "posebusters",
            "task_count": task_count,
            "candidate_count": len(candidates),
            "evaluable_candidate_count": pose_artifact_count,
            "evidence_count": len(pass_values),
            "coverage": len(pass_values) / pose_artifact_count if pose_artifact_count else None,
            "pass_count": pass_count,
            "pass_rate": pass_count / len(pass_values) if pass_values else None,
            "best_sa_score": None,
            "mean_sa_score": None,
            "pose_artifact_count": pose_artifact_count,
            "status": status,
            "notes": _status_note(status, preflight=preflight, runtime=runtime),
        }
    )


def _status_note(status: str, *, preflight: dict[str, Any], runtime: dict[str, Any]) -> str:
    if status == "available":
        return "Pose artifacts were stable and PoseBusters produced parseable evidence for all PDBbind+ poses."
    if status == "partial":
        return "Pose artifacts were stable, but PoseBusters produced evidence for only part of the PDBbind+ poses."
    if status == "not_available":
        return str(runtime.get("error") or "PoseBusters runtime is unavailable.")
    if status == "artifact_missing":
        return "At least one pose or protein artifact is missing; PoseBusters was not executed."
    if status == "conversion_not_stable":
        return (
            f"Only {preflight.get('convertible_count')}/{preflight.get('pose_input_count')} "
            "pose artifacts converted; PoseBusters was not executed."
        )
    if status == "error":
        return "Pose artifacts converted, but PoseBusters produced no complete parseable evidence."
    return "Pose sanity was not evaluated because no pose/protein input pairs were found."


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gated PoseBusters sanity checks for PDBbind+ docking traces.")
    parser.add_argument("--trace", default=DEFAULT_TRACE, help="Completed PDBbind+ trace JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output summary JSON path.")
    parser.add_argument("--project-root", default=None, help="Project root used for relative paths.")
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID, help="Benchmark id recorded in the output summary.")
    parser.add_argument("--posebusters-conda-env", help="Optional conda env used to run the PoseBusters CLI.")
    parser.add_argument(
        "--obabel-conda-env",
        default=DEFAULT_OBABEL_CONDA_ENV,
        help="Conda env used for Open Babel conversion when obabel is not on PATH.",
    )
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="Directory for converted pose files.")
    parser.add_argument(
        "--min-convertible-fraction",
        type=float,
        default=1.0,
        help="Minimum pose conversion fraction required before executing PoseBusters.",
    )
    args = parser.parse_args()
    payload = run_pdbbindplus_pose_sanity_summary(
        trace_path=args.trace,
        output_path=args.output,
        project_root=args.project_root,
        benchmark_id=args.benchmark_id,
        posebusters_conda_env=args.posebusters_conda_env,
        obabel_conda_env=args.obabel_conda_env,
        work_dir=args.work_dir,
        min_convertible_fraction=args.min_convertible_fraction,
    )
    row = payload["rows"][0]
    print(
        json.dumps(
            {
                "summary_json": str(args.output),
                "status": row.get("status"),
                "coverage": row.get("coverage"),
                "pass_rate": row.get("pass_rate"),
                "preflight": payload.get("preflight", {}),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
