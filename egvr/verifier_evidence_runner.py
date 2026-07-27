"""Build offline verifier-evidence summaries from completed trace artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .task_schema import CandidateRecord

DEFAULT_CROSSDOCKED_TRACE_DIR = "logs/baseline_runs/crossdocked_rxnflow_candidates5_targets15/traces"
DEFAULT_LITPCBA_TRACE_DIR = "logs/baseline_runs/litpcba_vina_prepared_15_elapsed/traces"
DEFAULT_OUTPUT = "logs/baseline_runs/verifier_enhancement_v1/verifier_evidence_summary.json"
DEFAULT_POSEBUSTERS_WORK_DIR = "logs/baseline_runs/verifier_enhancement_v1/posebusters_inputs"
DEFAULT_OBABEL_CONDA_ENV = "biomni_e1"
SA_SCORE_PASS_THRESHOLD = 6.0
SUPPORTED_POSEBUSTERS_EXTENSIONS = {".sdf", ".mol", ".mol2", ".pdb"}
POSEBUSTERS_FAILURE_MODE_COLUMNS = [
    "check_name",
    "category",
    "pose_count",
    "evaluated_count",
    "pass_count",
    "fail_count",
    "missing_count",
    "fail_rate",
    "example_task_ids",
    "interpretation",
]
POSEBUSTERS_CHECK_CATEGORIES = {
    "mol_pred_loaded": "input_loading",
    "mol_cond_loaded": "input_loading",
    "sanitization": "molecule_validity",
    "inchi_convertible": "molecule_validity",
    "all_atoms_connected": "molecule_validity",
    "no_radicals": "molecule_validity",
    "bond_lengths": "geometry",
    "bond_angles": "geometry",
    "internal_steric_clash": "geometry",
    "aromatic_ring_flatness": "geometry",
    "non-aromatic_ring_non-flatness": "geometry",
    "double_bond_flatness": "geometry",
    "internal_energy": "geometry",
    "protein-ligand_maximum_distance": "protein_ligand_geometry",
    "minimum_distance_to_protein": "protein_ligand_geometry",
    "minimum_distance_to_organic_cofactors": "protein_ligand_geometry",
    "minimum_distance_to_inorganic_cofactors": "protein_ligand_geometry",
    "minimum_distance_to_waters": "protein_ligand_geometry",
    "volume_overlap_with_protein": "protein_ligand_geometry",
    "volume_overlap_with_organic_cofactors": "protein_ligand_geometry",
    "volume_overlap_with_inorganic_cofactors": "protein_ligand_geometry",
    "volume_overlap_with_waters": "protein_ligand_geometry",
}


def calculate_sa_scores(smiles_values: list[str]) -> dict[str, Any]:
    """Return RDKit SA_Score rows for valid SMILES strings when available."""

    scorer = _load_sascorer()
    if scorer is None:
        return {
            "success": False,
            "status": "not_available",
            "results": [],
            "error": "RDKit SA_Score contrib scorer is unavailable.",
        }

    from rdkit import Chem

    rows: list[dict[str, Any]] = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append({"smiles": smiles, "sa_score": None, "status": "invalid_smiles"})
            continue
        rows.append(
            {
                "smiles": smiles,
                "sa_score": float(scorer.calculateScore(mol)),
                "status": "ok",
            }
        )
    return {"success": True, "status": "available", "results": rows}


def collect_posebusters_evidence(
    candidates: list[CandidateRecord],
    *,
    trace_records: list[dict[str, Any]] | None = None,
    posebusters_conda_env: str | None = None,
    obabel_conda_env: str = DEFAULT_OBABEL_CONDA_ENV,
    work_dir: str | Path = DEFAULT_POSEBUSTERS_WORK_DIR,
) -> dict[str, Any]:
    """Return PoseBusters availability and pose-artifact coverage without false passes."""

    pose_artifact_count = sum(1 for candidate in candidates if _pose_artifacts(candidate))
    external_env = posebusters_conda_env or os.environ.get("POSEBUSTERS_CONDA_ENV")
    if importlib.util.find_spec("posebusters") is None and not external_env:
        return {
            "success": False,
            "status": "not_available",
            "pose_artifact_count": pose_artifact_count,
            "results": [],
            "error": "PoseBusters is not installed in the current environment.",
        }
    pose_inputs = _trace_posebusters_inputs(trace_records or [])
    if not pose_inputs:
        return {
            "success": False,
            "status": "not_evaluated",
            "pose_artifact_count": pose_artifact_count,
            "results": [],
            "error": "PoseBusters package is available, but no pose/protein input pairs were found.",
        }
    rows = _run_posebusters_cases(
        pose_inputs,
        posebusters_conda_env=external_env,
        obabel_conda_env=obabel_conda_env,
        work_dir=Path(work_dir),
    )
    evidence_rows = [row for row in rows if row.get("posebusters_pass") is not None]
    status = "available" if evidence_rows else "error"
    return {
        "success": bool(evidence_rows),
        "status": status,
        "pose_artifact_count": pose_artifact_count,
        "results": rows,
        "error": None
        if evidence_rows
        else "PoseBusters execution did not produce parseable pass/fail evidence.",
    }


def run_verifier_evidence_summary(
    *,
    input_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT,
    crossdocked_trace_path: str | Path | None = None,
    litpcba_trace_path: str | Path | None = None,
    project_root: str | Path | None = None,
    benchmark_id: str = "verifier_enhancement_v1",
    posebusters_conda_env: str | None = None,
    obabel_conda_env: str = DEFAULT_OBABEL_CONDA_ENV,
    posebusters_work_dir: str | Path = DEFAULT_POSEBUSTERS_WORK_DIR,
) -> dict[str, Any]:
    """Compute verifier-evidence rows from existing real-run traces."""

    root = Path(project_root or Path.cwd())
    input_file = _resolve_optional(input_path, root)
    crossdocked_trace = _resolve_trace_path(
        crossdocked_trace_path,
        root / DEFAULT_CROSSDOCKED_TRACE_DIR,
    )
    litpcba_trace = _resolve_trace_path(
        litpcba_trace_path,
        root / DEFAULT_LITPCBA_TRACE_DIR,
    )
    crossdocked_traces = _load_trace_records(crossdocked_trace)
    litpcba_traces = _load_trace_records(litpcba_trace)
    crossdocked_candidates = _trace_candidates(crossdocked_traces)
    litpcba_candidates = _trace_candidates(litpcba_traces)

    sa_summary = _sa_score_row(crossdocked_candidates, task_count=len(crossdocked_traces))
    pose_payload = collect_posebusters_evidence(
        litpcba_candidates,
        trace_records=litpcba_traces,
        posebusters_conda_env=posebusters_conda_env,
        obabel_conda_env=obabel_conda_env,
        work_dir=root / Path(posebusters_work_dir),
    )
    pose_summary = _posebusters_row_from_payload(
        litpcba_candidates,
        task_count=len(litpcba_traces),
        payload=pose_payload,
    )
    posebusters_failure_modes = _posebusters_failure_mode_rows(pose_payload.get("results", []))
    rows = [sa_summary, pose_summary]
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_id": benchmark_id,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_path": str(input_file) if input_file else None,
        "artifacts": {
            "crossdocked_trace": str(crossdocked_trace),
            "litpcba_trace": str(litpcba_trace),
            "summary_json": str(output_file),
        },
        "environment": {
            "rdkit_available": importlib.util.find_spec("rdkit") is not None,
            "sa_score_available": _load_sascorer() is not None,
            "posebusters_available": importlib.util.find_spec("posebusters") is not None,
            "posebusters_conda_env": posebusters_conda_env or os.environ.get("POSEBUSTERS_CONDA_ENV"),
            "posebusters_cli_evaluated": pose_summary.get("status") == "available",
        },
        "rows": rows,
        "posebusters_failure_modes": {
            "columns": POSEBUSTERS_FAILURE_MODE_COLUMNS,
            "row_count": len(posebusters_failure_modes),
            "rows": posebusters_failure_modes,
        },
        "notes": [
            "SA_Score is computed offline from completed CrossDocked candidate SMILES.",
            "PoseBusters evidence is conservative: unavailable, conversion errors, or failed checks are recorded explicitly.",
            "This summary augments verifier evidence and does not rerun generation, docking, or repair benchmarks.",
        ],
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _sa_score_row(candidates: list[CandidateRecord], *, task_count: int) -> dict[str, Any]:
    smiles_values = [candidate.smiles for candidate in candidates if candidate.smiles and candidate.is_valid]
    score_payload = calculate_sa_scores(smiles_values)
    scores = [
        row.get("sa_score")
        for row in score_payload.get("results", [])
        if isinstance(row, dict) and row.get("sa_score") is not None
    ]
    candidate_count = len(candidates)
    evidence_count = len(scores)
    pass_count = sum(1 for score in scores if float(score) <= SA_SCORE_PASS_THRESHOLD)
    return _ordered_verifier_row(
        {
            "evidence_family": "crossdocked_generation",
            "dataset": "CrossDocked2020",
            "evidence_type": "sa_score",
            "task_count": task_count,
            "candidate_count": candidate_count,
            "evaluable_candidate_count": len(smiles_values),
            "evidence_count": evidence_count,
            "coverage": evidence_count / len(smiles_values) if smiles_values else None,
            "pass_count": pass_count,
            "pass_rate": pass_count / evidence_count if evidence_count else None,
            "best_sa_score": min(scores) if scores else None,
            "mean_sa_score": mean(scores) if scores else None,
            "pose_artifact_count": None,
            "status": score_payload.get("status"),
            "notes": f"SA_Score pass threshold is <= {SA_SCORE_PASS_THRESHOLD}; lower is easier synthesis.",
        }
    )


def _posebusters_row(
    candidates: list[CandidateRecord],
    *,
    task_count: int,
    trace_records: list[dict[str, Any]] | None = None,
    posebusters_conda_env: str | None = None,
    obabel_conda_env: str = DEFAULT_OBABEL_CONDA_ENV,
    work_dir: str | Path = DEFAULT_POSEBUSTERS_WORK_DIR,
) -> dict[str, Any]:
    payload = collect_posebusters_evidence(
        candidates,
        trace_records=trace_records,
        posebusters_conda_env=posebusters_conda_env,
        obabel_conda_env=obabel_conda_env,
        work_dir=work_dir,
    )
    return _posebusters_row_from_payload(candidates, task_count=task_count, payload=payload)


def _posebusters_row_from_payload(
    candidates: list[CandidateRecord],
    *,
    task_count: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    rows = [row for row in payload.get("results", []) if isinstance(row, dict)]
    pass_values = [row.get("posebusters_pass") for row in rows if row.get("posebusters_pass") is not None]
    pass_count = sum(1 for value in pass_values if bool(value))
    pose_artifact_count = int(payload.get("pose_artifact_count") or 0)
    return _ordered_verifier_row(
        {
            "evidence_family": "litpcba_docking",
            "dataset": "LIT-PCBA",
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
            "status": payload.get("status"),
            "notes": payload.get("error"),
        }
    )


VERIFIER_EVIDENCE_COLUMNS = [
    "evidence_family",
    "dataset",
    "evidence_type",
    "task_count",
    "candidate_count",
    "evaluable_candidate_count",
    "evidence_count",
    "coverage",
    "pass_count",
    "pass_rate",
    "best_sa_score",
    "mean_sa_score",
    "pose_artifact_count",
    "status",
    "notes",
]


def _ordered_verifier_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in VERIFIER_EVIDENCE_COLUMNS}


def _load_sascorer():
    try:
        from rdkit import RDConfig

        contrib_path = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(contrib_path) not in sys.path:
            sys.path.append(str(contrib_path))
        import sascorer

        return sascorer
    except Exception:
        return None


def _resolve_optional(path: str | Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def _resolve_trace_path(path: str | Path | None, default_dir: Path) -> Path:
    if path is not None:
        return Path(path)
    traces = sorted(default_dir.glob("*_traces.jsonl"))
    if not traces:
        raise FileNotFoundError(f"No trace JSONL files found in {default_dir}")
    return traces[-1]


def _load_trace_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Trace JSONL not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _trace_candidates(records: list[dict[str, Any]]) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    for record in records:
        for item in record.get("final_candidates", []):
            if isinstance(item, dict):
                candidates.append(CandidateRecord(**item))
    return candidates


def _trace_posebusters_inputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for record in records:
        parsed_task = record.get("parsed_task") or {}
        task_metadata = parsed_task.get("metadata") or {}
        protein_path = (
            task_metadata.get("source_protein_path")
            or parsed_task.get("protein_path")
            or task_metadata.get("protein_path")
        )
        for index, item in enumerate(record.get("final_candidates", [])):
            if not isinstance(item, dict):
                continue
            artifacts = item.get("artifacts") or {}
            pose_path = artifacts.get("minimized_pose_file_path") or artifacts.get("docked_poses_file_path")
            if pose_path and protein_path:
                inputs.append(
                    {
                        "task_id": record.get("task_id"),
                        "candidate_index": index,
                        "pose_path": str(pose_path),
                        "protein_path": str(protein_path),
                    }
                )
    return inputs


def _run_posebusters_cases(
    pose_inputs: list[dict[str, Any]],
    *,
    posebusters_conda_env: str | None,
    obabel_conda_env: str,
    work_dir: Path,
) -> list[dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in pose_inputs:
        prepared = _prepare_posebusters_pose(case, work_dir=work_dir, obabel_conda_env=obabel_conda_env)
        if prepared.get("error"):
            rows.append({**case, "posebusters_pass": None, "posebusters_checks": {}, **prepared})
            continue
        command = _posebusters_command(
            prepared["posebusters_input_path"],
            case["protein_path"],
            posebusters_conda_env=posebusters_conda_env,
        )
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
        except Exception as exc:
            rows.append(
                {
                    **case,
                    **prepared,
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
                    **case,
                    **prepared,
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
                **case,
                **prepared,
                "posebusters_pass": bool(checks) and all(value is True for value in checks.values()),
                "posebusters_checks": checks,
                "status": "ok",
                "stderr": completed.stderr.strip() or None,
            }
        )
    return rows


def _posebusters_failure_mode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_rows = [row for row in rows if isinstance(row, dict) and isinstance(row.get("posebusters_checks"), dict)]
    check_names = sorted({check for row in evidence_rows for check in row["posebusters_checks"]})
    output_rows: list[dict[str, Any]] = []
    pose_count = len(evidence_rows)
    for check_name in check_names:
        values = [row["posebusters_checks"].get(check_name) for row in evidence_rows]
        pass_count = sum(1 for value in values if value is True)
        fail_count = sum(1 for value in values if value is False)
        missing_count = sum(1 for value in values if value is None)
        evaluated_count = pass_count + fail_count
        failing_task_ids = [
            str(row.get("task_id"))
            for row in evidence_rows
            if row.get("posebusters_checks", {}).get(check_name) is False and row.get("task_id")
        ]
        output_rows.append(
            _ordered_posebusters_failure_mode_row(
                {
                    "check_name": check_name,
                    "category": POSEBUSTERS_CHECK_CATEGORIES.get(check_name, "other"),
                    "pose_count": pose_count,
                    "evaluated_count": evaluated_count,
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "missing_count": missing_count,
                    "fail_rate": fail_count / evaluated_count if evaluated_count else None,
                    "example_task_ids": ", ".join(failing_task_ids[:3]),
                    "interpretation": _posebusters_check_interpretation(check_name),
                }
            )
        )
    return sorted(output_rows, key=lambda row: (-(row.get("fail_count") or 0), row.get("check_name") or ""))


def _ordered_posebusters_failure_mode_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in POSEBUSTERS_FAILURE_MODE_COLUMNS}


def _posebusters_check_interpretation(check_name: str) -> str:
    mapping = {
        "sanitization": "The predicted ligand cannot be sanitized by RDKit.",
        "inchi_convertible": "The predicted ligand cannot be converted to InChI.",
        "internal_energy": "The predicted ligand has problematic internal strain or cannot be energy-checked.",
        "minimum_distance_to_protein": "The pose is too close to the protein under PoseBusters thresholds.",
        "volume_overlap_with_protein": "The pose overlaps protein volume under PoseBusters thresholds.",
        "protein-ligand_maximum_distance": "The ligand is outside the expected protein-ligand distance range.",
    }
    return mapping.get(check_name, "PoseBusters structural or protein-ligand sanity check.")


def _prepare_posebusters_pose(
    case: dict[str, Any],
    *,
    work_dir: Path,
    obabel_conda_env: str,
) -> dict[str, Any]:
    pose_path = Path(case["pose_path"])
    if not pose_path.exists():
        return {"status": "error", "error": f"Pose artifact not found: {pose_path}"}
    if pose_path.suffix.lower() in SUPPORTED_POSEBUSTERS_EXTENSIONS:
        return {"posebusters_input_path": str(pose_path), "status": "prepared"}
    if pose_path.suffix.lower() != ".pdbqt":
        return {"status": "error", "error": f"Unsupported pose artifact extension: {pose_path.suffix}"}
    safe_task_id = str(case.get("task_id") or "task").replace("/", "_")
    output_path = work_dir / f"{safe_task_id}_{case.get('candidate_index', 0)}.sdf"
    command = _obabel_command(pose_path, output_path, obabel_conda_env=obabel_conda_env)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode != 0 or not output_path.exists():
        error = (completed.stderr or completed.stdout or "Open Babel conversion failed.").strip()
        return {"status": "error", "error": error, "posebusters_input_path": str(output_path)}
    return {"posebusters_input_path": str(output_path), "status": "converted"}


def _obabel_command(input_path: Path, output_path: Path, *, obabel_conda_env: str) -> list[str]:
    obabel = shutil.which("obabel")
    if obabel:
        return [obabel, str(input_path), "-O", str(output_path)]
    return [
        os.environ.get("CONDA_EXE", "conda"),
        "run",
        "-n",
        obabel_conda_env,
        "obabel",
        str(input_path),
        "-O",
        str(output_path),
    ]


def _posebusters_command(
    pose_path: str,
    protein_path: str,
    *,
    posebusters_conda_env: str | None,
) -> list[str]:
    base = ["bust", pose_path, "-p", protein_path, "--outfmt", "csv"]
    if posebusters_conda_env:
        return [os.environ.get("CONDA_EXE", "conda"), "run", "-n", posebusters_conda_env, *base]
    return base


def _parse_posebusters_csv(output: str) -> list[dict[str, Any]]:
    lines = [line for line in output.splitlines() if line.strip()]
    header_index = next((index for index, line in enumerate(lines) if line.startswith("file,")), None)
    if header_index is None:
        return []
    reader = csv.DictReader(lines[header_index:])
    rows: list[dict[str, Any]] = []
    for row in reader:
        checks = {
            key: _csv_bool(value)
            for key, value in row.items()
            if key not in {"file", "molecule", "position"} and key is not None
        }
        rows.append(
            {
                "file": row.get("file"),
                "molecule": row.get("molecule"),
                "position": row.get("position"),
                "checks": checks,
            }
        )
    return rows


def _csv_bool(value: str | None) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _pose_artifacts(candidate: CandidateRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.artifacts.items()
        if key in {"docked_poses_file_path", "minimized_pose_file_path"} and value
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline verifier-evidence summaries from completed traces.")
    parser.add_argument("--input", help="Optional master baseline JSON used as provenance context.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output summary JSON path.")
    parser.add_argument("--crossdocked-trace", help="Optional CrossDocked trace JSONL path.")
    parser.add_argument("--litpcba-trace", help="Optional LIT-PCBA trace JSONL path.")
    parser.add_argument("--benchmark-id", default="verifier_enhancement_v1", help="Benchmark id stored in the summary.")
    parser.add_argument("--posebusters-conda-env", help="Optional conda env used to run the PoseBusters CLI.")
    parser.add_argument(
        "--obabel-conda-env",
        default=DEFAULT_OBABEL_CONDA_ENV,
        help="Conda env used for Open Babel conversion when obabel is not on PATH.",
    )
    parser.add_argument(
        "--posebusters-work-dir",
        default=DEFAULT_POSEBUSTERS_WORK_DIR,
        help="Directory for converted pose files used by PoseBusters.",
    )
    args = parser.parse_args()
    payload = run_verifier_evidence_summary(
        input_path=args.input,
        output_path=args.output,
        crossdocked_trace_path=args.crossdocked_trace,
        litpcba_trace_path=args.litpcba_trace,
        benchmark_id=args.benchmark_id,
        posebusters_conda_env=args.posebusters_conda_env,
        obabel_conda_env=args.obabel_conda_env,
        posebusters_work_dir=args.posebusters_work_dir,
    )
    print(
        json.dumps(
            {
                "summary_json": str(args.output),
                "row_count": len(payload["rows"]),
                "environment": payload["environment"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
