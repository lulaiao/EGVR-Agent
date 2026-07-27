"""Resolve simple PDBbind receptor template ambiguities and prepare PDBQT files."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pdbbind_pilot_generator import (
    PROTEIN_SUFFIXES,
    _first_existing_with_suffix,
    discover_ready_target_dirs,
)
from .pdbbind_receptor_prep_probe import classify_receptor_prep_failure


DEFAULT_OUTPUT = "logs/baseline_runs/pdbbind_receptor_template_prep_v1/pdbbind_receptor_template_prep_summary.json"
DEFAULT_PREP_DIR = "logs/baseline_runs/pdbbind_receptor_template_prep_v1/prepared_receptors"

CommandRunner = Callable[[Sequence[str], Path, int], subprocess.CompletedProcess[str]]


def run_pdbbind_receptor_template_prep(
    *,
    pdbbind_root: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    prepared_dir: str | Path = DEFAULT_PREP_DIR,
    limit: int | None = None,
    conda_env: str = "vina",
    timeout_sec: int = 90,
    max_template_attempts: int = 12,
    template_policy: str = "first",
    target_ids: list[str] | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Try to prepare receptor PDBQT files by iteratively assigning residue templates."""

    root = Path(pdbbind_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"PDBbind root not found: {root}")
    prep_dir = Path(prepared_dir).expanduser().resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path)
    runner = command_runner or _run_command
    target_dirs = discover_ready_target_dirs(root)
    if target_ids:
        wanted = {target_id.lower() for target_id in target_ids}
        target_dirs = [target_dir for target_dir in target_dirs if target_dir.name.lower() in wanted]
    if limit is not None:
        target_dirs = target_dirs[: max(limit, 0)]

    records = [
        _prepare_target(
            target_dir,
            prep_dir=prep_dir,
            conda_env=conda_env,
            timeout_sec=timeout_sec,
            max_template_attempts=max_template_attempts,
            template_policy=template_policy,
            runner=runner,
        )
        for target_dir in target_dirs
    ]
    payload = {
        "benchmark_id": "pdbbind_receptor_template_prep_v1",
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "pdbbind_root": str(root),
        "prepared_dir": str(prep_dir),
        "conda_env": conda_env,
        "timeout_sec": timeout_sec,
        "max_template_attempts": max_template_attempts,
        "template_policy": template_policy,
        "target_ids": [target_id.lower() for target_id in target_ids] if target_ids else [],
        "summary": _summarize(records),
        "ready_targets": [record for record in records if record["prep_success"]],
        "unresolved_targets": [record for record in records if not record["prep_success"]],
        "records": records,
        "notes": [
            "This step generates receptor PDBQT files only; it does not run docking.",
            "Template assignments are heuristic and auditable. They should be reported as preparation metadata, not chemistry ground truth.",
        ],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _prepare_target(
    target_dir: Path,
    *,
    prep_dir: Path,
    conda_env: str,
    timeout_sec: int,
    max_template_attempts: int,
    template_policy: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    pdb_id = target_dir.name.lower()
    protein_path = _first_existing_with_suffix(target_dir, pdb_id, PROTEIN_SUFFIXES)
    record: dict[str, Any] = {
        "pdb_id": pdb_id,
        "target_dir": str(target_dir),
        "protein_path": str(protein_path) if protein_path else None,
        "prep_success": False,
        "prepared_receptor_pdbqt_path": None,
        "template_assignments": {},
        "template_policy": template_policy,
        "attempt_count": 0,
        "attempts": [],
        "final_failure_type": None,
        "final_failure_family": None,
        "unresolved_residue_key": None,
        "unresolved_template_candidates": [],
    }
    if protein_path is None:
        record.update({"final_failure_type": "missing_input", "final_failure_family": "input"})
        return record
    if protein_path.suffix.lower() == ".pdbqt":
        record.update(
            {
                "prep_success": True,
                "prepared_receptor_pdbqt_path": str(protein_path),
                "template_assignments": {},
            }
        )
        return record

    output_basename = prep_dir / pdb_id
    assignments: OrderedDict[str, str] = OrderedDict()
    tried_by_residue: dict[str, list[str]] = {}
    last_failure: dict[str, Any] = {}
    for _ in range(max(max_template_attempts, 1)):
        command = _build_command(conda_env, protein_path, output_basename, assignments)
        attempt = _run_attempt(command, cwd=target_dir, timeout_sec=timeout_sec, runner=runner)
        record["attempts"].append(attempt)
        if attempt["success"]:
            record.update(
                {
                    "prep_success": True,
                    "prepared_receptor_pdbqt_path": attempt["prepared_receptor_pdbqt_path"],
                    "template_assignments": dict(assignments),
                    "attempt_count": len(record["attempts"]),
                }
            )
            return record

        last_failure = attempt["failure"]
        residue_key = last_failure.get("residue_key")
        candidates = last_failure.get("template_candidates", [])
        if last_failure.get("failure_type") != "histidine_template_ambiguity" or not residue_key or not candidates:
            break
        if len(record["attempts"]) >= max(max_template_attempts, 1):
            break

        tried = tried_by_residue.setdefault(residue_key, [])
        next_template = _choose_next_template(candidates, tried=tried, policy=template_policy)
        if not next_template:
            break
        assignments[residue_key] = next_template
        tried.append(next_template)

    record.update(
        {
            "template_assignments": dict(assignments),
            "attempt_count": len(record["attempts"]),
            "final_failure_type": last_failure.get("failure_type", "unknown_failure"),
            "final_failure_family": last_failure.get("failure_family", "unknown"),
            "unresolved_residue_key": last_failure.get("residue_key"),
            "unresolved_template_candidates": last_failure.get("template_candidates", []),
        }
    )
    return record


def _run_attempt(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_sec: int,
    runner: CommandRunner,
) -> dict[str, Any]:
    output_basename = Path(command[command.index("-o") + 1])
    expected_outputs = [output_basename.with_suffix(".pdbqt"), Path(str(output_basename) + "_rigid.pdbqt")]
    started = time.monotonic()
    try:
        proc = runner(command, cwd, timeout_sec)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        failure = classify_receptor_prep_failure(f"Timed out after {timeout_sec}s\n{exc.stderr or ''}", exc.stdout or "")
        return {
            "command": list(command),
            "elapsed_sec": round(elapsed, 3),
            "returncode": None,
            "success": False,
            "prepared_receptor_pdbqt_path": None,
            "failure": failure,
            "stdout_excerpt": _excerpt(str(exc.stdout or "")),
            "stderr_excerpt": _excerpt(str(exc.stderr or "")),
        }

    elapsed = time.monotonic() - started
    prepared = next((path for path in expected_outputs if path.exists() and path.stat().st_size > 0), None)
    success = proc.returncode == 0 and prepared is not None
    failure = {} if success else classify_receptor_prep_failure(proc.stderr, proc.stdout)
    return {
        "command": list(command),
        "elapsed_sec": round(elapsed, 3),
        "returncode": proc.returncode,
        "success": success,
        "prepared_receptor_pdbqt_path": str(prepared) if prepared else None,
        "failure": failure,
        "stdout_excerpt": _excerpt(proc.stdout),
        "stderr_excerpt": _excerpt(proc.stderr),
    }


def _build_command(
    conda_env: str,
    protein_path: Path,
    output_basename: Path,
    assignments: OrderedDict[str, str],
) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "mk_prepare_receptor.py",
        "-i",
        str(protein_path),
        "-o",
        str(output_basename),
        "-p",
    ]
    if assignments:
        assignment_text = ",".join(f"{residue}={template}" for residue, template in assignments.items())
        command.extend(["-n", assignment_text])
    return command


def _choose_next_template(candidates: list[str], *, tried: list[str], policy: str) -> str | None:
    ordered = list(candidates)
    if policy == "prefer_hid":
        ordered = sorted(candidates, key=lambda item: 0 if item.endswith("HID") else 1)
    elif policy == "prefer_hie":
        ordered = sorted(candidates, key=lambda item: 0 if item.endswith("HIE") else 1)
    elif policy != "first":
        raise ValueError("template_policy must be one of: first, prefer_hid, prefer_hie")
    return next((candidate for candidate in ordered if candidate not in tried), None)


def _run_command(command: Sequence[str], cwd: Path, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    success_count = sum(1 for record in records if record["prep_success"])
    failure_counts: dict[str, int] = {}
    for record in records:
        failure_type = record["final_failure_type"]
        if failure_type:
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
    return {
        "total": total,
        "prep_success_count": success_count,
        "prep_success_rate": success_count / total if total else 0.0,
        "prep_failure_count": total - success_count,
        "failure_counts": failure_counts,
        "stable_target_ids": [record["pdb_id"] for record in records if record["prep_success"]],
        "unresolved_target_ids": [record["pdb_id"] for record in records if not record["prep_success"]],
        "mean_attempt_count": sum(record["attempt_count"] for record in records) / total if total else 0.0,
    }


def _excerpt(text: str, *, limit: int = 1200) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "...<truncated>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PDBbind receptor PDBQT files with auditable template assignments.")
    parser.add_argument("--pdbbind-root", required=True, help="Local PDBbind root containing ready target dirs.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON summary path.")
    parser.add_argument("--prepared-dir", default=DEFAULT_PREP_DIR, help="Directory for prepared receptor PDBQT files.")
    parser.add_argument("--limit", type=int, help="Maximum targets to attempt.")
    parser.add_argument("--conda-env", default="vina", help="Conda environment containing mk_prepare_receptor.py.")
    parser.add_argument("--timeout-sec", type=int, default=90, help="Per-attempt receptor-prep timeout.")
    parser.add_argument("--max-template-attempts", type=int, default=12)
    parser.add_argument("--template-policy", default="first", choices=["first", "prefer_hid", "prefer_hie"])
    parser.add_argument("--target-id", action="append", default=[], help="Optional PDB ID to prepare. Can be repeated.")
    args = parser.parse_args()

    payload = run_pdbbind_receptor_template_prep(
        pdbbind_root=args.pdbbind_root,
        output_path=args.output,
        prepared_dir=args.prepared_dir,
        limit=args.limit,
        conda_env=args.conda_env,
        timeout_sec=args.timeout_sec,
        max_template_attempts=args.max_template_attempts,
        template_policy=args.template_policy,
        target_ids=args.target_id,
    )
    print(json.dumps({"summary_json": args.output, "summary": payload["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
