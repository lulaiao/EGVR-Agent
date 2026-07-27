"""Probe PDBbind receptor PDBQT preparation before running docking."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pdbbind_pilot_generator import (
    PROTEIN_SUFFIXES,
    _first_existing_with_suffix,
    discover_ready_target_dirs,
)


DEFAULT_OUTPUT = "logs/baseline_runs/pdbbind_receptor_prep_probe_v1/pdbbind_receptor_prep_summary.json"
DEFAULT_PREP_DIR = "logs/baseline_runs/pdbbind_receptor_prep_probe_v1/prepared_receptors"
HISTIDINE_AMBIGUITY_RE = re.compile(
    r"for residue_key='(?P<residue_key>[^']+)'.*?(?:HIE|HID|HIP).*?tied.*?: (?P<candidates>[A-Z0-9_ ]+)",
    re.IGNORECASE | re.DOTALL,
)

CommandRunner = Callable[[Sequence[str], Path, int], subprocess.CompletedProcess[str]]


def run_pdbbind_receptor_prep_probe(
    *,
    pdbbind_root: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    prepared_dir: str | Path = DEFAULT_PREP_DIR,
    limit: int | None = None,
    conda_env: str = "vina",
    timeout_sec: int = 90,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Attempt receptor PDBQT preparation for ready PDBbind targets."""

    root = Path(pdbbind_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"PDBbind root not found: {root}")

    output_file = Path(output_path)
    prep_dir = Path(prepared_dir).expanduser().resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)
    target_dirs = discover_ready_target_dirs(root)
    if limit is not None:
        target_dirs = target_dirs[: max(limit, 0)]

    runner = command_runner or _run_command
    records = [
        _probe_target(target_dir, prep_dir=prep_dir, conda_env=conda_env, timeout_sec=timeout_sec, runner=runner)
        for target_dir in target_dirs
    ]
    summary = _summarize(records)
    payload = {
        "benchmark_id": "pdbbind_receptor_prep_probe_v1",
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "pdbbind_root": str(root),
        "prepared_dir": str(prep_dir),
        "conda_env": conda_env,
        "timeout_sec": timeout_sec,
        "summary": summary,
        "ready_targets": [record for record in records if record["prep_success"]],
        "template_required_targets": [
            record for record in records if record["failure_type"] == "histidine_template_ambiguity"
        ],
        "records": records,
        "notes": [
            "This probe only prepares receptor PDBQT files; it does not run docking.",
            "Targets with histidine/template ambiguity should be handled by explicit residue templates or excluded from the first real pilot.",
        ],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def classify_receptor_prep_failure(stderr: str, stdout: str = "") -> dict[str, Any]:
    """Classify receptor-preparation failure text into a stable category."""

    text = f"{stderr}\n{stdout}".strip()
    ambiguity = HISTIDINE_AMBIGUITY_RE.search(text)
    if ambiguity:
        return {
            "failure_type": "histidine_template_ambiguity",
            "failure_family": "residue_template",
            "residue_key": ambiguity.group("residue_key"),
            "template_candidates": ambiguity.group("candidates").split(),
        }
    lowered = text.lower()
    if "no module named" in lowered or "not installed" in lowered:
        return {"failure_type": "missing_dependency", "failure_family": "environment"}
    if "timed out" in lowered or "timeout" in lowered:
        return {"failure_type": "timeout", "failure_family": "runtime"}
    if "no such file" in lowered or "does not exist" in lowered or "not found" in lowered:
        return {"failure_type": "missing_input", "failure_family": "input"}
    if "runtimeerror" in lowered:
        return {"failure_type": "runtime_error", "failure_family": "runtime"}
    if text:
        return {"failure_type": "command_failed", "failure_family": "runtime"}
    return {"failure_type": "unknown_failure", "failure_family": "unknown"}


def _probe_target(
    target_dir: Path,
    *,
    prep_dir: Path,
    conda_env: str,
    timeout_sec: int,
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
        "elapsed_sec": 0.0,
        "failure_type": None,
        "failure_family": None,
        "residue_key": None,
        "template_candidates": [],
        "stderr_excerpt": "",
        "stdout_excerpt": "",
        "command": [],
    }
    if protein_path is None:
        record.update({"failure_type": "missing_input", "failure_family": "input"})
        return record
    if protein_path.suffix.lower() == ".pdbqt":
        record.update(
            {
                "prep_success": True,
                "prepared_receptor_pdbqt_path": str(protein_path),
                "failure_type": None,
                "failure_family": None,
            }
        )
        return record

    output_basename = prep_dir / pdb_id
    expected_outputs = [output_basename.with_suffix(".pdbqt"), Path(str(output_basename) + "_rigid.pdbqt")]
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
    started = time.monotonic()
    try:
        proc = runner(command, target_dir, timeout_sec)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        failure = classify_receptor_prep_failure(f"Timed out after {timeout_sec}s\n{exc.stderr or ''}", exc.stdout or "")
        record.update(
            {
                "elapsed_sec": round(elapsed, 3),
                "failure_type": failure["failure_type"],
                "failure_family": failure["failure_family"],
                "stderr_excerpt": _excerpt(str(exc.stderr or "")),
                "stdout_excerpt": _excerpt(str(exc.stdout or "")),
                "command": command,
            }
        )
        return record

    elapsed = time.monotonic() - started
    prepared = next((path for path in expected_outputs if path.exists() and path.stat().st_size > 0), None)
    if proc.returncode == 0 and prepared is not None:
        record.update(
            {
                "prep_success": True,
                "prepared_receptor_pdbqt_path": str(prepared),
                "elapsed_sec": round(elapsed, 3),
                "stderr_excerpt": _excerpt(proc.stderr),
                "stdout_excerpt": _excerpt(proc.stdout),
                "command": command,
            }
        )
        return record

    failure = classify_receptor_prep_failure(proc.stderr, proc.stdout)
    record.update(
        {
            "elapsed_sec": round(elapsed, 3),
            "failure_type": failure["failure_type"],
            "failure_family": failure["failure_family"],
            "residue_key": failure.get("residue_key"),
            "template_candidates": failure.get("template_candidates", []),
            "stderr_excerpt": _excerpt(proc.stderr),
            "stdout_excerpt": _excerpt(proc.stdout),
            "command": command,
        }
    )
    return record


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
        failure_type = record["failure_type"]
        if failure_type:
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
    return {
        "total": total,
        "prep_success_count": success_count,
        "prep_success_rate": success_count / total if total else 0.0,
        "prep_failure_count": total - success_count,
        "failure_counts": failure_counts,
        "template_required_count": failure_counts.get("histidine_template_ambiguity", 0),
        "stable_target_ids": [record["pdb_id"] for record in records if record["prep_success"]],
        "template_required_target_ids": [
            record["pdb_id"] for record in records if record["failure_type"] == "histidine_template_ambiguity"
        ],
    }


def _excerpt(text: str, *, limit: int = 1200) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "...<truncated>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe PDBbind receptor PDBQT preparation readiness.")
    parser.add_argument("--pdbbind-root", required=True, help="Local PDBbind root containing ready target dirs.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON summary path.")
    parser.add_argument("--prepared-dir", default=DEFAULT_PREP_DIR, help="Directory for prepared receptor PDBQT files.")
    parser.add_argument("--limit", type=int, help="Maximum targets to probe.")
    parser.add_argument("--conda-env", default="vina", help="Conda environment containing mk_prepare_receptor.py.")
    parser.add_argument("--timeout-sec", type=int, default=90, help="Per-target receptor-prep timeout.")
    args = parser.parse_args()

    payload = run_pdbbind_receptor_prep_probe(
        pdbbind_root=args.pdbbind_root,
        output_path=args.output,
        prepared_dir=args.prepared_dir,
        limit=args.limit,
        conda_env=args.conda_env,
        timeout_sec=args.timeout_sec,
    )
    print(
        json.dumps(
            {
                "summary_json": args.output,
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
