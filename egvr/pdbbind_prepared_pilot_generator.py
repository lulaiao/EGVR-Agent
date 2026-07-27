"""Generate a PDBbind real-pilot JSONL from successfully prepared receptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark_task_generator import DEFAULT_BOX_SIZE, mol2_center, write_jsonl
from .pdbbind_pilot_generator import (
    LIGAND_SUFFIXES,
    _first_existing_with_suffix,
    discover_ready_target_dirs,
    load_refined_affinity_index,
)


DEFAULT_OUTPUT = "egvr/benchmarks/pdbbind_refined_prepared_pilot_5.jsonl"
GATE_READY = "ready_to_write"
GATE_BLOCKED_INSUFFICIENT_PREPARED_TARGETS = "blocked_insufficient_prepared_targets"


def generate_pdbbind_prepared_pilot_tasks(
    *,
    pdbbind_root: str | Path,
    template_prep_summary: str | Path,
    limit: int = 5,
    box_size: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Generate docking tasks from template-prepared receptor PDBQT files."""

    root = Path(pdbbind_root).expanduser()
    summary = json.loads(Path(template_prep_summary).read_text(encoding="utf-8"))
    prepared_records = [
        record
        for record in summary.get("ready_targets", [])
        if record.get("prep_success") and record.get("prepared_receptor_pdbqt_path")
    ]
    prepared_by_id = {record["pdb_id"].lower(): record for record in prepared_records}
    if not prepared_by_id:
        return []

    target_dirs = {path.name.lower(): path for path in discover_ready_target_dirs(root)}
    affinity_index = load_refined_affinity_index(root)
    box_size = box_size or list(DEFAULT_BOX_SIZE)
    tasks: list[dict[str, Any]] = []
    for pdb_id in sorted(prepared_by_id):
        target_dir = target_dirs.get(pdb_id)
        if target_dir is None:
            continue
        ligand_path = _first_existing_with_suffix(target_dir, pdb_id, LIGAND_SUFFIXES)
        center_ligand_path = _first_existing_with_suffix(target_dir, pdb_id, ("_ligand.mol2", "_ligand.sdf"))
        receptor_path = Path(prepared_by_id[pdb_id]["prepared_receptor_pdbqt_path"])
        if not receptor_path.exists() or ligand_path is None or center_ligand_path is None:
            continue
        if center_ligand_path.suffix.lower() != ".mol2":
            continue
        center = mol2_center(center_ligand_path)
        task_id = f"pdbbind_refined_prepared_pilot_{len(tasks):03d}_{pdb_id}"
        tasks.append(
            {
                "task_id": task_id,
                "raw_user_query": (
                    f"Dock receptor={receptor_path} ligand_path={ligand_path} "
                    f"center=[{center[0]},{center[1]},{center[2]}] "
                    f"box_size=[{box_size[0]},{box_size[1]},{box_size[2]}]."
                ),
                "expected_task_type": "docking_evaluation",
                "expected_tools": ["vina"],
                "should_succeed": True,
                "metadata": {
                    "dataset": "PDBbind_v2020",
                    "subset": "refined_prepared_pilot",
                    "task_family": "docking_evaluation",
                    "pdb_id": pdb_id,
                    "protein_path": str(receptor_path),
                    "source_protein_path": prepared_by_id[pdb_id].get("protein_path"),
                    "ligand_path": str(ligand_path),
                    "source_ligand_path": str(center_ligand_path),
                    "pocket_center": center,
                    "box_size": box_size,
                    "affinity_metadata": affinity_index.get(pdb_id, {}),
                    "receptor_template_assignments": prepared_by_id[pdb_id].get("template_assignments", {}),
                    "receptor_template_attempt_count": prepared_by_id[pdb_id].get("attempt_count"),
                    "template_prep_summary_path": str(template_prep_summary),
                    "real_run_notes": "Generated only from targets with successful receptor PDBQT preparation.",
                },
            }
        )
        if len(tasks) >= limit:
            break
    return tasks


def build_prepared_pilot_gate_report(
    *,
    tasks: list[dict[str, Any]],
    requested_limit: int,
    min_ready: int | None = None,
    output: str | Path | None = None,
    template_prep_summary: str | Path | None = None,
    benchmark_written: bool = False,
) -> dict[str, Any]:
    """Summarize whether a prepared PDBbind pilot is safe to materialize."""

    task_count = len(tasks)
    gate_status = GATE_READY
    if min_ready is not None and task_count < min_ready:
        gate_status = GATE_BLOCKED_INSUFFICIENT_PREPARED_TARGETS

    return {
        "gate_status": gate_status,
        "benchmark_written": benchmark_written,
        "output": str(output) if output is not None else None,
        "requested_limit": requested_limit,
        "min_ready": min_ready,
        "task_count": task_count,
        "task_ids": [task["task_id"] for task in tasks],
        "pdb_ids": [task.get("metadata", {}).get("pdb_id") for task in tasks],
        "template_prep_summary": str(template_prep_summary) if template_prep_summary is not None else None,
        "notes": _gate_notes(gate_status),
    }


def should_write_prepared_pilot(
    *,
    task_count: int,
    min_ready: int | None = None,
    allow_partial_output: bool = False,
) -> bool:
    """Return whether the CLI should write the benchmark JSONL."""

    if min_ready is None or task_count >= min_ready:
        return True
    return allow_partial_output


def _gate_notes(gate_status: str) -> str:
    if gate_status == GATE_BLOCKED_INSUFFICIENT_PREPARED_TARGETS:
        return (
            "Prepared-receptor gate blocked benchmark materialization because fewer stable "
            "targets were available than requested. Generate a larger prep summary before "
            "running the real pilot."
        )
    return "Prepared-receptor gate satisfied."


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDBbind docking tasks from prepared receptor PDBQT files.")
    parser.add_argument("--pdbbind-root", required=True)
    parser.add_argument("--template-prep-summary", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--min-ready",
        type=int,
        default=None,
        help=(
            "Optional gate: require at least this many generated prepared-receptor tasks before "
            "writing the benchmark JSONL. Use this for named scale pilots such as pilot_30."
        ),
    )
    parser.add_argument(
        "--allow-partial-output",
        action="store_true",
        help="Write the JSONL even when --min-ready is not met. Defaults to blocking partial named pilots.",
    )
    parser.add_argument(
        "--gate-summary-output",
        default=None,
        help="Optional path to write the gate decision as JSON.",
    )
    args = parser.parse_args()

    tasks = generate_pdbbind_prepared_pilot_tasks(
        pdbbind_root=args.pdbbind_root,
        template_prep_summary=args.template_prep_summary,
        limit=args.limit,
    )
    should_write = should_write_prepared_pilot(
        task_count=len(tasks),
        min_ready=args.min_ready,
        allow_partial_output=args.allow_partial_output,
    )
    output = write_jsonl(tasks, args.output) if should_write else None
    report = build_prepared_pilot_gate_report(
        tasks=tasks,
        requested_limit=args.limit,
        min_ready=args.min_ready,
        output=output,
        template_prep_summary=args.template_prep_summary,
        benchmark_written=should_write,
    )
    if args.gate_summary_output:
        summary_output = Path(args.gate_summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
