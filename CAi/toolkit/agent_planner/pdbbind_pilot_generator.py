"""Generate a small PDBbind refined/core pilot benchmark without running docking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark_task_generator import DEFAULT_BOX_SIZE, mol2_center, write_jsonl


DEFAULT_OUTPUT = "CAi/toolkit/agent_planner/benchmarks/pdbbind_refined_structure_pilot_5.jsonl"
PROTEIN_SUFFIXES = ("_protein.pdbqt", "_protein.pdb")
LIGAND_SUFFIXES = ("_ligand.pdbqt", "_ligand.sdf", "_ligand.mol2")


def generate_pdbbind_pilot_tasks(
    pdbbind_root: str | Path,
    *,
    limit: int = 5,
    box_size: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Generate docking-evaluation tasks from locally ready PDBbind target folders."""

    root = Path(pdbbind_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"PDBbind root not found: {root}")

    affinity_index = load_refined_affinity_index(root)
    target_dirs = discover_ready_target_dirs(root)
    if not target_dirs:
        raise FileNotFoundError(f"No ready PDBbind targets with protein and ligand files found under: {root}")

    box_size = box_size or list(DEFAULT_BOX_SIZE)
    tasks: list[dict[str, Any]] = []
    for target_dir in target_dirs:
        pdb_id = target_dir.name.lower()
        protein_path = _first_existing_with_suffix(target_dir, pdb_id, PROTEIN_SUFFIXES)
        ligand_path = _first_existing_with_suffix(target_dir, pdb_id, LIGAND_SUFFIXES)
        if protein_path is None or ligand_path is None:
            continue
        center_ligand_path = _first_existing_with_suffix(target_dir, pdb_id, ("_ligand.mol2", "_ligand.sdf"))
        if center_ligand_path is None or center_ligand_path.suffix.lower() != ".mol2":
            # The current lightweight centroid helper only parses MOL2 reliably.
            continue
        center = mol2_center(center_ligand_path)
        task_id = f"pdbbind_refined_structure_pilot_{len(tasks):03d}_{pdb_id}"
        affinity_metadata = affinity_index.get(pdb_id, {})
        tasks.append(
            {
                "task_id": task_id,
                "raw_user_query": (
                    f"Dock receptor={protein_path} ligand_path={ligand_path} "
                    f"center=[{center[0]},{center[1]},{center[2]}] "
                    f"box_size=[{box_size[0]},{box_size[1]},{box_size[2]}]."
                ),
                "expected_task_type": "docking_evaluation",
                "expected_tools": ["vina"],
                "should_succeed": True,
                "metadata": {
                    "dataset": "PDBbind_v2020",
                    "subset": "refined_structure_pilot",
                    "task_family": "docking_evaluation",
                    "pdb_id": pdb_id,
                    "protein_path": str(protein_path),
                    "ligand_path": str(ligand_path),
                    "source_ligand_path": str(center_ligand_path),
                    "pocket_path": str(target_dir / f"{pdb_id}_pocket.pdb")
                    if (target_dir / f"{pdb_id}_pocket.pdb").exists()
                    else None,
                    "pocket_center": center,
                    "box_size": box_size,
                    "affinity_metadata": affinity_metadata,
                    "real_run_notes": (
                        "Structure pilot generated from local PDBbind files. The active Vina wrapper "
                        "attempts PDB/SDF-to-PDBQT conversion, but PDBbind receptor preparation may require "
                        "explicit residue templates or precomputed receptor PDBQT files; MOL2 is used only "
                        "for pocket-center derivation."
                    ),
                },
            }
        )
        if len(tasks) >= limit:
            break
    return tasks


def discover_ready_target_dirs(root: str | Path, *, max_depth: int = 4) -> list[Path]:
    """Return PDB-id directories containing both protein and ligand files."""

    root = Path(root)
    targets: list[Path] = []
    for path in _safe_walk(root, max_depth=max_depth):
        if not path.is_dir() or not _looks_like_pdb_id(path.name):
            continue
        pdb_id = path.name.lower()
        if _first_existing_with_suffix(path, pdb_id, PROTEIN_SUFFIXES) and _first_existing_with_suffix(
            path, pdb_id, LIGAND_SUFFIXES
        ):
            targets.append(path)
    return sorted(targets, key=lambda item: item.name.lower())


def load_refined_affinity_index(root: str | Path) -> dict[str, dict[str, Any]]:
    """Parse PDBbind refined index metadata when available."""

    root = Path(root)
    index_path = _find_refined_index(root)
    if index_path is None:
        return {}

    records: dict[str, dict[str, Any]] = {}
    with index_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5 or not _looks_like_pdb_id(parts[0]):
                continue
            pdb_id = parts[0].lower()
            records[pdb_id] = {
                "pdb_id": pdb_id,
                "resolution": _float_or_none(parts[1]),
                "release_year": _int_or_none(parts[2]),
                "neg_log_kd_ki": _float_or_none(parts[3]),
                "kd_ki": parts[4],
                "index_path": str(index_path),
                "ligand_name": parts[-1] if len(parts) > 5 else None,
            }
    return records


def _find_refined_index(root: Path) -> Path | None:
    for path in _safe_walk(root, max_depth=4):
        if path.is_file() and path.name == "INDEX_refined_data.2020":
            return path
    for path in _safe_walk(root, max_depth=4):
        if path.is_file() and "refined" in path.name.lower() and "index" in path.name.lower():
            return path
    return None


def _first_existing_with_suffix(target_dir: Path, pdb_id: str, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        path = target_dir / f"{pdb_id}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _safe_walk(root: Path, *, max_depth: int) -> list[Path]:
    rows: list[Path] = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        rows.append(current)
        if depth >= max_depth or not current.is_dir():
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in children)
    return rows


def _looks_like_pdb_id(name: str) -> bool:
    return len(name) == 4 and name.isalnum()


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an offline PDBbind refined/core pilot JSONL.")
    parser.add_argument("--pdbbind-root", required=True, help="Local PDBbind root containing index and target dirs.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output benchmark JSONL path.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum ready targets to include.")
    args = parser.parse_args()

    tasks = generate_pdbbind_pilot_tasks(args.pdbbind_root, limit=args.limit)
    output = write_jsonl(tasks, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "task_count": len(tasks),
                "task_ids": [task["task_id"] for task in tasks],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
