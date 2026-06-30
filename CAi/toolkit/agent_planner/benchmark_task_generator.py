"""Generate real-path benchmark JSONL tasks from local molecular datasets."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_BOX_SIZE = [20.0, 20.0, 20.0]


@dataclass
class DataSourceManifest:
    """Validated paths for local benchmark data sources."""

    crossdocked_root: Path | None = None
    crossdocked_center_info_dir: Path | None = None
    crossdocked_protein_dir: Path | None = None
    lit_pcba_root: Path | None = None


@dataclass
class VinaPreparationConfig:
    """Optional local preprocessing for benchmark Vina inputs."""

    output_dir: Path
    receptor_command: tuple[str, ...] = ("prepare_receptor4.py",)
    ligand_command: tuple[str, ...] = ("mk_prepare_ligand.py",)
    ligand_fallback_command: tuple[str, ...] | None = None
    overwrite: bool = False


def load_data_source_manifest(path: str | Path) -> DataSourceManifest:
    """Load a JSON data-source manifest without adding YAML dependencies."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    crossdocked = payload.get("crossdocked2020", {})
    lit_pcba = payload.get("lit_pcba", {})
    return DataSourceManifest(
        crossdocked_root=_optional_path(crossdocked.get("root")),
        crossdocked_center_info_dir=_optional_path(crossdocked.get("center_info_dir")),
        crossdocked_protein_dir=_optional_path(crossdocked.get("protein_dir")),
        lit_pcba_root=_optional_path(lit_pcba.get("root")),
    )


def generate_real_benchmark_tasks(
    manifest: DataSourceManifest,
    *,
    crossdocked_split: str = "test",
    max_crossdocked: int = 5,
    max_lit_pcba: int = 5,
    num_candidates: int = 10,
    run_seed: int | None = None,
    lit_pcba_vina_preparation: VinaPreparationConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate benchmark cases with real local file paths."""

    cases: list[dict[str, Any]] = []
    cases.extend(
        generate_crossdocked_pocket_tasks(
            manifest,
            split=crossdocked_split,
            limit=max_crossdocked,
            num_candidates=num_candidates,
            run_seed=run_seed,
        )
    )
    cases.extend(
        generate_lit_pcba_docking_tasks(
            manifest,
            limit=max_lit_pcba,
            vina_preparation=lit_pcba_vina_preparation,
        )
    )
    return cases


def generate_crossdocked_pocket_tasks(
    manifest: DataSourceManifest,
    *,
    split: str = "test",
    limit: int = 5,
    num_candidates: int = 10,
    run_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate pocket-conditioned generation tasks from CrossDocked centers."""

    if limit <= 0:
        return []
    if not manifest.crossdocked_center_info_dir or not manifest.crossdocked_protein_dir:
        return []
    center_file = manifest.crossdocked_center_info_dir / f"{split}.csv"
    if not center_file.exists():
        raise FileNotFoundError(f"CrossDocked center file not found: {center_file}")

    tasks: list[dict[str, Any]] = []
    with center_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            protein_id = row[0]
            center = [round(float(row[1]), 3), round(float(row[2]), 3), round(float(row[3]), 3)]
            protein_path = manifest.crossdocked_protein_dir / split / f"{protein_id}.pdb"
            if not protein_path.exists():
                continue
            seed_suffix = f"_seed{int(run_seed):02d}" if run_seed is not None else ""
            seed_clause = f" rxnflow_seed={int(run_seed)}" if run_seed is not None else ""
            task_id = f"crossdocked_{split}_{len(tasks):03d}_{protein_id}{seed_suffix}"
            tasks.append(
                {
                    "task_id": task_id,
                    "raw_user_query": (
                        f"Generate {num_candidates} molecules for protein_path={protein_path} "
                        f"pocket_center=[{center[0]},{center[1]},{center[2]}] "
                        f"for synthesizability and toxicity.{seed_clause}"
                    ),
                    "expected_task_type": "pocket_conditioned_generation",
                    "expected_tools": ["rxnflow", "reinvent4_denovo", "scscore", "toxicity"],
                    "should_succeed": True,
                    "metadata": {
                        "dataset": "CrossDocked2020",
                        "split": split,
                        "task_family": "pocket_conditioned_generation",
                        "protein_id": protein_id,
                        "protein_path": str(protein_path),
                        "pocket_center": center,
                        "num_candidates": num_candidates,
                        **({"run_seed": int(run_seed), "rxnflow_seed": int(run_seed)} if run_seed is not None else {}),
                        "real_run_notes": "Requires rxnflow plus evaluator tools through the CAi tool server.",
                    },
                }
            )
            if len(tasks) >= limit:
                break
    return tasks


def generate_lit_pcba_docking_tasks(
    manifest: DataSourceManifest,
    *,
    limit: int = 5,
    box_size: list[float] | None = None,
    vina_preparation: VinaPreparationConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate docking evaluation tasks from LIT-PCBA protein/ligand pairs."""

    if limit <= 0:
        return []
    if not manifest.lit_pcba_root:
        return []
    if not manifest.lit_pcba_root.exists():
        raise FileNotFoundError(f"LIT-PCBA root not found: {manifest.lit_pcba_root}")

    box_size = box_size or DEFAULT_BOX_SIZE
    tasks: list[dict[str, Any]] = []
    for target_dir in sorted(path for path in manifest.lit_pcba_root.iterdir() if path.is_dir()):
        protein_path = target_dir / "protein.pdb"
        ligand_path = target_dir / "ligand.mol2"
        if not protein_path.exists() or not ligand_path.exists():
            continue
        center = mol2_center(ligand_path)
        task_id = f"litpcba_docking_{len(tasks):03d}_{target_dir.name}"
        benchmark_protein_path = protein_path
        benchmark_ligand_path = ligand_path
        preparation_metadata: dict[str, Any] = {}
        if vina_preparation:
            prepared = prepare_lit_pcba_vina_inputs(
                protein_path,
                ligand_path,
                output_dir=vina_preparation.output_dir,
                task_id=task_id,
                receptor_command=vina_preparation.receptor_command,
                ligand_command=vina_preparation.ligand_command,
                ligand_fallback_command=vina_preparation.ligand_fallback_command,
                overwrite=vina_preparation.overwrite,
            )
            benchmark_protein_path = Path(prepared["protein_path"])
            benchmark_ligand_path = Path(prepared["ligand_path"])
            preparation_metadata = prepared["metadata"]
        tasks.append(
            {
                "task_id": task_id,
                "raw_user_query": (
                    f"Dock receptor={benchmark_protein_path} ligand_path={benchmark_ligand_path} "
                    f"center=[{center[0]},{center[1]},{center[2]}] "
                    f"box_size=[{box_size[0]},{box_size[1]},{box_size[2]}]."
                ),
                "expected_task_type": "docking_evaluation",
                "expected_tools": ["vina"],
                "should_succeed": True,
                "metadata": {
                    "dataset": "LIT-PCBA",
                    "task_family": "docking_evaluation",
                    "target": target_dir.name,
                    "protein_path": str(benchmark_protein_path),
                    "ligand_path": str(benchmark_ligand_path),
                    "source_protein_path": str(protein_path),
                    "source_ligand_path": str(ligand_path),
                    "pocket_center": center,
                    "box_size": box_size,
                    "real_run_notes": "Requires vina through the CAi tool server.",
                    **preparation_metadata,
                },
            }
        )
        if len(tasks) >= limit:
            break
    return tasks


def prepare_lit_pcba_vina_inputs(
    protein_path: str | Path,
    ligand_path: str | Path,
    *,
    output_dir: str | Path,
    task_id: str,
    receptor_command: Sequence[str] = ("prepare_receptor4.py",),
    ligand_command: Sequence[str] = ("mk_prepare_ligand.py",),
    ligand_fallback_command: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare LIT-PCBA receptor/ligand files as PDBQT without touching tool code."""

    protein_path = Path(protein_path)
    ligand_path = Path(ligand_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_out = output_dir / f"{task_id}_receptor.pdbqt"
    ligand_out = output_dir / f"{task_id}_ligand.pdbqt"

    if overwrite or not receptor_out.exists():
        _run_prep_command(
            [
                *receptor_command,
                "-r",
                str(protein_path),
                "-o",
                str(receptor_out),
                "-A",
                "hydrogens",
            ],
            output_path=receptor_out,
            label="receptor",
        )
    ligand_fallback_used = False
    if overwrite or not ligand_out.exists():
        try:
            _run_prep_command(
                [
                    *ligand_command,
                    "-i",
                    str(ligand_path),
                    "-o",
                    str(ligand_out),
                ],
                output_path=ligand_out,
                label="ligand",
            )
        except RuntimeError:
            if not ligand_fallback_command:
                raise
            if ligand_out.exists():
                ligand_out.unlink()
            _run_obabel_ligand_command(
                ligand_fallback_command,
                input_path=ligand_path,
                output_path=ligand_out,
            )
            ligand_fallback_used = True

    return {
        "protein_path": str(receptor_out),
        "ligand_path": str(ligand_out),
        "metadata": {
            "vina_preparation": {
                "prepared": True,
                "receptor_command": list(receptor_command),
                "ligand_command": list(ligand_command),
                "ligand_fallback_command": (
                    list(ligand_fallback_command) if ligand_fallback_command else None
                ),
                "ligand_fallback_used": ligand_fallback_used,
                "receptor_pdbqt_path": str(receptor_out),
                "ligand_pdbqt_path": str(ligand_out),
            }
        },
    }


def _run_prep_command(command: Sequence[str], *, output_path: Path, label: str) -> None:
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to prepare {label} PDBQT.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Preparation command completed but did not create {label} output: {output_path}"
        )


def _run_obabel_ligand_command(
    command: Sequence[str],
    *,
    input_path: Path,
    output_path: Path,
) -> None:
    full_command = [
        *command,
        "-imol2",
        str(input_path),
        "-opdbqt",
        "-O",
        str(output_path),
        "-h",
    ]
    proc = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to prepare ligand PDBQT with Open Babel fallback.\n"
            f"Command: {' '.join(full_command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Open Babel fallback completed but did not create ligand output: {output_path}"
        )


def write_jsonl(cases: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    return output_path


def mol2_center(path: str | Path, *, include_hydrogen: bool = False) -> list[float]:
    """Compute the atom-coordinate centroid from a MOL2 file."""

    coords: list[tuple[float, float, float]] = []
    in_atoms = False
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "@<TRIPOS>ATOM":
                in_atoms = True
                continue
            if stripped.startswith("@<TRIPOS>") and in_atoms:
                break
            if not in_atoms or not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 6:
                continue
            atom_type = parts[5]
            if not include_hydrogen and atom_type.upper().startswith("H"):
                continue
            coords.append((float(parts[2]), float(parts[3]), float(parts[4])))
    if not coords:
        raise ValueError(f"No atom coordinates found in MOL2 file: {path}")
    count = len(coords)
    return [round(sum(coord[idx] for coord in coords) / count, 3) for idx in range(3)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-path CAi agent_planner benchmark tasks.")
    parser.add_argument("--data-sources", required=True, help="Path to data source manifest JSON.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--crossdocked-split", default="test", choices=["train", "test"])
    parser.add_argument("--max-crossdocked", type=int, default=5)
    parser.add_argument("--max-lit-pcba", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--run-seed", type=int, help="Optional RxnFlow seed recorded in CrossDocked tasks.")
    parser.add_argument(
        "--prepare-lit-pcba-vina-inputs",
        action="store_true",
        help="Prepare LIT-PCBA receptor/ligand files as PDBQT before writing docking tasks.",
    )
    parser.add_argument("--vina-prep-dir", help="Directory for prepared LIT-PCBA PDBQT files.")
    parser.add_argument(
        "--vina-receptor-prep-command",
        default="prepare_receptor4.py",
        help="Command prefix used for receptor PDBQT preparation.",
    )
    parser.add_argument(
        "--vina-ligand-prep-command",
        default="mk_prepare_ligand.py",
        help="Command prefix used for ligand PDBQT preparation.",
    )
    parser.add_argument(
        "--vina-ligand-fallback-command",
        help="Optional Open Babel command prefix used when ligand preparation fails.",
    )
    parser.add_argument(
        "--overwrite-vina-prep",
        action="store_true",
        help="Regenerate prepared PDBQT files even if they already exist.",
    )
    args = parser.parse_args()

    manifest = load_data_source_manifest(args.data_sources)
    vina_preparation = None
    if args.prepare_lit_pcba_vina_inputs:
        if not args.vina_prep_dir:
            parser.error("--vina-prep-dir is required with --prepare-lit-pcba-vina-inputs")
        vina_preparation = VinaPreparationConfig(
            output_dir=Path(args.vina_prep_dir),
            receptor_command=tuple(shlex.split(args.vina_receptor_prep_command)),
            ligand_command=tuple(shlex.split(args.vina_ligand_prep_command)),
            ligand_fallback_command=(
                tuple(shlex.split(args.vina_ligand_fallback_command))
                if args.vina_ligand_fallback_command
                else None
            ),
            overwrite=args.overwrite_vina_prep,
        )
    cases = generate_real_benchmark_tasks(
        manifest,
        crossdocked_split=args.crossdocked_split,
        max_crossdocked=args.max_crossdocked,
        max_lit_pcba=args.max_lit_pcba,
        num_candidates=args.num_candidates,
        run_seed=args.run_seed,
        lit_pcba_vina_preparation=vina_preparation,
    )
    output = write_jsonl(cases, args.output)
    print(json.dumps({"output": str(output), "tasks": len(cases)}, ensure_ascii=False, indent=2, sort_keys=True))


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


if __name__ == "__main__":
    main()
