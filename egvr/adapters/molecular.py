"""Thin molecular adapters for the optional external tool service.

The paper artifact owns the execution contract, normalization, verification,
and repair logic. Model implementations and their environments remain external.
"""

from __future__ import annotations

from typing import Any

from .tool_server import run_external_tool


def generate_molecules_for_pocket(
    protein_pdb_path: str,
    center_xyz: list[float] | None = None,
    ref_ligand_path: str | None = None,
    num_samples: int = 10,
    seed: int | None = None,
) -> dict[str, Any]:
    payload = {
        "protein_pdb_path": protein_pdb_path,
        "num_samples": num_samples,
        "save_reward": True,
    }
    if center_xyz is not None:
        payload["center"] = center_xyz
    if ref_ligand_path is not None:
        payload["ref_ligand_path"] = ref_ligand_path
    if seed is not None:
        payload["seed"] = seed
    result = run_external_tool("rxnflow", payload, timeout_sec=900)
    if _failed(result):
        return _failure(result)
    summary = result.get("summary", {})
    details = result.get("results", {})
    return {
        "success": True,
        "generated_count": summary.get("generated_count"),
        "sampling_time_sec": summary.get("sampling_time_sec"),
        "full_results_csv_path": summary.get("output_file"),
        "top_molecules_preview": details.get("generated_preview", []),
    }


def generate_molecules_reinvent4_denovo(num_variants: int = 100) -> dict[str, Any]:
    return _reinvent("de_novo", None, num_variants)


def generate_molecules_reinvent4_mol2mol(
    smiles: str,
    num_variants: int = 50,
    strategy: str = "beamsearch",
    temperature: float = 1.0,
) -> dict[str, Any]:
    return _reinvent(
        "mol2mol",
        smiles,
        num_variants,
        strategy=strategy,
        temperature=temperature,
    )


def generate_molecules_reinvent4_libinvent(
    smiles: str,
    num_variants: int = 50,
) -> dict[str, Any]:
    return _reinvent("libinvent", smiles, num_variants)


def generate_scaffold_analogs(smiles: str, num_analogs: int = 10) -> dict[str, Any]:
    result = run_external_tool(
        "scaffold",
        {"smiles": smiles, "num_analogs": num_analogs},
        timeout_sec=600,
    )
    if _failed(result):
        return _failure(result)
    summary = result.get("summary", {})
    return {
        "success": True,
        "input_scaffold": summary.get("input_scaffold", smiles),
        "requested_batch_size": num_analogs,
        "generated_count": summary.get("valid_unique_generated"),
        "molecules": [
            row.get("smiles")
            for row in result.get("results", [])
            if isinstance(row, dict) and row.get("smiles")
        ],
    }


def generate_libinvent_decorations(
    smiles: str,
    num_decorations: int = 3,
) -> dict[str, Any]:
    result = run_external_tool(
        "libinvent",
        {"smiles": smiles, "num_decorations": num_decorations},
        timeout_sec=600,
    )
    if _failed(result):
        return _failure(result)
    summary = result.get("summary", {})
    preview = summary.get("preview", [])
    return {
        "success": True,
        "input_scaffold": smiles,
        "requested_num_decorations": num_decorations,
        "generated_count": summary.get("row_count"),
        "molecules_smiles": [
            row.get("SMILES") or row.get("smiles")
            for row in preview
            if isinstance(row, dict) and (row.get("SMILES") or row.get("smiles"))
        ],
        "decorated_molecules_preview": preview,
    }


def calculate_scscore(
    smiles: str | None = None,
    smiles_list: list[str] | None = None,
    model_type: str = "1024bool",
) -> dict[str, Any]:
    values = [smiles] if smiles else list(smiles_list or [])
    return run_external_tool(
        "scscore",
        {"smiles_list": values, "model_type": model_type},
    )


def predict_molecule_toxicity(smiles: str) -> dict[str, Any]:
    result = run_external_tool("toxicity", {"smiles": smiles})
    if _failed(result):
        return _failure(result)
    summary = result.get("summary", {})
    return {
        "success": True,
        "smiles": smiles,
        "toxicity_probability": summary.get("toxicity_probability"),
        "is_toxic": summary.get("is_toxic"),
        "verdict": "Toxic" if summary.get("is_toxic") else "Non-Toxic",
    }


def predict_antibacterial_pmic(smiles: str) -> dict[str, Any]:
    result = run_external_tool("pmic", {"smiles": smiles})
    if _failed(result):
        return _failure(result)
    summary = result.get("summary", {})
    return {
        "success": True,
        "smiles": smiles,
        "pMIC_value": summary.get("pMIC_value"),
        "estimated_MIC_uM": summary.get("estimated_MIC_uM"),
    }


def perform_molecular_docking_vina(
    receptor_pdbqt_path: str,
    ligand_pdbqt_path: str,
    center_xyz: list[float],
    box_size_xyz: list[float],
    exhaustiveness: int = 32,
) -> dict[str, Any]:
    result = run_external_tool(
        "vina",
        {
            "receptor_file": receptor_pdbqt_path,
            "ligand_file": ligand_pdbqt_path,
            "center": center_xyz,
            "box_size": box_size_xyz,
            "exhaustiveness": exhaustiveness,
            "n_poses": 20,
        },
        timeout_sec=1200,
    )
    if _failed(result):
        return _failure(result)
    summary = result.get("summary", {})
    details = result.get("results", {})
    return {
        "success": True,
        "best_docking_score_kcal_mol": summary.get("best_docking_score"),
        "score_before_minimization_kcal_mol": details.get("score_before_minimization"),
        "score_after_minimization_kcal_mol": summary.get("score_after_minimization"),
        "docked_poses_file_path": details.get("docked_poses_file"),
        "minimized_pose_file_path": details.get("minimized_pose_file"),
    }


def _reinvent(
    action: str,
    smiles: str | None,
    num_variants: int,
    **parameters: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"num_variants": num_variants, **parameters}
    if smiles is not None:
        payload["smiles_list"] = [smiles]
    result = run_external_tool(
        "reinvent4",
        payload,
        action=action,
        timeout_sec=600,
    )
    if _failed(result):
        return _failure(result)
    rows = result.get("results", {}).get("molecules", [])
    molecules = [
        row.get("smiles")
        for row in rows
        if isinstance(row, dict) and row.get("smiles")
    ]
    return {
        "success": True,
        "mode": action,
        "input_smiles": smiles,
        "requested_variants": num_variants,
        "generated_count": result.get("summary", {}).get("generated_count", len(molecules)),
        "molecules_smiles": molecules,
    }


def _failed(result: dict[str, Any]) -> bool:
    return bool(result.get("error") or result.get("success") is False)


def _failure(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": False,
        "error": result.get("error") or "External tool failed without an error message.",
    }
