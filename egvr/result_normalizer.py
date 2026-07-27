"""Normalize heterogeneous tool outputs into candidate-level records."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from .task_schema import CandidateRecord


GENERATION_TOOLS = {
    "rxnflow",
    "reinvent4_denovo",
    "reinvent4_mol2mol",
    "reinvent4_libinvent",
    "scaffold",
    "libinvent",
}


def normalize_tool_output(
    tool_name: str,
    output: dict[str, Any] | None,
    *,
    existing_candidates: Iterable[CandidateRecord] | None = None,
    input_smiles: str | None = None,
) -> list[CandidateRecord]:
    """Return normalized candidate records after one tool output."""

    candidates = _clone_candidates(existing_candidates)
    output = dict(output or {})
    if not output:
        return candidates
    if output.get("success") is False or output.get("error"):
        return _attach_error(candidates, tool_name, str(output.get("error") or "Tool failed"))
    if tool_name in GENERATION_TOOLS:
        return _normalize_generation(tool_name, output)
    if tool_name == "scscore":
        return _merge_scscore(candidates, output)
    if tool_name == "sa_score":
        return _merge_sa_score(candidates, output)
    if tool_name == "toxicity":
        return _merge_toxicity(candidates, output, input_smiles=input_smiles)
    if tool_name == "pmic":
        return _merge_pmic(candidates, output, input_smiles=input_smiles)
    if tool_name == "vina":
        return _merge_vina(candidates, output, input_smiles=input_smiles)
    if tool_name == "posebusters":
        return _merge_posebusters(candidates, output)
    if tool_name == "rdkit_property_verifier":
        return _merge_rdkit_properties(candidates, output)
    return candidates


def rank_candidates(candidates: Iterable[CandidateRecord]) -> list[CandidateRecord]:
    """Assign deterministic ranks, preferring docking, toxicity, and SCScore."""

    ranked = _clone_candidates(candidates)

    def score_key(candidate: CandidateRecord) -> tuple[float, float, float, float, str]:
        docking = candidate.docking_score if candidate.docking_score is not None else float("inf")
        toxicity = candidate.toxicity_score if candidate.toxicity_score is not None else float("inf")
        scscore = candidate.scscore if candidate.scscore is not None else float("inf")
        sa_score = candidate.sa_score if candidate.sa_score is not None else float("inf")
        return (docking, toxicity, scscore, sa_score, candidate.smiles or "")

    ranked.sort(key=score_key)
    for idx, candidate in enumerate(ranked, start=1):
        candidate.rank = idx
    return ranked


def _normalize_generation(tool_name: str, output: dict[str, Any]) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for rank, item in enumerate(_extract_smiles_items(output), start=1):
        smiles = item.get("smiles")
        if not smiles:
            continue
        metadata = {key: value for key, value in item.items() if key != "smiles"}
        records.append(
            CandidateRecord(
                smiles=smiles,
                source_tool=tool_name,
                rank=rank,
                is_valid=_looks_like_smiles(smiles),
                artifacts=_extract_generation_artifacts(output),
                metadata=metadata,
            )
        )
    return records


def _extract_smiles_items(output: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("molecules_smiles", "molecules"):
        values = output.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str):
                    items.append({"smiles": value})
                elif isinstance(value, dict):
                    smiles = value.get("smiles") or value.get("SMILES") or value.get("canonical_smiles")
                    items.append({"smiles": smiles, **value})

    for key in ("top_molecules_preview", "decorated_molecules_preview"):
        values = output.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    smiles = value.get("smiles") or value.get("SMILES") or value.get("canonical_smiles")
                    items.append({"smiles": smiles, **value})

    results = output.get("results")
    if isinstance(results, list):
        for value in results:
            if isinstance(value, dict):
                smiles = value.get("smiles") or value.get("SMILES") or value.get("canonical_smiles")
                items.append({"smiles": smiles, **value})
    elif isinstance(results, dict):
        preview = results.get("generated_preview") or results.get("molecules")
        if isinstance(preview, list):
            for value in preview:
                if isinstance(value, dict):
                    smiles = value.get("smiles") or value.get("SMILES") or value.get("canonical_smiles")
                    items.append({"smiles": smiles, **value})

    return _dedupe_items(items)


def _merge_scscore(candidates: list[CandidateRecord], output: dict[str, Any]) -> list[CandidateRecord]:
    results = output.get("results", [])
    if isinstance(results, dict):
        results = [results]
    if not candidates:
        candidates = [
            CandidateRecord(
                smiles=row.get("input_smiles") or row.get("canonical_smiles"),
                source_tool="scscore",
                is_valid=bool(row.get("input_smiles") or row.get("canonical_smiles")),
            )
            for row in results
            if isinstance(row, dict)
        ]
    by_smiles = _index_by_smiles(candidates)
    for row in results:
        if not isinstance(row, dict):
            continue
        smiles = row.get("input_smiles") or row.get("canonical_smiles")
        candidate = by_smiles.get(smiles)
        if candidate is None:
            candidate = CandidateRecord(smiles=smiles, source_tool="scscore", is_valid=_looks_like_smiles(smiles))
            candidates.append(candidate)
            if smiles:
                by_smiles[smiles] = candidate
        candidate.scscore = _to_float(row.get("scscore"))
        candidate.metadata["canonical_smiles"] = row.get("canonical_smiles")
        candidate.metadata["scscore_interpretation"] = row.get("interpretation")
    return candidates


def _merge_sa_score(candidates: list[CandidateRecord], output: dict[str, Any]) -> list[CandidateRecord]:
    results = output.get("results", [])
    if isinstance(results, dict):
        results = [results]
    if not candidates:
        candidates = [
            CandidateRecord(
                smiles=row.get("smiles") or row.get("input_smiles") or row.get("canonical_smiles"),
                source_tool="sa_score",
                is_valid=bool(row.get("smiles") or row.get("input_smiles") or row.get("canonical_smiles")),
            )
            for row in results
            if isinstance(row, dict)
        ]
    by_smiles = _index_by_smiles(candidates)
    for row in results:
        if not isinstance(row, dict):
            continue
        smiles = row.get("smiles") or row.get("input_smiles") or row.get("canonical_smiles")
        candidate = by_smiles.get(smiles)
        if candidate is None:
            candidate = CandidateRecord(smiles=smiles, source_tool="sa_score", is_valid=_looks_like_smiles(smiles))
            candidates.append(candidate)
            if smiles:
                by_smiles[smiles] = candidate
        candidate.sa_score = _to_float(row.get("sa_score") or row.get("sascore"))
        if row.get("status"):
            candidate.metadata["sa_score_status"] = row.get("status")
        if row.get("error"):
            candidate.errors.append(str(row.get("error")))
    return candidates


def _merge_toxicity(
    candidates: list[CandidateRecord],
    output: dict[str, Any],
    *,
    input_smiles: str | None,
) -> list[CandidateRecord]:
    rows = _toxicity_rows(output, input_smiles=input_smiles)
    return _merge_single_value_rows(
        candidates,
        rows,
        source_tool="toxicity",
        score_key="toxicity_probability",
        assign=lambda candidate, row: _assign_toxicity(candidate, row),
    )


def _merge_pmic(
    candidates: list[CandidateRecord],
    output: dict[str, Any],
    *,
    input_smiles: str | None,
) -> list[CandidateRecord]:
    rows = _pmic_rows(output, input_smiles=input_smiles)
    return _merge_single_value_rows(
        candidates,
        rows,
        source_tool="pmic",
        score_key="pMIC_value",
        assign=lambda candidate, row: _assign_pmic(candidate, row),
    )


def _merge_vina(
    candidates: list[CandidateRecord],
    output: dict[str, Any],
    *,
    input_smiles: str | None,
) -> list[CandidateRecord]:
    score = _to_float(output.get("best_docking_score_kcal_mol") or output.get("best_docking_score"))
    if not candidates:
        candidates = [CandidateRecord(smiles=input_smiles, source_tool="vina", is_valid=bool(input_smiles))]
    target = candidates[0]
    target.docking_score = score
    target.artifacts.update(
        {
            key: value
            for key, value in {
                "docked_poses_file_path": output.get("docked_poses_file_path"),
                "minimized_pose_file_path": output.get("minimized_pose_file_path"),
            }.items()
            if value
        }
    )
    return candidates


def _merge_posebusters(candidates: list[CandidateRecord], output: dict[str, Any]) -> list[CandidateRecord]:
    rows = output.get("results", [])
    if isinstance(rows, dict):
        rows = [rows]
    if not rows and output:
        rows = [output]
    if not candidates:
        candidates = [
            CandidateRecord(
                smiles=row.get("smiles"),
                source_tool="posebusters",
                is_valid=bool(row.get("smiles")),
            )
            for row in rows
            if isinstance(row, dict)
        ]
    for idx, row in enumerate(row for row in rows if isinstance(row, dict)):
        candidate = _posebusters_candidate(candidates, row, idx)
        if candidate is None:
            continue
        status = row.get("status")
        pass_value = row.get("posebusters_pass")
        if pass_value is None:
            pass_value = row.get("all_checks_passed")
        candidate.posebusters_pass = _to_bool_or_none(pass_value)
        checks = row.get("checks")
        if isinstance(checks, dict):
            candidate.posebusters_checks = dict(checks)
        if status:
            candidate.metadata["posebusters_status"] = status
        if row.get("error"):
            candidate.errors.append(str(row.get("error")))
    return candidates


def _merge_rdkit_properties(candidates: list[CandidateRecord], output: dict[str, Any]) -> list[CandidateRecord]:
    rows = output.get("results", [])
    if isinstance(rows, dict):
        rows = [rows]
    if not candidates:
        candidates = [
            CandidateRecord(
                smiles=row.get("smiles") or row.get("input_smiles") or row.get("canonical_smiles"),
                source_tool="rdkit_property_verifier",
                is_valid=bool(row.get("smiles") or row.get("input_smiles") or row.get("canonical_smiles")),
            )
            for row in rows
            if isinstance(row, dict)
        ]
    by_smiles = _index_by_smiles(candidates)
    for row in rows:
        if not isinstance(row, dict):
            continue
        smiles = row.get("smiles") or row.get("input_smiles") or row.get("canonical_smiles")
        candidate = by_smiles.get(smiles)
        if candidate is None:
            candidate = CandidateRecord(
                smiles=smiles,
                source_tool="rdkit_property_verifier",
                is_valid=_looks_like_smiles(smiles),
            )
            candidates.append(candidate)
            if smiles:
                by_smiles[smiles] = candidate
        if row.get("status") == "invalid_smiles":
            candidate.is_valid = False
        properties = {
            key: row.get(key)
            for key in (
                "qed",
                "molwt",
                "logp",
                "tpsa",
                "hbd",
                "hba",
                "rotatable_bonds",
                "lipinski_violations",
                "lipinski_pass",
                "pains_flags",
                "brenk_flags",
            )
            if key in row
        }
        candidate.metadata["rdkit_properties"] = properties
        for key in ("qed", "logp", "lipinski_violations", "pains_flags", "brenk_flags"):
            if key in properties:
                candidate.metadata[key] = properties[key]
        if row.get("error"):
            candidate.errors.append(str(row["error"]))
    return candidates


def _posebusters_candidate(candidates: list[CandidateRecord], row: dict[str, Any], idx: int) -> CandidateRecord | None:
    smiles = row.get("smiles") or row.get("input_smiles")
    if smiles:
        by_smiles = _index_by_smiles(candidates)
        candidate = by_smiles.get(smiles)
        if candidate is not None:
            return candidate
    if candidates:
        return candidates[min(idx, len(candidates) - 1)]
    return None


def _toxicity_rows(output: dict[str, Any], *, input_smiles: str | None) -> list[dict[str, Any]]:
    if isinstance(output.get("results"), list):
        return [row for row in output["results"] if isinstance(row, dict)]
    row = dict(output)
    row.setdefault("smiles", input_smiles)
    return [row]


def _pmic_rows(output: dict[str, Any], *, input_smiles: str | None) -> list[dict[str, Any]]:
    if isinstance(output.get("results"), list):
        return [row for row in output["results"] if isinstance(row, dict)]
    row = dict(output)
    row.setdefault("smiles", input_smiles or output.get("smiles"))
    return [row]


def _merge_single_value_rows(candidates, rows, *, source_tool, score_key, assign):
    if not candidates:
        candidates = [
            CandidateRecord(smiles=row.get("smiles"), source_tool=source_tool, is_valid=_looks_like_smiles(row.get("smiles")))
            for row in rows
        ]
    by_smiles = _index_by_smiles(candidates)
    for row in rows:
        smiles = row.get("smiles") or row.get("input_smiles")
        candidate = by_smiles.get(smiles)
        if candidate is None:
            candidate = CandidateRecord(smiles=smiles, source_tool=source_tool, is_valid=_looks_like_smiles(smiles))
            candidates.append(candidate)
            if smiles:
                by_smiles[smiles] = candidate
        if row.get(score_key) is not None:
            assign(candidate, row)
    return candidates


def _assign_toxicity(candidate: CandidateRecord, row: dict[str, Any]) -> None:
    candidate.toxicity_score = _to_float(row.get("toxicity_probability") or row.get("toxicity_score"))
    candidate.metadata["toxicity_verdict"] = row.get("verdict")
    candidate.metadata["is_toxic"] = row.get("is_toxic")
    if row.get("image_saved_at"):
        candidate.artifacts["toxicity_image"] = row["image_saved_at"]


def _assign_pmic(candidate: CandidateRecord, row: dict[str, Any]) -> None:
    candidate.pmic_score = _to_float(row.get("pMIC_value") or row.get("pmic_score"))
    if row.get("estimated_MIC_uM") is not None:
        candidate.metadata["estimated_MIC_uM"] = row["estimated_MIC_uM"]


def _attach_error(candidates: list[CandidateRecord], tool_name: str, error: str) -> list[CandidateRecord]:
    if not candidates:
        return [CandidateRecord(source_tool=tool_name, is_valid=False, errors=[error])]
    for candidate in candidates:
        candidate.errors.append(error)
    return candidates


def _extract_generation_artifacts(output: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "full_results_csv_path": output.get("full_results_csv_path"),
            "input_scaffold": output.get("input_scaffold"),
        }.items()
        if value
    }


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        smiles = item.get("smiles")
        if not smiles or smiles in seen:
            continue
        result.append(item)
        seen.add(smiles)
    return result


def _clone_candidates(candidates: Iterable[CandidateRecord] | None) -> list[CandidateRecord]:
    return [CandidateRecord(**deepcopy(candidate.to_dict())) for candidate in (candidates or [])]


def _index_by_smiles(candidates: Iterable[CandidateRecord]) -> dict[str, CandidateRecord]:
    return {candidate.smiles: candidate for candidate in candidates if candidate.smiles}


def _looks_like_smiles(smiles: str | None) -> bool:
    return bool(smiles and isinstance(smiles, str) and not any(char.isspace() for char in smiles))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "pass", "passed"}:
        return True
    if normalized in {"false", "0", "no", "fail", "failed"}:
        return False
    return None
