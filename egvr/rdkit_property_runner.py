"""Build RDKit property-verifier summaries from completed trace artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .task_schema import CandidateRecord
from .verifier_evidence_runner import DEFAULT_CROSSDOCKED_TRACE_DIR


DEFAULT_OUTPUT = "logs/baseline_runs/verifier_enhancement_v1/property_verifier_summary.json"

PROPERTY_VERIFIER_COLUMNS = [
    "evidence_family",
    "dataset",
    "task_count",
    "candidate_count",
    "valid_smiles_count",
    "property_coverage",
    "mean_qed",
    "mean_logp",
    "mean_molwt",
    "lipinski_pass_count",
    "lipinski_pass_rate",
    "pains_flag_count",
    "pains_flag_rate",
    "brenk_flag_count",
    "brenk_flag_rate",
    "status",
    "notes",
]

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


def calculate_rdkit_properties(smiles_list: list[str] | None = None, **kwargs) -> dict[str, Any]:
    """Return RDKit property rows for SMILES strings without making pass/fail claims."""

    values = smiles_list or kwargs.get("smiles_values") or []
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    except Exception as exc:
        return {
            "success": False,
            "status": "not_available",
            "results": [],
            "error": f"RDKit property calculation unavailable: {exc}",
        }

    pains_catalog = _build_filter_catalog("PAINS_A", "PAINS_B", "PAINS_C")
    brenk_catalog = _build_filter_catalog("BRENK")

    rows: list[dict[str, Any]] = []
    for smiles in values:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append({"smiles": smiles, "status": "invalid_smiles", "error": "RDKit failed to parse SMILES."})
            continue
        canonical = Chem.MolToSmiles(mol)
        molwt = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        rotatable_bonds = int(Lipinski.NumRotatableBonds(mol))
        violations = int(molwt > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
        pains_flags = _filter_matches(pains_catalog, mol)
        brenk_flags = _filter_matches(brenk_catalog, mol)
        rows.append(
            {
                "smiles": smiles,
                "canonical_smiles": canonical,
                "qed": float(QED.qed(mol)),
                "molwt": molwt,
                "logp": logp,
                "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                "hbd": hbd,
                "hba": hba,
                "rotatable_bonds": rotatable_bonds,
                "lipinski_violations": violations,
                "lipinski_pass": violations == 0,
                "pains_flags": pains_flags,
                "brenk_flags": brenk_flags,
                "status": "ok",
            }
        )
    return {"success": True, "status": "available", "results": rows}


def run_property_verifier_summary(
    *,
    trace_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute RDKit property evidence from completed CrossDocked traces."""

    root = Path(project_root or Path.cwd())
    trace_file = _resolve_trace_path(trace_path, root / DEFAULT_CROSSDOCKED_TRACE_DIR)
    trace_records = _load_trace_records(trace_file)
    candidates = _trace_candidates(trace_records)
    valid_smiles = [candidate.smiles for candidate in candidates if candidate.smiles and candidate.is_valid]
    property_payload = calculate_rdkit_properties(valid_smiles)
    ok_rows = [
        row
        for row in property_payload.get("results", [])
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    property_rows = [
        _ordered_property_row(
            {
                "evidence_family": "crossdocked_generation",
                "dataset": "CrossDocked2020",
                "task_count": len(trace_records),
                "candidate_count": len(candidates),
                "valid_smiles_count": len(valid_smiles),
                "property_coverage": len(ok_rows) / len(valid_smiles) if valid_smiles else None,
                "mean_qed": _mean(row.get("qed") for row in ok_rows),
                "mean_logp": _mean(row.get("logp") for row in ok_rows),
                "mean_molwt": _mean(row.get("molwt") for row in ok_rows),
                "lipinski_pass_count": sum(1 for row in ok_rows if row.get("lipinski_pass")),
                "lipinski_pass_rate": _rate(row.get("lipinski_pass") for row in ok_rows),
                "pains_flag_count": sum(1 for row in ok_rows if row.get("pains_flags")),
                "pains_flag_rate": _rate(bool(row.get("pains_flags")) for row in ok_rows),
                "brenk_flag_count": sum(1 for row in ok_rows if row.get("brenk_flags")),
                "brenk_flag_rate": _rate(bool(row.get("brenk_flags")) for row in ok_rows),
                "status": property_payload.get("status"),
                "notes": "RDKit properties are evidence accounting, not biological activity or ADMET claims.",
            }
        )
    ]
    verifier_evidence_rows = [_property_row_to_verifier_evidence(property_rows[0])] if property_rows else []
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_id": "rdkit_property_verifier_v1",
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "artifacts": {
            "trace": str(trace_file),
            "summary_json": str(output_file),
        },
        "property_columns": PROPERTY_VERIFIER_COLUMNS,
        "property_rows": property_rows,
        "verifier_evidence_columns": VERIFIER_EVIDENCE_COLUMNS,
        "verifier_evidence_rows": verifier_evidence_rows,
        "candidate_rows": property_payload.get("results", []),
        "environment": {
            "rdkit_property_status": property_payload.get("status"),
            "error": property_payload.get("error"),
        },
        "notes": [
            "This runner reads completed trace candidates and does not rerun molecular generation.",
            "Lipinski, PAINS, and Brenk are reported as conservative verifier evidence, not activity claims.",
        ],
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _build_filter_catalog(*catalog_names: str):
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        params = FilterCatalogParams()
        for name in catalog_names:
            catalog = getattr(FilterCatalogParams.FilterCatalogs, name)
            params.AddCatalog(catalog)
        return FilterCatalog(params)
    except Exception:
        return None


def _filter_matches(catalog: Any, mol: Any) -> list[str]:
    if catalog is None:
        return []
    try:
        return [match.GetDescription() for match in catalog.GetMatches(mol)]
    except Exception:
        return []


def _resolve_trace_path(path: str | Path | None, default_dir: Path) -> Path:
    if path is not None:
        return Path(path)
    traces = sorted(default_dir.glob("*_traces.jsonl"))
    if not traces:
        raise FileNotFoundError(f"No trace JSONL files found in {default_dir}")
    return traces[-1]


def _load_trace_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _trace_candidates(records: list[dict[str, Any]]) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    for record in records:
        for item in record.get("final_candidates", []):
            if isinstance(item, dict):
                candidates.append(CandidateRecord(**item))
    return candidates


def _property_row_to_verifier_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return _ordered_verifier_evidence_row(
        {
            "evidence_family": row.get("evidence_family"),
            "dataset": row.get("dataset"),
            "evidence_type": "rdkit_property_verifier",
            "task_count": row.get("task_count"),
            "candidate_count": row.get("candidate_count"),
            "evaluable_candidate_count": row.get("valid_smiles_count"),
            "evidence_count": row.get("valid_smiles_count"),
            "coverage": row.get("property_coverage"),
            "pass_count": row.get("lipinski_pass_count"),
            "pass_rate": row.get("lipinski_pass_rate"),
            "best_sa_score": None,
            "mean_sa_score": None,
            "pose_artifact_count": None,
            "status": row.get("status"),
            "notes": "Pass rate reports Lipinski pass; QED, LogP, PAINS, and Brenk are in the property verifier table.",
        }
    )


def _ordered_property_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in PROPERTY_VERIFIER_COLUMNS}


def _ordered_verifier_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in VERIFIER_EVIDENCE_COLUMNS}


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _rate(values) -> float | None:
    flags = [bool(value) for value in values if value is not None]
    return sum(1 for value in flags if value) / len(flags) if flags else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RDKit property-verifier evidence from completed traces.")
    parser.add_argument("--trace", help="Optional trace JSONL path. Defaults to the latest CrossDocked trace.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output summary JSON path.")
    parser.add_argument("--project-root", help="Project root; defaults to the current working directory.")
    args = parser.parse_args()
    payload = run_property_verifier_summary(
        trace_path=args.trace,
        output_path=args.output,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "summary_json": args.output,
                "property_rows": payload["property_rows"],
                "environment": payload["environment"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
