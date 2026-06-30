from __future__ import annotations

from CAi.toolkit.agent_planner.result_normalizer import normalize_tool_output, rank_candidates
from CAi.toolkit.agent_planner.task_schema import CandidateRecord


def test_normalizer_extracts_generation_smiles():
    records = normalize_tool_output(
        "reinvent4_denovo",
        {"success": True, "molecules_smiles": ["CCO", "CCC", "CCO"]},
    )

    assert [record.smiles for record in records] == ["CCO", "CCC"]
    assert [record.rank for record in records] == [1, 2]
    assert records[0].source_tool == "reinvent4_denovo"


def test_normalizer_merges_scscore_by_smiles():
    records = [CandidateRecord(smiles="CCO"), CandidateRecord(smiles="CCC")]

    updated = normalize_tool_output(
        "scscore",
        {
            "success": True,
            "results": [
                {"input_smiles": "CCO", "canonical_smiles": "CCO", "scscore": 1.5},
                {"input_smiles": "CCC", "canonical_smiles": "CCC", "scscore": 2.1},
            ],
        },
        existing_candidates=records,
    )

    assert [candidate.scscore for candidate in updated] == [1.5, 2.1]
    assert records[0].scscore is None


def test_normalizer_merges_toxicity_and_pmic():
    records = [CandidateRecord(smiles="CCO")]
    records = normalize_tool_output(
        "toxicity",
        {"success": True, "smiles": "CCO", "toxicity_probability": 0.12, "verdict": "Non-Toxic"},
        existing_candidates=records,
        input_smiles="CCO",
    )
    records = normalize_tool_output(
        "pmic",
        {"success": True, "smiles": "CCO", "pMIC_value": 5.8, "estimated_MIC_uM": 1.6},
        existing_candidates=records,
        input_smiles="CCO",
    )

    assert records[0].toxicity_score == 0.12
    assert records[0].pmic_score == 5.8
    assert records[0].metadata["estimated_MIC_uM"] == 1.6


def test_normalizer_merges_sa_score_and_posebusters_evidence():
    records = [CandidateRecord(smiles="CCO")]
    records = normalize_tool_output(
        "sa_score",
        {"success": True, "results": [{"smiles": "CCO", "sa_score": 2.4, "status": "ok"}]},
        existing_candidates=records,
    )
    records = normalize_tool_output(
        "posebusters",
        {
            "success": True,
            "results": [
                {
                    "smiles": "CCO",
                    "posebusters_pass": True,
                    "checks": {"mol_pred_loaded": True},
                    "status": "ok",
                }
            ],
        },
        existing_candidates=records,
    )

    assert records[0].sa_score == 2.4
    assert records[0].metadata["sa_score_status"] == "ok"
    assert records[0].posebusters_pass is True
    assert records[0].posebusters_checks == {"mol_pred_loaded": True}


def test_normalizer_merges_rdkit_property_evidence():
    records = [CandidateRecord(smiles="CCO")]

    updated = normalize_tool_output(
        "rdkit_property_verifier",
        {
            "success": True,
            "results": [
                {
                    "smiles": "CCO",
                    "qed": 0.42,
                    "molwt": 46.07,
                    "logp": -0.01,
                    "tpsa": 20.23,
                    "hbd": 1,
                    "hba": 1,
                    "rotatable_bonds": 0,
                    "lipinski_violations": 0,
                    "lipinski_pass": True,
                    "pains_flags": [],
                    "brenk_flags": ["alcohol"],
                    "status": "ok",
                }
            ],
        },
        existing_candidates=records,
    )

    assert updated[0].metadata["rdkit_properties"]["qed"] == 0.42
    assert updated[0].metadata["qed"] == 0.42
    assert updated[0].metadata["lipinski_violations"] == 0
    assert updated[0].metadata["brenk_flags"] == ["alcohol"]


def test_rank_candidates_prefers_better_scores():
    records = [
        CandidateRecord(smiles="bad", docking_score=-7.0, toxicity_score=0.8, scscore=3.0),
        CandidateRecord(smiles="good", docking_score=-8.0, toxicity_score=0.1, scscore=2.0),
    ]

    ranked = rank_candidates(records)

    assert [candidate.smiles for candidate in ranked] == ["good", "bad"]
    assert [candidate.rank for candidate in ranked] == [1, 2]
