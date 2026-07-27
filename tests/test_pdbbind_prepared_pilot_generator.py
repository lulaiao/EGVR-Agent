from __future__ import annotations

import json

from egvr.pdbbind_prepared_pilot_generator import (
    GATE_BLOCKED_INSUFFICIENT_PREPARED_TARGETS,
    GATE_READY,
    build_prepared_pilot_gate_report,
    generate_pdbbind_prepared_pilot_tasks,
    should_write_prepared_pilot,
)


def test_generate_prepared_pilot_uses_only_successfully_prepared_receptors(tmp_path):
    root = tmp_path / "PDBbind_v2020"
    index_dir = root / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "INDEX_refined_data.2020").write_text(
        "1abc 2.10 2018 7.52 3.0nM // ref ligand-a\n"
        "2def 1.90 2019 6.00 1.0uM // ref ligand-b\n",
        encoding="utf-8",
    )
    _make_target(root, "1abc")
    _make_target(root, "2def")
    prepared = tmp_path / "prepared" / "1abc.pdbqt"
    prepared.parent.mkdir()
    prepared.write_text("RECEPTOR\n", encoding="utf-8")
    summary = tmp_path / "template_summary.json"
    summary.write_text(
        json.dumps(
            {
                "ready_targets": [
                    {
                        "pdb_id": "1abc",
                        "prep_success": True,
                        "protein_path": str(root / "extracted" / "v2020-refined_1" / "1abc" / "1abc_protein.pdb"),
                        "prepared_receptor_pdbqt_path": str(prepared),
                        "template_assignments": {"A:12": "HIE"},
                        "attempt_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = generate_pdbbind_prepared_pilot_tasks(
        pdbbind_root=root,
        template_prep_summary=summary,
        limit=5,
    )

    assert len(tasks) == 1
    task = tasks[0]
    assert task["expected_task_type"] == "docking_evaluation"
    assert task["expected_tools"] == ["vina"]
    assert task["metadata"]["pdb_id"] == "1abc"
    assert task["metadata"]["protein_path"] == str(prepared)
    assert task["metadata"]["receptor_template_assignments"] == {"A:12": "HIE"}
    assert task["metadata"]["pocket_center"] == [1.0, 1.0, 1.0]
    assert "2def" not in task["task_id"]


def test_generate_prepared_pilot_returns_empty_when_no_ready_targets(tmp_path):
    root = tmp_path / "PDBbind_v2020"
    root.mkdir()
    summary = tmp_path / "template_summary.json"
    summary.write_text(json.dumps({"ready_targets": []}), encoding="utf-8")

    tasks = generate_pdbbind_prepared_pilot_tasks(
        pdbbind_root=root,
        template_prep_summary=summary,
        limit=5,
    )

    assert tasks == []


def test_prepared_pilot_gate_blocks_named_scale_when_too_few_tasks():
    tasks = [{"task_id": "pdbbind_refined_prepared_pilot_000_1abc", "metadata": {"pdb_id": "1abc"}}]

    report = build_prepared_pilot_gate_report(
        tasks=tasks,
        requested_limit=30,
        min_ready=30,
        template_prep_summary="summary.json",
    )

    assert report["gate_status"] == GATE_BLOCKED_INSUFFICIENT_PREPARED_TARGETS
    assert report["benchmark_written"] is False
    assert report["task_count"] == 1
    assert report["requested_limit"] == 30
    assert report["min_ready"] == 30
    assert report["pdb_ids"] == ["1abc"]
    assert not should_write_prepared_pilot(task_count=1, min_ready=30)


def test_prepared_pilot_gate_keeps_default_backwards_compatible():
    tasks = [{"task_id": "pdbbind_refined_prepared_pilot_000_1abc", "metadata": {"pdb_id": "1abc"}}]

    report = build_prepared_pilot_gate_report(tasks=tasks, requested_limit=30, min_ready=None, benchmark_written=True)

    assert report["gate_status"] == GATE_READY
    assert should_write_prepared_pilot(task_count=1, min_ready=None)
    assert should_write_prepared_pilot(task_count=1, min_ready=30, allow_partial_output=True)


def _make_target(root, pdb_id: str):
    target = root / "extracted" / "v2020-refined_1" / pdb_id
    target.mkdir(parents=True)
    (target / f"{pdb_id}_protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target / f"{pdb_id}_ligand.sdf").write_text("sdf\n", encoding="utf-8")
    (target / f"{pdb_id}_ligand.mol2").write_text(_mol2(), encoding="utf-8")


def _mol2() -> str:
    return """@<TRIPOS>MOLECULE
ligand
@<TRIPOS>ATOM
1 C1 0.000 0.000 0.000 C.3 1 LIG 0.0
2 C2 2.000 2.000 2.000 C.3 1 LIG 0.0
@<TRIPOS>BOND
"""
