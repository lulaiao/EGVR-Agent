from __future__ import annotations

import json

from CAi.toolkit.agent_planner.pdbbind_pilot_generator import (
    discover_ready_target_dirs,
    generate_pdbbind_pilot_tasks,
    load_refined_affinity_index,
)


def test_load_refined_affinity_index_parses_records(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "INDEX_refined_data.2020").write_text(
        "# header\n"
        "1abc 2.10 2018 7.52 3.0nM // ref ligand-a\n"
        "2def 1.90 2019 6.00 1.0uM // ref ligand-b\n",
        encoding="utf-8",
    )

    records = load_refined_affinity_index(tmp_path)

    assert records["1abc"]["resolution"] == 2.10
    assert records["1abc"]["release_year"] == 2018
    assert records["1abc"]["neg_log_kd_ki"] == 7.52
    assert records["1abc"]["kd_ki"] == "3.0nM"
    assert records["2def"]["ligand_name"] == "ligand-b"


def test_discover_ready_target_dirs_detects_extracted_targets(tmp_path):
    target = tmp_path / "extracted" / "v2020-refined_1" / "1abc"
    target.mkdir(parents=True)
    (target / "1abc_protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target / "1abc_ligand.mol2").write_text(_mol2(), encoding="utf-8")

    targets = discover_ready_target_dirs(tmp_path)

    assert targets == [target]


def test_generate_pdbbind_pilot_tasks_writes_docking_metadata(tmp_path):
    index_dir = tmp_path / "index"
    target = tmp_path / "extracted" / "v2020-refined_1" / "1abc"
    index_dir.mkdir()
    target.mkdir(parents=True)
    (index_dir / "INDEX_refined_data.2020").write_text(
        "1abc 2.10 2018 7.52 3.0nM // ref ligand-a\n",
        encoding="utf-8",
    )
    (target / "1abc_protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target / "1abc_ligand.mol2").write_text(_mol2(), encoding="utf-8")
    (target / "1abc_ligand.sdf").write_text("sdf\n", encoding="utf-8")
    (target / "1abc_pocket.pdb").write_text("ATOM\n", encoding="utf-8")

    tasks = generate_pdbbind_pilot_tasks(tmp_path, limit=5)

    assert len(tasks) == 1
    task = tasks[0]
    assert task["expected_task_type"] == "docking_evaluation"
    assert task["expected_tools"] == ["vina"]
    assert "receptor=" in task["raw_user_query"]
    assert "ligand_path=" in task["raw_user_query"]
    assert task["metadata"]["pdb_id"] == "1abc"
    assert task["metadata"]["ligand_path"].endswith("1abc_ligand.sdf")
    assert task["metadata"]["source_ligand_path"].endswith("1abc_ligand.mol2")
    assert task["metadata"]["pocket_center"] == [1.0, 1.0, 1.0]
    assert task["metadata"]["affinity_metadata"]["neg_log_kd_ki"] == 7.52
    json.dumps(task)


def _mol2() -> str:
    return """@<TRIPOS>MOLECULE
ligand
@<TRIPOS>ATOM
1 C1 0.000 0.000 0.000 C.3 1 LIG 0.0
2 C2 2.000 2.000 2.000 C.3 1 LIG 0.0
@<TRIPOS>BOND
"""
