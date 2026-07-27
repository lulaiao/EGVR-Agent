from __future__ import annotations

import json

from egvr.benchmark_task_generator import (
    DataSourceManifest,
    VinaPreparationConfig,
    generate_crossdocked_pocket_tasks,
    generate_lit_pcba_docking_tasks,
    generate_real_benchmark_tasks,
    load_data_source_manifest,
    mol2_center,
    write_jsonl,
)
from egvr.rule_planner import plan_workflow
from egvr.task_parser import parse_task


def test_mol2_center_uses_heavy_atoms_by_default(tmp_path):
    ligand = tmp_path / "ligand.mol2"
    ligand.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "demo",
                "@<TRIPOS>ATOM",
                "1 H1 100.0 100.0 100.0 H 1 <0> 0.0",
                "2 C1 0.0 0.0 0.0 C.3 1 <0> 0.0",
                "3 O1 2.0 4.0 6.0 O.3 1 <0> 0.0",
                "@<TRIPOS>BOND",
            ]
        ),
        encoding="utf-8",
    )

    assert mol2_center(ligand) == [1.0, 2.0, 3.0]


def test_generate_crossdocked_tasks_from_manifest(tmp_path):
    center_dir = tmp_path / "center_info"
    protein_dir = tmp_path / "protein"
    center_dir.mkdir()
    (protein_dir / "test").mkdir(parents=True)
    (center_dir / "test.csv").write_text("1abc_A,1.0,2.0,3.0\nmissing,4,5,6\n", encoding="utf-8")
    (protein_dir / "test" / "1abc_A.pdb").write_text("ATOM\n", encoding="utf-8")
    manifest = DataSourceManifest(crossdocked_center_info_dir=center_dir, crossdocked_protein_dir=protein_dir)

    tasks = generate_crossdocked_pocket_tasks(manifest, split="test", limit=5, num_candidates=7)

    assert len(tasks) == 1
    assert tasks[0]["expected_task_type"] == "pocket_conditioned_generation"
    assert tasks[0]["metadata"]["pocket_center"] == [1.0, 2.0, 3.0]
    parsed = parse_task(tasks[0]["raw_user_query"], task_id=tasks[0]["task_id"])
    workflow = plan_workflow(parsed)
    assert {"rxnflow", "reinvent4_denovo", "scscore", "toxicity"}.issubset(workflow.selected_tools)


def test_generate_crossdocked_tasks_records_run_seed(tmp_path):
    center_dir = tmp_path / "center_info"
    protein_dir = tmp_path / "protein"
    center_dir.mkdir()
    (protein_dir / "test").mkdir(parents=True)
    (center_dir / "test.csv").write_text("1abc_A,1.0,2.0,3.0\n", encoding="utf-8")
    (protein_dir / "test" / "1abc_A.pdb").write_text("ATOM\n", encoding="utf-8")
    manifest = DataSourceManifest(crossdocked_center_info_dir=center_dir, crossdocked_protein_dir=protein_dir)

    tasks = generate_crossdocked_pocket_tasks(manifest, split="test", limit=1, num_candidates=5, run_seed=2)

    assert tasks[0]["task_id"].endswith("_seed02")
    assert "rxnflow_seed=2" in tasks[0]["raw_user_query"]
    assert tasks[0]["metadata"]["run_seed"] == 2
    parsed = parse_task(tasks[0]["raw_user_query"], task_id=tasks[0]["task_id"], metadata=tasks[0]["metadata"])
    workflow = plan_workflow(parsed)
    assert workflow.tool_sequence[0].parameters["seed"] == 2


def test_generate_lit_pcba_tasks_from_manifest(tmp_path):
    target_dir = tmp_path / "LIT-PCBA" / "ADRB2"
    target_dir.mkdir(parents=True)
    (target_dir / "protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target_dir / "ligand.mol2").write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "demo",
                "@<TRIPOS>ATOM",
                "1 C1 -1.0 0.0 1.0 C.3 1 <0> 0.0",
                "2 N1 1.0 2.0 3.0 N.3 1 <0> 0.0",
                "@<TRIPOS>BOND",
            ]
        ),
        encoding="utf-8",
    )
    manifest = DataSourceManifest(lit_pcba_root=tmp_path / "LIT-PCBA")

    tasks = generate_lit_pcba_docking_tasks(manifest, limit=1)

    assert len(tasks) == 1
    assert tasks[0]["expected_task_type"] == "docking_evaluation"
    assert tasks[0]["metadata"]["pocket_center"] == [0.0, 1.0, 2.0]
    parsed = parse_task(tasks[0]["raw_user_query"], task_id=tasks[0]["task_id"])
    workflow = plan_workflow(parsed)
    assert workflow.selected_tools == ["vina"]


def test_generate_lit_pcba_tasks_can_prepare_vina_pdbqt_inputs(tmp_path):
    target_dir = tmp_path / "LIT-PCBA" / "ADRB2"
    target_dir.mkdir(parents=True)
    (target_dir / "protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target_dir / "ligand.mol2").write_text(
        "@<TRIPOS>ATOM\n1 C1 0 0 0 C.3 1 <0> 0.0\n@<TRIPOS>BOND\n",
        encoding="utf-8",
    )
    prep_script = tmp_path / "prep_stub.sh"
    prep_script.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "out=",
                "while [ $# -gt 0 ]; do",
                "  case \"$1\" in",
                "    -o) out=\"$2\"; shift 2 ;;",
                "    *) shift ;;",
                "  esac",
                "done",
                "printf 'PDBQT\\n' > \"$out\"",
            ]
        ),
        encoding="utf-8",
    )
    prep_script.chmod(0o755)
    manifest = DataSourceManifest(lit_pcba_root=tmp_path / "LIT-PCBA")

    tasks = generate_lit_pcba_docking_tasks(
        manifest,
        limit=1,
        vina_preparation=VinaPreparationConfig(
            output_dir=tmp_path / "prepared",
            receptor_command=(str(prep_script),),
            ligand_command=(str(prep_script),),
        ),
    )

    assert len(tasks) == 1
    assert tasks[0]["metadata"]["protein_path"].endswith("_receptor.pdbqt")
    assert tasks[0]["metadata"]["ligand_path"].endswith("_ligand.pdbqt")
    assert tasks[0]["metadata"]["source_protein_path"].endswith("protein.pdb")
    assert tasks[0]["metadata"]["source_ligand_path"].endswith("ligand.mol2")
    assert tasks[0]["metadata"]["vina_preparation"]["prepared"] is True
    assert "receptor=" + tasks[0]["metadata"]["protein_path"] in tasks[0]["raw_user_query"]
    assert (tmp_path / "prepared" / "litpcba_docking_000_ADRB2_receptor.pdbqt").exists()
    assert (tmp_path / "prepared" / "litpcba_docking_000_ADRB2_ligand.pdbqt").exists()


def test_generate_lit_pcba_tasks_can_fallback_to_obabel_ligand_prep(tmp_path):
    target_dir = tmp_path / "LIT-PCBA" / "IDH1"
    target_dir.mkdir(parents=True)
    (target_dir / "protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target_dir / "ligand.mol2").write_text(
        "@<TRIPOS>ATOM\n1 C1 0 0 0 C.3 1 <0> 0.0\n@<TRIPOS>BOND\n",
        encoding="utf-8",
    )
    receptor_script = tmp_path / "receptor_prep_stub.sh"
    receptor_script.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "out=",
                "while [ $# -gt 0 ]; do",
                "  case \"$1\" in",
                "    -o) out=\"$2\"; shift 2 ;;",
                "    *) shift ;;",
                "  esac",
                "done",
                "printf 'RECEPTOR\\n' > \"$out\"",
            ]
        ),
        encoding="utf-8",
    )
    failing_ligand_script = tmp_path / "failing_ligand_prep_stub.sh"
    failing_ligand_script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    obabel_script = tmp_path / "obabel_stub.sh"
    obabel_script.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "out=",
                "while [ $# -gt 0 ]; do",
                "  case \"$1\" in",
                "    -O) out=\"$2\"; shift 2 ;;",
                "    *) shift ;;",
                "  esac",
                "done",
                "printf 'LIGAND\\n' > \"$out\"",
            ]
        ),
        encoding="utf-8",
    )
    for script in (receptor_script, failing_ligand_script, obabel_script):
        script.chmod(0o755)
    manifest = DataSourceManifest(lit_pcba_root=tmp_path / "LIT-PCBA")

    tasks = generate_lit_pcba_docking_tasks(
        manifest,
        limit=1,
        vina_preparation=VinaPreparationConfig(
            output_dir=tmp_path / "prepared",
            receptor_command=(str(receptor_script),),
            ligand_command=(str(failing_ligand_script),),
            ligand_fallback_command=(str(obabel_script),),
        ),
    )

    assert tasks[0]["metadata"]["vina_preparation"]["ligand_fallback_used"] is True
    assert (tmp_path / "prepared" / "litpcba_docking_000_IDH1_ligand.pdbqt").read_text(
        encoding="utf-8"
    ) == "LIGAND\n"


def test_load_manifest_generate_and_write_jsonl(tmp_path):
    root = tmp_path / "CrossDocked2020"
    center_dir = root / "center_info"
    protein_dir = root / "protein"
    center_dir.mkdir(parents=True)
    (protein_dir / "test").mkdir(parents=True)
    (center_dir / "test.csv").write_text("2xyz_A,1,2,3\n", encoding="utf-8")
    (protein_dir / "test" / "2xyz_A.pdb").write_text("ATOM\n", encoding="utf-8")
    lit_root = tmp_path / "LIT-PCBA"
    (lit_root / "MAPK1").mkdir(parents=True)
    (lit_root / "MAPK1" / "protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (lit_root / "MAPK1" / "ligand.mol2").write_text(
        "@<TRIPOS>ATOM\n1 C1 0 0 0 C.3 1 <0> 0.0\n@<TRIPOS>BOND\n",
        encoding="utf-8",
    )
    manifest_file = tmp_path / "data_sources.json"
    manifest_file.write_text(
        json.dumps(
            {
                "crossdocked2020": {
                    "center_info_dir": str(center_dir),
                    "protein_dir": str(protein_dir),
                },
                "lit_pcba": {"root": str(lit_root)},
            }
        ),
        encoding="utf-8",
    )

    manifest = load_data_source_manifest(manifest_file)
    tasks = generate_real_benchmark_tasks(manifest, max_crossdocked=1, max_lit_pcba=1)
    output = write_jsonl(tasks, tmp_path / "tasks.jsonl")

    assert len(tasks) == 2
    assert output.read_text(encoding="utf-8").count("\n") == 2


def test_generator_respects_zero_limits(tmp_path):
    manifest = DataSourceManifest(
        crossdocked_center_info_dir=tmp_path / "center_info",
        crossdocked_protein_dir=tmp_path / "protein",
        lit_pcba_root=tmp_path / "LIT-PCBA",
    )

    assert generate_crossdocked_pocket_tasks(manifest, limit=0) == []
    assert generate_lit_pcba_docking_tasks(manifest, limit=0) == []
    assert generate_real_benchmark_tasks(manifest, max_crossdocked=0, max_lit_pcba=0) == []
