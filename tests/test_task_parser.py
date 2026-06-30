from __future__ import annotations

from CAi.toolkit.agent_planner.task_parser import RuleBasedTaskParser, parse_task


def test_parser_extracts_pocket_conditioned_generation_fields():
    task = parse_task(
        "Generate 20 molecules for protein_path=agent_workspace/1HVR.pdb "
        "pocket_center=[1.0, 2.5, 3.0] with docking and toxicity checks.",
        task_id="task-pocket",
    )

    assert task.task_id == "task-pocket"
    assert task.task_type == "pocket_conditioned_generation"
    assert task.protein_path == "agent_workspace/1HVR.pdb"
    assert task.pocket_center == [1.0, 2.5, 3.0]
    assert task.constraints.max_candidates == 20
    assert task.constraints.require_docking is True
    assert task.constraints.require_toxicity is True
    assert task.objectives == ["binding", "toxicity"]


def test_parser_ignores_docking_words_inside_file_paths():
    task = parse_task(
        "Generate 1 molecules for "
        "protein_path=external_data/CrossDocked2020/protein/test/2z3h_A.pdb "
        "pocket_center=[24.195,35.026,105.003] for synthesizability and toxicity.",
        task_id="task-crossdocked",
    )

    assert task.task_type == "pocket_conditioned_generation"
    assert task.objectives == ["synthesizability", "toxicity"]
    assert task.constraints.require_docking is False
    assert task.constraints.require_synthesizability is True
    assert task.constraints.require_toxicity is True


def test_parser_extracts_run_seed_from_query_and_metadata():
    from_query = parse_task(
        "Generate 5 molecules for protein_path=target.pdb pocket_center=[1,2,3] rxnflow_seed=2.",
        task_id="seed-query",
    )
    from_metadata = parse_task(
        "Generate 5 molecules for protein_path=target.pdb pocket_center=[1,2,3].",
        task_id="seed-metadata",
        metadata={"random_seed": "3"},
    )

    assert from_query.metadata["run_seed"] == 2
    assert "run_seed" in from_query.metadata["explicit_fields"]
    assert from_metadata.metadata["run_seed"] == 3


def test_parser_extracts_hit_to_lead_optimization():
    task = parse_task(
        "Optimize hit_smiles=CCO for synthesizability and toxicity, num_variants=25.",
    )

    assert task.task_type == "hit_to_lead_optimization"
    assert task.input_smiles == ["CCO"]
    assert task.constraints.max_candidates == 25
    assert task.constraints.require_synthesizability is True
    assert task.constraints.require_toxicity is True


def test_parser_extracts_docking_evaluation_inputs():
    task = RuleBasedTaskParser().parse(
        "Dock receptor=agent_workspace/1HVR.pdb ligand_path=agent_workspace/ligands/lig_0.pdb "
        "center=[4,5,6] box_size=[20,20,22]."
    )

    assert task.task_type == "docking_evaluation"
    assert task.protein_path == "agent_workspace/1HVR.pdb"
    assert task.ligand_path == "agent_workspace/ligands/lig_0.pdb"
    assert task.pocket_center == [4.0, 5.0, 6.0]
    assert task.box_size == [20.0, 20.0, 22.0]
    assert task.constraints.require_docking is True


def test_parser_extracts_multi_objective_screening_and_smiles_list():
    task = parse_task("Rank smiles_list=[CCO, CCC] by docking, scscore, toxicity, and pMIC.")

    assert task.task_type == "multi_objective_screening"
    assert task.input_smiles == ["CCO", "CCC"]
    assert task.constraints.require_ranking is True
    assert task.constraints.require_synthesizability is True
    assert task.constraints.require_toxicity is True
    assert task.objectives == ["binding", "synthesizability", "toxicity", "bioactivity"]


def test_parser_extracts_scaffold_conditioned_generation():
    task = parse_task("Decorate scaffold_smiles=c1cc([*])ccc1 for synthesizability.")

    assert task.task_type == "scaffold_conditioned_generation"
    assert task.input_smiles == ["c1cc([*])ccc1"]
    assert task.constraints.require_synthesizability is True


def test_parser_falls_back_to_unknown_conservatively():
    task = parse_task("Can you think about this project direction?")

    assert task.task_type == "unknown"
    assert task.input_smiles == []
    assert task.metadata["parser_type"] == "rule_based"
