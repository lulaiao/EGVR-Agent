from __future__ import annotations

from CAi.toolkit.agent_planner.pdbbind_readiness_probe import run_pdbbind_readiness_probe


def test_pdbbind_readiness_probe_detects_ready_candidate(tmp_path):
    root = tmp_path / "PDBbind_v2020_refined"
    target = root / "1abc"
    target.mkdir(parents=True)
    (root / "INDEX_refined_data.2020").write_text("index\n", encoding="utf-8")
    (target / "1abc_protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target / "1abc_ligand.sdf").write_text("mol\n", encoding="utf-8")

    payload = run_pdbbind_readiness_probe(search_roots=[tmp_path], output_path=tmp_path / "summary.json")

    assert payload["status"] == "ready"
    assert payload["best_candidate"]["ready_target_count"] == 1
    assert payload["best_candidate"]["index_file_count"] == 1
    assert payload["best_candidate"]["sample_ready_targets"] == ["1abc"]
    assert (tmp_path / "summary.json").exists()


def test_pdbbind_readiness_probe_detects_split_index_and_extracted_targets(tmp_path):
    root = tmp_path / "PDBbind_v2020"
    index_dir = root / "index"
    target = root / "extracted" / "v2020-refined_1" / "1abc"
    index_dir.mkdir(parents=True)
    target.mkdir(parents=True)
    (index_dir / "INDEX_refined_data.2020").write_text("index\n", encoding="utf-8")
    (target / "1abc_protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target / "1abc_ligand.sdf").write_text("mol\n", encoding="utf-8")

    payload = run_pdbbind_readiness_probe(search_roots=[root], output_path=tmp_path / "summary.json")

    assert payload["status"] == "ready"
    assert payload["best_candidate"]["root"] == str(root)
    assert payload["best_candidate"]["ready_target_count"] == 1
    assert payload["best_candidate"]["index_file_count"] == 1


def test_pdbbind_readiness_probe_marks_missing_data_not_found(tmp_path):
    payload = run_pdbbind_readiness_probe(search_roots=[tmp_path], output_path=tmp_path / "summary.json")

    assert payload["status"] == "not_found"
    assert payload["best_candidate"] is None
