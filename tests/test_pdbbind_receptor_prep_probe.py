from __future__ import annotations

import subprocess
from pathlib import Path

from CAi.toolkit.agent_planner.pdbbind_receptor_prep_probe import (
    classify_receptor_prep_failure,
    run_pdbbind_receptor_prep_probe,
)


def test_classify_histidine_template_ambiguity():
    failure = classify_receptor_prep_failure(
        "RuntimeError: for residue_key='A:12', 3 have passed: ['HIE', 'HID', 'HIP'] "
        "and tied for fewest missing and excess H: HIE HID"
    )

    assert failure["failure_type"] == "histidine_template_ambiguity"
    assert failure["failure_family"] == "residue_template"
    assert failure["residue_key"] == "A:12"
    assert failure["template_candidates"] == ["HIE", "HID"]


def test_receptor_prep_probe_records_success(tmp_path):
    root = tmp_path / "PDBbind_v2020"
    target = root / "extracted" / "v2020-refined_1" / "1abc"
    target.mkdir(parents=True)
    (target / "1abc_protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target / "1abc_ligand.sdf").write_text("sdf\n", encoding="utf-8")

    def runner(command, cwd, timeout_sec):
        output_basename = Path(command[command.index("-o") + 1])
        assert output_basename.is_absolute()
        output_basename.with_suffix(".pdbqt").write_text("RECEPTOR\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    payload = run_pdbbind_receptor_prep_probe(
        pdbbind_root=root,
        output_path=tmp_path / "summary.json",
        prepared_dir=tmp_path / "prepared",
        command_runner=runner,
    )

    assert payload["summary"]["total"] == 1
    assert payload["summary"]["prep_success_count"] == 1
    assert payload["summary"]["stable_target_ids"] == ["1abc"]
    assert payload["ready_targets"][0]["prepared_receptor_pdbqt_path"].endswith("1abc.pdbqt")
    assert (tmp_path / "summary.json").exists()


def test_receptor_prep_probe_records_template_required_failure(tmp_path):
    root = tmp_path / "PDBbind_v2020"
    target = root / "extracted" / "v2020-refined_1" / "1abc"
    target.mkdir(parents=True)
    (target / "1abc_protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target / "1abc_ligand.sdf").write_text("sdf\n", encoding="utf-8")

    stderr = (
        "RuntimeError: for residue_key='A:45', 3 have passed: ['HIE', 'HID', 'HIP'] "
        "and tied for fewest missing and excess H: HIE HID"
    )

    def runner(command, cwd, timeout_sec):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr)

    payload = run_pdbbind_receptor_prep_probe(
        pdbbind_root=root,
        output_path=tmp_path / "summary.json",
        prepared_dir=tmp_path / "prepared",
        command_runner=runner,
    )

    record = payload["records"][0]
    assert payload["summary"]["prep_success_count"] == 0
    assert payload["summary"]["template_required_count"] == 1
    assert payload["summary"]["template_required_target_ids"] == ["1abc"]
    assert record["failure_type"] == "histidine_template_ambiguity"
    assert record["residue_key"] == "A:45"
    assert record["template_candidates"] == ["HIE", "HID"]
    assert payload["template_required_targets"][0]["pdb_id"] == "1abc"


def test_receptor_prep_probe_records_missing_input_without_runner(tmp_path):
    root = tmp_path / "PDBbind_v2020"
    target = root / "extracted" / "v2020-refined_1" / "1abc"
    target.mkdir(parents=True)
    (target / "1abc_ligand.sdf").write_text("sdf\n", encoding="utf-8")

    payload = run_pdbbind_receptor_prep_probe(
        pdbbind_root=root,
        output_path=tmp_path / "summary.json",
        prepared_dir=tmp_path / "prepared",
        command_runner=lambda command, cwd, timeout_sec: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    assert payload["summary"]["total"] == 0
    assert payload["summary"]["prep_success_count"] == 0
