from __future__ import annotations

import subprocess
from pathlib import Path

from egvr.pdbbind_receptor_template_prep import run_pdbbind_receptor_template_prep


def test_template_prep_resolves_histidine_ambiguity(tmp_path):
    root = _make_target(tmp_path, "1abc")

    def runner(command, cwd, timeout_sec):
        if not _has_template_assignment(command):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=_his_error("A:12"))
        output_basename = Path(command[command.index("-o") + 1])
        assert output_basename.is_absolute()
        output_basename.with_suffix(".pdbqt").write_text("RECEPTOR\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    payload = run_pdbbind_receptor_template_prep(
        pdbbind_root=root,
        output_path=tmp_path / "summary.json",
        prepared_dir=tmp_path / "prepared",
        max_template_attempts=3,
        command_runner=runner,
    )

    record = payload["records"][0]
    assert payload["summary"]["prep_success_count"] == 1
    assert payload["summary"]["stable_target_ids"] == ["1abc"]
    assert record["template_assignments"] == {"A:12": "HIE"}
    assert record["attempt_count"] == 2
    assert record["prepared_receptor_pdbqt_path"].endswith("1abc.pdbqt")


def test_template_prep_records_unresolved_failure(tmp_path):
    root = _make_target(tmp_path, "1abc")

    def runner(command, cwd, timeout_sec):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=_his_error("A:12"))

    payload = run_pdbbind_receptor_template_prep(
        pdbbind_root=root,
        output_path=tmp_path / "summary.json",
        prepared_dir=tmp_path / "prepared",
        max_template_attempts=2,
        command_runner=runner,
    )

    record = payload["records"][0]
    assert payload["summary"]["prep_success_count"] == 0
    assert payload["summary"]["failure_counts"] == {"histidine_template_ambiguity": 1}
    assert record["final_failure_type"] == "histidine_template_ambiguity"
    assert record["unresolved_residue_key"] == "A:12"
    assert record["template_assignments"] == {"A:12": "HIE"}


def test_template_prep_prefer_hid_policy(tmp_path):
    root = _make_target(tmp_path, "1abc")

    def runner(command, cwd, timeout_sec):
        if not _has_template_assignment(command):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=_his_error("A:12"))
        output_basename = Path(command[command.index("-o") + 1])
        assert output_basename.is_absolute()
        output_basename.with_suffix(".pdbqt").write_text("RECEPTOR\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    payload = run_pdbbind_receptor_template_prep(
        pdbbind_root=root,
        output_path=tmp_path / "summary.json",
        prepared_dir=tmp_path / "prepared",
        max_template_attempts=3,
        template_policy="prefer_hid",
        command_runner=runner,
    )

    assert payload["records"][0]["template_assignments"] == {"A:12": "HID"}


def test_template_prep_filters_target_ids(tmp_path):
    root = _make_target(tmp_path, "1abc")
    _make_target(tmp_path, "2def")

    def runner(command, cwd, timeout_sec):
        output_basename = Path(command[command.index("-o") + 1])
        assert output_basename.is_absolute()
        output_basename.with_suffix(".pdbqt").write_text("RECEPTOR\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    payload = run_pdbbind_receptor_template_prep(
        pdbbind_root=root,
        output_path=tmp_path / "summary.json",
        prepared_dir=tmp_path / "prepared",
        target_ids=["2DEF"],
        command_runner=runner,
    )

    assert payload["summary"]["total"] == 1
    assert payload["summary"]["stable_target_ids"] == ["2def"]


def _make_target(tmp_path, pdb_id: str):
    root = tmp_path / "PDBbind_v2020"
    target = root / "extracted" / "v2020-refined_1" / pdb_id
    target.mkdir(parents=True)
    (target / f"{pdb_id}_protein.pdb").write_text("ATOM\n", encoding="utf-8")
    (target / f"{pdb_id}_ligand.sdf").write_text("sdf\n", encoding="utf-8")
    return root


def _his_error(residue_key: str) -> str:
    return (
        f"RuntimeError: for residue_key='{residue_key}', 3 have passed: ['HIE', 'HID', 'HIP'] "
        "and tied for fewest missing and excess H: HIE HID"
    )


def _has_template_assignment(command) -> bool:
    script_index = command.index("mk_prepare_receptor.py")
    return "-n" in command[script_index + 1 :]
