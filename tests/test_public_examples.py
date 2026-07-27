from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_minimal_mock_example_runs_end_to_end(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.minimal_mock",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["task_success"] is True
    assert payload["candidate_count"] == 2
    assert payload["selected_tools"] == [
        "reinvent4_denovo",
        "scscore",
        "toxicity",
    ]
    assert Path(payload["trace_path"]).exists()


def test_custom_tool_adapter_example_runs_end_to_end():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.custom_tool_adapter"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["task_success"] is True
    assert payload["candidate_smiles"] == ["CCO", "CCN"]
