from __future__ import annotations

import json
from pathlib import Path


MANIFEST_PATH = Path("CAi/toolkit/agent_planner/configs/paper_experiment_manifest.json")
ALLOWED_BASELINES = {
    "all_tool_agent",
    "fixed_pipeline",
    "rule_based_planner",
    "full_copilot",
    "scheduled_fallback_no_verifier",
    "verifier_only_no_repair",
}


def test_public_paper_experiment_manifest_is_portable():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert payload["manifest_id"] == "paper_experiment_manifest_v1"
    assert payload["project"] == "FullCopilot"
    assert payload["scope"]["public_release"] is True
    assert set(payload["baselines"]) == ALLOWED_BASELINES
    assert payload["experiment_matrix"]

    experiment_ids = [item["experiment_id"] for item in payload["experiment_matrix"]]
    assert len(experiment_ids) == len(set(experiment_ids))
    assert set(payload["recommended_next_run_order"]).issubset(experiment_ids)

    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    forbidden_fragments = [
        "/" + "data/ssd1/lla",
        "/" + "home/lula",
        "CAi" + "_copilot",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in manifest_text

    for experiment in payload["experiment_matrix"]:
        assert experiment["status"] == "ready_to_run"
        assert experiment["execution_mode"] == "mock"
        assert set(experiment["baselines"]).issubset(ALLOWED_BASELINES)
        assert Path(experiment["benchmark_path"]).exists()
        assert experiment["primary_metrics"]
        _assert_copyable_command(experiment.get("run_command"))


def _assert_copyable_command(command: str | None) -> None:
    assert command
    assert not command.startswith("$")
    assert "set -euo pipefail" not in command
