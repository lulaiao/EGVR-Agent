from __future__ import annotations

import json

from egvr.gemini_planner_artifact_validator import (
    validate_gemini_artifacts,
)


def test_validator_checks_locked_provenance_and_secret_absence(tmp_path):
    summary = tmp_path / "summary.json"
    responses = tmp_path / "responses.jsonl"
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=not-in-artifacts\n", encoding="utf-8")
    task_result = {
        "task_id": "task_1",
        "raw_response": "{}",
        "prompt_hash": "p",
        "request_hash": "r",
        "plan_hash": "w",
        "api_response_recorded_at": "2026-07-25T00:00:00Z",
        "api_metadata": {
            "api_model": "gemini-2.5-pro",
            "usage": {"total_tokens": 3},
            "request_payload": {
                "model": "gemini-2.5-pro",
                "temperature": 0.0,
                "reasoning_effort": "medium",
            },
        },
    }
    summary.write_text(
        json.dumps(
            {
                "row": {"valid_json_count": 1, "valid_schema_count": 1},
                "task_results": [task_result],
            }
        ),
        encoding="utf-8",
    )
    responses.write_text(
        json.dumps({"task_id": "task_1", "raw_response": "{}", "plan_hash": "w"}) + "\n",
        encoding="utf-8",
    )

    payload = validate_gemini_artifacts(
        summary_path=summary,
        response_log_path=responses,
        dotenv_path=dotenv,
        expected_task_count=1,
    )

    assert payload["passed"] is True
    assert payload["api_key_present_in_artifacts"] is False

