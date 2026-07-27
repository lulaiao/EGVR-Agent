"""Validate locked Gemini planner artifacts without exposing credentials."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def validate_gemini_artifacts(
    *,
    summary_path: str | Path,
    response_log_path: str | Path,
    dotenv_path: str | Path,
    expected_task_count: int,
    expected_model: str = "gemini-2.5-pro",
    expected_reasoning_effort: str = "medium",
    expected_temperature: float = 0.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    summary_file = Path(summary_path)
    response_file = Path(response_log_path)
    dotenv_file = Path(dotenv_path)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    task_results = [row for row in summary.get("task_results", []) if isinstance(row, dict)]
    response_rows = _read_jsonl(response_file)
    task_ids = [str(row.get("task_id")) for row in task_results]
    errors: list[str] = []

    _expect(len(task_results) == expected_task_count, "unexpected_summary_task_count", errors)
    _expect(len(set(task_ids)) == expected_task_count, "task_ids_not_unique_or_missing", errors)
    _expect(len(response_rows) == expected_task_count, "unexpected_response_log_count", errors)

    model_match = 0
    temperature_match = 0
    reasoning_match = 0
    usage_coverage = 0
    hash_coverage = 0
    raw_response_coverage = 0
    recorded_at_coverage = 0
    for row in task_results:
        metadata = row.get("api_metadata") or {}
        request = metadata.get("request_payload") or {}
        if metadata.get("api_model") == expected_model and request.get("model") == expected_model:
            model_match += 1
        if request.get("temperature") == expected_temperature:
            temperature_match += 1
        if request.get("reasoning_effort") == expected_reasoning_effort:
            reasoning_match += 1
        if metadata.get("usage") is not None:
            usage_coverage += 1
        if all(row.get(name) for name in ("prompt_hash", "request_hash", "plan_hash")):
            hash_coverage += 1
        if row.get("raw_response"):
            raw_response_coverage += 1
        if row.get("api_response_recorded_at"):
            recorded_at_coverage += 1

    for count, name in (
        (model_match, "model_id_mismatch"),
        (temperature_match, "temperature_mismatch"),
        (reasoning_match, "reasoning_effort_mismatch"),
        (usage_coverage, "usage_missing"),
        (hash_coverage, "provenance_hash_missing"),
        (raw_response_coverage, "raw_response_missing"),
        (recorded_at_coverage, "timestamp_missing"),
    ):
        _expect(count == expected_task_count, name, errors)

    secret = _dotenv_value(dotenv_file, "GEMINI_API_KEY")
    artifact_text = summary_file.read_text(encoding="utf-8") + response_file.read_text(encoding="utf-8")
    secret_present = bool(secret and secret in artifact_text)
    _expect(not secret_present, "api_key_present_in_artifacts", errors)

    payload = {
        "audit_id": "gemini_planner_artifact_validation_v1",
        "recorded_at": _utc_timestamp(),
        "summary_path": str(summary_file),
        "response_log_path": str(response_file),
        "expected_task_count": expected_task_count,
        "summary_task_count": len(task_results),
        "response_log_count": len(response_rows),
        "unique_task_id_count": len(set(task_ids)),
        "model_match_count": model_match,
        "temperature_match_count": temperature_match,
        "reasoning_effort_match_count": reasoning_match,
        "usage_coverage_count": usage_coverage,
        "provenance_hash_coverage_count": hash_coverage,
        "raw_response_coverage_count": raw_response_coverage,
        "timestamp_coverage_count": recorded_at_coverage,
        "api_key_present_in_artifacts": secret_present,
        "valid_json_count": (summary.get("row") or {}).get("valid_json_count"),
        "valid_schema_count": (summary.get("row") or {}).get("valid_schema_count"),
        "passed": not errors,
        "errors": errors,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dotenv_value(path: Path, name: str) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def _expect(condition: bool, error: str, errors: list[str]) -> None:
    if not condition:
        errors.append(error)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate locked Gemini planner artifacts.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--response-log", required=True)
    parser.add_argument("--dotenv-path", required=True)
    parser.add_argument("--expected-task-count", type=int, required=True)
    parser.add_argument("--expected-model", default="gemini-2.5-pro")
    parser.add_argument("--expected-reasoning-effort", default="medium")
    parser.add_argument("--expected-temperature", type=float, default=0.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = validate_gemini_artifacts(
        summary_path=args.summary,
        response_log_path=args.response_log,
        dotenv_path=args.dotenv_path,
        expected_task_count=args.expected_task_count,
        expected_model=args.expected_model,
        expected_reasoning_effort=args.expected_reasoning_effort,
        expected_temperature=args.expected_temperature,
        output_path=args.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

