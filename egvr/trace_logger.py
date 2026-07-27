"""JSONL trace logging for chemistry agent workflows."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .task_schema import CandidateRecord, ParsedTask, PlannedWorkflow, ToolCallRecord, VerifierResult


class JSONLTraceLogger:
    """Append compact task execution traces to daily JSONL files."""

    def __init__(self, log_dir: str | Path = "logs/copilot_traces") -> None:
        self.log_dir = Path(log_dir)

    def log_trace(
        self,
        *,
        parsed_task: ParsedTask,
        planned_workflow: PlannedWorkflow,
        tool_calls: list[ToolCallRecord],
        candidate_records: list[CandidateRecord],
        verifier_result: VerifierResult,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc)
        payload = build_trace_payload(
            parsed_task=parsed_task,
            planned_workflow=planned_workflow,
            tool_calls=tool_calls,
            candidate_records=candidate_records,
            verifier_result=verifier_result,
            trace_id=trace_id or f"trace_{uuid.uuid4().hex[:12]}",
            timestamp=timestamp.isoformat(),
            metadata=metadata or {},
        )
        path = self.log_dir / f"{timestamp:%Y%m%d}_traces.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return path


def build_trace_payload(
    *,
    parsed_task: ParsedTask,
    planned_workflow: PlannedWorkflow,
    tool_calls: list[ToolCallRecord],
    candidate_records: list[CandidateRecord],
    verifier_result: VerifierResult,
    trace_id: str,
    timestamp: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_smiles = [candidate.smiles for candidate in candidate_records if candidate.smiles]
    return {
        "trace_id": trace_id,
        "task_id": parsed_task.task_id,
        "timestamp": timestamp,
        "raw_user_query": parsed_task.raw_user_query,
        "parsed_task": parsed_task.to_dict(),
        "selected_tools": planned_workflow.selected_tools,
        "tool_sequence": [step.to_dict() for step in planned_workflow.tool_sequence],
        "tool_calls": [record.to_dict() for record in tool_calls],
        "generated_smiles": generated_smiles,
        "final_candidates": [candidate.to_dict() for candidate in candidate_records],
        "verifier_result": verifier_result.to_dict(),
        "task_success": verifier_result.success,
        "failure_reason": verifier_result.failure_reason,
        "metadata": metadata or {},
    }
