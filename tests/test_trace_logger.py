from __future__ import annotations

import json

from egvr.task_schema import CandidateRecord, ParsedTask, PlannedWorkflow, ToolCallRecord, VerifierResult
from egvr.trace_logger import JSONLTraceLogger, build_trace_payload


def test_build_trace_payload_has_required_fields():
    task = ParsedTask(task_id="task", raw_user_query="generate", task_type="de_novo_generation")
    workflow = PlannedWorkflow(task_id="task", planner_type="rule_based", selected_tools=["reinvent4_denovo"])
    calls = [ToolCallRecord(tool_name="reinvent4_denovo", success=True)]
    candidates = [CandidateRecord(smiles="CCO")]
    verifier = VerifierResult(success=True)

    payload = build_trace_payload(
        parsed_task=task,
        planned_workflow=workflow,
        tool_calls=calls,
        candidate_records=candidates,
        verifier_result=verifier,
        trace_id="trace-1",
        timestamp="2026-05-27T00:00:00+00:00",
    )

    assert payload["trace_id"] == "trace-1"
    assert payload["generated_smiles"] == ["CCO"]
    assert payload["task_success"] is True


def test_jsonl_trace_logger_writes_one_line(tmp_path):
    logger = JSONLTraceLogger(log_dir=tmp_path)
    task = ParsedTask(task_id="task", raw_user_query="generate", task_type="de_novo_generation")
    workflow = PlannedWorkflow(task_id="task", planner_type="rule_based", selected_tools=["reinvent4_denovo"])
    calls = [ToolCallRecord(tool_name="reinvent4_denovo", success=True)]
    candidates = [CandidateRecord(smiles="CCO")]
    verifier = VerifierResult(success=True)

    path = logger.log_trace(
        parsed_task=task,
        planned_workflow=workflow,
        tool_calls=calls,
        candidate_records=candidates,
        verifier_result=verifier,
        trace_id="trace-1",
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["final_candidates"][0]["smiles"] == "CCO"
