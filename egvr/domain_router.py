"""Domain routing for molecular and offline biomedical planning slices."""

from __future__ import annotations

from typing import Any

from .biomedical_offline import (
    detect_biomedical_domain,
    execute_biomedical_offline,
    parse_biomedical_task,
    plan_biomedical_workflow,
    verify_biomedical_workflow,
)
from .executor import WorkflowExecutor
from .rule_planner import plan_workflow
from .task_parser import parse_task
from .task_schema import CandidateRecord, ParsedTask, PlannedWorkflow, ToolCallRecord, VerifierResult
from .verifier import verify_workflow


MOLECULAR_DOMAIN = "molecular"


def route_domain(raw_user_query: str, metadata: dict[str, Any] | None = None) -> str:
    """Route a user request to molecular or offline biomedical planning."""

    return detect_biomedical_domain(raw_user_query, metadata) or MOLECULAR_DOMAIN


def parse_domain_task(
    raw_user_query: str,
    *,
    task_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParsedTask:
    """Parse a task with the domain-specific parser selected by route_domain."""

    domain = route_domain(raw_user_query, metadata)
    if domain == MOLECULAR_DOMAIN:
        parsed = parse_task(raw_user_query, task_id=task_id, metadata=metadata)
        parsed.metadata.setdefault("domain", MOLECULAR_DOMAIN)
        return parsed
    return parse_biomedical_task(raw_user_query, task_id=task_id, metadata=metadata)


def plan_domain_workflow(parsed_task: ParsedTask) -> PlannedWorkflow:
    """Plan a workflow for a parsed molecular or biomedical task."""

    if parsed_task.metadata.get("domain") == MOLECULAR_DOMAIN:
        return plan_workflow(parsed_task)
    return plan_biomedical_workflow(parsed_task)


def execute_and_verify_domain(
    parsed_task: ParsedTask,
    workflow: PlannedWorkflow,
    *,
    executor: WorkflowExecutor | None = None,
) -> tuple[list[ToolCallRecord], list[CandidateRecord], VerifierResult]:
    """Execute and verify a workflow using domain-appropriate evidence records."""

    if parsed_task.metadata.get("domain") == MOLECULAR_DOMAIN:
        active_executor = executor or WorkflowExecutor()
        tool_calls, candidates = active_executor.execute(parsed_task, workflow)
        return tool_calls, candidates, verify_workflow(parsed_task, workflow, tool_calls, candidates)

    tool_calls, evidence = execute_biomedical_offline(parsed_task, workflow)
    verifier_result = verify_biomedical_workflow(parsed_task, workflow, tool_calls, evidence)
    return tool_calls, [], verifier_result
