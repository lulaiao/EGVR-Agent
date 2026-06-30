"""Chemistry-grounded verification for planned workflow outputs."""

from __future__ import annotations

from collections.abc import Iterable

from .task_schema import CandidateRecord, ParsedTask, PlannedWorkflow, ToolCallRecord, VerifierResult


def verify_workflow(
    parsed_task: ParsedTask,
    planned_workflow: PlannedWorkflow,
    tool_call_records: Iterable[ToolCallRecord],
    candidate_records: Iterable[CandidateRecord],
) -> VerifierResult:
    """Verify task completion using structured execution artifacts."""

    return WorkflowVerifier().verify(parsed_task, planned_workflow, tool_call_records, candidate_records)


class WorkflowVerifier:
    """Deterministic verifier for baseline experiments."""

    def verify(
        self,
        parsed_task: ParsedTask,
        planned_workflow: PlannedWorkflow,
        tool_call_records: Iterable[ToolCallRecord],
        candidate_records: Iterable[CandidateRecord],
    ) -> VerifierResult:
        tool_calls = list(tool_call_records)
        candidates = list(candidate_records)
        completion_candidates = _completion_candidates(planned_workflow, candidates)
        valid_smiles = [
            candidate.smiles
            for candidate in completion_candidates
            if candidate.smiles and candidate.is_valid
        ]
        unique_smiles = set(valid_smiles)
        needs_candidates = _needs_candidates(parsed_task, planned_workflow)

        checks = {
            "has_tool_success": any(record.success and not record.metadata.get("skipped") for record in tool_calls),
            "has_valid_smiles": bool(valid_smiles),
            "has_unique_molecules": bool(valid_smiles) and len(unique_smiles) == len(valid_smiles),
            "has_docking_scores": _has_docking_scores(tool_calls, candidates),
            "passes_synthesizability": _passes_synthesizability(parsed_task, candidates),
            "passes_toxicity": _passes_toxicity(parsed_task, candidates),
            "has_sa_score_evidence": _has_sa_score_evidence(candidates),
            "has_posebusters_evidence": _has_posebusters_evidence(candidates),
            "has_ranked_output": _has_ranked_output(parsed_task, planned_workflow, candidates),
        }
        required = ["has_tool_success"]
        if needs_candidates:
            required.extend(["has_valid_smiles", "has_unique_molecules"])
        if _needs_docking(parsed_task, planned_workflow):
            required.append("has_docking_scores")
        if parsed_task.constraints.require_synthesizability or "synthesizability" in parsed_task.objectives:
            required.append("passes_synthesizability")
        if parsed_task.constraints.require_toxicity or "toxicity" in parsed_task.objectives:
            required.append("passes_toxicity")
        if parsed_task.constraints.require_ranking or "ranked_candidates" in planned_workflow.expected_outputs:
            required.append("has_ranked_output")

        failed = [check for check in required if not checks.get(check)]
        metrics = {
            "candidate_count": len(candidates),
            "valid_smiles_count": len(valid_smiles),
            "unique_smiles_count": len(unique_smiles),
            "tool_call_count": len(tool_calls),
            "successful_tool_call_count": sum(1 for record in tool_calls if record.success),
            "best_docking_score": _min_or_none(candidate.docking_score for candidate in candidates),
            "best_scscore": _min_or_none(candidate.scscore for candidate in candidates),
            "best_sa_score": _min_or_none(candidate.sa_score for candidate in candidates),
            "sa_score_coverage": _coverage(candidate.sa_score is not None for candidate in candidates),
            "max_toxicity_score": _max_or_none(candidate.toxicity_score for candidate in candidates),
            "posebusters_pass_rate": _posebusters_pass_rate(candidates),
            "posebusters_coverage": _coverage(candidate.posebusters_pass is not None for candidate in candidates),
            "rdkit_property_coverage": _coverage(_rdkit_properties(candidate) is not None for candidate in candidates),
            "mean_qed": _mean_or_none(_property_number(candidate, "qed") for candidate in candidates),
            "mean_logp": _mean_or_none(_property_number(candidate, "logp") for candidate in candidates),
            "lipinski_pass_rate": _coverage(_property_bool(candidate, "lipinski_pass") for candidate in candidates),
            "pains_flag_rate": _coverage(_has_property_flags(candidate, "pains_flags") for candidate in candidates),
        }
        return VerifierResult(
            success=not failed,
            checks=checks,
            metrics=metrics,
            failure_reason=", ".join(failed) if failed else None,
        )


def _completion_candidates(workflow: PlannedWorkflow, candidates: list[CandidateRecord]) -> list[CandidateRecord]:
    if "generated_smiles" not in workflow.expected_outputs:
        return candidates
    generated = [candidate for candidate in candidates if candidate.source_tool and candidate.source_tool != "input"]
    return generated


def _needs_candidates(parsed_task: ParsedTask, workflow: PlannedWorkflow) -> bool:
    if parsed_task.task_type == "docking_evaluation" and "generated_smiles" not in workflow.expected_outputs:
        return False
    return parsed_task.task_type != "unknown"


def _needs_docking(parsed_task: ParsedTask, workflow: PlannedWorkflow) -> bool:
    return (
        parsed_task.constraints.require_docking
        or "binding" in parsed_task.objectives
        or any(step.tool_name == "vina" for step in workflow.tool_sequence)
    )


def _has_docking_scores(tool_calls: list[ToolCallRecord], candidates: list[CandidateRecord]) -> bool:
    if any(candidate.docking_score is not None for candidate in candidates):
        return True
    for record in tool_calls:
        if record.tool_name != "vina" or not record.success or not record.outputs:
            continue
        if record.outputs.get("best_docking_score_kcal_mol") is not None or record.outputs.get("best_docking_score") is not None:
            return True
    return False


def _passes_synthesizability(parsed_task: ParsedTask, candidates: list[CandidateRecord]) -> bool:
    if not (parsed_task.constraints.require_synthesizability or "synthesizability" in parsed_task.objectives):
        return True
    scored = [candidate.scscore for candidate in candidates if candidate.scscore is not None]
    sa_scored = [candidate.sa_score for candidate in candidates if candidate.sa_score is not None]
    if not scored and not sa_scored:
        return False
    threshold = parsed_task.constraints.max_scscore
    if threshold is not None and scored:
        return any(score <= threshold for score in scored)
    return True


def _has_sa_score_evidence(candidates: list[CandidateRecord]) -> bool:
    return any(candidate.sa_score is not None for candidate in candidates)


def _has_posebusters_evidence(candidates: list[CandidateRecord]) -> bool:
    return any(candidate.posebusters_pass is not None for candidate in candidates)


def _passes_toxicity(parsed_task: ParsedTask, candidates: list[CandidateRecord]) -> bool:
    if not (parsed_task.constraints.require_toxicity or "toxicity" in parsed_task.objectives):
        return True
    scored = [candidate.toxicity_score for candidate in candidates if candidate.toxicity_score is not None]
    if not scored:
        return False
    threshold = parsed_task.constraints.max_toxicity_score
    return any(score <= threshold for score in scored) if threshold is not None else True


def _has_ranked_output(parsed_task: ParsedTask, workflow: PlannedWorkflow, candidates: list[CandidateRecord]) -> bool:
    if not (parsed_task.constraints.require_ranking or "ranked_candidates" in workflow.expected_outputs):
        return True
    return bool(candidates) and all(candidate.rank is not None for candidate in candidates)


def _min_or_none(values):
    clean = [value for value in values if value is not None]
    return min(clean) if clean else None


def _max_or_none(values):
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _mean_or_none(values):
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _coverage(values) -> float | None:
    flags = list(values)
    if not flags:
        return None
    return sum(1 for value in flags if value) / len(flags)


def _posebusters_pass_rate(candidates: list[CandidateRecord]) -> float | None:
    scored = [candidate.posebusters_pass for candidate in candidates if candidate.posebusters_pass is not None]
    if not scored:
        return None
    return sum(1 for value in scored if value) / len(scored)


def _rdkit_properties(candidate: CandidateRecord) -> dict | None:
    properties = candidate.metadata.get("rdkit_properties")
    return properties if isinstance(properties, dict) else None


def _property_number(candidate: CandidateRecord, key: str) -> float | None:
    properties = _rdkit_properties(candidate)
    if not properties:
        return None
    value = properties.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _property_bool(candidate: CandidateRecord, key: str) -> bool | None:
    properties = _rdkit_properties(candidate)
    if not properties or key not in properties:
        return None
    value = properties.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "pass", "passed"}:
        return True
    if normalized in {"false", "0", "no", "fail", "failed"}:
        return False
    return None


def _has_property_flags(candidate: CandidateRecord, key: str) -> bool | None:
    properties = _rdkit_properties(candidate)
    if not properties or key not in properties:
        return None
    value = properties.get(key)
    if value is None:
        return None
    if isinstance(value, list | tuple | set):
        return bool(value)
    return bool(value)
