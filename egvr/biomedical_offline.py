"""Offline biomedical task planning and verification.

These helpers intentionally do not call external clinical or DTI prediction
systems. They provide lightweight generalization slices for testing whether the
EGVR-Agent planning/verification abstractions transfer beyond molecular design
without making clinical or DTI performance claims.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .biomedical_schema import EvidenceRecord, evidence_records_to_dicts
from .task_schema import ParsedTask, PlannedToolCall, PlannedWorkflow, ToolCallRecord, VerifierResult


CLINICAL_DOMAIN = "clinical_trial"
DRUG_TARGET_DOMAIN = "drug_target"
CLINICAL_TASK_TYPE = "clinical_trial_outcome_prediction"
CLINICAL_PREDICTION_TASK_TYPE = "clinical_trial_prediction"
DRUG_TARGET_TASK_TYPE = "drug_target_evidence"

CLINICAL_TOOLS = (
    "clinical_trial_metadata_parser",
    "eligibility_evidence_checker",
    "trial_outcome_evidence_checker",
)
CLINICAL_PREDICTION_TOOLS = (
    "clinical_trial_metadata_parser",
    "clinicalagent_evidence_retriever",
    "clinicalagent_enrollment_predictor",
    "clinicalagent_drug_risk_checker",
    "clinicalagent_disease_risk_checker",
    "clinicalagent_outcome_predictor",
)
DRUG_TARGET_TOOLS = (
    "drug_target_evidence_checker",
    "repurposing_evidence_checker",
)

EVIDENCE_TYPE_TO_FAMILY = {
    "clinical_trial_id": "trial_metadata",
    "clinical_phase": "trial_metadata",
    "clinical_condition": "trial_metadata",
    "clinical_intervention": "trial_metadata",
    "clinical_endpoint": "trial_metadata",
    "clinical_eligibility": "eligibility",
    "clinical_enrollment": "eligibility",
    "clinical_outcome": "outcome",
    "clinical_outcome_provenance": "outcome",
    "external_knowledge_evidence": "external_provenance",
    "enrollment_evidence": "enrollment_prediction",
    "drug_risk_evidence": "risk_evidence",
    "disease_risk_evidence": "risk_evidence",
    "clinical_outcome_prediction": "prediction",
    "clinical_prediction_confidence": "prediction",
    "drug_name": "drug_target_identity",
    "target_name": "drug_target_identity",
    "mechanism_evidence": "mechanism",
    "kg_provenance": "provenance",
    "literature_provenance": "provenance",
    "disease_indication": "repurposing",
    "repurposing_rationale": "repurposing",
}

_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
_KNOWN_FIELD_NAMES = (
    "phase",
    "trial_phase",
    "condition",
    "disease",
    "intervention",
    "drug",
    "endpoint",
    "primary_endpoint",
    "eligibility",
    "criteria",
    "enrollment",
    "sample_size",
    "outcome",
    "label",
    "outcome_source",
    "source",
    "compound",
    "target",
    "protein",
    "mechanism",
    "moa",
    "kg_source",
    "kg",
    "knowledge_graph",
    "literature_source",
    "paper",
    "literature",
    "rationale",
    "repurposing_rationale",
    "indication",
)


def detect_biomedical_domain(raw_user_query: str, metadata: dict[str, Any] | None = None) -> str | None:
    """Return a biomedical domain label when a query is clearly non-molecular."""

    explicit = (metadata or {}).get("domain")
    if explicit in {CLINICAL_DOMAIN, DRUG_TARGET_DOMAIN}:
        return str(explicit)
    lower = (raw_user_query or "").lower()
    if _NCT_RE.search(raw_user_query or "") or any(
        token in lower for token in ("clinical trial", "trial outcome", "eligibility", "enrollment", "phase ii")
    ):
        return CLINICAL_DOMAIN
    if any(
        token in lower
        for token in (
            "drug-target",
            "drug target",
            "dti",
            "repurposing",
            "target evidence",
            "disease indication",
        )
    ):
        return DRUG_TARGET_DOMAIN
    return None


def parse_biomedical_task(
    raw_user_query: str,
    *,
    task_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParsedTask:
    """Parse a clinical-trial or drug-target evidence task into ParsedTask."""

    query = raw_user_query or ""
    domain = detect_biomedical_domain(query, metadata) or CLINICAL_DOMAIN
    parsed_metadata = dict(metadata or {})
    parsed_metadata["domain"] = domain
    parsed_metadata["parser_type"] = "biomedical_offline_rule_based"

    if domain == CLINICAL_DOMAIN:
        parsed_metadata.update(_extract_clinical_entities(query, parsed_metadata))
        if _is_clinical_prediction_task(query, parsed_metadata):
            task_type = CLINICAL_PREDICTION_TASK_TYPE
            objectives = ["clinical_trial_prediction", "evidence_completeness"]
        else:
            task_type = CLINICAL_TASK_TYPE
            objectives = ["clinical_outcome_evidence", "evidence_completeness"]
    else:
        parsed_metadata.update(_extract_drug_target_entities(query, parsed_metadata))
        task_type = DRUG_TARGET_TASK_TYPE
        objectives = ["drug_target_evidence", "provenance"]
        if parsed_metadata.get("disease"):
            objectives.append("repurposing_evidence")

    return ParsedTask(
        task_id=task_id or _stable_task_id(query, prefix=domain),
        raw_user_query=query,
        task_type=task_type,
        objectives=objectives,
        metadata=parsed_metadata,
    )


def plan_biomedical_workflow(parsed_task: ParsedTask) -> PlannedWorkflow:
    """Build a deterministic offline evidence workflow for a biomedical task."""

    domain = parsed_task.metadata.get("domain")
    if parsed_task.task_type == CLINICAL_PREDICTION_TASK_TYPE:
        steps = [
            PlannedToolCall(
                tool_name="clinical_trial_metadata_parser",
                reason="Extract trial identifiers and structured trial metadata before invoking the clinical backend.",
                expected_outputs=["trial_metadata_evidence"],
            ),
            PlannedToolCall(
                tool_name="clinicalagent_evidence_retriever",
                reason="Retrieve external clinical evidence and provenance used by the prediction backend.",
                expected_outputs=["external_knowledge_evidence"],
            ),
            PlannedToolCall(
                tool_name="clinicalagent_enrollment_predictor",
                reason="Estimate enrollment feasibility evidence through the clinical backend.",
                expected_outputs=["enrollment_evidence"],
            ),
            PlannedToolCall(
                tool_name="clinicalagent_drug_risk_checker",
                reason="Retrieve drug-level historical risk evidence used by the clinical backend.",
                expected_outputs=["drug_risk_evidence"],
            ),
            PlannedToolCall(
                tool_name="clinicalagent_disease_risk_checker",
                reason="Retrieve disease-level historical risk evidence used by the clinical backend.",
                expected_outputs=["disease_risk_evidence"],
            ),
            PlannedToolCall(
                tool_name="clinicalagent_outcome_predictor",
                reason="Invoke the clinical prediction backend and record prediction evidence.",
                expected_outputs=["clinical_outcome_prediction", "clinical_prediction_confidence"],
            ),
        ]
        return PlannedWorkflow(
            task_id=parsed_task.task_id,
            planner_type="clinicalagent_backend_rule_planner",
            selected_tools=[step.tool_name for step in steps],
            tool_sequence=steps,
            expected_outputs=["clinical_prediction_evidence_records", "verifier_result"],
            notes=["ClinicalAgent backend wrapper slice; no clinical prediction SOTA claim."],
        )

    if domain == CLINICAL_DOMAIN or parsed_task.task_type == CLINICAL_TASK_TYPE:
        steps = [
            PlannedToolCall(
                tool_name="clinical_trial_metadata_parser",
                reason="Extract trial identifiers, phase, condition, intervention, and endpoint evidence.",
                expected_outputs=["trial_metadata_evidence"],
            ),
            PlannedToolCall(
                tool_name="eligibility_evidence_checker",
                reason="Check whether eligibility and enrollment evidence are present.",
                expected_outputs=["eligibility_evidence"],
            ),
            PlannedToolCall(
                tool_name="trial_outcome_evidence_checker",
                reason="Check whether outcome label/evidence and provenance are present.",
                expected_outputs=["outcome_evidence"],
            ),
        ]
        return PlannedWorkflow(
            task_id=parsed_task.task_id,
            planner_type="biomedical_offline_rule_planner",
            selected_tools=[step.tool_name for step in steps],
            tool_sequence=steps,
            expected_outputs=["clinical_evidence_records", "verifier_result"],
            notes=["Offline clinical-trial evidence generalization slice; no clinical performance claim."],
        )

    steps = [
        PlannedToolCall(
            tool_name="drug_target_evidence_checker",
            reason="Check drug, target, mechanism, and provenance evidence.",
            expected_outputs=["drug_target_evidence"],
        )
    ]
    if parsed_task.metadata.get("disease"):
        steps.append(
            PlannedToolCall(
                tool_name="repurposing_evidence_checker",
                reason="Check disease/indication evidence for repurposing-style tasks.",
                expected_outputs=["repurposing_evidence"],
            )
        )
    return PlannedWorkflow(
        task_id=parsed_task.task_id,
        planner_type="biomedical_offline_rule_planner",
        selected_tools=[step.tool_name for step in steps],
        tool_sequence=steps,
        expected_outputs=["drug_target_evidence_records", "verifier_result"],
        notes=["Offline drug-target evidence generalization slice; no DTI or repurposing SOTA claim."],
    )


def execute_biomedical_offline(
    parsed_task: ParsedTask,
    workflow: PlannedWorkflow,
) -> tuple[list[ToolCallRecord], list[EvidenceRecord]]:
    """Run deterministic offline evidence checkers for a biomedical workflow."""

    records: list[ToolCallRecord] = []
    evidence: list[EvidenceRecord] = []
    for step in workflow.tool_sequence:
        started = _now_iso()
        produced, missing = _run_offline_tool(parsed_task, step.tool_name)
        evidence.extend(produced)
        records.append(
            ToolCallRecord(
                tool_name=step.tool_name,
                action=step.action,
                inputs={"task_id": parsed_task.task_id, "metadata": dict(parsed_task.metadata)},
                outputs={
                    "evidence_records": evidence_records_to_dicts(produced),
                    "missing_fields": missing,
                    "offline": True,
                },
                success=True,
                started_at=started,
                finished_at=_now_iso(),
                elapsed_time_sec=0.0,
                metadata={"domain": parsed_task.metadata.get("domain"), "missing_fields": missing},
            )
        )
    return records, evidence


def verify_biomedical_workflow(
    parsed_task: ParsedTask,
    planned_workflow: PlannedWorkflow,
    tool_call_records: Iterable[ToolCallRecord],
    evidence_records: Iterable[EvidenceRecord],
) -> VerifierResult:
    """Verify biomedical completion from concrete evidence, not tool success alone."""

    tool_calls = list(tool_call_records)
    evidence = list(evidence_records)
    required = _required_evidence_types(parsed_task, planned_workflow)
    present = {record.evidence_type for record in evidence if record.is_complete()}
    missing = [item for item in required if item not in present]
    missing_reasons = _missing_reasons_by_type(evidence, missing)
    coverage = (len(required) - len(missing)) / len(required) if required else 1.0
    checks = {
        "has_tool_success": any(record.success for record in tool_calls),
        "has_required_evidence": not missing,
        "has_provenance": all(record.has_provenance() for record in evidence if record.has_value()),
        "no_missing_evidence": not any(record.metadata.get("missing") for record in evidence),
    }
    if parsed_task.metadata.get("domain") == CLINICAL_DOMAIN or parsed_task.task_type in {
        CLINICAL_TASK_TYPE,
        CLINICAL_PREDICTION_TASK_TYPE,
    }:
        checks.update(
            {
                "has_trial_metadata": _has_all(
                    present,
                    [
                        "clinical_trial_id",
                        "clinical_phase",
                        "clinical_condition",
                        "clinical_intervention",
                        "clinical_endpoint",
                    ],
                ),
            }
        )
    if parsed_task.task_type == CLINICAL_TASK_TYPE:
        checks.update(
            {
                "has_eligibility_evidence": _has_all(present, ["clinical_eligibility", "clinical_enrollment"]),
                "has_outcome_evidence": _has_all(present, ["clinical_outcome", "clinical_outcome_provenance"]),
            }
        )
    if parsed_task.task_type == CLINICAL_PREDICTION_TASK_TYPE:
        checks.update(
            {
                "has_prediction_output": "clinical_outcome_prediction" in present,
                "has_prediction_confidence_or_score": "clinical_prediction_confidence" in present,
                "has_enrollment_evidence": "enrollment_evidence" in present,
                "has_drug_or_disease_risk_evidence": (
                    "drug_risk_evidence" in present and "disease_risk_evidence" in present
                ),
                "has_external_provenance": "external_knowledge_evidence" in present,
                "no_backend_error": not any(record.error for record in tool_calls),
            }
        )
    success = all(checks.values())
    return VerifierResult(
        success=success,
        checks=checks,
        metrics={
            "evidence_count": len(evidence),
            "required_evidence_count": len(required),
            "evidence_coverage": coverage,
            "missing_evidence_count": len(missing),
            "tool_call_count": len(tool_calls),
        },
        failure_reason=", ".join(missing) if missing else None,
        metadata={
            "domain": parsed_task.metadata.get("domain"),
            "required_evidence": required,
            "present_evidence": sorted(present),
            "missing_evidence": missing,
            "missing_reasons": missing_reasons,
            "missing_evidence_families": _families_for_types(missing),
            "failed_checks": sorted(key for key, value in checks.items() if not value),
            "evidence_records": evidence_records_to_dicts(evidence),
        },
    )


def _run_offline_tool(parsed_task: ParsedTask, tool_name: str) -> tuple[list[EvidenceRecord], list[str]]:
    metadata = parsed_task.metadata
    if tool_name == "clinical_trial_metadata_parser":
        return _records_from_fields(
            metadata,
            {
                "trial_id": "clinical_trial_id",
                "phase": "clinical_phase",
                "condition": "clinical_condition",
                "intervention": "clinical_intervention",
                "endpoint": "clinical_endpoint",
            },
            default_source="trial_record",
        )
    if tool_name == "eligibility_evidence_checker":
        return _records_from_fields(
            metadata,
            {
                "eligibility_criteria": "clinical_eligibility",
                "enrollment": "clinical_enrollment",
            },
            default_source="trial_record",
        )
    if tool_name == "trial_outcome_evidence_checker":
        return _records_from_fields(
            metadata,
            {
                "outcome_label": "clinical_outcome",
                "outcome_source": "clinical_outcome_provenance",
            },
            default_source="trial_record",
        )
    if tool_name == "clinicalagent_evidence_retriever":
        return _records_from_fields(
            metadata,
            {
                "external_knowledge_source": "external_knowledge_evidence",
            },
            default_source="clinicalagent_backend",
        )
    if tool_name == "clinicalagent_enrollment_predictor":
        return _records_from_fields(
            metadata,
            {
                "enrollment_prediction": "enrollment_evidence",
            },
            default_source="clinicalagent_backend",
        )
    if tool_name == "clinicalagent_drug_risk_checker":
        return _records_from_fields(
            metadata,
            {
                "drug_risk_evidence": "drug_risk_evidence",
            },
            default_source="clinicalagent_backend",
        )
    if tool_name == "clinicalagent_disease_risk_checker":
        return _records_from_fields(
            metadata,
            {
                "disease_risk_evidence": "disease_risk_evidence",
            },
            default_source="clinicalagent_backend",
        )
    if tool_name == "clinicalagent_outcome_predictor":
        return _records_from_fields(
            metadata,
            {
                "prediction_label": "clinical_outcome_prediction",
                "prediction_confidence": "clinical_prediction_confidence",
            },
            default_source="clinicalagent_backend",
        )
    if tool_name == "drug_target_evidence_checker":
        return _records_from_fields(
            metadata,
            {
                "drug": "drug_name",
                "target": "target_name",
                "mechanism": "mechanism_evidence",
                "kg_source": "kg_provenance",
                "literature_source": "literature_provenance",
            },
            default_source="drug_target_record",
        )
    if tool_name == "repurposing_evidence_checker":
        return _records_from_fields(
            metadata,
            {
                "disease": "disease_indication",
                "repurposing_rationale": "repurposing_rationale",
            },
            default_source="drug_target_record",
        )
    return [], [f"unknown_tool:{tool_name}"]


def _records_from_fields(
    metadata: dict[str, Any],
    field_to_type: dict[str, str],
    *,
    default_source: str,
) -> tuple[list[EvidenceRecord], list[str]]:
    records: list[EvidenceRecord] = []
    missing: list[str] = []
    source = metadata.get("source") or metadata.get("provenance") or default_source
    for field, evidence_type in field_to_type.items():
        value = metadata.get(field)
        is_missing = value in (None, "")
        if value in (None, ""):
            missing.append(field)
        records.append(
            EvidenceRecord(
                evidence_type=evidence_type,
                value=value,
                evidence_family=_evidence_family(evidence_type),
                source=str(source) if value not in (None, "") else None,
                supports=value not in (None, ""),
                required=True,
                missing_reason=f"missing_{field}" if is_missing else None,
                provenance={"field": field, "source": source} if value not in (None, "") else {},
                metadata={"field": field, "missing": is_missing},
            )
        )
    return records, missing


def _required_evidence_types(parsed_task: ParsedTask, workflow: PlannedWorkflow) -> list[str]:
    domain = parsed_task.metadata.get("domain")
    if parsed_task.task_type == CLINICAL_PREDICTION_TASK_TYPE:
        return [
            "clinical_trial_id",
            "clinical_phase",
            "clinical_condition",
            "clinical_intervention",
            "clinical_endpoint",
            "external_knowledge_evidence",
            "enrollment_evidence",
            "drug_risk_evidence",
            "disease_risk_evidence",
            "clinical_outcome_prediction",
            "clinical_prediction_confidence",
        ]
    if domain == CLINICAL_DOMAIN or parsed_task.task_type == CLINICAL_TASK_TYPE:
        return [
            "clinical_trial_id",
            "clinical_phase",
            "clinical_condition",
            "clinical_intervention",
            "clinical_endpoint",
            "clinical_eligibility",
            "clinical_enrollment",
            "clinical_outcome",
            "clinical_outcome_provenance",
        ]
    required = [
        "drug_name",
        "target_name",
        "mechanism_evidence",
        "kg_provenance",
        "literature_provenance",
    ]
    if any(step.tool_name == "repurposing_evidence_checker" for step in workflow.tool_sequence):
        required.extend(["disease_indication", "repurposing_rationale"])
    return required


def _extract_clinical_entities(query: str, metadata: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    match = _NCT_RE.search(query)
    if match and "trial_id" not in metadata:
        extracted["trial_id"] = match.group(0).upper()
    for key, aliases in {
        "phase": ("phase", "trial_phase"),
        "condition": ("condition", "disease"),
        "intervention": ("intervention", "drug"),
        "endpoint": ("endpoint", "primary_endpoint"),
        "eligibility_criteria": ("eligibility", "criteria"),
        "enrollment": ("enrollment", "sample_size"),
        "outcome_label": ("outcome", "label"),
        "outcome_source": ("outcome_source", "source"),
    }.items():
        if key not in metadata:
            value = _extract_field(query, aliases)
            if value:
                extracted[key] = value
    return extracted


def _is_clinical_prediction_task(query: str, metadata: dict[str, Any]) -> bool:
    if metadata.get("task_type") == CLINICAL_PREDICTION_TASK_TYPE:
        return True
    if metadata.get("use_clinicalagent_backend") or metadata.get("clinical_backend") == "clinicalagent":
        return True
    lower = query.lower()
    return any(
        phrase in lower
        for phrase in (
            "clinical trial prediction",
            "trial outcome prediction",
            "predict clinical trial",
            "predict trial outcome",
        )
    )


def _extract_drug_target_entities(query: str, metadata: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for key, aliases in {
        "drug": ("drug", "compound"),
        "target": ("target", "protein"),
        "disease": ("disease", "indication"),
        "mechanism": ("mechanism", "moa"),
        "kg_source": ("kg_source", "kg", "knowledge_graph"),
        "literature_source": ("literature_source", "paper", "literature"),
        "repurposing_rationale": ("rationale", "repurposing_rationale"),
    }.items():
        if key not in metadata:
            value = _extract_field(query, aliases)
            if value:
                extracted[key] = value
    return extracted


def _extract_field(query: str, fields: tuple[str, ...]) -> str | None:
    field_pattern = "|".join(re.escape(field) for field in fields)
    stop_pattern = "|".join(re.escape(field) for field in _KNOWN_FIELD_NAMES)
    pattern = rf"\b(?:{field_pattern})\b\s*[:=]\s*(?P<value>.*?)(?=\s+\b(?:{stop_pattern})\b\s*[:=]|[,;\n]|$)"
    match = re.search(pattern, query, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group("value").strip().strip("\"'.")


def _has_all(present: set[str], required: list[str]) -> bool:
    return all(item in present for item in required)


def _evidence_family(evidence_type: str) -> str:
    return EVIDENCE_TYPE_TO_FAMILY.get(evidence_type, "other")


def _families_for_types(evidence_types: list[str]) -> list[str]:
    return sorted({_evidence_family(item) for item in evidence_types})


def _missing_reasons_by_type(records: list[EvidenceRecord], missing_types: list[str]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    missing_set = set(missing_types)
    for record in records:
        if record.evidence_type in missing_set and record.missing_reason:
            reasons[record.evidence_type] = record.missing_reason
    for evidence_type in missing_types:
        reasons.setdefault(evidence_type, f"missing_{evidence_type}")
    return reasons


def _stable_task_id(query: str, *, prefix: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
