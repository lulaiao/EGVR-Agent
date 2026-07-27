"""Biomedical evidence records for offline generalization slices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clean_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass
class EvidenceRecord:
    """A small provenance-carrying evidence unit used by biomedical verifiers."""

    evidence_type: str
    value: Any = None
    evidence_family: str | None = None
    source: str | None = None
    supports: bool = True
    required: bool = True
    confidence: float | None = None
    missing_reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.evidence_type = str(self.evidence_type)
        self.provenance = _clean_mapping(self.provenance)
        self.metadata = _clean_mapping(self.metadata)

    def has_value(self) -> bool:
        return self.value is not None and self.value != ""

    def has_provenance(self) -> bool:
        return bool(self.source or self.provenance)

    def is_complete(self) -> bool:
        return self.supports and self.has_value() and self.has_provenance() and not self.missing_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "value": self.value,
            "evidence_family": self.evidence_family,
            "source": self.source,
            "supports": self.supports,
            "required": self.required,
            "confidence": self.confidence,
            "missing_reason": self.missing_reason,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }


def evidence_records_to_dicts(records: list[EvidenceRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]
