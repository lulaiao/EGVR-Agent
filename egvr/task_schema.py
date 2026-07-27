"""Structured records shared by parser, planner, executor, and verifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _clean_list(value: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    return list(value or [])


def _clean_xyz(value: list[Any] | tuple[Any, ...] | None, field_name: str) -> list[float] | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three coordinates")
    return [float(coord) for coord in value]


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@dataclass
class TaskConstraints:
    """Conservative machine-readable constraints parsed from a user task."""

    max_candidates: int | None = None
    min_candidates: int | None = None
    require_docking: bool = False
    require_synthesizability: bool = False
    require_toxicity: bool = False
    require_ranking: bool = False
    max_scscore: float | None = None
    max_toxicity_score: float | None = None
    min_pmic: float | None = None
    allowed_elements: list[str] = field(default_factory=list)
    disallowed_substructures: list[str] = field(default_factory=list)
    time_budget_minutes: float | None = None
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class ParsedTask:
    """Normalized user intent before tool planning."""

    task_id: str
    raw_user_query: str
    task_type: str = "unknown"
    objectives: list[str] = field(default_factory=list)
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    protein_path: str | None = None
    pocket_center: list[float] | None = None
    ligand_path: str | None = None
    ref_ligand_path: str | None = None
    box_size: list[float] | None = None
    input_smiles: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.objectives = [str(item) for item in _clean_list(self.objectives)]
        self.input_smiles = [str(item) for item in _clean_list(self.input_smiles)]
        self.context_files = [str(item) for item in _clean_list(self.context_files)]
        self.metadata = _clean_mapping(self.metadata)
        self.pocket_center = _clean_xyz(self.pocket_center, "pocket_center")
        self.box_size = _clean_xyz(self.box_size, "box_size")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class PlannedToolCall:
    """One planned tool step in a workflow."""

    tool_name: str
    reason: str
    action: str = "default"
    expected_outputs: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    optional_inputs: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expected_outputs = [str(item) for item in _clean_list(self.expected_outputs)]
        self.required_inputs = [str(item) for item in _clean_list(self.required_inputs)]
        self.optional_inputs = [str(item) for item in _clean_list(self.optional_inputs)]
        self.parameters = _clean_mapping(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class PlannedWorkflow:
    """Structured output from a planner."""

    task_id: str
    planner_type: str
    selected_tools: list[str] = field(default_factory=list)
    tool_sequence: list[PlannedToolCall] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        normalized_steps: list[PlannedToolCall] = []
        for step in self.tool_sequence:
            if isinstance(step, PlannedToolCall):
                normalized_steps.append(step)
            elif isinstance(step, dict):
                normalized_steps.append(PlannedToolCall(**step))
            else:
                raise TypeError("tool_sequence entries must be PlannedToolCall or dict")
        self.tool_sequence = normalized_steps
        if not self.selected_tools:
            self.selected_tools = [step.tool_name for step in self.tool_sequence]
        else:
            self.selected_tools = [str(item) for item in _clean_list(self.selected_tools)]
        self.expected_outputs = [str(item) for item in _clean_list(self.expected_outputs)]
        self.notes = [str(item) for item in _clean_list(self.notes)]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class CandidateRecord:
    """Candidate-level representation shared across generation and evaluation tools."""

    smiles: str | None = None
    source_tool: str | None = None
    is_valid: bool = True
    rank: int | None = None
    docking_score: float | None = None
    scscore: float | None = None
    sa_score: float | None = None
    toxicity_score: float | None = None
    pmic_score: float | None = None
    posebusters_pass: bool | None = None
    posebusters_checks: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.posebusters_checks = _clean_mapping(self.posebusters_checks)
        self.artifacts = _clean_mapping(self.artifacts)
        self.metadata = _clean_mapping(self.metadata)
        self.errors = [str(item) for item in _clean_list(self.errors)]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class ToolMetadata:
    """Chemistry-aware tool metadata used by rule-based and learned planners."""

    tool_name: str
    description: str
    supported_task_types: list[str]
    required_inputs: list[str]
    optional_inputs: list[str]
    outputs: list[str]
    typical_failures: list[str]
    estimated_cost: str
    downstream_tools: list[str]
    chemistry_role: str
    backend_tool_name: str
    backend_action: str = "default"
    wrapper_function: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.supported_task_types = [str(item) for item in _clean_list(self.supported_task_types)]
        self.required_inputs = [str(item) for item in _clean_list(self.required_inputs)]
        self.optional_inputs = [str(item) for item in _clean_list(self.optional_inputs)]
        self.outputs = [str(item) for item in _clean_list(self.outputs)]
        self.typical_failures = [str(item) for item in _clean_list(self.typical_failures)]
        self.downstream_tools = [str(item) for item in _clean_list(self.downstream_tools)]
        self.tags = [str(item) for item in _clean_list(self.tags)]
        self.metadata = _clean_mapping(self.metadata)

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class ToolCallRecord:
    """Execution record for one concrete tool invocation."""

    tool_name: str
    action: str = "default"
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] | None = None
    success: bool = False
    error: str | None = None
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    elapsed_time_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.inputs = _clean_mapping(self.inputs)
        self.metadata = _clean_mapping(self.metadata)
        if self.outputs is not None:
            self.outputs = _clean_mapping(self.outputs)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class VerifierResult:
    """Structured result from chemistry-grounded workflow verification."""

    success: bool
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, int | float | str | bool | None] = field(default_factory=dict)
    failure_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.checks = {str(key): bool(value) for key, value in _clean_mapping(self.checks).items()}
        self.metrics = _clean_mapping(self.metrics)
        self.warnings = [str(item) for item in _clean_list(self.warnings)]
        self.metadata = _clean_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
