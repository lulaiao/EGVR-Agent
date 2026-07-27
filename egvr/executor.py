"""Structured workflow executor for planned chemistry tool calls."""

from __future__ import annotations

import importlib
import inspect
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from .result_normalizer import normalize_tool_output
from .task_schema import CandidateRecord, ParsedTask, PlannedToolCall, PlannedWorkflow, ToolCallRecord
from .tool_registry import EvidenceToolRegistry, build_default_tool_registry


ToolFunction = Callable[..., dict[str, Any]]


class WorkflowExecutor:
    """Execute planned tool calls with structured records and mockable tools."""

    def __init__(
        self,
        *,
        registry: EvidenceToolRegistry | None = None,
        tool_functions: dict[str, ToolFunction] | None = None,
    ) -> None:
        self.registry = registry or build_default_tool_registry()
        self.tool_functions = dict(tool_functions or {})

    def execute(
        self,
        parsed_task: ParsedTask,
        workflow: PlannedWorkflow,
        *,
        initial_candidates: Iterable[CandidateRecord] | None = None,
    ) -> tuple[list[ToolCallRecord], list[CandidateRecord]]:
        tool_calls: list[ToolCallRecord] = []
        candidates = list(initial_candidates or _candidates_from_input_smiles(parsed_task))

        for step in workflow.tool_sequence:
            if self._should_skip_step(step, tool_calls, candidates):
                tool_calls.append(_skipped_record(step))
                continue
            step_records, candidates = self._execute_step(parsed_task, step, candidates)
            tool_calls.extend(step_records)

        return tool_calls, candidates

    def _execute_step(
        self,
        parsed_task: ParsedTask,
        step: PlannedToolCall,
        candidates: list[CandidateRecord],
    ) -> tuple[list[ToolCallRecord], list[CandidateRecord]]:
        if step.tool_name in {"toxicity", "pmic"}:
            return self._execute_per_candidate(parsed_task, step, candidates)

        started = _now_iso()
        start_time = time.perf_counter()
        inputs: dict[str, Any] = {}
        try:
            inputs = self._build_inputs(parsed_task, step, candidates)
            output = self._resolve_function(step.tool_name)(**inputs)
            elapsed = time.perf_counter() - start_time
            record = ToolCallRecord(
                tool_name=step.tool_name,
                action=step.action,
                inputs=inputs,
                outputs=output,
                success=not bool(output.get("error") or output.get("success") is False),
                error=output.get("error"),
                started_at=started,
                finished_at=_now_iso(),
                elapsed_time_sec=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            record = ToolCallRecord(
                tool_name=step.tool_name,
                action=step.action,
                inputs=inputs,
                outputs=None,
                success=False,
                error=str(exc),
                started_at=started,
                finished_at=_now_iso(),
                elapsed_time_sec=elapsed,
            )

        if record.success:
            candidates = normalize_tool_output(step.tool_name, record.outputs, existing_candidates=candidates)
        return [record], candidates

    def _execute_per_candidate(
        self,
        parsed_task: ParsedTask,
        step: PlannedToolCall,
        candidates: list[CandidateRecord],
    ) -> tuple[list[ToolCallRecord], list[CandidateRecord]]:
        records: list[ToolCallRecord] = []
        if not candidates and parsed_task.input_smiles:
            candidates = _candidates_from_input_smiles(parsed_task)
        if not candidates:
            return [
                ToolCallRecord(
                    tool_name=step.tool_name,
                    action=step.action,
                    inputs={},
                    success=False,
                    error="No candidate SMILES available for per-molecule evaluation.",
                    finished_at=_now_iso(),
                )
            ], candidates

        for idx, candidate in enumerate(list(candidates)):
            if not candidate.smiles:
                continue
            started = _now_iso()
            start_time = time.perf_counter()
            inputs = {"smiles": candidate.smiles}
            try:
                output = self._resolve_function(step.tool_name)(**inputs)
                elapsed = time.perf_counter() - start_time
                record = ToolCallRecord(
                    tool_name=step.tool_name,
                    action=step.action,
                    inputs=inputs,
                    outputs=output,
                    success=not bool(output.get("error") or output.get("success") is False),
                    error=output.get("error"),
                    started_at=started,
                    finished_at=_now_iso(),
                    elapsed_time_sec=elapsed,
                    metadata={"candidate_index": idx},
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                record = ToolCallRecord(
                    tool_name=step.tool_name,
                    action=step.action,
                    inputs=inputs,
                    success=False,
                    error=str(exc),
                    started_at=started,
                    finished_at=_now_iso(),
                    elapsed_time_sec=elapsed,
                    metadata={"candidate_index": idx},
                )
            records.append(record)
            if record.success:
                candidates = normalize_tool_output(
                    step.tool_name,
                    record.outputs,
                    existing_candidates=candidates,
                    input_smiles=candidate.smiles,
                )
        return records, candidates

    def _build_inputs(
        self,
        parsed_task: ParsedTask,
        step: PlannedToolCall,
        candidates: list[CandidateRecord],
    ) -> dict[str, Any]:
        parameters = _runtime_parameters(step.parameters)
        tool_name = step.tool_name

        if tool_name == "rxnflow":
            return _drop_none(
                {
                    "protein_pdb_path": parsed_task.protein_path,
                    "center_xyz": parsed_task.pocket_center,
                    "ref_ligand_path": parsed_task.ref_ligand_path,
                    **parameters,
                }
            )
        if tool_name == "reinvent4_denovo":
            return parameters
        if tool_name in {"reinvent4_mol2mol", "reinvent4_libinvent", "scaffold", "libinvent"}:
            smiles = _first_smiles(step.parameters.get("input_smiles") or parsed_task.input_smiles)
            if not smiles:
                raise ValueError(f"{tool_name} requires input_smiles")
            return {"smiles": smiles, **parameters}
        if tool_name == "scscore":
            smiles_list = [candidate.smiles for candidate in candidates if candidate.smiles] or parsed_task.input_smiles
            if not smiles_list:
                raise ValueError("scscore requires candidate SMILES")
            return {"smiles_list": smiles_list, **parameters}
        if tool_name == "rdkit_property_verifier":
            smiles_list = [candidate.smiles for candidate in candidates if candidate.smiles] or parsed_task.input_smiles
            if not smiles_list:
                raise ValueError("rdkit_property_verifier requires candidate SMILES")
            return {"smiles_list": smiles_list, **parameters}
        if tool_name == "vina":
            ligand_path = parsed_task.ligand_path or _first_candidate_artifact(candidates, "ligand_path")
            if not ligand_path:
                if tool_name in self.tool_functions:
                    return _drop_none(
                        {
                            "receptor_pdbqt_path": parsed_task.protein_path,
                            "candidate_smiles": [candidate.smiles for candidate in candidates if candidate.smiles],
                            "center_xyz": parsed_task.pocket_center,
                            "box_size_xyz": parsed_task.box_size,
                            **parameters,
                        }
                    )
                raise ValueError("vina requires ligand_path or candidate ligand_path artifacts")
            return _drop_none(
                {
                    "receptor_pdbqt_path": parsed_task.protein_path,
                    "ligand_pdbqt_path": ligand_path,
                    "center_xyz": parsed_task.pocket_center,
                    "box_size_xyz": parsed_task.box_size,
                    **parameters,
                }
            )
        return parameters

    def _resolve_function(self, tool_name: str) -> ToolFunction:
        if tool_name in self.tool_functions:
            return self.tool_functions[tool_name]
        tool = self.registry.require(tool_name)
        if not tool.wrapper_function:
            raise ValueError(f"No wrapper_function registered for {tool_name}")
        module_name, attr_name = tool.wrapper_function.rsplit(".", 1)
        func = getattr(importlib.import_module(module_name), attr_name)
        return _signature_filtered(func)

    def _should_skip_step(
        self,
        step: PlannedToolCall,
        tool_calls: list[ToolCallRecord],
        candidates: list[CandidateRecord],
    ) -> bool:
        if step.parameters.get("defer_until_repair") is True:
            return True
        execute_if = step.parameters.get("execute_if")
        fallback_for = step.parameters.get("fallback_for")
        if execute_if != "rxnflow_failed_or_empty" or fallback_for != "rxnflow":
            return False
        upstream_calls = [record for record in tool_calls if record.tool_name == "rxnflow"]
        upstream_failed = upstream_calls and not upstream_calls[-1].success
        return bool(upstream_calls and not upstream_failed and candidates)


def _signature_filtered(func: ToolFunction) -> ToolFunction:
    signature = inspect.signature(func)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return func

    def _wrapped(**kwargs):
        filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return func(**filtered)

    return _wrapped


def _runtime_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if key
        not in {
            "execute_if",
            "fallback_for",
            "defer_until_repair",
            "input_source",
            "input_smiles",
            "repair_retry",
            "retry_reason",
        }
    }


def _candidates_from_input_smiles(parsed_task: ParsedTask) -> list[CandidateRecord]:
    return [
        CandidateRecord(smiles=smiles, source_tool="input", rank=idx)
        for idx, smiles in enumerate(parsed_task.input_smiles, start=1)
    ]


def _skipped_record(step: PlannedToolCall) -> ToolCallRecord:
    deferred = step.parameters.get("defer_until_repair") is True
    return ToolCallRecord(
        tool_name=step.tool_name,
        action=step.action,
        inputs={},
        outputs={
            "skipped": True,
            "deferred": deferred,
            "reason": "deferred_until_repair" if deferred else step.parameters.get("execute_if"),
        },
        success=True,
        finished_at=_now_iso(),
        metadata={
            "skipped": True,
            "deferred": deferred,
            "fallback_for": step.parameters.get("fallback_for"),
        },
    )


def _first_smiles(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple) and value:
        return value[0]
    return None


def _first_candidate_artifact(candidates: Iterable[CandidateRecord], key: str) -> Any:
    for candidate in candidates:
        if candidate.artifacts.get(key):
            return candidate.artifacts[key]
    return None


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
