"""Planning-only LLM-as-router baseline evaluator.

The runner evaluates whether a router returns a valid structured workflow from
a natural-language task and a structured tool registry. It does not execute
tools. Real LLM outputs can be supplied through --responses-jsonl or collected
through --router-mode api. The default heuristic mode is for CI and
prompt/interface debugging only; it must not be reported as a real LLM result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .rule_planner import plan_workflow
from .task_parser import parse_task
from .task_schema import PlannedWorkflow
from .tool_registry import EvidenceToolRegistry, build_default_tool_registry


DEFAULT_BENCHMARKS = (
    "egvr/benchmarks/crossdocked_rxnflow_candidates5_targets30.jsonl",
    "egvr/benchmarks/litpcba_vina_prepared_15.jsonl",
    "egvr/benchmarks/task_generalization_real_v1.jsonl",
)
DEFAULT_OUTPUT = "logs/baseline_runs/llm_as_router_planning_v1/llm_router_baseline_summary.json"

LLM_ROUTER_BASELINE_COLUMNS = [
    "benchmark_id",
    "dataset",
    "router_mode",
    "task_count",
    "valid_json_count",
    "valid_schema_count",
    "invalid_json_rate",
    "invalid_schema_rate",
    "hallucinated_tool_count",
    "hallucinated_tool_rate",
    "missing_required_input_count",
    "missing_required_input_rate",
    "tool_precision",
    "tool_recall",
    "tool_f1",
    "workflow_order_match_rate",
    "mean_selected_tool_count",
    "mean_extra_tool_count",
    "notes",
]


def run_llm_router_baseline(
    *,
    benchmark_paths: list[str | Path] | tuple[str | Path, ...] = DEFAULT_BENCHMARKS,
    responses_jsonl: str | Path | None = None,
    response_log_jsonl: str | Path | None = None,
    router_mode: str = "heuristic",
    output_path: str | Path = DEFAULT_OUTPUT,
    project_root: str | Path | None = None,
    llm_model: str | None = None,
    llm_model_env: str | None = None,
    llm_base_url: str | None = None,
    llm_base_url_env: str | None = None,
    llm_api_key: str | None = None,
    llm_api_key_env: str | None = None,
    llm_extra_body_json: str | None = None,
    dotenv_path: str | Path | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    request_timeout_sec: float = 60.0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate planning-only router outputs and write a summary JSON."""

    root = _project_root(project_root)
    registry = build_default_tool_registry()
    out_path = _resolve_path(output_path, root)
    response_log_path = _resolve_response_log_path(
        response_log_jsonl=response_log_jsonl,
        output_path=out_path,
        router_mode=router_mode,
        project_root=root,
    )
    response_records: dict[str, dict[str, Any]] = {}
    responses: dict[str, str] = {}
    if responses_jsonl:
        response_records.update(_load_response_record_map(_resolve_path(responses_jsonl, root)))
        responses.update(_raw_response_map(response_records))
    if response_log_path and response_log_path.exists():
        response_records.update(_load_response_record_map(response_log_path))
        responses.update(_raw_response_map(response_records))
    api_config = None
    if router_mode == "api":
        _load_project_dotenv(root)
        if dotenv_path is not None:
            _load_dotenv_path(_resolve_path(dotenv_path, root))
        api_config = _resolve_api_config(
            llm_model=llm_model,
            llm_model_env=llm_model_env,
            llm_base_url=llm_base_url,
            llm_base_url_env=llm_base_url_env,
            llm_api_key=llm_api_key,
            llm_api_key_env=llm_api_key_env,
            llm_extra_body_json=llm_extra_body_json,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout_sec=request_timeout_sec,
        )
    task_results: list[dict[str, Any]] = []
    for benchmark_path in benchmark_paths:
        resolved = _resolve_path(benchmark_path, root)
        for task in _read_jsonl(resolved):
            if limit is not None and len(task_results) >= limit:
                break
            task_results.append(
                evaluate_router_task(
                    task,
                    benchmark_path=resolved,
                    registry=registry,
                    responses=responses,
                    response_records=response_records,
                    router_mode=router_mode,
                    project_root=root,
                    api_config=api_config,
                    response_log_path=response_log_path,
                )
            )
        if limit is not None and len(task_results) >= limit:
            break

    summary_row = _summarize_task_results(task_results, router_mode=router_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    _write_csv([summary_row], csv_path, LLM_ROUTER_BASELINE_COLUMNS)
    payload = {
        "benchmark_id": "llm_as_router_planning_v1",
        "execution_mode": "planning_only",
        "router_mode": router_mode,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "columns": LLM_ROUTER_BASELINE_COLUMNS,
        "row": summary_row,
        "task_count": len(task_results),
        "task_results": task_results,
        "artifacts": {
            "json": _display_path(out_path, root),
            "csv": _display_path(csv_path, root),
            "response_log_jsonl": _display_path(response_log_path, root) if response_log_path else None,
        },
        "notes": [
            "This is a planning-only baseline and does not execute chemistry tools.",
            "Heuristic mode is for offline interface testing; use --router-mode api or --responses-jsonl for real LLM evidence.",
            "API mode uses a fixed prompt, fixed structured tool registry, temperature 0 by default, and stores raw LLM responses.",
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def evaluate_router_task(
    task: dict[str, Any],
    *,
    benchmark_path: Path,
    registry: EvidenceToolRegistry,
    responses: dict[str, str],
    response_records: dict[str, dict[str, Any]] | None = None,
    router_mode: str,
    project_root: Path,
    api_config: dict[str, Any] | None = None,
    response_log_path: Path | None = None,
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or f"task_{len(responses)}")
    raw_query = str(task.get("raw_user_query") or "")
    parsed_task = parse_task(raw_query, task_id=task_id, metadata=task.get("metadata") or {})
    gold_workflow = plan_workflow(parsed_task, registry=registry)
    expected_tools = list(task.get("expected_tools") or gold_workflow.selected_tools)
    raw_response = responses.get(task_id)
    response_source = "responses_jsonl" if raw_response is not None else router_mode
    cached_record = (response_records or {}).get(task_id) or {}
    api_metadata: dict[str, Any] = dict(cached_record.get("api_metadata") or {})
    request_payload = api_metadata.get("request_payload") or {}
    cached_messages = request_payload.get("messages")
    prompt_messages: list[dict[str, str]] | None = (
        cached_messages if isinstance(cached_messages, list) else None
    )
    api_response_recorded_at: str | None = cached_record.get("recorded_at")
    new_api_response = False
    if raw_response is None:
        if router_mode == "api":
            if api_config is None:
                raise ValueError("router_mode='api' requires resolved API configuration")
            prompt_messages = _build_router_prompt_messages(task=task, registry=registry)
            raw_response, api_metadata = _call_openai_compatible_chat_api(
                messages=prompt_messages,
                api_config=api_config,
            )
            api_response_recorded_at = _utc_timestamp()
            new_api_response = True
        else:
            raw_response = _heuristic_router_response(task, gold_workflow, router_mode=router_mode, registry=registry)
    parsed_response, json_error = _parse_json_object(raw_response)
    response_normalization = _response_normalization(raw_response)
    workflow, schema_error = _workflow_from_response(parsed_response)
    prompt_hash = cached_record.get("prompt_hash") or (
        _stable_hash(prompt_messages) if prompt_messages else None
    )
    request_hash = cached_record.get("request_hash") or api_metadata.get("request_hash")
    plan_hash = _plan_hash(parsed_response, raw_response)
    if new_api_response and api_response_recorded_at and response_log_path is not None:
        _append_response_log(
            response_log_path,
            {
                "task_id": task_id,
                "raw_response": raw_response,
                "api_metadata": api_metadata,
                "prompt_hash": prompt_hash,
                "request_hash": request_hash,
                "plan_hash": plan_hash,
                "response_normalization": response_normalization,
                "recorded_at": api_response_recorded_at,
            },
        )
    selected_tools = workflow.selected_tools if workflow else []
    known_tools = set(registry.names())
    hallucinated_tools = [tool for tool in selected_tools if tool not in known_tools]
    missing_required_inputs = _missing_required_inputs(workflow, parsed_task, registry) if workflow else []
    precision, recall, f1 = _prf(selected_tools, expected_tools)
    order_match = _order_match(selected_tools, expected_tools)
    extra_tools = sorted(set(selected_tools) - set(expected_tools))
    return {
        "task_id": task_id,
        "benchmark_path": _display_path(benchmark_path, project_root),
        "dataset": (task.get("metadata") or {}).get("dataset") or task.get("dataset"),
        "expected_task_type": task.get("expected_task_type"),
        "parsed_task_type": parsed_task.task_type,
        "expected_tools": expected_tools,
        "selected_tools": selected_tools,
        "extra_tools": extra_tools,
        "hallucinated_tools": hallucinated_tools,
        "missing_required_inputs": missing_required_inputs,
        "valid_json": json_error is None,
        "valid_schema": workflow is not None and schema_error is None,
        "json_error": json_error,
        "schema_error": schema_error,
        "raw_response": raw_response,
        "response_source": response_source,
        "prompt_messages": prompt_messages,
        "api_metadata": api_metadata,
        "prompt_hash": prompt_hash,
        "request_hash": request_hash,
        "plan_hash": plan_hash,
        "api_response_recorded_at": api_response_recorded_at,
        "response_normalization": response_normalization,
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": f1,
        "workflow_order_match": order_match,
        "selected_tool_count": len(selected_tools),
        "extra_tool_count": len(extra_tools),
    }


def _resolve_api_config(
    *,
    llm_model: str | None,
    llm_model_env: str | None = None,
    llm_base_url: str | None,
    llm_base_url_env: str | None = None,
    llm_api_key: str | None,
    llm_api_key_env: str | None,
    llm_extra_body_json: str | None,
    temperature: float,
    max_tokens: int,
    request_timeout_sec: float,
) -> dict[str, Any]:
    """Resolve OpenAI-compatible API settings without falling back to heuristics."""

    model = llm_model or (os.getenv(llm_model_env) if llm_model_env else None) or os.getenv("LLM_MODEL")
    base_url = (
        llm_base_url
        or (os.getenv(llm_base_url_env) if llm_base_url_env else None)
        or os.getenv("LLM_BASE_URL")
    )
    api_key = llm_api_key
    if api_key is None and llm_api_key_env:
        api_key = os.getenv(llm_api_key_env)
    if api_key is None:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

    missing: list[str] = []
    if not model:
        missing.append("LLM_MODEL or --llm-model")
    if not base_url:
        missing.append("LLM_BASE_URL or --llm-base-url")
    if not api_key:
        missing.append("LLM_API_KEY/OPENAI_API_KEY/DEEPSEEK_API_KEY, --llm-api-key, or --llm-api-key-env")
    if missing:
        raise ValueError(
            "router_mode='api' requires explicit OpenAI-compatible API settings; missing: "
            + ", ".join(missing)
            + ". Use LLM_API_KEY=EMPTY only for unauthenticated local endpoints."
        )
    extra_body = _parse_extra_body_json(llm_extra_body_json)

    return {
        "model": model,
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "extra_body": extra_body,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "request_timeout_sec": float(request_timeout_sec),
    }


def _parse_extra_body_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--llm-extra-body-json must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--llm-extra-body-json must decode to a JSON object")
    return payload


def _load_project_dotenv(project_root: Path) -> None:
    """Load a repository-local ignored dotenv file when present."""

    try:
        from dotenv import load_dotenv
    except Exception:  # noqa: BLE001 - dotenv is optional for the runner.
        return
    load_dotenv(project_root / ".env", override=False)


def _load_dotenv_path(path: Path) -> None:
    """Load an explicitly selected dotenv file without exposing its values."""

    try:
        from dotenv import load_dotenv
    except Exception as exc:  # noqa: BLE001 - dependency absence should be explicit here.
        raise RuntimeError("python-dotenv is required when --dotenv-path is used") from exc
    if not path.is_file():
        raise FileNotFoundError(f"dotenv file not found: {path}")
    load_dotenv(path, override=False)


def _resolve_response_log_path(
    *,
    response_log_jsonl: str | Path | None,
    output_path: Path,
    router_mode: str,
    project_root: Path,
) -> Path | None:
    if response_log_jsonl:
        return _resolve_path(response_log_jsonl, project_root)
    if router_mode == "api":
        return output_path.with_suffix(".responses.jsonl")
    return None


def _append_response_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _build_router_prompt_messages(
    *,
    task: dict[str, Any],
    registry: EvidenceToolRegistry,
) -> list[dict[str, str]]:
    """Create the fixed planning prompt used for the API LLM-router baseline."""

    system_prompt = (
        "You are a planning-only router for molecular design workflows. "
        "Return only one JSON object, with no markdown or explanatory text. "
        "The top-level JSON object must have exactly these workflow keys: "
        "task_id, planner_type, selected_tools, tool_sequence, expected_outputs, notes. "
        "Do not wrap the answer inside any other key such as schema, output_schema, required_output_schema, task, or workflow. "
        "Select tools only from the provided tool_registry and use tool names exactly. "
        "Do not execute tools. Do not invent tools, files, scores, or candidate molecules. "
        "Prefer the minimal task-conditioned tool set that satisfies the user request. "
        "If the request asks for evidence such as docking, synthesizability, toxicity, "
        "pose sanity, or molecular properties, include the corresponding evaluator/verifier tools. "
        "If a required input is missing, still produce the best conservative workflow and mention the missing input in notes."
    )
    user_payload = {
        "task": {
            "task_id": task.get("task_id"),
            "raw_user_query": task.get("raw_user_query"),
            "metadata_hints": _safe_metadata_hints(task.get("metadata") or {}),
        },
        "tool_registry": _compact_tool_registry(registry),
        "output_contract_text": (
            "Return a single workflow JSON object, not a wrapper. "
            "Top-level keys: task_id, planner_type, selected_tools, tool_sequence, expected_outputs, notes. "
            "planner_type must be 'llm_as_router'. "
            "selected_tools must be a list of known tool names. "
            "tool_sequence must be a list of objects with keys: "
            "tool_name, reason, action, expected_outputs, required_inputs, optional_inputs, parameters. "
            "expected_outputs and notes must be lists. parameters must be an object. "
            "No extra top-level keys are allowed."
        ),
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
    ]


def _compact_tool_registry(registry: EvidenceToolRegistry) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in registry.all():
        tools.append(
            {
                "tool_name": tool.tool_name,
                "description": tool.description,
                "supported_task_types": tool.supported_task_types,
                "required_inputs": tool.required_inputs,
                "optional_inputs": tool.optional_inputs,
                "outputs": tool.outputs,
                "downstream_tools": tool.downstream_tools,
                "chemistry_role": tool.chemistry_role,
                "backend_action": tool.backend_action,
                "tags": tool.tags,
            }
        )
    return tools


def _safe_metadata_hints(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep operational hints without leaking expected labels into the prompt."""

    allowed = {
        "dataset",
        "task_family",
        "num_candidates",
        "protein_path",
        "ligand_path",
        "ref_ligand_path",
        "pocket_center",
        "box_size",
        "target",
        "protein_id",
        "run_seed",
        "seed",
        "rxnflow_seed",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def _call_openai_compatible_chat_api(
    *,
    messages: list[dict[str, str]],
    api_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    url = f"{api_config['base_url']}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_config["api_key"] != "EMPTY":
        headers["Authorization"] = f"Bearer {api_config['api_key']}"
    payload = {
        "model": api_config["model"],
        "messages": messages,
        "temperature": api_config["temperature"],
        "max_tokens": api_config["max_tokens"],
    }
    payload.update(api_config.get("extra_body") or {})
    request_hash = _stable_hash(payload)
    started = time.monotonic()
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=api_config["request_timeout_sec"],
    )
    elapsed = time.monotonic() - started
    response.raise_for_status()
    data = response.json()
    try:
        raw_content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat completion response shape: {type(exc).__name__}: {exc}") from exc
    if not isinstance(raw_content, str):
        raise ValueError("Unexpected chat completion response: message content is not a string")
    metadata = {
        "api_model": api_config["model"],
        "api_base_url": api_config["base_url"],
        "api_protocol": "openai_compatible_chat_completions",
        "api_elapsed_sec": elapsed,
        "response_id": data.get("id"),
        "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
        "usage": data.get("usage"),
        "request_payload": payload,
        "request_hash": request_hash,
        "response_payload": data,
    }
    return raw_content, metadata


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_hash(parsed_response: dict[str, Any] | None, raw_response: str) -> str:
    if parsed_response is not None:
        return _stable_hash(parsed_response)
    return hashlib.sha256(raw_response.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _heuristic_router_response(
    task: dict[str, Any],
    gold_workflow: PlannedWorkflow,
    *,
    router_mode: str,
    registry: EvidenceToolRegistry,
) -> str:
    workflow = gold_workflow.to_dict()
    workflow["planner_type"] = "llm_as_router"
    workflow["notes"] = [
        "Generated by local heuristic router mode for planning-interface testing.",
        "This row should not be interpreted as a real LLM API result.",
    ]
    if router_mode == "heuristic_overexpose":
        workflow["selected_tools"] = registry.names()
    elif router_mode == "heuristic_invalid_tool":
        workflow["selected_tools"] = [*workflow.get("selected_tools", []), "imaginary_docking_tool"]
    elif router_mode == "heuristic_invalid_json":
        return "{not valid json"
    elif router_mode != "heuristic":
        raise ValueError(f"Unsupported router mode without --responses-jsonl: {router_mode}")
    return json.dumps(workflow, ensure_ascii=False, sort_keys=True)


def _workflow_from_response(parsed_response: dict[str, Any] | None) -> tuple[PlannedWorkflow | None, str | None]:
    if parsed_response is None:
        return None, "missing_json_object"
    try:
        return PlannedWorkflow(**parsed_response), None
    except Exception as exc:  # noqa: BLE001 - schema errors are recorded for auditing.
        return None, f"{type(exc).__name__}: {exc}"


def _parse_json_object(raw_response: str) -> tuple[dict[str, Any] | None, str | None]:
    normalized = _normalize_json_response(raw_response)
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if not isinstance(value, dict):
        return None, "router_response_not_object"
    return value, None


def _normalize_json_response(raw_response: str) -> str:
    """Remove one optional Markdown JSON fence without repairing JSON content."""

    stripped = raw_response.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    opening = lines[0].strip().lower()
    if opening not in {"```", "```json"}:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _response_normalization(raw_response: str) -> str:
    return "markdown_json_fence_removed" if _normalize_json_response(raw_response) != raw_response.strip() else "none"


def _missing_required_inputs(
    workflow: PlannedWorkflow,
    parsed_task: Any,
    registry: EvidenceToolRegistry,
) -> list[str]:
    available = {
        "protein_path": parsed_task.protein_path,
        "ligand_path": parsed_task.ligand_path,
        "ref_ligand_path": parsed_task.ref_ligand_path,
        "pocket_center": parsed_task.pocket_center,
        "box_size": parsed_task.box_size,
        "input_smiles": parsed_task.input_smiles,
    }
    missing: list[str] = []
    produced_outputs: set[str] = set()
    for step in workflow.tool_sequence:
        tool = registry.get(step.tool_name)
        required_inputs = step.required_inputs or (tool.required_inputs if tool else [])
        for required in required_inputs:
            required_key = _canonical_required_input(required)
            if required_key in {"generated_smiles", "candidate_records"} and produced_outputs:
                continue
            if required_key.startswith("candidate_") and "smiles" in required_key and produced_outputs:
                continue
            if required_key == "input_smiles" and produced_outputs & {"generated_smiles", "candidate_records"}:
                continue
            if available.get(required_key) in (None, [], "") and step.parameters.get(required_key) in (None, [], ""):
                missing.append(f"{step.tool_name}:{required}")
        produced_outputs.update(str(output) for output in step.expected_outputs)
        if tool:
            produced_outputs.update(str(output) for output in tool.outputs)
    return sorted(set(missing))


def _canonical_required_input(required_input: str) -> str:
    """Normalize explanatory LLM strings like 'input_smiles (from rxnflow)'."""

    normalized = str(required_input).split("(", 1)[0].strip()
    normalized = normalized.split("=", 1)[0].strip()
    return normalized


def _summarize_task_results(task_results: list[dict[str, Any]], *, router_mode: str) -> dict[str, Any]:
    task_count = len(task_results)
    valid_json_count = sum(1 for item in task_results if item.get("valid_json"))
    valid_schema_count = sum(1 for item in task_results if item.get("valid_schema"))
    hallucinated_tool_count = sum(len(item.get("hallucinated_tools") or []) for item in task_results)
    missing_required_input_count = sum(len(item.get("missing_required_inputs") or []) for item in task_results)
    selected_total = sum(len(item.get("selected_tools") or []) for item in task_results)
    extra_total = sum(len(item.get("extra_tools") or []) for item in task_results)
    return {
        "benchmark_id": "llm_as_router_planning_v1",
        "dataset": "mixed_molecular_planning_tasks",
        "router_mode": router_mode,
        "task_count": task_count,
        "valid_json_count": valid_json_count,
        "valid_schema_count": valid_schema_count,
        "invalid_json_rate": _rate(task_count - valid_json_count, task_count),
        "invalid_schema_rate": _rate(task_count - valid_schema_count, task_count),
        "hallucinated_tool_count": hallucinated_tool_count,
        "hallucinated_tool_rate": _rate(hallucinated_tool_count, selected_total),
        "missing_required_input_count": missing_required_input_count,
        "missing_required_input_rate": _rate(missing_required_input_count, task_count),
        "tool_precision": _mean(item.get("tool_precision") for item in task_results),
        "tool_recall": _mean(item.get("tool_recall") for item in task_results),
        "tool_f1": _mean(item.get("tool_f1") for item in task_results),
        "workflow_order_match_rate": _mean(1.0 if item.get("workflow_order_match") else 0.0 for item in task_results),
        "mean_selected_tool_count": selected_total / task_count if task_count else None,
        "mean_extra_tool_count": extra_total / task_count if task_count else None,
        "notes": "Planning-only LLM-router baseline; no real tool execution.",
    }


def _prf(selected_tools: list[str], expected_tools: list[str]) -> tuple[float | None, float | None, float | None]:
    selected = set(selected_tools)
    expected = set(expected_tools)
    if not selected and not expected:
        return 1.0, 1.0, 1.0
    precision = len(selected & expected) / len(selected) if selected else 0.0
    recall = len(selected & expected) / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _order_match(selected_tools: list[str], expected_tools: list[str]) -> bool:
    if not expected_tools:
        return not selected_tools
    selected_expected = [tool for tool in selected_tools if tool in expected_tools]
    return selected_expected == expected_tools


def _load_response_map(path: Path) -> dict[str, str]:
    return _raw_response_map(_load_response_record_map(path))


def _load_response_record_map(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in _read_jsonl(path):
        task_id = item.get("task_id")
        response = item.get("raw_response", item.get("response"))
        if task_id and response is not None:
            record = dict(item)
            record["raw_response"] = str(response)
            records[str(task_id)] = record
    return records


def _raw_response_map(records: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        task_id: str(record["raw_response"])
        for task_id, record in records.items()
        if record.get("raw_response") is not None
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _rate(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if denominator in (None, 0):
        return None
    return float(numerator or 0) / float(denominator)


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path | None, project_root: Path) -> Path:
    if path is None:
        raise ValueError("Path cannot be None")
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a planning-only LLM-as-router baseline.")
    parser.add_argument("--benchmark", action="append", dest="benchmarks", default=None)
    parser.add_argument("--responses-jsonl", default=None)
    parser.add_argument(
        "--router-mode",
        default="heuristic",
        choices=(
            "heuristic",
            "heuristic_overexpose",
            "heuristic_invalid_tool",
            "heuristic_invalid_json",
            "api",
            "api_replay",
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--response-log-jsonl",
        default=None,
        help="Append raw API responses here and reuse existing rows to resume interrupted API runs.",
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-model-env", default=None, help="Environment variable containing the model ID.")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument(
        "--llm-base-url-env",
        default=None,
        help="Environment variable containing the OpenAI-compatible base URL.",
    )
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help="Name of an environment variable containing the API key.",
    )
    parser.add_argument(
        "--llm-extra-body-json",
        default=None,
        help="Optional JSON object merged into the OpenAI-compatible chat completion request body.",
    )
    parser.add_argument(
        "--dotenv-path",
        default=None,
        help="Optional dotenv file loaded in addition to .env; values are never written to artifacts.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout-sec", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional task limit for smoke tests.")
    args = parser.parse_args()
    payload = run_llm_router_baseline(
        benchmark_paths=args.benchmarks or list(DEFAULT_BENCHMARKS),
        responses_jsonl=args.responses_jsonl,
        response_log_jsonl=args.response_log_jsonl,
        router_mode=args.router_mode,
        output_path=args.output,
        project_root=args.project_root,
        llm_model=args.llm_model,
        llm_model_env=args.llm_model_env,
        llm_base_url=args.llm_base_url,
        llm_base_url_env=args.llm_base_url_env,
        llm_api_key=args.llm_api_key,
        llm_api_key_env=args.llm_api_key_env,
        llm_extra_body_json=args.llm_extra_body_json,
        dotenv_path=args.dotenv_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout_sec=args.request_timeout_sec,
        limit=args.limit,
    )
    print(json.dumps(payload["row"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
