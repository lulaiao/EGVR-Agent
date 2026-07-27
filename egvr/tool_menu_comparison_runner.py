"""Matched all-tool versus task-conditioned LLM-router menu comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_router_baseline_runner import (
    _build_router_prompt_messages,
    _load_project_dotenv,
    _load_response_record_map,
    _raw_response_map,
    _resolve_api_config,
    evaluate_router_task,
)
from .rule_planner import plan_workflow
from .task_parser import parse_task
from .tool_registry import EvidenceToolRegistry, build_default_tool_registry


MENU_CONDITIONS = ("all_tool", "task_conditioned")


def run_tool_menu_comparison(
    *,
    benchmark_paths: list[str | Path],
    output_dir: str | Path,
    router_mode: str = "api",
    project_root: str | Path = ".",
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
    request_timeout_sec: float = 120.0,
    limit: int | None = None,
) -> dict[str, Any]:
    if router_mode not in {"api", "heuristic"}:
        raise ValueError("router_mode must be 'api' or 'heuristic'")
    root = Path(project_root).resolve()
    output = _resolve(output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    full_registry = build_default_tool_registry()
    api_config = None
    if router_mode == "api":
        _load_project_dotenv(root)
        if dotenv_path is not None:
            from .llm_router_baseline_runner import _load_dotenv_path

            _load_dotenv_path(_resolve(dotenv_path, root))
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

    tasks: list[tuple[Path, dict[str, Any]]] = []
    for benchmark in benchmark_paths:
        path = _resolve(benchmark, root)
        tasks.extend((path, row) for row in _read_jsonl(path))
        if limit is not None and len(tasks) >= limit:
            tasks = tasks[:limit]
            break

    condition_results: dict[str, list[dict[str, Any]]] = {condition: [] for condition in MENU_CONDITIONS}
    response_record_maps = {
        condition: _load_response_record_map(output / f"{condition}.responses.jsonl")
        if (output / f"{condition}.responses.jsonl").exists()
        else {}
        for condition in MENU_CONDITIONS
    }
    response_maps = {
        condition: _raw_response_map(records)
        for condition, records in response_record_maps.items()
    }
    for task_index, (benchmark_path, task) in enumerate(tasks):
        task_registry = _task_conditioned_registry(task, full_registry)
        fixed_hashes: set[str] = set()
        for condition, registry in (("all_tool", full_registry), ("task_conditioned", task_registry)):
            response_log = output / f"{condition}.responses.jsonl" if router_mode == "api" else None
            result = evaluate_router_task(
                task,
                benchmark_path=benchmark_path,
                registry=registry,
                responses=response_maps[condition],
                response_records=response_record_maps[condition],
                router_mode=router_mode,
                project_root=root,
                api_config=api_config,
                response_log_path=response_log,
            )
            if result.get("raw_response"):
                response_maps[condition][str(task.get("task_id"))] = str(result["raw_response"])
            if result.get("prompt_messages") is None:
                result["prompt_messages"] = _build_router_prompt_messages(task=task, registry=registry)
            result.update(
                {
                    "menu_condition": condition,
                    "model": api_config["model"] if api_config else "heuristic",
                    "temperature": temperature,
                    "exposed_tools": registry.names(),
                    "exposed_tool_count": len(registry),
                    "required_tool_recall": result["tool_recall"],
                    "dependency_order_valid": _dependency_order_valid(
                        result["selected_tools"], result["expected_tools"]
                    ),
                    "prompt_hash": _prompt_hash(result.get("prompt_messages")),
                    "fixed_context_hash": _fixed_context_hash(result.get("prompt_messages")),
                    "task_pair_index": task_index,
                    "source": "new_api_run" if router_mode == "api" else "test_only_heuristic",
                }
            )
            fixed_hashes.add(result["fixed_context_hash"])
            condition_results[condition].append(result)
        if len(fixed_hashes) != 1:
            raise RuntimeError(f"Non-registry prompt content changed between menu conditions for {task.get('task_id')}")

    summaries = [
        _summarize(condition, rows, router_mode=router_mode, temperature=temperature)
        for condition, rows in condition_results.items()
    ]
    recorded_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    for condition, rows in condition_results.items():
        payload = {
            "benchmark_id": "strict_tool_menu_comparison_v1",
            "execution_mode": "planning_only",
            "router_mode": router_mode,
            "menu_condition": condition,
            "recorded_at": recorded_at,
            "task_count": len(rows),
            "task_results": rows,
            "row": next(item for item in summaries if item["menu_condition"] == condition),
            "notes": [
                "Matched comparison: task, model, temperature, system prompt, output contract, and evaluator are fixed.",
                "Only the serialized tool_registry differs between menu conditions.",
                "Task-conditioned menus are selected deterministically from the parsed task and rule planner.",
            ],
        }
        _write_json(output / f"{condition}.planning.json", payload)
        _write_csv(output / f"{condition}.per_task.csv", rows)

    result = {
        "experiment_id": "strict_tool_menu_comparison_v1",
        "recorded_at": recorded_at,
        "router_mode": router_mode,
        "model": api_config["model"] if api_config else "heuristic",
        "temperature": temperature,
        "task_count": len(tasks),
        "rows": summaries,
        "source": "new_api_run" if router_mode == "api" else "test_only_heuristic",
    }
    _write_json(output / "tool_menu_comparison_summary.json", result)
    _write_csv(output / "tool_menu_comparison_summary.csv", summaries)
    (output / "tool_menu_comparison_summary.tex").write_text(_latex(summaries), encoding="utf-8")
    return result


def _task_conditioned_registry(task: dict[str, Any], full_registry: EvidenceToolRegistry) -> EvidenceToolRegistry:
    task_id = str(task.get("task_id") or "task")
    parsed = parse_task(
        str(task.get("raw_user_query") or ""),
        task_id=task_id,
        metadata=dict(task.get("metadata") or {}),
    )
    workflow = plan_workflow(parsed, registry=full_registry)
    names = list(dict.fromkeys(workflow.selected_tools))
    return EvidenceToolRegistry(full_registry.require(name) for name in names)


def _dependency_order_valid(selected_tools: list[str], expected_tools: list[str]) -> bool:
    expected_position = {tool: index for index, tool in enumerate(expected_tools)}
    observed = [tool for tool in selected_tools if tool in expected_position]
    positions = [expected_position[tool] for tool in observed]
    return positions == sorted(positions)


def _summarize(
    condition: str,
    rows: list[dict[str, Any]],
    *,
    router_mode: str,
    temperature: float,
) -> dict[str, Any]:
    total = len(rows)
    return {
        "menu_condition": condition,
        "model": rows[0]["model"] if rows else None,
        "router_mode": router_mode,
        "temperature": temperature,
        "task_count": total,
        "schema_validity": _rate(row["valid_schema"] for row in rows),
        "required_tool_recall": _mean(row["required_tool_recall"] for row in rows),
        "tool_precision": _mean(row["tool_precision"] for row in rows),
        "dependency_order_validity": _rate(row["dependency_order_valid"] for row in rows),
        "exact_order_match_rate": _rate(row["workflow_order_match"] for row in rows),
        "mean_exposed_tools": _mean(row["exposed_tool_count"] for row in rows),
        "mean_selected_tools": _mean(row["selected_tool_count"] for row in rows),
        "hallucinated_tool_count": sum(len(row["hallucinated_tools"]) for row in rows),
        "missing_required_input_count": sum(len(row["missing_required_inputs"]) for row in rows),
        "source": rows[0]["source"] if rows else None,
    }


def _prompt_hash(messages: list[dict[str, str]] | None) -> str | None:
    if not messages:
        return None
    data = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _fixed_context_hash(messages: list[dict[str, str]] | None) -> str:
    if not messages:
        return "heuristic"
    clean = copy_messages = json.loads(json.dumps(messages))
    for message in copy_messages:
        if message.get("role") != "user":
            continue
        payload = json.loads(message["content"])
        payload["tool_registry"] = "<menu-varies>"
        message["content"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    data = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated strict tool-menu comparison.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Menu & Exposed & Schema & Recall & Precision & Dep. order \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['menu_condition'])} & {row['mean_exposed_tools']:.1f} & "
            f"{_pct(row['schema_validity'])} & {_pct(row['required_tool_recall'])} & "
            f"{_pct(row['tool_precision'])} & {_pct(row['dependency_order_validity'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Matched LLM-router comparison under all-tool and task-conditioned menus.}",
            "\\label{tab:strict-tool-menu}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_response_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    responses: dict[str, str] = {}
    for row in _read_jsonl(path):
        if row.get("task_id") and row.get("raw_response"):
            responses[str(row["task_id"])] = str(row["raw_response"])
    return responses


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _rate(values) -> float | None:
    flags = [bool(value) for value in values]
    return sum(flags) / len(flags) if flags else None


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}\\%"


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a matched LLM-router tool-menu comparison.")
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--router-mode", choices=["api", "heuristic"], default="api")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-model-env")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-base-url-env")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-extra-body-json")
    parser.add_argument(
        "--dotenv-path",
        help="Optional dotenv file loaded without writing any secret values to experiment artifacts.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    payload = run_tool_menu_comparison(
        benchmark_paths=args.benchmark,
        output_dir=args.output_dir,
        router_mode=args.router_mode,
        project_root=args.project_root,
        llm_model=args.llm_model,
        llm_model_env=args.llm_model_env,
        llm_base_url=args.llm_base_url,
        llm_base_url_env=args.llm_base_url_env,
        llm_api_key_env=args.llm_api_key_env,
        llm_extra_body_json=args.llm_extra_body_json,
        dotenv_path=args.dotenv_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout_sec=args.request_timeout_sec,
        limit=args.limit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
