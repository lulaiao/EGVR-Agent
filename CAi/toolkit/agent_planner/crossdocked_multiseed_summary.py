"""Aggregate CrossDocked generation summaries across repeated RxnFlow seeds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_OUTPUT = "logs/baseline_runs/crossdocked_multiseed_v1/crossdocked_multiseed_summary.json"


def build_crossdocked_multiseed_summary(
    *,
    seed_runs: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT,
    benchmark_id: str = "crossdocked_rxnflow_candidates5_targets30_multiseed_v1",
) -> dict[str, Any]:
    rows = [_seed_row(run) for run in seed_runs]
    aggregate = _aggregate_row(rows, benchmark_id=benchmark_id)
    payload = {
        "benchmark_id": benchmark_id,
        "dataset": "CrossDocked2020",
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "rows": [aggregate],
        "seed_rows": rows,
        "notes": [
            "Rows aggregate completed CrossDocked30 generation runs across RxnFlow seeds.",
            "Mean/std summarize workflow repeatability and verifier evidence, not molecular-design SOTA.",
        ],
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _seed_row(run: dict[str, Any]) -> dict[str, Any]:
    seed = int(run["seed"])
    record_path = Path(run["record_path"])
    record = _load_json(record_path)
    verifier = _load_optional_json(Path(run["verifier_summary_path"])) if run.get("verifier_summary_path") else {}
    prop = _load_optional_json(Path(run["property_summary_path"])) if run.get("property_summary_path") else {}
    global_summary = record.get("global_candidate_summary", {})
    task_count = _as_int(record.get("task_count"))
    generated_count = _as_int(global_summary.get("generated_candidate_count"))
    valid_count = _as_int(global_summary.get("valid_candidate_count"))
    unique_count = _as_int(global_summary.get("unique_smiles_count_across_tasks"))
    summary = record.get("summary", {})
    elapsed = record.get("elapsed_summary_sec", {})
    sa_row = _find_row(verifier.get("rows", []), "evidence_type", "sa_score")
    property_row = _first(prop.get("property_rows", []))
    return {
        "seed": seed,
        "benchmark_id": record.get("benchmark_id"),
        "task_count": task_count,
        "generated_candidate_count": generated_count,
        "valid_candidate_count": valid_count,
        "valid_candidate_rate": valid_count / generated_count if generated_count else None,
        "unique_smiles_count": unique_count,
        "unique_smiles_rate": unique_count / generated_count if generated_count else None,
        "task_success_rate": _as_float(summary.get("task_success_rate")),
        "false_success_count": _as_int(summary.get("false_success_count")),
        "verifier_expectation_match": _as_float(summary.get("verifier_expectation_match")),
        "mean_total_elapsed_sec": _as_float(elapsed.get("mean_total")),
        "best_scscore": _as_float(global_summary.get("best_scscore_overall")),
        "max_toxicity_score": _as_float(global_summary.get("max_toxicity_score_overall")),
        "sa_score_coverage": _as_float(sa_row.get("coverage")),
        "sa_score_pass_rate": _as_float(sa_row.get("pass_rate")),
        "rdkit_property_coverage": _as_float(property_row.get("property_coverage")),
        "mean_qed": _as_float(property_row.get("mean_qed")),
        "record_path": str(record_path),
        "verifier_summary_path": run.get("verifier_summary_path"),
        "property_summary_path": run.get("property_summary_path"),
    }


def _aggregate_row(rows: list[dict[str, Any]], *, benchmark_id: str) -> dict[str, Any]:
    total_target_runs = sum(_as_int(row.get("task_count")) for row in rows)
    total_candidates = sum(_as_int(row.get("generated_candidate_count")) for row in rows)
    false_success_count = sum(_as_int(row.get("false_success_count")) for row in rows)
    return {
        "benchmark_id": benchmark_id,
        "dataset": "CrossDocked2020",
        "seed_count": len(rows),
        "seeds": ",".join(str(row.get("seed")) for row in rows),
        "total_target_runs": total_target_runs,
        "total_candidates": total_candidates,
        "mean_task_success_rate": _mean(row.get("task_success_rate") for row in rows),
        "std_task_success_rate": _std(row.get("task_success_rate") for row in rows),
        "mean_valid_candidate_rate": _mean(row.get("valid_candidate_rate") for row in rows),
        "std_valid_candidate_rate": _std(row.get("valid_candidate_rate") for row in rows),
        "mean_unique_smiles_rate": _mean(row.get("unique_smiles_rate") for row in rows),
        "std_unique_smiles_rate": _std(row.get("unique_smiles_rate") for row in rows),
        "mean_sa_score_coverage": _mean(row.get("sa_score_coverage") for row in rows),
        "mean_sa_score_pass_rate": _mean(row.get("sa_score_pass_rate") for row in rows),
        "mean_rdkit_property_coverage": _mean(row.get("rdkit_property_coverage") for row in rows),
        "mean_qed": _mean(row.get("mean_qed") for row in rows),
        "mean_seconds_per_task": _mean(row.get("mean_total_elapsed_sec") for row in rows),
        "std_seconds_per_task": _std(row.get("mean_total_elapsed_sec") for row in rows),
        "false_success_count": false_success_count,
        "notes": (
            "Aggregates CrossDocked30 seed repeats; diversity is reported as diagnostic evidence, "
            "not as a molecular-quality claim."
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _find_row(rows: list[Any], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict) and row.get(key) == value:
            return row
    return {}


def _first(rows: list[Any]) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict):
            return row
    return {}


def _clean(values) -> list[float]:
    cleaned = []
    for value in values:
        number = _as_float(value)
        if number is not None:
            cleaned.append(number)
    return cleaned


def _mean(values) -> float | None:
    cleaned = _clean(values)
    return mean(cleaned) if cleaned else None


def _std(values) -> float | None:
    cleaned = _clean(values)
    return stdev(cleaned) if len(cleaned) > 1 else 0.0 if cleaned else None


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_seed_run(value: str) -> dict[str, Any]:
    fields = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(f"Seed run entries must use key=value fields: {value}")
        key, field_value = item.split("=", 1)
        fields[key.strip()] = field_value.strip()
    if "seed" not in fields or "record_path" not in fields:
        raise ValueError("--seed-run requires at least seed= and record_path=")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CrossDocked multi-seed summary tables.")
    parser.add_argument("--seed-run", action="append", default=[], help="Comma fields: seed=1,record_path=...,verifier_summary_path=...,property_summary_path=...")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output summary JSON.")
    parser.add_argument("--benchmark-id", default="crossdocked_rxnflow_candidates5_targets30_multiseed_v1")
    args = parser.parse_args()
    payload = build_crossdocked_multiseed_summary(
        seed_runs=[_parse_seed_run(item) for item in args.seed_run],
        output_path=args.output,
        benchmark_id=args.benchmark_id,
    )
    print(json.dumps({"summary_json": args.output, "rows": payload["rows"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
