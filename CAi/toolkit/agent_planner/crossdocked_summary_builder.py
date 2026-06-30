"""Build paper-table summary records for CrossDocked generation runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


def build_crossdocked_summary_record(
    *,
    benchmark_path: str | Path,
    result_path: str | Path,
    trace_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Summarize a completed CrossDocked benchmark run into a reusable record."""

    benchmark_file = Path(benchmark_path)
    result_file = Path(result_path)
    trace_file = Path(trace_path)
    output_file = Path(output_path)

    benchmark_cases = _load_benchmark_cases(benchmark_file)
    result_payload = _load_json(result_file)
    traces = {str(record.get("task_id")): record for record in _load_jsonl(trace_file)}
    results = result_payload.get("results", [])
    per_target = []
    all_smiles: set[str] = set()

    for result in results:
        task_id = str(result.get("task_id"))
        metadata = benchmark_cases.get(task_id, {}).get("metadata", {})
        trace = traces.get(task_id, {})
        candidates = trace.get("final_candidates", [])
        smiles = [str(candidate.get("smiles")) for candidate in candidates if candidate.get("smiles")]
        all_smiles.update(smiles)
        scscores = [_as_float(candidate.get("scscore")) for candidate in candidates]
        toxicity_scores = [_as_float(candidate.get("toxicity_score")) for candidate in candidates]
        proxy_scores = [
            _as_float(candidate.get("metadata", {}).get("proxy_score"))
            for candidate in candidates
            if isinstance(candidate.get("metadata"), dict)
        ]
        candidate_count = _as_int(result.get("metrics", {}).get("candidate_count"), len(candidates))
        tool_elapsed = result.get("tool_elapsed_sec", {}) if isinstance(result.get("tool_elapsed_sec"), dict) else {}
        toxicity_total = _as_float(tool_elapsed.get("toxicity"), 0.0)
        total_elapsed = _as_float(result.get("total_elapsed_sec"), 0.0)
        per_target.append(
            {
                "task_id": task_id,
                "protein_id": metadata.get("protein_id"),
                "candidate_count": candidate_count,
                "unique_smiles_count": _as_int(result.get("metrics", {}).get("unique_smiles_count"), len(set(smiles))),
                "proxy_score_range": _range(proxy_scores),
                "best_scscore": _min_non_null(scscores),
                "max_toxicity_score": _max_non_null(toxicity_scores),
                "tool_elapsed_sec": {
                    "rxnflow": _round_or_none(_as_float(tool_elapsed.get("rxnflow"))),
                    "scscore": _round_or_none(_as_float(tool_elapsed.get("scscore"))),
                    "toxicity_total": _round_or_none(toxicity_total),
                    "toxicity_avg_per_candidate": _round_or_none(
                        toxicity_total / candidate_count if candidate_count else None
                    ),
                    "total": _round_or_none(total_elapsed),
                },
            }
        )

    candidate_counts = [_as_int(item.get("candidate_count"), 0) for item in per_target]
    unique_counts = [_as_int(item.get("unique_smiles_count"), 0) for item in per_target]
    best_scscores = [_as_float(item.get("best_scscore")) for item in per_target]
    max_toxicity_scores = [_as_float(item.get("max_toxicity_score")) for item in per_target]
    elapsed_total = [_as_float(item.get("tool_elapsed_sec", {}).get("total")) for item in per_target]
    elapsed_rxnflow = [_as_float(item.get("tool_elapsed_sec", {}).get("rxnflow")) for item in per_target]
    elapsed_scscore = [_as_float(item.get("tool_elapsed_sec", {}).get("scscore")) for item in per_target]
    elapsed_toxicity = [_as_float(item.get("tool_elapsed_sec", {}).get("toxicity_total")) for item in per_target]
    fastest = _elapsed_extreme(per_target, "min")
    slowest = _elapsed_extreme(per_target, "max")

    record = {
        "benchmark_id": benchmark_file.stem,
        "benchmark_path": str(benchmark_file),
        "dataset": "CrossDocked2020",
        "execution_mode": result_payload.get("execution_mode", "real"),
        "planner_baseline": result_payload.get("planner_baseline", "rule_based_planner"),
        "result_path": str(result_file),
        "trace_log_path": str(trace_file),
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "task_count": len(results),
        "num_candidates_per_task": _num_candidates_per_task(benchmark_cases),
        "summary": result_payload.get("summary", {}),
        "global_candidate_summary": {
            "generated_candidate_count": sum(candidate_counts),
            "valid_candidate_count": sum(
                _as_int(result.get("metrics", {}).get("valid_smiles_count"), 0) for result in results
            ),
            "unique_smiles_count_across_tasks": len(all_smiles),
            "per_task_unique_smiles_min": min(unique_counts) if unique_counts else None,
            "per_task_unique_smiles_max": max(unique_counts) if unique_counts else None,
            "best_scscore_overall": _min_non_null(best_scscores),
            "max_toxicity_score_overall": _max_non_null(max_toxicity_scores),
        },
        "elapsed_summary_sec": {
            "mean_total": _mean_non_null(elapsed_total),
            "median_total": _median_non_null(elapsed_total),
            "mean_rxnflow": _mean_non_null(elapsed_rxnflow),
            "mean_scscore": _mean_non_null(elapsed_scscore),
            "mean_toxicity_total": _mean_non_null(elapsed_toxicity),
            "fastest_target": fastest.get("protein_id"),
            "fastest_target_total": fastest.get("total"),
            "slowest_target": slowest.get("protein_id"),
            "slowest_target_total": slowest.get("total"),
        },
        "per_target": per_target,
        "notes": [
            (
                f"All {len(results)} CrossDocked pocket-conditioned generation tasks completed with "
                f"{sum(candidate_counts)} generated candidates."
            ),
            "Toxicity evaluation remains the dominant elapsed-time component.",
        ],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return record


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _load_benchmark_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            cases[str(payload.get("task_id"))] = payload
    return cases


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def _clean(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _mean_non_null(values: list[float | None]) -> float | None:
    clean = _clean(values)
    return mean(clean) if clean else None


def _median_non_null(values: list[float | None]) -> float | None:
    clean = _clean(values)
    return median(clean) if clean else None


def _min_non_null(values: list[float | None]) -> float | None:
    clean = _clean(values)
    return min(clean) if clean else None


def _max_non_null(values: list[float | None]) -> float | None:
    clean = _clean(values)
    return max(clean) if clean else None


def _range(values: list[float | None]) -> list[float] | None:
    clean = _clean(values)
    if not clean:
        return None
    return [min(clean), max(clean)]


def _elapsed_extreme(per_target: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = [
        {
            "protein_id": item.get("protein_id"),
            "total": _as_float(item.get("tool_elapsed_sec", {}).get("total")),
        }
        for item in per_target
        if isinstance(item.get("tool_elapsed_sec"), dict)
    ]
    rows = [row for row in rows if row["total"] is not None]
    if not rows:
        return {}
    return min(rows, key=lambda row: row["total"]) if mode == "min" else max(rows, key=lambda row: row["total"])


def _num_candidates_per_task(benchmark_cases: dict[str, dict[str, Any]]) -> int | None:
    counts = {
        _as_int(case.get("metadata", {}).get("num_candidates"), 0)
        for case in benchmark_cases.values()
        if isinstance(case.get("metadata"), dict)
    }
    counts.discard(0)
    return counts.pop() if len(counts) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CrossDocked generation summary record.")
    parser.add_argument("--benchmark", required=True, help="Input CrossDocked benchmark JSONL.")
    parser.add_argument("--result", required=True, help="Completed benchmark_runner result JSON.")
    parser.add_argument("--trace", required=True, help="Trace JSONL emitted by benchmark_runner.")
    parser.add_argument("--output", required=True, help="Output summary record JSON.")
    args = parser.parse_args()
    record = build_crossdocked_summary_record(
        benchmark_path=args.benchmark,
        result_path=args.result,
        trace_path=args.trace,
        output_path=args.output,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
