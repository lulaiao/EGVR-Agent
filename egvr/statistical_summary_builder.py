"""Build statistical summaries for paper-facing benchmark claims.

The builder consumes existing master-table views. It provides lightweight
Wilson confidence intervals for rate metrics and preserves repeated-run
standard deviations when they are already present in benchmark summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MASTER_TABLE = "logs/baseline_runs/master_baseline_table/master_baseline_table.json"
DEFAULT_OUTPUT_DIR = "logs/baseline_runs/master_baseline_table"

STATISTICAL_SUMMARY_COLUMNS = [
    "claim_id",
    "benchmark_id",
    "dataset",
    "planner_baseline",
    "metric",
    "n",
    "estimate",
    "std",
    "ci95_low",
    "ci95_high",
    "false_success_count",
    "source_table",
    "interpretation",
]


def build_statistical_summary_rows(
    *,
    master_payload: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Create compact statistical rows from master-table views."""

    root = _project_root(project_root)
    payload = master_payload or _load_optional_json(_resolve_path(DEFAULT_MASTER_TABLE, root))
    views = payload.get("views", {}) if isinstance(payload.get("views"), dict) else {}
    rows: list[dict[str, Any]] = []

    for row in _view_rows(views, "robustness_repeated"):
        task_count = _as_int(row.get("task_count"))
        n = task_count
        rows.append(
            _rate_row(
                claim_id="C1",
                source_table="robustness_repeated",
                source=row,
                metric="task_success_rate",
                n=n,
                estimate=_as_float(row.get("mean_task_success_rate")),
                std=_as_float(row.get("std_task_success_rate")),
                interpretation="Repeated controlled robustness; n counts unique scenarios, not repeated observations.",
            )
        )
        rows.append(
            _rate_row(
                claim_id="C1",
                source_table="robustness_repeated",
                source=row,
                metric="verifier_expectation_match",
                n=n,
                estimate=_as_float(row.get("mean_verifier_expectation_match")),
                std=_as_float(row.get("std_verifier_expectation_match")),
                interpretation="Expectation match audits false success avoidance under controlled failures.",
            )
        )

    for row in _view_rows(views, "repair_ablation_repeated"):
        task_count = _as_int(row.get("task_count"))
        n = task_count
        rows.append(
            _rate_row(
                claim_id="C3",
                source_table="repair_ablation_repeated",
                source=row,
                metric="task_success_rate",
                n=n,
                estimate=_as_float(row.get("mean_task_success_rate")),
                std=_as_float(row.get("std_task_success_rate")),
                interpretation="Repeated ambiguous-evidence ablation separates verifier-guided repair from scheduled fallback.",
            )
        )
        rows.append(
            _rate_row(
                claim_id="C3",
                source_table="repair_ablation_repeated",
                source=row,
                metric="repair_success_rate",
                n=n,
                estimate=_as_float(row.get("mean_repair_success_rate")),
                std=_as_float(row.get("std_repair_success_rate")),
                interpretation="Repair rate is reported only for the controlled repair-ablation setting.",
            )
        )

    for row in _view_rows(views, "crossdocked_multiseed"):
        target_runs = _as_int(row.get("total_target_runs"))
        candidates = _as_int(row.get("total_candidates"))
        rows.append(
            _rate_row(
                claim_id="C4",
                source_table="crossdocked_multiseed",
                source=row,
                metric="task_success_rate",
                n=target_runs,
                estimate=_as_float(row.get("mean_task_success_rate")),
                std=_as_float(row.get("std_task_success_rate")),
                interpretation="CrossDocked30 repeatability over target-runs across seeds.",
            )
        )
        rows.append(
            _rate_row(
                claim_id="C4",
                source_table="crossdocked_multiseed",
                source=row,
                metric="valid_candidate_rate",
                n=candidates,
                estimate=_as_float(row.get("mean_valid_candidate_rate")),
                std=_as_float(row.get("std_valid_candidate_rate")),
                interpretation="Candidate validity over all generated candidates across seed repeats.",
            )
        )
        rows.append(
            _rate_row(
                claim_id="C4",
                source_table="crossdocked_multiseed",
                source=row,
                metric="sa_score_pass_rate",
                n=candidates,
                estimate=_as_float(row.get("mean_sa_score_pass_rate")),
                std=None,
                interpretation="SA pass rate is verifier evidence, not molecular-design SOTA.",
            )
        )

    for row in _view_rows(views, "verifier_evidence"):
        evidence_count = _as_int(row.get("evidence_count")) or _as_int(row.get("evaluable_candidate_count"))
        candidate_count = _as_int(row.get("candidate_count"))
        rows.append(
            _rate_row(
                claim_id="C6",
                source_table="verifier_evidence",
                source=row,
                metric=f"{row.get('evidence_type')}_coverage",
                n=candidate_count,
                estimate=_as_float(row.get("coverage")),
                std=None,
                interpretation="Coverage is reported separately from pass/fail evidence.",
            )
        )
        rows.append(
            _rate_row(
                claim_id="C6",
                source_table="verifier_evidence",
                source=row,
                metric=f"{row.get('evidence_type')}_pass_rate",
                n=evidence_count,
                estimate=_as_float(row.get("pass_rate")),
                std=None,
                interpretation="Pass rate is evaluated only over candidates with available evidence.",
            )
        )

    for row in _view_rows(views, "pdbbind_prep_gate"):
        prep_n = _as_int(row.get("receptor_prep_target_count"))
        pilot_n = _as_int(row.get("real_pilot_task_count"))
        rows.append(
            _rate_row(
                claim_id="C8",
                source_table="pdbbind_prep_gate",
                source=row,
                metric="receptor_prep_success_rate",
                n=prep_n,
                estimate=_as_float(row.get("prep_success_rate")),
                std=None,
                interpretation="Prepared-receptor gate is infrastructure evidence, not an affinity benchmark.",
            )
        )
        rows.append(
            _rate_row(
                claim_id="C8",
                source_table="pdbbind_prep_gate",
                source=row,
                metric="real_pilot_success_rate",
                n=pilot_n,
                estimate=_as_float(row.get("real_pilot_success_rate")),
                std=None,
                interpretation="Real pilot success counts failed docking tasks explicitly.",
            )
        )

    return [_ordered_row(row) for row in rows if row.get("estimate") is not None]


def write_statistical_summary_table(
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    table_name: str = "statistical_summary_table",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    out_dir = _resolve_path(output_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{table_name}.csv"
    json_path = out_dir / f"{table_name}.json"
    tex_path = out_dir / f"{table_name}.tex"
    _write_csv(rows, csv_path, STATISTICAL_SUMMARY_COLUMNS)
    payload = {
        "table_id": table_name,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "columns": STATISTICAL_SUMMARY_COLUMNS,
        "row_count": len(rows),
        "rows": rows,
        "notes": [
            "Wilson intervals are approximate and intended for reporting uncertainty, not for claiming statistical significance.",
            "Repeated-run standard deviations are preserved when available in benchmark summaries.",
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tex_path.write_text(render_latex(rows), encoding="utf-8")
    payload["artifacts"] = {
        "csv": _display_path(csv_path, root),
        "json": _display_path(json_path, root),
        "tex": _display_path(tex_path, root),
    }
    return payload


def build_and_write_statistical_summary_table(
    *,
    master_table_path: str | Path = DEFAULT_MASTER_TABLE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    payload = _load_optional_json(_resolve_path(master_table_path, root))
    rows = build_statistical_summary_rows(master_payload=payload, project_root=root)
    return write_statistical_summary_table(rows, output_dir=output_dir, project_root=root)


def render_latex(rows: list[dict[str, Any]]) -> str:
    compact_rows = [
        row
        for row in rows
        if row.get("metric")
        in {
            "task_success_rate",
            "repair_success_rate",
            "real_pilot_success_rate",
            "receptor_prep_success_rate",
            "valid_candidate_rate",
        }
    ][:10]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Statistical summary for key rate metrics.}",
        r"\label{tab:statistical-summary}",
        r"\begin{tabular}{lllll}",
        r"\hline",
        r"Benchmark & Metric & Method & Estimate & 95\% CI \\",
        r"\hline",
    ]
    for row in compact_rows:
        interval = _format_interval(row.get("ci95_low"), row.get("ci95_high"))
        lines.append(
            " & ".join(
                _escape_latex(value)
                for value in (
                    row.get("benchmark_id"),
                    row.get("metric"),
                    row.get("planner_baseline") or "--",
                    _format_rate(row.get("estimate")),
                    interval,
                )
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _rate_row(
    *,
    claim_id: str,
    source_table: str,
    source: dict[str, Any],
    metric: str,
    n: int | None,
    estimate: float | None,
    std: float | None,
    interpretation: str,
) -> dict[str, Any]:
    ci_low, ci_high = _wilson_interval(estimate, n)
    return {
        "claim_id": claim_id,
        "benchmark_id": source.get("benchmark_id") or source.get("source_benchmark_id"),
        "dataset": source.get("dataset"),
        "planner_baseline": source.get("planner_baseline"),
        "metric": metric,
        "n": n,
        "estimate": estimate,
        "std": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "false_success_count": source.get("false_success_count"),
        "source_table": source_table,
        "interpretation": interpretation,
    }


def _wilson_interval(rate: float | None, n: int | None) -> tuple[float | None, float | None]:
    if rate is None or n is None or n <= 0:
        return None, None
    p = min(max(rate, 0.0), 1.0)
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _view_rows(views: dict[str, Any], name: str) -> list[dict[str, Any]]:
    view = views.get(name, {}) if isinstance(views.get(name), dict) else {}
    return [item for item in view.get("rows", []) if isinstance(item, dict)]


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _ordered_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in STATISTICAL_SUMMARY_COLUMNS}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _format_rate(value: Any) -> str:
    number = _as_float(value)
    return "--" if number is None else f"{number * 100:.1f}%"


def _format_interval(low: Any, high: Any) -> str:
    low_f = _as_float(low)
    high_f = _as_float(high)
    if low_f is None or high_f is None:
        return "--"
    return f"[{low_f * 100:.1f}, {high_f * 100:.1f}]"


def _escape_latex(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


CLUSTERED_COLUMNS = [
    "benchmark_id",
    "row_type",
    "planner_baseline",
    "comparator",
    "metric",
    "unique_scenario_count",
    "observation_count",
    "repeat_count",
    "estimate",
    "ci95_low",
    "ci95_high",
    "false_success_count",
]


def build_clustered_statistical_rows(
    result_paths: list[str | Path],
    *,
    project_root: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    random_seed: int = 20260707,
) -> list[dict[str, Any]]:
    """Build scenario-clustered method and paired-difference estimates."""

    root = _project_root(project_root)
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    false_success: dict[tuple[str, str], int] = defaultdict(int)
    for raw_path in result_paths:
        path = _resolve_path(raw_path, root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = str(payload.get("planner_baseline") or "unknown")
        results = [item for item in payload.get("results", []) if isinstance(item, dict)]
        benchmark_id = str(payload.get("benchmark_id") or _infer_benchmark_id(path, results))
        for item in results:
            scenario = str(item.get("scenario_template_id") or item.get("task_id"))
            grouped[(benchmark_id, baseline)][scenario].append(item)
            if item.get("agent_claimed_success", item.get("task_success")) and not item.get(
                "evidence_verified_success", item.get("task_success")
            ):
                false_success[(benchmark_id, baseline)] += 1

    rng = random.Random(random_seed)
    rows: list[dict[str, Any]] = []
    benchmarks: dict[str, list[str]] = defaultdict(list)
    for benchmark_id, baseline in sorted(grouped):
        benchmarks[benchmark_id].append(baseline)
        clusters = grouped[(benchmark_id, baseline)]
        values = {scenario: _mean_bool(items, "task_success") for scenario, items in clusters.items()}
        estimate = sum(values.values()) / len(values) if values else None
        low, high = _cluster_bootstrap_interval(values, rng, bootstrap_samples)
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "row_type": "method",
                "planner_baseline": baseline,
                "comparator": None,
                "metric": "task_success_rate",
                "unique_scenario_count": len(values),
                "observation_count": sum(len(items) for items in clusters.values()),
                "repeat_count": max((len(items) for items in clusters.values()), default=0),
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "false_success_count": false_success[(benchmark_id, baseline)],
            }
        )

    for benchmark_id, baselines in sorted(benchmarks.items()):
        repair_baseline = _preferred_repair_baseline(baselines)
        if repair_baseline is None:
            continue
        full = grouped[(benchmark_id, repair_baseline)]
        full_values = {scenario: _mean_bool(items, "task_success") for scenario, items in full.items()}
        for comparator in sorted(set(baselines) - {repair_baseline}):
            other = grouped[(benchmark_id, comparator)]
            other_values = {scenario: _mean_bool(items, "task_success") for scenario, items in other.items()}
            common = sorted(set(full_values) & set(other_values))
            deltas = {scenario: full_values[scenario] - other_values[scenario] for scenario in common}
            estimate = sum(deltas.values()) / len(deltas) if deltas else None
            low, high = _cluster_bootstrap_interval(deltas, rng, bootstrap_samples)
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "row_type": "paired_delta",
                    "planner_baseline": repair_baseline,
                    "comparator": comparator,
                    "metric": "task_success_rate_delta",
                    "unique_scenario_count": len(deltas),
                    "observation_count": sum(len(full[scenario]) + len(other[scenario]) for scenario in common),
                    "repeat_count": None,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "false_success_count": false_success[(benchmark_id, repair_baseline)],
                }
            )
    return rows


def _preferred_repair_baseline(baselines: list[str]) -> str | None:
    if "egvr_agent" in baselines:
        return "egvr_agent"
    if "full_copilot" in baselines:
        return "full_copilot"
    return None


def write_clustered_statistical_summary(
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    project_root: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    random_seed: int = 20260707,
) -> dict[str, Any]:
    root = _project_root(project_root)
    target = _resolve_path(output_dir, root)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "table_id": "statistical_summary_clustered",
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "row_count": len(rows),
        "rows": rows,
        "notes": [
            "The unit of resampling is the unique scenario template.",
            "Repeated observations contribute to a scenario mean but do not inflate the cluster count.",
            "Paired deltas compare EGVR-Agent with baselines on shared scenarios.",
        ],
    }
    _write_csv(rows, target / "statistical_summary_clustered.csv", CLUSTERED_COLUMNS)
    (target / "statistical_summary_clustered.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (target / "statistical_summary_clustered.tex").write_text(
        _clustered_latex(rows), encoding="utf-8"
    )
    return payload


def _mean_bool(items: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 if item.get(key) else 0.0 for item in items) / len(items)


def _cluster_bootstrap_interval(
    values: dict[str, float],
    rng: random.Random,
    samples: int,
) -> tuple[float | None, float | None]:
    keys = sorted(values)
    if not keys:
        return None, None
    estimates: list[float] = []
    for _ in range(samples):
        selected = [keys[rng.randrange(len(keys))] for _ in keys]
        estimates.append(sum(values[key] for key in selected) / len(selected))
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _percentile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _infer_benchmark_id(path: Path, results: list[dict[str, Any]]) -> str:
    if results and results[0].get("scenario_template_id"):
        return str(results[0]["scenario_template_id"]).split(":", 1)[0]
    name = path.stem
    for suffix in (
        ".egvr_agent",
        ".full_copilot",
        ".rule_based_planner",
        ".verifier_only_no_repair",
        ".scheduled_fallback_no_verifier",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _clustered_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated scenario-clustered statistical summary.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llllrr}",
        r"\hline",
        r"Benchmark & Row & Method & Comparator & Estimate & 95\% CI \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                _escape_latex(value)
                for value in (
                    row.get("benchmark_id"),
                    row.get("row_type"),
                    row.get("planner_baseline"),
                    row.get("comparator") or "--",
                    _format_rate(row.get("estimate")),
                    _format_interval(row.get("ci95_low"), row.get("ci95_high")),
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Scenario-clustered uncertainty for controlled reliability experiments.}",
            r"\label{tab:statistical-summary-clustered}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build statistical summary table from master views.")
    parser.add_argument("--master-table", default=DEFAULT_MASTER_TABLE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cluster-result", action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260707)
    args = parser.parse_args()
    payload = build_and_write_statistical_summary_table(
        master_table_path=args.master_table,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    output = {"standard": payload["artifacts"]}
    if args.cluster_result:
        clustered_rows = build_clustered_statistical_rows(
            args.cluster_result,
            project_root=args.project_root,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed,
        )
        clustered = write_clustered_statistical_summary(
            clustered_rows,
            output_dir=args.output_dir,
            project_root=args.project_root,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed,
        )
        output["clustered"] = {"row_count": clustered["row_count"]}
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
