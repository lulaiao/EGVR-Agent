"""Build a reviewer-facing claim-to-evidence audit table.

This module is intentionally analysis-only: it reads existing benchmark table
artifacts and writes compact CSV/JSON/LaTeX summaries. It does not rerun tools.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MASTER_TABLE = "logs/baseline_runs/master_baseline_table/master_baseline_table.json"
DEFAULT_OUTPUT_DIR = "paper"

EVIDENCE_AUDIT_COLUMNS = [
    "claim_id",
    "claim",
    "claim_strength",
    "evidence_role",
    "table_name",
    "table_label",
    "benchmark_id",
    "dataset",
    "execution_mode",
    "evidence_type",
    "is_real_result",
    "is_controlled",
    "is_supporting",
    "row_count",
    "result_path",
    "notes",
]


_CLAIM_SPECS = [
    {
        "claim_id": "C1",
        "claim": "Verifier-guided repair improves robustness under controlled tool failures.",
        "claim_strength": "strong",
        "evidence_role": "main",
        "table_name": "robustness_repeated",
        "table_label": "tab:robustness-repeated",
        "evidence_type": "repeated_real_controlled",
        "is_real_result": True,
        "is_controlled": True,
        "is_supporting": False,
        "notes": "Mean +/- std over repeated failure taxonomy runs; controlled failures test mechanism, not production failure frequency.",
    },
    {
        "claim_id": "C2",
        "claim": "The gain is not explained by exposing more tools to the language model.",
        "claim_strength": "strong",
        "evidence_role": "main",
        "table_name": "ablation",
        "table_label": "tab:ablation-tool-repair",
        "evidence_type": "ablation_mixed",
        "is_real_result": True,
        "is_controlled": True,
        "is_supporting": False,
        "notes": "Combines tool-exposure budget with robustness repair metrics.",
    },
    {
        "claim_id": "C3",
        "claim": "Verifier-triggered repair is separable from scheduled fallback.",
        "claim_strength": "strong",
        "evidence_role": "main",
        "table_name": "repair_ablation_repeated",
        "table_label": "tab:repair-ablation-repeated",
        "evidence_type": "repeated_real_controlled",
        "is_real_result": True,
        "is_controlled": True,
        "is_supporting": False,
        "notes": "Ambiguous-evidence wrapper injection keeps initial task distribution fixed while varying repair policy.",
    },
    {
        "claim_id": "C4",
        "claim": "CrossDocked generation scale and verifier logging remain stable across seeds.",
        "claim_strength": "strong",
        "evidence_role": "main",
        "table_name": "crossdocked_multiseed",
        "table_label": "tab:crossdocked-multiseed",
        "evidence_type": "real_multiseed",
        "is_real_result": True,
        "is_controlled": False,
        "is_supporting": False,
        "notes": "Reports 30-target CrossDocked repeats across true RxnFlow seeds; diversity is diagnostic, not a SOTA quality claim.",
    },
    {
        "claim_id": "C5",
        "claim": "The planner is not limited to a single RxnFlow/Vina workflow family.",
        "claim_strength": "strong",
        "evidence_role": "main",
        "table_name": "task_generalization",
        "table_label": "tab:task-generalization",
        "evidence_type": "real_task_generalization",
        "is_real_result": True,
        "is_controlled": False,
        "is_supporting": False,
        "notes": "Real hit-to-lead and scaffold-conditioned generation slices exercise non-pocket workflows.",
    },
    {
        "claim_id": "C6",
        "claim": "Chemistry-grounded verification records evidence beyond raw tool-call success.",
        "claim_strength": "strong",
        "evidence_role": "main",
        "table_name": "verifier_evidence",
        "table_label": "tab:verifier-evidence",
        "evidence_type": "real_verifier_evidence",
        "is_real_result": True,
        "is_controlled": False,
        "is_supporting": False,
        "notes": "SA_Score, RDKit properties, and PoseBusters evidence are reported as coverage/pass signals, not as biological activity.",
    },
    {
        "claim_id": "C7",
        "claim": "Docking completion is not treated as sufficient pose-quality evidence.",
        "claim_strength": "supporting",
        "evidence_role": "appendix",
        "table_name": "posebusters_top_failures",
        "table_label": "tab:posebusters-top-failures",
        "evidence_type": "real_pose_sanity",
        "is_real_result": True,
        "is_controlled": False,
        "is_supporting": True,
        "notes": "Pose artifacts pass a conversion gate before PoseBusters failures are counted.",
    },
    {
        "claim_id": "C8",
        "claim": "PDBbind+ is handled through an explicit prepared-receptor and evidence gate.",
        "claim_strength": "supporting",
        "evidence_role": "appendix",
        "table_name": "pdbbind_prep_gate",
        "table_label": "tab:pdbbind-prep-gate",
        "evidence_type": "real_infrastructure_gate",
        "is_real_result": True,
        "is_controlled": False,
        "is_supporting": True,
        "notes": "Supports docking-infrastructure generalization only; no affinity-correlation or drug-discovery SOTA claim.",
    },
    {
        "claim_id": "C9",
        "claim": "LLM-as-router is evaluated as a planning baseline rather than a learned planner claim.",
        "claim_strength": "supporting",
        "evidence_role": "baseline",
        "table_name": "llm_router_baseline",
        "table_label": "tab:llm-router-baseline",
        "evidence_type": "real_planning_only_api_replay",
        "is_real_result": True,
        "is_controlled": False,
        "is_supporting": True,
        "notes": "DeepSeek v4 pro and Qwen3.7-Plus API replays use a fixed prompt and tool registry; planning-only evidence should not be used for end-to-end robustness claims.",
    },
    {
        "claim_id": "C10",
        "claim": "Natural toolchain failures are audited separately from controlled injections.",
        "claim_strength": "supporting",
        "evidence_role": "appendix",
        "table_name": "natural_failure_audit",
        "table_label": "tab:natural-failure-audit",
        "evidence_type": "real_trace_audit",
        "is_real_result": True,
        "is_controlled": False,
        "is_supporting": True,
        "notes": "Aggregates observed real trace failures; it is not a complete production failure distribution.",
    },
    {
        "claim_id": "C11",
        "claim": "The framework can wrap an external clinical backend and audit prediction evidence without false success.",
        "claim_strength": "supporting",
        "evidence_role": "appendix",
        "table_name": "clinical_backend_pilot",
        "table_label": "tab:clinical-backend-pilot",
        "benchmark_id": "clinical_prediction_v1_private",
        "dataset": "ClinicalTrials.gov + TOP/HINT-compatible labels",
        "execution_mode": "private_executed_backend",
        "evidence_type": "clinical_evidence_audit",
        "is_real_result": True,
        "is_controlled": True,
        "is_supporting": True,
        "row_count": 2,
        "result_path": "logs/baseline_runs/clinical_prediction_v1_private/clinical_prediction_supporting_table.csv",
        "notes": "Private executed clinical-backend pilot; supports evidence/provenance auditing only, not ClinicalAgent reproduction or clinical prediction performance.",
    },
]


def build_evidence_audit_rows(
    *,
    master_payload: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return one audit row per paper claim."""

    root = _project_root(project_root)
    payload = master_payload or _load_optional_json(_resolve_path(DEFAULT_MASTER_TABLE, root))
    views = payload.get("views", {}) if isinstance(payload.get("views"), dict) else {}
    artifacts = payload.get("artifacts", {}).get("views", {}) if isinstance(payload.get("artifacts"), dict) else {}

    rows: list[dict[str, Any]] = []
    for spec in _CLAIM_SPECS:
        table_name = spec["table_name"]
        view = views.get(table_name, {}) if isinstance(views.get(table_name), dict) else {}
        view_rows = view.get("rows", []) if isinstance(view.get("rows"), list) else []
        first = next((item for item in view_rows if isinstance(item, dict)), {})
        artifact = artifacts.get(table_name, {}) if isinstance(artifacts.get(table_name), dict) else {}
        rows.append(
            {
                **spec,
                "benchmark_id": first.get("benchmark_id") or first.get("source_benchmark_id") or spec.get("benchmark_id") or "--",
                "dataset": first.get("dataset") or spec.get("dataset") or "--",
                "execution_mode": first.get("execution_mode") or spec.get("execution_mode") or "--",
                "row_count": len(view_rows) if view_rows else spec.get("row_count", 0),
                "result_path": artifact.get("json") or artifact.get("csv") or spec.get("result_path") or "--",
            }
        )
    return [_ordered_row(row) for row in rows]


def write_evidence_audit_table(
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    table_name: str = "evidence_audit_table",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write CSV, JSON, and a compact LaTeX draft for the evidence audit."""

    root = _project_root(project_root)
    out_dir = _resolve_path(output_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{table_name}.csv"
    json_path = out_dir / f"{table_name}.json"
    tex_path = out_dir / f"{table_name}.tex"
    _write_csv(rows, csv_path, EVIDENCE_AUDIT_COLUMNS)
    payload = {
        "table_id": table_name,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "columns": EVIDENCE_AUDIT_COLUMNS,
        "row_count": len(rows),
        "rows": rows,
        "notes": [
            "This table audits claim-to-evidence alignment; it does not rerun benchmarks.",
            "Planning-only or supporting rows must not be used as end-to-end robustness evidence.",
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


def render_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Claim-to-evidence audit for the paper's main reliability claims.}",
        r"\label{tab:evidence-audit}",
        r"\begin{tabular}{lllll}",
        r"\hline",
        r"Claim & Strength & Evidence & Table & Role \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                _escape_latex(value)
                for value in (
                    row.get("claim_id"),
                    row.get("claim_strength"),
                    row.get("evidence_type"),
                    row.get("table_label"),
                    row.get("evidence_role"),
                )
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_and_write_evidence_audit_table(
    *,
    master_table_path: str | Path = DEFAULT_MASTER_TABLE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    payload = _load_optional_json(_resolve_path(master_table_path, root))
    rows = build_evidence_audit_rows(master_payload=payload, project_root=root)
    return write_evidence_audit_table(rows, output_dir=output_dir, project_root=root)


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _ordered_row(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in EVIDENCE_AUDIT_COLUMNS}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper evidence-audit table.")
    parser.add_argument("--master-table", default=DEFAULT_MASTER_TABLE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    payload = build_and_write_evidence_audit_table(
        master_table_path=args.master_table,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    print(json.dumps(payload["artifacts"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
