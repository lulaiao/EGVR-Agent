#!/usr/bin/env python3
"""Run the public, network-free EGVR-Agent artifact workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egvr.baseline_planners import (
    EGVR_AGENT,
    SCHEDULED_FALLBACK_NO_VERIFIER,
    TOOL_STATUS_ONLY,
    VERIFIER_ONLY_NO_REPAIR,
    VERIFIER_TARGETED_RETRY_NO_FALLBACK,
)
from egvr.benchmark_runner import BenchmarkRunner
from egvr.biomedical_generalization_table import (
    build_biomedical_generalization_table,
    write_biomedical_generalization_table,
)
from egvr.failure_taxonomy_v3_generator import generate_failure_taxonomy_v3
from egvr.repair_quality_builder import build_and_write_repair_quality_table
from egvr.statistical_summary_builder import (
    build_clustered_statistical_rows,
    write_clustered_statistical_summary,
)


BASELINES = (
    TOOL_STATUS_ONLY,
    VERIFIER_ONLY_NO_REPAIR,
    VERIFIER_TARGETED_RETRY_NO_FALLBACK,
    SCHEDULED_FALLBACK_NO_VERIFIER,
    EGVR_AGENT,
)


def run_artifact(
    *,
    output_dir: str | Path,
    bootstrap_samples: int = 10_000,
    random_seed: int = 20260707,
) -> dict:
    root = PROJECT_ROOT
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    benchmark_dir = root / "egvr" / "benchmarks"
    taxonomy_path = output / "failure_taxonomy_v3.jsonl"
    taxonomy = generate_failure_taxonomy_v3(
        docking_benchmark=benchmark_dir / "failure_taxonomy_v3_docking_inputs.example.jsonl",
        output_path=taxonomy_path,
    )

    result_paths: list[Path] = []
    for baseline in BASELINES:
        result_path = output / "results" / f"{baseline}.json"
        trace_dir = output / "traces" / baseline
        BenchmarkRunner(
            execution_mode="mock",
            planner_baseline=baseline,
            trace_log_dir=trace_dir,
        ).run_file(taxonomy_path, output_path=result_path)
        result_paths.append(result_path)

    repair_payload = build_and_write_repair_quality_table(
        result_paths,
        output_dir=output / "repair_quality",
        project_root=root,
    )
    statistical_rows = build_clustered_statistical_rows(
        result_paths,
        project_root=root,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    statistical_payload = write_clustered_statistical_summary(
        statistical_rows,
        output_dir=output / "statistics",
        project_root=root,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )

    biomedical_paths = [
        benchmark_dir / "clinical_trial_outcome_prediction_v2_offline.jsonl",
        benchmark_dir / "drug_target_evidence_v2_offline.jsonl",
    ]
    biomedical_table = build_biomedical_generalization_table(biomedical_paths)
    biomedical_path = output / "biomedical" / "biomedical_generalization_table.json"
    write_biomedical_generalization_table(biomedical_table, biomedical_path)

    manifest = {
        "artifact": "EGVR-Agent offline reliability artifact",
        "execution_mode": "mock_and_offline",
        "network_used": False,
        "task_count": len(taxonomy),
        "scenario_template_count": len(
            {item["metadata"]["scenario_template_id"] for item in taxonomy}
        ),
        "baselines": list(BASELINES),
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "repair_quality_row_count": repair_payload["row_count"],
        "statistical_row_count": statistical_payload["row_count"],
        "biomedical_row_count": biomedical_table["row_count"],
        "claim_boundary": (
            "Controlled mechanism and evidence-workflow checks only; "
            "not a real-tool, clinical prediction, DTI, or drug-discovery SOTA reproduction."
        ),
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260707)
    args = parser.parse_args()
    manifest = run_artifact(
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
