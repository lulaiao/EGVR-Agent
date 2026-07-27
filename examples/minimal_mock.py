"""Run a complete EGVR-Agent workflow without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egvr import JSONLTraceLogger, WorkflowExecutor, parse_task, plan_workflow, verify_workflow


def _generate_candidates(**_: object) -> dict[str, object]:
    return {"success": True, "molecules_smiles": ["CCO", "CCN"]}


def _score_synthesis(smiles_list: list[str], **_: object) -> dict[str, object]:
    return {
        "success": True,
        "results": [
            {
                "input_smiles": smiles,
                "canonical_smiles": smiles,
                "scscore": 1.4 + 0.2 * index,
            }
            for index, smiles in enumerate(smiles_list)
        ],
    }


def _score_toxicity(smiles: str, **_: object) -> dict[str, object]:
    scores = {"CCO": 0.08, "CCN": 0.12}
    return {
        "success": True,
        "smiles": smiles,
        "toxicity_probability": scores.get(smiles, 0.5),
        "verdict": "Non-Toxic",
    }


def run_demo(output_dir: Path) -> dict[str, object]:
    task = parse_task(
        "Generate num_candidates=2 de novo molecules and evaluate "
        "synthesizability and toxicity.",
        task_id="public_minimal_demo",
    )
    workflow = plan_workflow(task)
    executor = WorkflowExecutor(
        tool_functions={
            "reinvent4_denovo": _generate_candidates,
            "scscore": _score_synthesis,
            "toxicity": _score_toxicity,
        }
    )
    tool_calls, candidates = executor.execute(task, workflow)
    verifier_result = verify_workflow(task, workflow, tool_calls, candidates)
    trace_path = JSONLTraceLogger(output_dir).log_trace(
        parsed_task=task,
        planned_workflow=workflow,
        tool_calls=tool_calls,
        candidate_records=candidates,
        verifier_result=verifier_result,
        metadata={"example": "minimal_mock"},
    )

    return {
        "task_id": task.task_id,
        "selected_tools": workflow.selected_tools,
        "tool_call_count": len(tool_calls),
        "candidate_count": len(candidates),
        "task_success": verifier_result.success,
        "failure_reason": verifier_result.failure_reason,
        "trace_path": str(trace_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/egvr-agent-demo"),
        help="Directory for the generated JSONL trace.",
    )
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
