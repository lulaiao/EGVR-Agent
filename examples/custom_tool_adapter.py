"""Show how application-owned Python tools plug into EGVR-Agent."""

from __future__ import annotations

import json

from egvr import WorkflowExecutor, parse_task, plan_workflow, verify_workflow


class ApplicationTools:
    """Small example adapter; replace these methods with real backend calls."""

    def generate(self, num_variants: int = 2, **_: object) -> dict[str, object]:
        molecules = ["CCO", "CCN", "CCC"][:num_variants]
        return {"success": True, "molecules_smiles": molecules}

    def scscore(self, smiles_list: list[str], **_: object) -> dict[str, object]:
        return {
            "success": True,
            "results": [
                {
                    "input_smiles": smiles,
                    "canonical_smiles": smiles,
                    "scscore": 1.5,
                }
                for smiles in smiles_list
            ],
        }

    def executor_functions(self) -> dict[str, object]:
        return {
            "reinvent4_denovo": self.generate,
            "scscore": self.scscore,
        }


def main() -> None:
    task = parse_task(
        "Generate num_candidates=2 de novo molecules and evaluate synthesizability.",
        task_id="custom_adapter_demo",
    )
    workflow = plan_workflow(task)
    executor = WorkflowExecutor(tool_functions=ApplicationTools().executor_functions())
    calls, candidates = executor.execute(task, workflow)
    result = verify_workflow(task, workflow, calls, candidates)

    print(
        json.dumps(
            {
                "selected_tools": workflow.selected_tools,
                "candidate_smiles": [candidate.smiles for candidate in candidates],
                "task_success": result.success,
                "checks": result.checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
