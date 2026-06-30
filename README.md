# FullCopilot

FullCopilot is a trustworthy biomedical tool-using agent framework. It parses a user request into a structured task, selects task-relevant tools, executes planned calls, and verifies whether the resulting evidence is sufficient to mark the task as complete.

## Core Idea

FullCopilot is not about exposing more tools to a language model. It focuses on execution evidence: which tools were selected, why they were called, what they returned, which checks passed, and whether missing evidence should trigger repair, fallback, or an explicit incomplete status.

## Features

- Parse natural-language requests into `ParsedTask`.
- Build task-conditioned `PlannedWorkflow` objects.
- Execute tools or offline wrappers through a structured executor.
- Normalize tool outputs into candidates and evidence records.
- Verify outputs, scores, evidence fields, and provenance before declaring success.
- Apply conservative repair or fallback when evidence is missing.
- Save JSONL execution traces for reproducibility, audit, and future planner learning.
- Provide benchmark runners, baseline runners, and release hygiene checks.

## Repository Layout

```text
FullCopilot/
├── CAi/
│   ├── CAi_agent/                 # compatible agent shell
│   └── toolkit/
│       ├── functions/             # agent-facing tool wrappers
│       ├── server/                # optional tool server wrappers
│       └── agent_planner/         # planning, execution, verification, tracing
├── tests/                         # unit tests, no API keys required
├── docs/                          # architecture and release notes
├── scripts/                       # release hygiene checks
└── pyproject.toml
```

## Quick Start

```bash
conda create -n fullcopilot python=3.11
conda activate fullcopilot
pip install -e ".[dev]"
python -m pytest tests/test_domain_router.py tests/test_clinical_trial_verifier.py tests/test_drug_target_verifier.py
```

Run an offline benchmark example:

```bash
python -m CAi.toolkit.agent_planner.biomedical_benchmark_runner \
  --benchmark CAi/toolkit/agent_planner/benchmarks/clinical_trial_outcome_prediction_v1_offline.jsonl \
  --output /tmp/fullcopilot_offline_summary.json
```

Run a mock benchmark example:

```bash
python -m CAi.toolkit.agent_planner.benchmark_runner \
  --benchmark CAi/toolkit/agent_planner/benchmarks/molecular_agent_tasks.example.jsonl \
  --execution-mode mock \
  --output /tmp/fullcopilot_mock_summary.json
```

## Optional Tool Server

Some real tool executions can use the local tool server:

```bash
python -m CAi.toolkit.server.app
```

Tool source code, model weights, generated workspaces, and large datasets are not included in this repository. Configure those paths locally and keep them outside version control.

## Release Hygiene

Before publishing:

```bash
python scripts/audit_release_tree.py --root .
```

The audit checks for local absolute paths, credential-like tokens, runtime logs, tool workspaces, and large binary/model artifacts.
