# EGVR-Agent

[![CI](https://github.com/lulaiao/EGVR-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/lulaiao/EGVR-Agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

EGVR-Agent is the research artifact for **Evidence Before Success:
Execution-Grounded Verification and Targeted Repair for Biomedical Tool-Using
Agents**.

It provides an evidence-first reliability layer around external tools. A task
is complete only when recorded execution outputs satisfy task-specific
verifier checks. Missing or malformed evidence may authorize a bounded retry
or declared fallback; otherwise the run remains explicitly incomplete.

## Why EGVR-Agent?

Tool completion is not the same as task completion. EGVR-Agent separates:

```text
Task -> Plan -> Execute -> Normalize -> Verify -> Repair or Incomplete -> Trace
```

- **Task-conditioned planning** exposes only tools relevant to the task.
- **Deterministic execution** dispatches declared calls instead of running
  LLM-written code.
- **Execution-grounded verification** checks the outputs and artifacts that
  actually exist.
- **Targeted repair** is bounded by verifier reasons and an explicit budget.
- **Traceable decisions** are written as structured JSONL records.

This repository is a research artifact, not a chat application. It intentionally
contains no Web UI, conversation manager, model weights, private traces,
licensed datasets, or bundled scientific tools.

## Requirements

- Python 3.11 or newer
- Linux or macOS for the public offline workflow
- No API key, GPU, model weight, or tool server for the quick demo
- Optional real-tool execution requires an independently deployed backend

## Installation

Clone the repository and create an environment named after the method:

```bash
git clone https://github.com/lulaiao/EGVR-Agent.git
cd EGVR-Agent

conda create -n egvr-agent python=3.11 -y
conda activate egvr-agent
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Without Conda, use a method-named virtual environment directory:

```bash
python3.11 -m venv .egvr-agent-venv
source .egvr-agent-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 30-Second Demo

Run a complete parser-planner-executor-verifier-trace workflow with local mock
tools:

```bash
python -m examples.minimal_mock \
  --output-dir /tmp/egvr-agent-demo
```

Expected summary:

```json
{
  "candidate_count": 2,
  "failure_reason": null,
  "selected_tools": [
    "reinvent4_denovo",
    "scscore",
    "toxicity"
  ],
  "task_id": "public_minimal_demo",
  "task_success": true,
  "tool_call_count": 4,
  "trace_path": "/tmp/egvr-agent-demo/YYYYMMDD_traces.jsonl"
}
```

The example is network-free. It demonstrates that nominal tool outputs are
normalized and checked before success is recorded.

## Use Your Own Python Tools

Application-owned functions can be injected without changing the executor:

```bash
python -m examples.custom_tool_adapter
```

The complete example is in
[`examples/custom_tool_adapter.py`](examples/custom_tool_adapter.py). The core
pattern is:

```python
from egvr import WorkflowExecutor, parse_task, plan_workflow, verify_workflow

task = parse_task("Generate de novo molecules and evaluate synthesizability.")
workflow = plan_workflow(task)
executor = WorkflowExecutor(
    tool_functions={
        "reinvent4_denovo": my_generator,
        "scscore": my_synthesis_evaluator,
    }
)
calls, candidates = executor.execute(task, workflow)
result = verify_workflow(task, workflow, calls, candidates)
print(result.success, result.failure_reason)
```

Tool functions return dictionaries. EGVR-Agent records their inputs, outputs,
errors, elapsed time, normalized candidates, and verifier decision.

## CLI

After installation, run a lightweight public benchmark:

```bash
egvr-benchmark \
  --benchmark egvr/benchmarks/molecular_agent_tasks.example.jsonl \
  --execution-mode mock \
  --planner-baseline egvr_agent \
  --output /tmp/egvr-mock-summary.json
```

Equivalent module entry point:

```bash
python -m egvr.benchmark_runner --help
```

Run the larger network-free paper artifact:

```bash
python scripts/run_offline_artifact.py \
  --output-dir /tmp/egvr-artifact
```

See [`ARTIFACT.md`](ARTIFACT.md) for generated files and scientific boundaries.

## Connect an External Tool Service

Set the service URL:

```bash
export EGVR_TOOL_SERVER_URL=http://127.0.0.1:8001
```

The client in
[`egvr/adapters/tool_server.py`](egvr/adapters/tool_server.py) uses this small
asynchronous HTTP contract:

```http
GET /health
```

```json
{"status": "ok", "tools": ["generator", "evaluator"]}
```

```http
POST /run/{tool}/{action}
Content-Type: application/json

{"input": "application-defined payload"}
```

```json
{"job_id": "job-123"}
```

```http
GET /job/job-123
```

While running:

```json
{"status": "running"}
```

When complete:

```json
{
  "status": "finished",
  "data": {
    "success": true,
    "summary": {},
    "results": {}
  }
}
```

Failed jobs return `{"status": "failed", "data": ...}`. Missing, malformed,
timed-out, or failed outputs remain failed evidence; the adapter never converts
backend availability into scientific success.

## Repository Layout

```text
egvr/                  # Planning, execution, verification, repair, and runners
egvr/adapters/         # Optional external backend contract
egvr/benchmarks/       # Lightweight public benchmark definitions
examples/              # Network-free user-facing examples
tests/                 # Network-free regression tests
scripts/               # Offline artifact and release audits
docs/                  # Architecture and reproducibility documentation
```

## Reproducibility

- [Artifact overview](ARTIFACT.md)
- [Architecture](docs/architecture.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Paper-to-code map](docs/paper_artifact_mapping.md)
- [Release checklist](docs/release_checklist.md)
- [Naming and compatibility notes](docs/naming_and_migration.md)

Run the public checks:

```bash
python -m pytest
python scripts/audit_release_tree.py --root .
```

## Scope

Public offline tasks validate mechanism behavior and evidence-interface
transfer. They do not reproduce private real-tool measurements or claim
molecular-generation quality, clinical prediction, DTI, ADMET, or
drug-discovery state of the art. Provider responses, private backend I/O, model
weights, and licensed datasets are not included.

## Troubleshooting

- **`ModuleNotFoundError: egvr`**: activate `egvr-agent` and rerun
  `python -m pip install -e ".[dev]"` from the repository root.
- **Unsupported Python version**: verify `python --version` is 3.11 or newer.
- **Tool server unavailable**: use `--execution-mode mock`, or check
  `EGVR_TOOL_SERVER_URL` and `GET /health`.
- **Real tool returns success but verification fails**: inspect
  `failure_reason`, required checks, tool outputs, and referenced artifacts in
  the JSONL trace. This is expected conservative behavior.
- **Replaying older result files**: see the compatibility policy in
  [`docs/naming_and_migration.md`](docs/naming_and_migration.md).

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff). The final
paper BibTeX entry will be added when the paper becomes public.

## License And Attribution

EGVR-Agent is released under Apache-2.0. See [`NOTICE`](NOTICE) for historical
compatibility boundaries and third-party attribution.
