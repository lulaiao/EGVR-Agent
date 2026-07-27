# EGVR-Agent

EGVR-Agent is the research artifact for **Evidence Before Success:
Execution-Grounded Verification and Targeted Repair for Biomedical Tool-Using
Agents**.

The repository implements an evidence-first reliability layer around external
tools. A task is complete only when the executor's recorded outputs satisfy the
task-specific verifier. Missing or malformed evidence can authorize a bounded
retry or declared fallback; otherwise the run remains explicitly incomplete.

## What Is Implemented

- Structured task, workflow, tool-call, candidate, evidence, and verifier records.
- Task-conditioned planning over a typed tool registry.
- Deterministic dispatch of declared calls, without runtime LLM-written code.
- Output normalization across heterogeneous backends.
- Execution-grounded evidence checks and conservative success gating.
- Verifier-guided retry/fallback with explicit repair budgets.
- JSONL traces and consistency audits.
- Controlled reliability, tool-menu, LLM-router, and biomedical evidence runners.

The repository is intentionally a research artifact rather than a general chat
application. It contains no Web UI, conversation manager, REPL shell, model
weights, private traces, or bundled scientific tools.

## Repository Layout

```text
egvr/
├── task_schema.py                # Typed execution records
├── task_parser.py                # Deterministic task normalization
├── rule_planner.py               # Task-conditioned workflow construction
├── tool_registry.py              # Typed tool capabilities and dependencies
├── executor.py                   # Structured dispatch and call recording
├── result_normalizer.py          # Backend output normalization
├── verifier.py                   # Evidence-gated task completion
├── repair.py                     # Bounded retry and fallback policies
├── trace_logger.py               # JSONL provenance
├── adapters/                     # Optional external backend contract
├── benchmarks/                   # Lightweight public benchmark definitions
└── *_runner.py                   # Evaluation and artifact builders
tests/                            # Network-free regression tests
scripts/                          # Offline artifact and release audits
docs/                             # Architecture and reproducibility notes
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest \
  tests/test_benchmark_runner.py \
  tests/test_verifier.py \
  tests/test_repair.py \
  tests/test_trace_consistency_audit_runner.py
```

Run the network-free paper artifact:

```bash
python scripts/run_offline_artifact.py \
  --output-dir /tmp/egvr_artifact
```

Run one lightweight benchmark:

```bash
python -m egvr.benchmark_runner \
  --benchmark egvr/benchmarks/molecular_agent_tasks.example.jsonl \
  --execution-mode mock \
  --planner-baseline egvr_agent \
  --output /tmp/egvr_mock_summary.json
```

## Optional External Tools

Real execution is connected through the small HTTP contract in
`egvr.adapters.tool_server`. Set:

```bash
export EGVR_TOOL_SERVER_URL=http://127.0.0.1:8001
```

The server must expose:

- `POST /run/{tool}/{action}`
- `GET /job/{job_id}`
- `GET /health`

Scientific tool implementations, environments, model weights, and datasets are
not distributed here. This keeps the reliability framework separable from any
particular molecular or clinical backend.

## Scope

Public offline tasks demonstrate mechanism behavior and evidence-interface
transfer. They do not claim molecular-generation quality, clinical prediction,
DTI, ADMET, or drug-discovery state of the art. Real-tool paper measurements,
provider responses, private backend I/O, and licensed datasets are not included.

The legacy result identifier `full_copilot` remains readable for artifact
compatibility. New runs should use `egvr_agent`.

## Reproducibility

- [Artifact overview](ARTIFACT.md)
- [Architecture](docs/architecture.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Paper-to-code map](docs/paper_artifact_mapping.md)
- [Release checklist](docs/release_checklist.md)
- [Naming and standalone migration](docs/naming_and_migration.md)

Before release:

```bash
python scripts/audit_release_tree.py --root .
```

## License And Attribution

The repository is released under Apache-2.0. See [NOTICE](NOTICE) for the
historical compatibility boundary and third-party attribution.
