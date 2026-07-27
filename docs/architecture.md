# EGVR-Agent Architecture

## Reliability Pipeline

EGVR-Agent separates planning, execution, verification, and repair into
structured stages:

```text
User query
    |
    v
ParsedTask -> PlannedWorkflow -> Executor
                                  |
                                  v
                    ToolCallRecords / CandidateRecords
                                  |
                                  v
                              Verifier
                         / success | missing evidence
                        v          v
                    JSONL trace  Repair / fallback
                                      |
                                      +----> bounded re-execution
```

The LLM-facing agent shell is optional. The reliability pipeline can be tested
deterministically without an LLM, network, or real tool server.

## Structured Objects

`task_schema.py` defines the molecular execution contract:

- `ParsedTask`: task type, objectives, inputs, constraints, and metadata.
- `PlannedWorkflow`: selected tools, ordered calls, and expected outputs.
- `ToolCallRecord`: concrete inputs, outputs, status, error, and elapsed time.
- `CandidateRecord`: normalized molecular candidate and evaluator evidence.
- `VerifierResult`: required checks, metrics, status, and failure reason.

`biomedical_schema.py` adds `EvidenceRecord` for provenance-bearing clinical
and drug-target evidence workflows.

## Task-Conditioned Planning

`task_parser.py`, `rule_planner.py`, and `tool_registry.py` convert a request
into a minimal task-relevant workflow. The registry stores structured metadata,
input requirements, output types, and tool roles. Unrelated tools are not added
to the executable plan.

The LLM-router runners provide a separate planning baseline. They receive a
locked tool menu and return a schema-validated JSON workflow; they do not bypass
the executor or verifier.

## Structured Execution

`executor.py` dispatches declared tool calls with explicit parameters. It does
not ask an LLM to write arbitrary tool code during benchmark execution.
`result_normalizer.py` maps heterogeneous backend outputs into stable candidate
and evidence fields.

Execution can use:

- mock functions for controlled mechanism tests;
- local tool-server wrappers for real molecular tools;
- user-provided adapter commands for private external backends.

Nominal backend success is recorded, but it is not sufficient for task success.

## Execution-Grounded Verification

`verifier.py` checks the evidence required by the parsed task. Examples include
valid candidate identity, evaluator scores, ranking output, artifacts, and
provenance. Missing or malformed evidence yields `incomplete` or `failed`
status, even if a backend reported success.

`evidence_corruption_runner.py` tests verifier sensitivity by removing or
breaking evidence, provenance, artifacts, scores, ordering, and tool-call
consistency in successful traces.

## Bounded Repair and Fallback

`repair.py` maps recoverable verifier failures to declared actions:

- targeted retry;
- missing-evaluator rerun;
- declared fallback tool.

`benchmark_runner.py` applies policy-specific authorization. It records proposed,
authorized, rejected, and executed actions. Repair budgets bound repeated
execution, and irrecoverable failures remain explicit failures.

## Traceability

`trace_logger.py` serializes parsed tasks, workflows, calls, candidates,
verifier outcomes, repair decisions, and failure reasons to JSONL.
`trace_consistency_audit_runner.py` checks schema completeness, status/verifier
agreement, failure-reason coverage, repair-plan/call consistency, required
evidence, artifact references, and duplicate records.

## Domain Routing

`domain_router.py` routes requests to molecular, clinical-trial, or drug-target
evidence workflows. The latter two are supporting evidence-interface examples,
not clinical prediction or DTI performance claims.

Private prediction backends can be connected through
`clinical_prediction_runner.py`. The readiness gate rejects missing code,
models, data, entrypoints, or licenses before execution.

## External Backend Boundary

The `egvr` package does not embed a conversational shell or scientific model
server. Real tools are optional external dependencies connected through
`egvr.adapters`. This keeps evidence verification and repair independently
testable and prevents backend availability from being mistaken for verified
task completion.
