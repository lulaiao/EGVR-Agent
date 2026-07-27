# EGVR-Agent Paper Artifact

This repository contains the implementation and lightweight reproducibility
entry points for an evidence-first biomedical tool-execution framework.

## Artifact Scope

The public artifact includes:

- structured task, workflow, tool-call, candidate, evidence, and verifier
  schemas;
- task-conditioned planning and baseline planners;
- execution, normalization, verification, repair, and JSONL trace logging;
- controlled reliability benchmark generators and offline analysis runners;
- lightweight molecular and biomedical example tasks;
- unit tests and a release-tree audit.

The public artifact does not include:

- API keys, provider response logs, or private endpoint configuration;
- real experiment traces or paper-result tables;
- CrossDocked, LIT-PCBA, PDBbind, ClinicalTrials, or other large datasets;
- third-party tool checkouts, private clinical backends, or model weights.

Those exclusions are intentional. They protect credentials, licenses, and
private provenance while keeping the reliability mechanism independently
testable.

## Quick Artifact Check

Create a Python 3.11 environment named after the method and install the
development dependencies:

```bash
conda create -n egvr-agent python=3.11 -y
conda activate egvr-agent
pip install -e ".[dev]"
```

Run the offline artifact workflow:

```bash
python scripts/run_offline_artifact.py \
  --output-dir /tmp/egvr_artifact
```

This command performs no network or real-tool calls. It:

1. generates the 72-task controlled failure taxonomy from 24 scenario
   templates and three input variants;
2. evaluates five reliability mechanisms in mock mode;
3. builds repair-quality, cost-normalized, failure-family, and
   scenario-clustered statistical summaries;
4. runs the public clinical-trial and drug-target evidence-workflow examples.

Run the focused artifact tests and release audit:

```bash
python -m pytest \
  tests/test_benchmark_runner.py \
  tests/test_failure_taxonomy_v3_generator.py \
  tests/test_evidence_corruption_runner.py \
  tests/test_repair_budget_runner.py \
  tests/test_repair_quality_builder.py \
  tests/test_trace_consistency_audit_runner.py \
  tests/test_tool_menu_comparison_runner.py \
  tests/test_clinical_trial_verifier.py \
  tests/test_drug_target_verifier.py

python scripts/audit_release_tree.py --root .
```

## Result Boundaries

The offline workflow validates code paths and controlled mechanism behavior; it
does not reproduce the paper's private real-tool measurements. Real execution
requires separately licensed tools and datasets. Provider-based LLM routing
also requires user-supplied credentials. EGVR-Agent never treats backend
availability or nominal tool completion as scientific success without the
required verifier evidence.

See [docs/reproducibility.md](docs/reproducibility.md) for command-level details
and [docs/paper_artifact_mapping.md](docs/paper_artifact_mapping.md) for the
mapping from paper claims to implementation modules.
