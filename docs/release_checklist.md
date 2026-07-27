# EGVR-Agent Release Checklist

Use this checklist before pushing the public GitHub repository.

## Include

- `egvr/` planning, execution, verification, repair,
  trace, controlled reliability, and offline analysis modules.
- Optional external-backend adapters with no bundled scientific models.
- Lightweight mock/offline benchmark examples.
- Unit tests that do not require API keys or large datasets.
- Documentation explaining external data/tool setup.
- `ARTIFACT.md`, `CITATION.cff`, and the paper-to-code implementation map.

## Exclude

- `.env`, API keys, raw LLM responses, and private endpoint configuration.
- `logs/`, `agent_workspace/`, tool server workspaces, and JSONL traces from real runs.
- Raw datasets such as CrossDocked, LIT-PCBA, PDBbind, BindingDB, and TDC.
- Third-party tool source checkouts, model weights, checkpoints, and generated artifacts.
- Historical paper upload zip files.
- Private benchmark definitions and formal provider response logs.

## Required Checks

```bash
python scripts/audit_release_tree.py --root .
python -m pytest \
  tests/test_benchmark_runner.py \
  tests/test_failure_taxonomy_v3_generator.py \
  tests/test_repair_quality_builder.py \
  tests/test_trace_consistency_audit_runner.py \
  tests/test_domain_router.py \
  tests/test_clinical_trial_verifier.py \
  tests/test_drug_target_verifier.py
python scripts/run_offline_artifact.py --output-dir /tmp/egvr_artifact
rm -rf build /tmp/egvr-dist
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/egvr-dist
python scripts/verify_wheel_contents.py /tmp/egvr-dist/*.whl
```

The release audit must report zero blocking findings before pushing, and the
wheel verifier must confirm that no legacy application package is included.

## Paper Scope Reminder

The public repository should describe EGVR-Agent as an evidence-first
biomedical tool-execution framework. Molecular design is the primary real-tool evaluation domain.
Clinical-trial and drug-target tasks are evidence-workflow generalization
examples and should not be described as predictive benchmarks or replications
of external systems. Controlled failures test mechanisms; they do not estimate
production failure prevalence.
