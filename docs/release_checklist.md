# FullCopilot Release Checklist

Use this checklist before pushing the public GitHub repository.

## Include

- Core CAi-compatible agent code.
- `CAi/toolkit/agent_planner/` planning, execution, verification, repair, and trace modules.
- Lightweight mock/offline benchmark examples.
- Unit tests that do not require API keys or large datasets.
- Documentation explaining external data/tool setup.

## Exclude

- `.env`, API keys, raw LLM responses, and private endpoint configuration.
- `logs/`, `agent_workspace/`, tool server workspaces, and JSONL traces from real runs.
- Raw datasets such as CrossDocked, LIT-PCBA, PDBbind, BindingDB, and TDC.
- Third-party tool source checkouts, model weights, checkpoints, and generated artifacts.
- Historical paper upload zip files.

## Required Checks

```bash
python scripts/audit_release_tree.py --root .
python -m pytest tests/test_domain_router.py tests/test_clinical_trial_verifier.py tests/test_drug_target_verifier.py
```

The release audit must report zero blocking findings before pushing.

## Paper Scope Reminder

The public repository should describe FullCopilot as a trustworthy biomedical
agent framework. Molecular design is the primary real-tool evaluation domain.
Clinical-trial and drug-target tasks are offline evidence-workflow
generalization examples and should not be described as predictive benchmarks or
replications of external systems.
