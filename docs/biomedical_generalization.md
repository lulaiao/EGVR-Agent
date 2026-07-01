# Biomedical Generalization Slices

FullCopilot is positioned as a trustworthy biomedical tool-using agent
framework. Molecular workflows remain the primary real-tool evaluation domain,
while clinical-trial and drug-target tasks are lightweight offline slices for
testing whether the same planning and verification abstractions transfer across
biomedical evidence workflows.

These slices are intentionally not predictive benchmarks. They do not report
clinical outcome accuracy, DTI accuracy, repurposing performance, AUC, or
drug-discovery SOTA metrics. They only test whether the agent can select the
right evidence tools, record concrete evidence fields, verify provenance
coverage, avoid false success, and expose missing evidence.

## Offline Slices

- `clinical_trial_outcome_prediction_v2_offline.jsonl` checks trial metadata,
  eligibility/enrollment evidence, outcome labels, and outcome provenance.
- `drug_target_evidence_v2_offline.jsonl` checks drug, target, mechanism,
  knowledge-graph provenance, literature provenance, and repurposing rationale
  when a disease indication is part of the task.

Run the slices and write per-task summaries:

```bash
python -m CAi.toolkit.agent_planner.biomedical_benchmark_runner \
  --benchmark CAi/toolkit/agent_planner/benchmarks/clinical_trial_outcome_prediction_v2_offline.jsonl \
  --output /tmp/clinical_trial_v2_summary.json

python -m CAi.toolkit.agent_planner.biomedical_benchmark_runner \
  --benchmark CAi/toolkit/agent_planner/benchmarks/drug_target_evidence_v2_offline.jsonl \
  --output /tmp/drug_target_v2_summary.json
```

Build the compact table used for paper-facing evidence summaries:

```bash
python -m CAi.toolkit.agent_planner.biomedical_generalization_table \
  --benchmark CAi/toolkit/agent_planner/benchmarks/clinical_trial_outcome_prediction_v2_offline.jsonl \
  --benchmark CAi/toolkit/agent_planner/benchmarks/drug_target_evidence_v2_offline.jsonl \
  --output /tmp/biomedical_generalization_table.json
```
