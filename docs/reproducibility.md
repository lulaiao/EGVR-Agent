# Reproducibility Guide

## Environment

- Python 3.11
- Linux is recommended for optional real-tool execution.
- Offline artifact checks do not require an API key, GPU, model weight, or tool
  server.

Install in a Conda environment named after the method:

```bash
conda create -n egvr-agent python=3.11 -y
conda activate egvr-agent
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Or use a method-named virtual environment directory:

```bash
python3.11 -m venv .egvr-agent-venv
source .egvr-agent-venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## One-Command Offline Check

```bash
python scripts/run_offline_artifact.py \
  --output-dir /tmp/egvr_artifact \
  --bootstrap-samples 10000 \
  --random-seed 20260707
```

Expected output families:

- `failure_taxonomy_v3.jsonl`
- one result JSON per planner baseline
- `repair_quality/repair_quality_table.{csv,json,tex}`
- `repair_quality/cost_normalized_table.{csv,json,tex}`
- `repair_quality/failure_taxonomy_family_table.{csv,json,tex}`
- `statistics/statistical_summary_clustered.{csv,json,tex}`
- `biomedical/biomedical_generalization_table.{csv,json}`
- `artifact_manifest.json`

The generated directory is an output directory and should not be committed.

## Individual Reliability Runners

The following modules expose command-line help:

```bash
python -m egvr.failure_taxonomy_v3_generator --help
python -m egvr.repair_quality_builder --help
python -m egvr.repair_budget_runner --help
python -m egvr.evidence_corruption_runner --help
python -m egvr.trace_consistency_audit_runner --help
python -m egvr.tool_menu_comparison_runner --help
python -m egvr.llm_router_baseline_runner --help
```

## LLM-Router Runs

LLM-router experiments are optional and use an OpenAI-compatible endpoint.
Credentials must be supplied at runtime or through an ignored dotenv file.
Never commit dotenv files or raw response logs.

Lock the model ID, prompt, registry, temperature, output schema, and task file
before a formal run. Preserve the generated prompt, request, plan, and response
hashes. Do not tune the parser or evaluator after inspecting formal results.

## Real Tools and Private Backends

Real molecular execution requires separately installed tools and datasets.
Private clinical execution requires a user-provided adapter command and must
pass `clinical_prediction_backend.py` readiness checks. Missing dependencies
produce an explicit blocked or incomplete result; they are never converted into
success.

## Release Audit

```bash
python scripts/audit_release_tree.py --root .
```

The audit rejects common secrets, local machine paths, archives, model files,
runtime traces, private backend directories, and large files.
