# Paper Code Availability Text

## Main-Paper Version

**Code availability.** The EGVR-Agent implementation is available at
`https://github.com/lulaiao/EGVR-Agent`. The repository contains the
structured planner, executor, result normalizer, verifier, bounded repair and
fallback policies, JSONL trace schema, controlled reliability benchmark
generators, offline analysis runners, tests, and a network-free artifact
workflow. Large third-party datasets, separately licensed molecular tools,
private backend checkouts, model weights, credentials, raw provider responses,
and private execution traces are not redistributed.

## Supplement Version

The public artifact provides a one-command offline check:

```bash
python scripts/run_offline_artifact.py \
  --output-dir /tmp/egvr_artifact \
  --bootstrap-samples 10000 \
  --random-seed 20260707
```

It regenerates the controlled 24-template/72-task reliability slice, evaluates
the five verification and repair mechanisms in mock mode, and exports
repair-quality, cost-normalized, failure-family, scenario-clustered statistical,
and biomedical evidence-interface summaries. This workflow validates the
released mechanism code without requiring private data or external services.
Exact private real-tool results reported in the paper remain traceable to the
authors' archived experiment manifests but are not included in the public
repository because of dataset, tool-license, credential, and provider-response
constraints.

## Claim Boundary

The repository supports inspection and reproduction of the agent reliability
mechanisms. It does not redistribute or claim reproduction of clinical
prediction, DTI, ADMET, molecular generation, docking, or drug-discovery SOTA
systems.
