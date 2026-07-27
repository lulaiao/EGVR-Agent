# Naming And Artifact Boundary

## Public Identity

- Method: **EGVR-Agent**
- Expansion: **Execution-Grounded Verification and Repair**
- Python distribution: `egvr-agent`
- Python package: `egvr`
- Recommended repository name: `EGVR-Agent`
- Recommended paper title: **Evidence Before Success: Execution-Grounded
  Verification and Targeted Repair for Biomedical Tool-Using Agents**

The public name intentionally describes the paper's reliability mechanism. It
does not present the artifact as a conversational copilot or a molecular design
application.

## Standalone Scope

The submission repository contains the implementation needed to inspect and
reproduce the paper's reliability claims:

- typed task, workflow, call, candidate, evidence, and verifier records;
- task-conditioned workflow construction;
- deterministic execution and output normalization;
- evidence-gated completion;
- bounded verifier-guided repair and declared fallback;
- trace logging, consistency audits, and controlled benchmarks.

The former application shell is outside this artifact boundary. The repository
does not distribute a chat UI, general-purpose REPL, conversation manager,
scientific model server, tool implementations, model weights, or datasets.
Real scientific backends are connected through the small HTTP adapter contract.

## Compatibility And Attribution

Previously generated experiment files may use the method identifier
`full_copilot`. Readers can still load and replay those artifacts. New runs use
`egvr_agent`; the legacy identifier is not the public method name.

The Apache-2.0 license and `NOTICE` file preserve the historical attribution
required for earlier application-scaffold use. This migration changes the
artifact boundary and implementation ownership; it does not hide provenance.

## GitHub Rename

After this tree is committed and tested, rename the GitHub repository to
`EGVR-Agent` and update the local remote:

```bash
git remote set-url origin https://github.com/lulaiao/EGVR-Agent.git
```

GitHub normally redirects the previous repository URL, but release badges,
paper links, and archival metadata should use the new URL.
