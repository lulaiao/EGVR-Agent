# Paper Table Utilities

This public release keeps table builders and LaTeX exporters, but it does not
commit private real-run logs or local dataset paths.

Regenerate tables from local summaries:

```bash
python -m egvr.master_table_builder --output-dir logs/baseline_runs/master_baseline_table
python -m egvr.latex_table_export --input-dir logs/baseline_runs/master_baseline_table --output-dir logs/baseline_runs/master_baseline_table/latex
```

For public smoke tests, prefer the offline biomedical runner and mock molecular
benchmarks described in the repository README.
