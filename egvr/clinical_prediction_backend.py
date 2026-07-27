"""Readiness checks for an external ClinicalAgent backend.

This module intentionally does not vendor or import ClinicalAgent. It only
checks whether a private backend checkout appears callable enough for a smoke
run. Real execution is handled by a user-provided adapter command in
``clinical_prediction_runner``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_TOOL_DIRS = {
    "drugbank": Path("algo/agents/tools/drugbank"),
    "enrollment": Path("algo/agents/tools/enrollment"),
    "hetionet": Path("algo/agents/tools/hetionet"),
    "risk_model": Path("algo/agents/tools/risk_model"),
}

CALLABLE_ENTRYPOINTS = (
    Path("algo/main.ipynb"),
    Path("algo/main.py"),
    Path("main.ipynb"),
)

MODEL_SUFFIXES = {".pt", ".pth", ".pkl", ".pickle", ".joblib", ".bin", ".ckpt", ".safetensors"}
DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".db", ".sqlite", ".pkl", ".pickle"}


def default_backend_root() -> Path:
    """Return the default private backend path without assuming it exists."""

    return Path(os.environ.get("CLINICAL_AGENT_BACKEND_ROOT", "external_backends/clinical_agent"))


def build_clinicalagent_readiness(backend_root: str | Path | None = None) -> dict[str, Any]:
    """Inspect a private ClinicalAgent checkout and summarize missing items."""

    root = Path(backend_root) if backend_root is not None else default_backend_root()
    root = root.expanduser().resolve()
    repo_found = root.exists() and root.is_dir()
    license_found = _any_exists(root, ("LICENSE", "LICENSE.md", "LICENSE.txt")) if repo_found else False
    required_dirs_found = {
        name: bool(repo_found and (root / rel_path).is_dir())
        for name, rel_path in REQUIRED_TOOL_DIRS.items()
    }
    callable_entrypoints_found = [
        str(path)
        for path in _relative_existing_paths(root, CALLABLE_ENTRYPOINTS)
    ] if repo_found else []
    model_files_found = _find_by_suffix(root, MODEL_SUFFIXES, limit=50) if repo_found else []
    data_files_found = _find_by_suffix(root, DATA_SUFFIXES, limit=50) if repo_found else []

    missing_items: list[str] = []
    if not repo_found:
        missing_items.append("clinicalagent_repo")
    if repo_found and not license_found:
        missing_items.append("license")
    for name, found in required_dirs_found.items():
        if not found:
            missing_items.append(f"required_dir:{name}")
    if repo_found and not callable_entrypoints_found:
        missing_items.append("callable_entrypoint")
    if repo_found and not model_files_found:
        missing_items.append("model_files")
    if repo_found and not data_files_found:
        missing_items.append("data_files")

    return {
        "backend_name": "clinicalagent",
        "backend_root": str(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_found": repo_found,
        "license_found": license_found,
        "required_dirs_found": required_dirs_found,
        "callable_entrypoint_found": bool(callable_entrypoints_found),
        "callable_entrypoints": callable_entrypoints_found,
        "model_files_found": model_files_found,
        "data_files_found": data_files_found,
        "missing_items": missing_items,
        "ready_for_smoke": not missing_items,
        "notes": (
            "Readiness gate only. ClinicalAgent source, data, models, and raw outputs "
            "must remain private and should not be committed to the public EGVR-Agent repository."
        ),
    }


def write_readiness_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _any_exists(root: Path, names: tuple[str, ...]) -> bool:
    return any((root / name).exists() for name in names)


def _relative_existing_paths(root: Path, paths: tuple[Path, ...]) -> list[Path]:
    return [path for path in paths if (root / path).exists()]


def _find_by_suffix(root: Path, suffixes: set[str], *, limit: int) -> list[str]:
    found: list[str] = []
    for path in root.rglob("*"):
        if len(found) >= limit:
            break
        if path.is_file() and path.suffix.lower() in suffixes:
            found.append(str(path.relative_to(root)))
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a private ClinicalAgent backend checkout.")
    parser.add_argument("--backend-root", default=str(default_backend_root()))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = build_clinicalagent_readiness(args.backend_root)
    write_readiness_summary(summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
