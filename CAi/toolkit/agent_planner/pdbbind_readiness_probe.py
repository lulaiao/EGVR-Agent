"""Probe local PDBbind refined/core readiness without running docking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "logs/baseline_runs/pdbbind_readiness_probe_v1/pdbbind_readiness_summary.json"
PROTEIN_SUFFIXES = ("_protein.pdb", "_protein.pdbqt")
LIGAND_SUFFIXES = ("_ligand.sdf", "_ligand.mol2", "_ligand.pdbqt")


def run_pdbbind_readiness_probe(
    *,
    search_roots: list[str | Path],
    output_path: str | Path = DEFAULT_OUTPUT,
    max_depth: int = 4,
) -> dict[str, Any]:
    roots = [Path(root).expanduser() for root in search_roots]
    candidates = [_probe_candidate_root(path) for root in roots for path in _candidate_roots(root, max_depth=max_depth)]
    candidates = [candidate for candidate in candidates if _has_signal(candidate)]
    best = _best_candidate(candidates)
    status = _status(best)
    payload = {
        "benchmark_id": "pdbbind_readiness_probe_v1",
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "search_roots": [str(root) for root in roots],
        "best_candidate": best,
        "candidate_count": len(candidates),
        "candidates": candidates[:50],
        "recommendation": _recommendation(status),
        "notes": [
            "This probe only checks local data readiness; it does not run docking or affinity evaluation.",
            "A ready PDBbind pilot requires index files and at least one target with protein and ligand files.",
        ],
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _candidate_roots(root: Path, *, max_depth: int) -> list[Path]:
    if not root.exists():
        return []
    candidates: list[Path] = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        name = current.name.lower()
        if "pdbbind" in name or "refined" in name or "core" in name:
            candidates.append(current)
        if depth >= max_depth:
            continue
        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in children)
    return _unique_paths(candidates)


def _probe_candidate_root(root: Path) -> dict[str, Any]:
    index_files = _index_files(root)
    target_dirs = _target_dirs(root)
    ready_targets = [target for target in target_dirs if _target_ready(target)]
    return {
        "root": str(root),
        "index_file_count": len(index_files),
        "index_files": [str(path) for path in index_files[:10]],
        "target_dir_count": len(target_dirs),
        "ready_target_count": len(ready_targets),
        "sample_ready_targets": [target.name for target in ready_targets[:10]],
        "has_refined_signal": "refined" in root.name.lower() or any("refined" in path.name.lower() for path in index_files),
        "has_core_signal": "core" in root.name.lower() or any("core" in path.name.lower() for path in index_files),
    }


def _index_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for child in _safe_rglob(root, max_depth=3):
        if child.is_file() and ("index" in child.name.lower() or child.name.startswith("INDEX")):
            files.append(child)
    return sorted(files)


def _target_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for child in _safe_rglob(root, max_depth=4):
        if child.is_dir() and _looks_like_pdb_id(child.name):
            dirs.append(child)
    return sorted(dirs)


def _target_ready(target_dir: Path) -> bool:
    names = {path.name for path in target_dir.iterdir() if path.is_file()}
    return any(name.endswith(PROTEIN_SUFFIXES) for name in names) and any(
        name.endswith(LIGAND_SUFFIXES) for name in names
    )


def _safe_rglob(root: Path, *, max_depth: int) -> list[Path]:
    rows: list[Path] = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        rows.append(current)
        if depth >= max_depth or not current.is_dir():
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in children)
    return rows


def _looks_like_pdb_id(name: str) -> bool:
    return len(name) == 4 and name.isalnum()


def _has_signal(candidate: dict[str, Any]) -> bool:
    return bool(candidate["index_file_count"] or candidate["target_dir_count"] or candidate["ready_target_count"])


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            int(item["ready_target_count"] > 0 and item["index_file_count"] > 0),
            item["ready_target_count"],
            item["index_file_count"],
            item["target_dir_count"],
        ),
    )


def _status(best: dict[str, Any] | None) -> str:
    if not best:
        return "not_found"
    if best["ready_target_count"] and best["index_file_count"]:
        return "ready"
    if best["ready_target_count"] or best["index_file_count"] or best["target_dir_count"]:
        return "partial"
    return "not_found"


def _recommendation(status: str) -> str:
    if status == "ready":
        return "PDBbind pilot can be planned as a separate docking/pose generalization slice."
    if status == "partial":
        return "PDBbind deferred pending local data confirmation; inspect missing index or protein/ligand files."
    return "PDBbind deferred pending local data download or path configuration."


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe local PDBbind refined/core data readiness.")
    parser.add_argument("--search-root", action="append", default=[], help="Root directory to scan. Can be repeated.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output readiness summary JSON.")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum directory depth for candidate discovery.")
    args = parser.parse_args()
    roots = args.search_root or [Path.cwd()]
    payload = run_pdbbind_readiness_probe(search_roots=roots, output_path=args.output, max_depth=args.max_depth)
    print(
        json.dumps(
            {
                "summary_json": args.output,
                "status": payload["status"],
                "best_candidate": payload["best_candidate"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
