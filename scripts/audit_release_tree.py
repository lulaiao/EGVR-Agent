"""Audit a FullCopilot release tree for files that should not be published."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


BLOCKED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "logs",
    "agent_workspace",
    "workspace",
    "tool_fill_logs",
    "datasets",
    "paper/upload",
}
BLOCKED_FILE_NAMES = {".env", "main.pdf"}
BLOCKED_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".pkg",
    ".zip",
    ".tar",
    ".gz",
    ".pkl",
    ".pickle",
}
LOCAL_PATH_PATTERNS = (
    re.compile(r"/data/ssd1/lla"),
    re.compile(r"/home/lula"),
    re.compile(r"/mnt/shared-storage"),
    re.compile(r"CAi_copilot"),
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bapi[_-]?key\s*[:=]\s*[A-Za-z0-9_-]{24,}\b", re.IGNORECASE),
)
MAX_FILE_BYTES = 5_000_000


def audit(root: Path) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        rel_text = rel.as_posix()
        if _is_ignored(rel_text):
            continue
        if path.is_dir():
            continue
        if path.name in BLOCKED_FILE_NAMES:
            findings.append(f"blocked file: {rel_text}")
        if path.suffix in BLOCKED_SUFFIXES:
            findings.append(f"blocked binary/archive suffix: {rel_text}")
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_BYTES:
            findings.append(f"large file > {MAX_FILE_BYTES} bytes: {rel_text}")
            continue
        text = _read_text(path)
        if text is None:
            continue
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(text):
                findings.append(f"local path marker {pattern.pattern!r}: {rel_text}")
                break
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"credential-like token {pattern.pattern!r}: {rel_text}")
                break
    return findings


def _is_ignored(rel_text: str) -> bool:
    if rel_text == "scripts/audit_release_tree.py":
        return True
    parts = rel_text.split("/")
    if any(part in BLOCKED_DIR_NAMES for part in parts):
        return True
    return any(rel_text.startswith(blocked + "/") for blocked in BLOCKED_DIR_NAMES if "/" in blocked)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a FullCopilot release tree.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    findings = audit(root)
    if findings:
        print("Release audit failed:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print(f"Release audit passed: {root}")


if __name__ == "__main__":
    main()
