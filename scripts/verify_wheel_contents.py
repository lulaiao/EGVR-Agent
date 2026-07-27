#!/usr/bin/env python3
"""Verify that a release wheel contains only the standalone EGVR package."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def verify_wheel(path: str | Path) -> list[str]:
    wheel_path = Path(path)
    findings: list[str] = []
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()

    if not any(name.startswith("egvr/") for name in names):
        findings.append("wheel does not contain the egvr package")

    unexpected_roots = sorted(
        {
            name.split("/", 1)[0]
            for name in names
            if "/" in name
            and not name.startswith("egvr/")
            and ".dist-info/" not in name
        }
    )
    if unexpected_roots:
        findings.append(f"unexpected package roots: {', '.join(unexpected_roots)}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel")
    args = parser.parse_args()

    findings = verify_wheel(args.wheel)
    if findings:
        print("Wheel verification failed:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print(f"Wheel verification passed: {Path(args.wheel).resolve()}")


if __name__ == "__main__":
    main()
