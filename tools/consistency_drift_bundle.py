#!/usr/bin/env python3
"""Build a single offline vime consistency-drift desktop-viewer bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.consistency_drift_report import load_consistency_artifacts
from vime.utils.consistency_drift_report import (
    build_consistency_drift_report,
    write_consistency_drift_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="vime debug .pt dump(s)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output .vime-drift bundle path")
    parser.add_argument("--title", default=None, help="report title")
    parser.add_argument("--no-preview", action="store_true", help="omit the PNG snapshot from the bundle")
    args = parser.parse_args()

    if args.output.suffix.lower() != ".vime-drift":
        parser.error("the desktop viewer bundle must use a .vime-drift output path")
    manifest, cube, provenance = load_consistency_artifacts(args.inputs)
    report = build_consistency_drift_report(
        replay_manifest=manifest,
        result_cube=cube,
        runtime_provenance=provenance,
        title=args.title,
    )
    output = write_consistency_drift_bundle(report, args.output, include_preview=not args.no_preview)
    print(f"Wrote vime consistency drift bundle to {output}")


if __name__ == "__main__":
    main()
