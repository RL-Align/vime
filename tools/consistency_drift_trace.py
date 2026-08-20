#!/usr/bin/env python3
"""Export vime consistency audit dumps as an expandable trace file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.consistency_drift_report import load_consistency_artifacts
from vime.utils.consistency_drift_report import (
    build_consistency_drift_report,
    write_consistency_drift_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="vime debug .pt dump(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output Chrome Trace Event JSON path (.json)",
    )
    parser.add_argument("--title", default=None, help="report title")
    args = parser.parse_args()

    if args.output.suffix.lower() != ".json":
        parser.error("the expandable trace uses Chrome Trace Event JSON; use a .json output path")

    manifest, cube, provenance = load_consistency_artifacts(args.inputs)
    report = build_consistency_drift_report(
        replay_manifest=manifest,
        result_cube=cube,
        runtime_provenance=provenance,
        title=args.title,
    )
    output = write_consistency_drift_trace(report, args.output)
    print(f"Wrote consistency drift trace to {output}")


if __name__ == "__main__":
    main()
