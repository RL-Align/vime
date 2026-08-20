#!/usr/bin/env python3
"""Render vime consistency audit dumps as a self-contained drift report."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from vime.utils.consistency_drift_report import (
    build_consistency_drift_report,
    write_consistency_drift_report,
)


def load_consistency_artifacts(paths: Sequence[str | Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and merge audit artifacts from one or more vime debug dumps."""

    manifests: list[dict[str, Any]] = []
    cubes: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for raw_path in paths:
        payload = _torch_load(Path(raw_path))
        manifest, cube, payload_provenance = _find_artifacts(payload)
        if manifest:
            manifests.append(manifest)
        if cube:
            cubes.append(cube)
        if payload_provenance and not provenance:
            provenance = payload_provenance

    return _merge_manifests(manifests), _merge_cubes(cubes), provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="vime debug .pt dump(s)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output HTML path")
    parser.add_argument("--title", default=None, help="report title")
    args = parser.parse_args()

    manifest, cube, provenance = load_consistency_artifacts(args.inputs)
    report = build_consistency_drift_report(
        replay_manifest=manifest,
        result_cube=cube,
        runtime_provenance=provenance,
        title=args.title,
    )
    output = write_consistency_drift_report(report, args.output)
    print(f"Wrote consistency drift report to {output}")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_artifacts(payload: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return {}, {}, {}
    candidates = [payload]
    for key in ("rollout_data", "data", "payload"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    manifest: dict[str, Any] = {}
    cube: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for candidate in candidates:
        if not manifest and isinstance(candidate.get("consistency_replay_manifest"), Mapping):
            manifest = dict(candidate["consistency_replay_manifest"])
        if not manifest and isinstance(candidate.get("replay_manifest"), Mapping):
            manifest = dict(candidate["replay_manifest"])
        if not cube and isinstance(candidate.get("consistency_result_cube"), Mapping):
            cube = dict(candidate["consistency_result_cube"])
        if not cube and isinstance(candidate.get("result_cube"), Mapping):
            cube = dict(candidate["result_cube"])
        if not provenance:
            for key in ("runtime_provenance", "consistency_runtime_provenance"):
                if isinstance(candidate.get(key), Mapping):
                    provenance = dict(candidate[key])
                    break
    if not cube:
        for candidate in candidates:
            cube = _derive_result_cube(candidate, manifest)
            if cube:
                break
    if not provenance and isinstance(cube.get("runtime_provenance"), Mapping):
        provenance = dict(cube["runtime_provenance"])
    if not provenance and isinstance(manifest.get("runtime_provenance"), Mapping):
        provenance = dict(manifest["runtime_provenance"])
    return manifest, cube, provenance


def _derive_result_cube(candidate: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive basic drift metrics from a train dump when no result cube was serialized."""

    train_log_probs = _first_mapping_value(candidate, "log_probs", "train_log_probs")
    rollout_log_probs = _first_mapping_value(candidate, "rollout_log_probs")
    loss_masks = _first_mapping_value(candidate, "loss_masks")
    if train_log_probs is None or rollout_log_probs is None or loss_masks is None:
        return {}

    train_values = _tensor_list(train_log_probs)
    rollout_values = _tensor_list(rollout_log_probs)
    mask_values = _tensor_list(loss_masks)
    sample_count = min(len(train_values), len(rollout_values), len(mask_values))
    active_token_count = 0
    warning_count = abs(float(len(train_values) - len(rollout_values)))
    max_abs_dlogp = 0.0
    worst_token = None
    for position in range(sample_count):
        train = train_values[position].flatten()
        rollout = rollout_values[position].flatten()
        mask = mask_values[position].flatten().to(dtype=torch.bool)
        if train.numel() != rollout.numel() or train.numel() != mask.numel():
            warning_count += 1.0
            continue
        active_token_count += int(mask.sum().item())
        values = (train.to(dtype=torch.float32) - rollout.to(dtype=torch.float32)).abs()[mask]
        if values.numel() == 0:
            continue
        sample_max, active_index = values.max(dim=0)
        sample_max_value = float(sample_max.item())
        if sample_max_value > max_abs_dlogp:
            active_positions = mask.nonzero(as_tuple=False).flatten()
            worst_token = {
                "abs_dlogp": sample_max_value,
                "sample_position": position,
                "token_position": int(active_positions[int(active_index.item())].item()),
            }
            max_abs_dlogp = sample_max_value

    validation = manifest.get("validation") if isinstance(manifest.get("validation"), Mapping) else {}
    return {
        "schema_version": 1,
        "mode": manifest.get("mode", "audit"),
        "axes": {},
        "metrics": {
            "active_token_count": float(active_token_count),
            "max_abs_dlogp": max_abs_dlogp,
            "warning_count": warning_count,
            "metadata_warning_count": float(len(validation.get("warnings", []) or [])),
            "metadata_failure_count": float(len(validation.get("failures", []) or [])),
            "sample_count": float(sample_count),
        },
        "worst_token": worst_token,
        "metadata_validation": dict(validation),
        "runtime_provenance": dict(manifest.get("runtime_provenance") or {}),
    }


def _first_mapping_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _tensor_list(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item if isinstance(item, torch.Tensor) else torch.as_tensor(item) for item in value]
    return []


def _merge_manifests(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not manifests:
        return {}
    first = dict(manifests[0])
    samples: list[Any] = []
    cases: list[Any] = []
    warnings: list[Any] = []
    failures: list[Any] = []
    for manifest in manifests:
        samples.extend(manifest.get("samples", []) or [])
        cases.extend(manifest.get("batch_invariance_cases", []) or [])
        validation = manifest.get("validation") if isinstance(manifest.get("validation"), Mapping) else {}
        warnings.extend(validation.get("warnings", []) or [])
        failures.extend(validation.get("failures", []) or [])
    first["samples"] = samples
    first["sample_count"] = len(samples)
    first["batch_invariance_cases"] = cases
    validation = dict(first.get("validation") or {})
    if warnings or failures:
        validation["warnings"] = warnings
        validation["failures"] = failures
    first["validation"] = validation
    return first


def _merge_cubes(cubes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cubes:
        return {}
    merged = dict(cubes[0])
    metrics = dict(merged.get("metrics") or {})
    for cube in cubes[1:]:
        for key, value in (cube.get("metrics") or {}).items():
            if key in {"max_abs_dlogp"}:
                metrics[key] = max(float(metrics.get(key, 0.0) or 0.0), float(value or 0.0))
            elif key in {"warning_count", "metadata_warning_count", "metadata_failure_count", "active_token_count", "sample_count"}:
                metrics[key] = float(metrics.get(key, 0.0) or 0.0) + float(value or 0.0)
            elif key in {"runtime_fallback", "runtime_strict_failure"}:
                metrics[key] = bool(metrics.get(key)) or bool(value)
    merged["metrics"] = metrics
    worst = merged.get("worst_token")
    for cube in cubes[1:]:
        candidate = cube.get("worst_token")
        if isinstance(candidate, Mapping) and (
            not isinstance(worst, Mapping) or float(candidate.get("abs_dlogp", 0.0) or 0.0) > float(worst.get("abs_dlogp", 0.0) or 0.0)
        ):
            worst = dict(candidate)
    if worst:
        merged["worst_token"] = worst
    return merged


if __name__ == "__main__":
    main()
