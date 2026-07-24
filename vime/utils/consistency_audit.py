"""Reusable rollout-to-training consistency audit harness.

The Phase 1 modules own compact metadata and raw diagnostic math. This module
ties those pieces together for Phase 3: validate comparison preconditions,
compute read-only dlogp diagnostics, and emit lightweight replay/result-cube
records that existing debug dumps can carry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vime.utils.consistency_metadata import (
    CONSISTENCY_METADATA_SCHEMA_VERSION,
    ConsistencyMetadataIssue,
    ConsistencyMetadataValidation,
    raise_for_consistency_metadata_failures,
    stable_fingerprint,
    validate_samples_consistency_metadata,
)
from vime.utils.dlogp_diagnostics import DlogpAuditReport, compute_dlogp_diagnostics, get_rlk_consistency_mode

CONSISTENCY_AUDIT_SCHEMA_VERSION = 1
AUDIT_REQUIRED_METADATA_FIELDS = (
    ("sample.session_id", "session_id_missing"),
    ("batch_layout.fingerprint", "batch_layout_missing"),
    ("position_cache.fingerprint", "position_cache_metadata_missing"),
    ("quantization.fingerprint", "quantization_metadata_missing"),
    ("weight.pre_update", "weight_update_status_missing"),
)


@dataclass(frozen=True)
class ConsistencyAuditResult:
    metrics: dict[str, torch.Tensor]
    dlogp_report: DlogpAuditReport
    metadata_validation: ConsistencyMetadataValidation
    replay_manifest: dict[str, Any]
    result_cube: dict[str, Any]
    diagnostic_metadata: tuple[dict[str, Any] | None, ...] = field(default_factory=tuple)


@dataclass
class _BatchMetadataSample:
    consistency_metadata: dict[str, Any] | None
    index: int | None = None
    rollout_id: int | None = None


def run_consistency_audit(
    train_log_probs: Sequence[torch.Tensor] | torch.Tensor,
    rollout_log_probs: Sequence[torch.Tensor] | torch.Tensor | None,
    loss_masks: Sequence[torch.Tensor] | torch.Tensor,
    *,
    args: Any | None = None,
    batch: Mapping[str, Any] | None = None,
    sample_indices: Sequence[int | torch.Tensor] | torch.Tensor | None = None,
    rollout_ids: Sequence[int | torch.Tensor] | torch.Tensor | None = None,
    rank: int | None = None,
    model_name: str | None = None,
    backend_id: str | None = None,
    contract_id: str | None = None,
    batch_layout_fingerprint: str | None = None,
    provenance_fingerprint: str | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    eps_clip: float = 0.2,
    prefix: str = "rlk_audit_",
) -> ConsistencyAuditResult:
    """Run the read-only consistency audit over already teacher-forced logprobs."""

    mode = get_rlk_consistency_mode(args)
    runtime_provenance = _plain_mapping(runtime_provenance)
    validation = validate_consistency_audit_batch(
        batch,
        mode=mode,
        runtime_provenance=runtime_provenance,
    )
    if mode == "strict":
        raise_for_consistency_metadata_failures(validation)

    diagnostic_metadata = build_consistency_diagnostic_metadata(batch)
    report = compute_dlogp_diagnostics(
        train_log_probs,
        rollout_log_probs,
        loss_masks,
        sample_indices=sample_indices if sample_indices is not None else _batch_get(batch, "sample_indices"),
        rollout_ids=rollout_ids if rollout_ids is not None else _batch_get(batch, "rollout_ids"),
        metadata=diagnostic_metadata or None,
        rank=rank,
        model_name=model_name if model_name is not None else getattr(args, "model_name", None),
        backend_id=backend_id if backend_id is not None else getattr(args, "train_backend", "megatron"),
        contract_id=contract_id if contract_id is not None else getattr(args, "rlk_contract_id", None),
        batch_layout_fingerprint=(
            batch_layout_fingerprint
            if batch_layout_fingerprint is not None
            else getattr(args, "rlk_batch_layout_fingerprint", None)
        ),
        provenance_fingerprint=(
            provenance_fingerprint
            if provenance_fingerprint is not None
            else _first_present(
                getattr(args, "rlk_provenance_fingerprint", None),
                stable_fingerprint(runtime_provenance) if runtime_provenance else None,
            )
        ),
        eps_clip=eps_clip,
        prefix=prefix,
    )
    replay_manifest = build_consistency_replay_manifest(
        batch,
        mode=mode,
        rank=rank,
        validation=validation.to_dict(),
        diagnostic_metadata=diagnostic_metadata,
        runtime_provenance=runtime_provenance,
    )
    result_cube = build_consistency_result_cube(
        args=args,
        batch=batch,
        mode=mode,
        rank=rank,
        dlogp_report=report,
        validation=validation,
        diagnostic_metadata=diagnostic_metadata,
        runtime_provenance=runtime_provenance,
    )

    metrics = dict(report.metrics)
    metrics.update(
        _audit_bookkeeping_metrics(
            report,
            validation=validation,
            replay_manifest=replay_manifest,
            result_cube=result_cube,
            runtime_provenance=runtime_provenance,
            prefix=prefix,
        )
    )
    return ConsistencyAuditResult(
        metrics={key: value.clone().detach() for key, value in metrics.items()},
        dlogp_report=report,
        metadata_validation=validation,
        replay_manifest=replay_manifest,
        result_cube=result_cube,
        diagnostic_metadata=tuple(diagnostic_metadata),
    )


def validate_consistency_audit_batch(
    batch: Mapping[str, Any] | None,
    *,
    mode: str,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> ConsistencyMetadataValidation:
    """Validate batch-level consistency metadata before drift attribution."""

    mode = str(mode).lower()
    if mode == "off":
        return ConsistencyMetadataValidation(mode=mode)

    if batch is None:
        return _single_issue_validation(
            mode,
            code="consistency_batch_missing",
            message="Training batch is unavailable for consistency metadata validation.",
        )

    precomputed = batch.get("consistency_metadata_validation")
    if isinstance(precomputed, Mapping):
        validation = _metadata_validation_from_dict(precomputed, mode=mode)
        validation = _with_audit_required_metadata_issues(batch, validation, mode=mode)
        return _with_runtime_provenance_issues(validation, runtime_provenance, mode=mode)

    records = _as_optional_mapping_list(batch.get("consistency_metadata"))
    if records:
        samples = [
            _BatchMetadataSample(
                consistency_metadata=record,
                index=_sequence_value(batch.get("sample_indices"), i),
                rollout_id=_sequence_value(batch.get("rollout_ids"), i),
            )
            for i, record in enumerate(records)
        ]
        validation = validate_samples_consistency_metadata(samples, mode=mode)
        validation = _with_audit_required_metadata_issues(batch, validation, mode=mode)
        return _with_runtime_provenance_issues(validation, runtime_provenance, mode=mode)

    validation = _single_issue_validation(
        mode,
        code="consistency_metadata_missing",
        message="Training batch is missing consistency metadata required for audit/strict comparison.",
    )
    return _with_runtime_provenance_issues(validation, runtime_provenance, mode=mode)


def build_consistency_diagnostic_metadata(batch: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, ...]:
    """Flatten Phase 1 metadata into the context fields dlogp diagnostics need."""

    if batch is None:
        return ()

    records = _as_optional_mapping_list(batch.get("consistency_metadata"))
    layouts = _as_optional_mapping_list(batch.get("consistency_batch_layout_fingerprints"))
    sample_count = max(len(records), len(layouts), _batch_sample_count(batch))
    if sample_count == 0:
        return ()

    result: list[dict[str, Any] | None] = []
    for i in range(sample_count):
        record = records[i] if i < len(records) else None
        layout = layouts[i] if i < len(layouts) else None
        if record is None and layout is None:
            result.append(None)
            continue

        provenance = _mapping_at(record, "provenance")
        actual = _mapping_at(provenance, "actual")
        old_logp = _mapping_at(record, "old_logp")
        model = _mapping_at(record, "model")
        flattened = {
            "model_name": model.get("name"),
            "backend_id": _first_present(
                actual.get("backend"),
                actual.get("actual_backend"),
                actual.get("backend_id"),
                old_logp.get("source"),
            ),
            "contract_id": old_logp.get("contract_id"),
            "batch_layout_fingerprint": _first_present(
                None if layout is None else layout.get("fingerprint"),
                _path(record, "batch_layout.fingerprint"),
            ),
            "provenance_fingerprint": _first_present(
                provenance.get("actual_fingerprint"),
                provenance.get("requested_fingerprint"),
            ),
            "router_policy": actual.get("router_policy"),
            "vllm_enable_prefix_caching": actual.get("vllm_enable_prefix_caching"),
            "vllm_enable_deterministic_inference": actual.get("vllm_enable_deterministic_inference"),
            "tensor_model_parallel_size": actual.get("tensor_model_parallel_size"),
            "megatron_tensor_parallel_size": actual.get("megatron_tensor_parallel_size"),
            "context_parallel_size": actual.get("context_parallel_size"),
            "megatron_context_parallel_size": actual.get("megatron_context_parallel_size"),
            "consistency_metadata_fingerprint": None if record is None else record.get("fingerprint"),
            "dynamic_sampling": None if record is None else record.get("dynamic_sampling"),
        }
        result.append(flattened)
    return tuple(result)


def build_consistency_replay_manifest(
    batch: Mapping[str, Any] | None,
    *,
    mode: str,
    rank: int | None = None,
    validation: Mapping[str, Any] | None = None,
    diagnostic_metadata: Sequence[Mapping[str, Any] | None] | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Build a lightweight replay/debug manifest for existing train-data dumps."""

    sample_count = _batch_sample_count(batch)
    if max_samples is not None:
        sample_count = min(sample_count, max(0, int(max_samples)))
    diagnostic_metadata = tuple(diagnostic_metadata or build_consistency_diagnostic_metadata(batch))
    runtime_provenance = _plain_mapping(runtime_provenance)

    samples = []
    for position in range(sample_count):
        record = _record_at(batch, "consistency_metadata", position)
        layout = _record_at(batch, "consistency_batch_layout_fingerprints", position)
        response_length = _int_or_none(_sequence_value(_batch_get(batch, "response_lengths"), position))
        loss_mask = _batch_sequence_value(batch, "loss_masks", position)
        diag = diagnostic_metadata[position] if position < len(diagnostic_metadata) else None
        samples.append(
            {
                "sample_position": position,
                "sample_index": _sequence_value(_batch_get(batch, "sample_indices"), position),
                "rollout_id": _sequence_value(_batch_get(batch, "rollout_ids"), position),
                "total_length": _int_or_none(_sequence_value(_batch_get(batch, "total_lengths"), position)),
                "response_length": response_length,
                "active_token_count": _active_token_count(loss_mask, response_length),
                "has_rollout_log_probs": _batch_sequence_value(batch, "rollout_log_probs", position) is not None,
                "consistency_metadata_fingerprint": None if record is None else record.get("fingerprint"),
                "batch_layout_fingerprint": _first_present(
                    None if layout is None else layout.get("fingerprint"),
                    None if diag is None else diag.get("batch_layout_fingerprint"),
                ),
                "provenance_fingerprint": None if diag is None else diag.get("provenance_fingerprint"),
                "dynamic_sampling": _path(record, "dynamic_sampling"),
            }
        )

    manifest = {
        "schema_version": CONSISTENCY_AUDIT_SCHEMA_VERSION,
        "metadata_schema_version": CONSISTENCY_METADATA_SCHEMA_VERSION,
        "mode": mode,
        "rank": rank,
        "sample_count": len(samples),
        "samples": samples,
        "batch_invariance_cases": build_batch_invariance_replay_cases(
            batch,
            diagnostic_metadata=diagnostic_metadata,
            max_samples=max_samples,
        ),
        "validation": dict(validation or {}),
        "runtime_provenance": runtime_provenance,
        "runtime_provenance_fingerprint": stable_fingerprint(runtime_provenance) if runtime_provenance else None,
    }
    manifest["fingerprint"] = stable_fingerprint(manifest)
    return manifest


def build_batch_invariance_replay_cases(
    batch: Mapping[str, Any] | None,
    *,
    diagnostic_metadata: Sequence[Mapping[str, Any] | None] | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Describe replay cases that keep one sample fixed while varying layout."""

    sample_count = _batch_sample_count(batch)
    if max_samples is not None:
        sample_count = min(sample_count, max(0, int(max_samples)))
    diagnostic_metadata = tuple(diagnostic_metadata or build_consistency_diagnostic_metadata(batch))

    cases: list[dict[str, Any]] = []
    for position in range(sample_count):
        sample_ref = {
            "sample_position": position,
            "sample_index": _sequence_value(_batch_get(batch, "sample_indices"), position),
            "rollout_id": _sequence_value(_batch_get(batch, "rollout_ids"), position),
        }
        layout = _record_at(batch, "consistency_batch_layout_fingerprints", position) or {}
        record = _record_at(batch, "consistency_metadata", position) or {}
        diag = diagnostic_metadata[position] if position < len(diagnostic_metadata) else None
        base = {
            **sample_ref,
            "batch_layout_fingerprint": _first_present(
                layout.get("fingerprint"),
                None if diag is None else diag.get("batch_layout_fingerprint"),
            ),
            "consistency_metadata_fingerprint": record.get("fingerprint"),
        }
        cases.extend(
            [
                {
                    **base,
                    "case": "same_sample_alone",
                    "varied_axes": ("batch_size", "neighboring_samples"),
                    "expected": "same-sample dlogp stays within the declared tolerance",
                },
                {
                    **base,
                    "case": "same_sample_mixed_batch",
                    "varied_axes": ("batch_order", "microbatch_membership"),
                    "expected": "mixed-batch placement does not change the fixed sample",
                },
                {
                    **base,
                    "case": "padding_packing_variant",
                    "varied_axes": ("padding_side", "packed_order", "microbatch_offset"),
                    "expected": "padding and packing changes do not alter active-token logprobs",
                },
                {
                    **base,
                    "case": "active_token_density_variant",
                    "varied_axes": ("active_mask_density",),
                    "active_mask_density": layout.get("active_mask_density"),
                    "expected": "neighboring active-token density does not alter the fixed sample",
                },
                {
                    **base,
                    "case": "dynamic_sampling_keep_drop_variant",
                    "varied_axes": ("dynamic_sampling_keep_drop",),
                    "dynamic_sampling": record.get("dynamic_sampling"),
                    "expected": "keep/drop decisions do not hide same-sample drift",
                },
            ]
        )
    return cases


def build_consistency_result_cube(
    *,
    args: Any | None,
    batch: Mapping[str, Any] | None,
    mode: str,
    rank: int | None,
    dlogp_report: DlogpAuditReport,
    validation: ConsistencyMetadataValidation,
    diagnostic_metadata: Sequence[Mapping[str, Any] | None],
    runtime_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact result-cube entry indexed by normalized audit axes."""

    runtime_provenance = _plain_mapping(runtime_provenance)
    axes = {
        "batch_layout": _unique_axis(diagnostic_metadata, "batch_layout_fingerprint"),
        "dtype": _dtype_axis(args),
        "router_policy": _axis_from_args_or_metadata(args, diagnostic_metadata, "router_policy"),
        "cache_policy": _first_present(
            getattr(args, "vllm_enable_prefix_caching", None),
            _axis_from_args_or_metadata(args, diagnostic_metadata, "vllm_enable_prefix_caching"),
        ),
        "tp": _first_present(
            getattr(args, "tensor_model_parallel_size", None),
            getattr(args, "megatron_tensor_parallel_size", None),
            _unique_axis(diagnostic_metadata, "tensor_model_parallel_size"),
            _unique_axis(diagnostic_metadata, "megatron_tensor_parallel_size"),
        ),
        "sp": _first_present(getattr(args, "sequence_parallel", None), getattr(args, "use_sequence_parallel", None)),
        "cp": _first_present(
            getattr(args, "context_parallel_size", None),
            getattr(args, "megatron_context_parallel_size", None),
            _unique_axis(diagnostic_metadata, "context_parallel_size"),
            _unique_axis(diagnostic_metadata, "megatron_context_parallel_size"),
        ),
        "logp_backend": _first_present(
            runtime_provenance.get("backend_id"),
            runtime_provenance.get("actual_backend"),
            _unique_axis(diagnostic_metadata, "backend_id"),
        ),
        "deterministic_policy": _first_present(
            getattr(args, "vllm_enable_deterministic_inference", None),
            getattr(args, "rlk_deterministic_logp", None),
        ),
    }
    metric_summary = {
        "active_token_count": _metric_float(dlogp_report, "rlk_audit_active_token_count"),
        "max_abs_dlogp": _metric_float(dlogp_report, "rlk_audit_dlogp_abs_max"),
        "warning_count": _metric_float(dlogp_report, "rlk_audit_warning_count"),
        "metadata_warning_count": float(len(validation.warnings)),
        "metadata_failure_count": float(len(validation.failures)),
        "sample_count": float(_batch_sample_count(batch)),
        "runtime_fallback": 1.0 if runtime_provenance.get("fallback") else 0.0,
        "runtime_strict_failure": 1.0 if runtime_provenance.get("strict_failure") else 0.0,
    }
    entry = {
        "schema_version": CONSISTENCY_AUDIT_SCHEMA_VERSION,
        "mode": mode,
        "rank": rank,
        "axes": axes,
        "metrics": metric_summary,
        "worst_token": dlogp_report.worst_token,
        "metadata_validation": validation.to_dict(),
        "runtime_provenance": runtime_provenance,
        "runtime_provenance_fingerprint": stable_fingerprint(runtime_provenance) if runtime_provenance else None,
    }
    entry["fingerprint"] = stable_fingerprint(entry)
    return entry


def _audit_bookkeeping_metrics(
    report: DlogpAuditReport,
    *,
    validation: ConsistencyMetadataValidation,
    replay_manifest: Mapping[str, Any],
    result_cube: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    prefix: str,
) -> dict[str, torch.Tensor]:
    device, dtype = _metric_device_dtype(report)
    return {
        f"{prefix}metadata_warning_count": torch.tensor(float(len(validation.warnings)), device=device, dtype=dtype),
        f"{prefix}metadata_failure_count": torch.tensor(float(len(validation.failures)), device=device, dtype=dtype),
        f"{prefix}metadata_active_token_count": torch.tensor(
            float(validation.active_token_count),
            device=device,
            dtype=dtype,
        ),
        f"{prefix}replay_case_count": torch.tensor(
            float(len(replay_manifest.get("batch_invariance_cases", ()))),
            device=device,
            dtype=dtype,
        ),
        f"{prefix}result_cube_axis_count": torch.tensor(
            float(len(result_cube.get("axes", {}))),
            device=device,
            dtype=dtype,
        ),
        f"{prefix}runtime_fallback": torch.tensor(
            1.0 if runtime_provenance.get("fallback") else 0.0,
            device=device,
            dtype=dtype,
        ),
        f"{prefix}runtime_strict_failure": torch.tensor(
            1.0 if runtime_provenance.get("strict_failure") else 0.0,
            device=device,
            dtype=dtype,
        ),
    }


def _metadata_validation_from_dict(record: Mapping[str, Any], *, mode: str) -> ConsistencyMetadataValidation:
    warnings = [_issue_from_dict(issue, default_severity="warning") for issue in record.get("warnings", ())]
    failures = [_issue_from_dict(issue, default_severity="error") for issue in record.get("failures", ())]
    return ConsistencyMetadataValidation(
        mode=mode,
        active_token_count=int(record.get("active_token_count") or 0),
        zero_active_token_samples=list(record.get("zero_active_token_samples") or ()),
        warnings=warnings,
        failures=failures,
    )


def _issue_from_dict(record: Mapping[str, Any], *, default_severity: str) -> ConsistencyMetadataIssue:
    return ConsistencyMetadataIssue(
        code=str(record.get("code") or "consistency_metadata_issue"),
        message=str(record.get("message") or ""),
        severity=str(record.get("severity") or default_severity),
        sample_index=_int_or_none(record.get("sample_index")),
        rollout_id=_int_or_none(record.get("rollout_id")),
        field=record.get("field"),
    )


def _single_issue_validation(mode: str, *, code: str, message: str) -> ConsistencyMetadataValidation:
    severity = "error" if mode == "strict" else "warning"
    issue = ConsistencyMetadataIssue(code=code, message=message, severity=severity)
    if mode == "strict":
        return ConsistencyMetadataValidation(mode=mode, failures=[issue])
    return ConsistencyMetadataValidation(mode=mode, warnings=[issue])


def _with_audit_required_metadata_issues(
    batch: Mapping[str, Any],
    validation: ConsistencyMetadataValidation,
    *,
    mode: str,
) -> ConsistencyMetadataValidation:
    records = _as_optional_mapping_list(batch.get("consistency_metadata"))
    if not records:
        return validation

    warnings = list(validation.warnings)
    failures = list(validation.failures)
    seen = {(issue.code, issue.sample_index, issue.rollout_id, issue.field) for issue in (*warnings, *failures)}

    for position, record in enumerate(records):
        if record is None:
            continue
        layout = _record_at(batch, "consistency_batch_layout_fingerprints", position) or {}
        sample_index = _first_present(
            _path(record, "sample.index"),
            _sequence_value(batch.get("sample_indices"), position),
        )
        rollout_id = _first_present(
            _path(record, "sample.rollout_id"),
            _sequence_value(batch.get("rollout_ids"), position),
        )
        for field_path, code in AUDIT_REQUIRED_METADATA_FIELDS:
            value = _first_present(
                layout.get("fingerprint") if field_path == "batch_layout.fingerprint" else None,
                _path(record, field_path),
            )
            if value not in (None, ""):
                continue
            key = (code, _int_or_none(sample_index), _int_or_none(rollout_id), field_path)
            if key in seen:
                continue
            issue = ConsistencyMetadataIssue(
                code=code,
                message=f"Consistency audit metadata field {field_path!r} is missing.",
                severity="error" if mode == "strict" else "warning",
                sample_index=_int_or_none(sample_index),
                rollout_id=_int_or_none(rollout_id),
                field=field_path,
            )
            if mode == "strict":
                failures.append(issue)
            else:
                warnings.append(issue)
            seen.add(key)

    return ConsistencyMetadataValidation(
        mode=validation.mode,
        active_token_count=validation.active_token_count,
        zero_active_token_samples=list(validation.zero_active_token_samples),
        warnings=warnings,
        failures=failures,
    )


def _with_runtime_provenance_issues(
    validation: ConsistencyMetadataValidation,
    runtime_provenance: Mapping[str, Any] | None,
    *,
    mode: str,
) -> ConsistencyMetadataValidation:
    if mode == "off" or not runtime_provenance:
        return validation

    warnings = list(validation.warnings)
    failures = list(validation.failures)
    issues: list[ConsistencyMetadataIssue] = []
    if runtime_provenance.get("fallback"):
        issues.append(
            ConsistencyMetadataIssue(
                code="undeclared_linear_logp_runtime_fallback",
                message="Training linear_logp runtime reported fallback during consistency audit.",
                severity="error" if mode == "strict" else "warning",
                field="linear_logp_runtime.fallback",
            )
        )
    if runtime_provenance.get("strict_failure"):
        issues.append(
            ConsistencyMetadataIssue(
                code="linear_logp_runtime_strict_failure",
                message="Training linear_logp runtime reported a strict failure.",
                severity="error",
                field="linear_logp_runtime.strict_failure",
            )
        )

    for issue in issues:
        if issue.severity == "error":
            failures.append(issue)
        else:
            warnings.append(issue)

    return ConsistencyMetadataValidation(
        mode=validation.mode,
        active_token_count=validation.active_token_count,
        zero_active_token_samples=list(validation.zero_active_token_samples),
        warnings=warnings,
        failures=failures,
    )


def _metric_device_dtype(report: DlogpAuditReport) -> tuple[torch.device, torch.dtype]:
    for metric in report.metrics.values():
        return metric.device, metric.dtype
    return torch.device("cpu"), torch.float32


def _metric_float(report: DlogpAuditReport, key: str) -> float:
    value = report.metrics.get(key)
    if value is None:
        return 0.0
    return float(value.detach().cpu().item())


def _as_optional_mapping_list(value: Any) -> tuple[dict[str, Any] | None, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (dict(value),)
    result = []
    for item in value:
        if item is None:
            result.append(None)
        elif isinstance(item, Mapping):
            result.append(dict(item))
        else:
            result.append(None)
    return tuple(result)


def _record_at(batch: Mapping[str, Any] | None, key: str, position: int) -> dict[str, Any] | None:
    records = _as_optional_mapping_list(_batch_get(batch, key))
    if position >= len(records):
        return None
    return records[position]


def _batch_get(batch: Mapping[str, Any] | None, key: str) -> Any:
    if batch is None:
        return None
    return batch.get(key)


def _batch_sample_count(batch: Mapping[str, Any] | None) -> int:
    if batch is None:
        return 0
    for key in (
        "tokens",
        "unconcat_tokens",
        "response_lengths",
        "loss_masks",
        "rollout_log_probs",
        "sample_indices",
        "rollout_ids",
        "consistency_metadata",
    ):
        value = batch.get(key)
        if value is not None:
            return _sample_count_from_value(value, key)
    return 0


def _sample_count_from_value(value: Any, key: str) -> int:
    if isinstance(value, torch.Tensor):
        if key in {"tokens", "unconcat_tokens", "loss_masks", "rollout_log_probs"}:
            return 1 if value.ndim <= 1 else int(value.shape[0])
        return int(value.numel()) if value.ndim <= 1 else int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return 1


def _batch_sequence_value(batch: Mapping[str, Any] | None, key: str, position: int) -> Any:
    value = _batch_get(batch, key)
    if isinstance(value, torch.Tensor) and key in {"tokens", "unconcat_tokens", "loss_masks", "rollout_log_probs"}:
        if value.ndim == 0:
            return value.detach().cpu().item() if position == 0 else None
        if value.ndim == 1:
            return value if position == 0 else None
        if position >= value.shape[0]:
            return None
        return value[position]
    return _sequence_value(value, position)


def _sequence_value(value: Any, position: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().flatten()
        if position >= flat.numel():
            return None
        return flat[position].item()
    try:
        if position >= len(value):
            return None
        item = value[position]
    except TypeError:
        return value if position == 0 else None
    if isinstance(item, torch.Tensor):
        if item.numel() == 1:
            return item.detach().cpu().item()
        return item.detach().cpu().tolist()
    return item


def _active_token_count(loss_mask: Any, response_length: int | None) -> int | None:
    if loss_mask is None:
        return response_length
    if isinstance(loss_mask, torch.Tensor):
        return int(loss_mask.detach().cpu().to(dtype=torch.bool).sum().item())
    try:
        return sum(1 for item in loss_mask if item)
    except TypeError:
        return None


def _mapping_at(value: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    item = value.get(key)
    return dict(item) if isinstance(item, Mapping) else {}


def _path(value: Mapping[str, Any] | None, path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _unique_axis(records: Sequence[Mapping[str, Any] | None], field: str) -> Any:
    values = []
    for record in records:
        if record is not None and record.get(field) is not None:
            values.append(record[field])
    unique = list(dict.fromkeys(values))
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return unique


def _axis_from_args_or_metadata(args: Any | None, records: Sequence[Mapping[str, Any] | None], field: str) -> Any:
    value = getattr(args, field, None) if args is not None else None
    if value is not None:
        return value
    return _unique_axis(records, field)


def _dtype_axis(args: Any | None) -> Any:
    if args is None:
        return None
    for attr in ("params_dtype", "dtype"):
        value = getattr(args, attr, None)
        if value is not None:
            return _normalize_dtype_value(value)
    if getattr(args, "bf16", False):
        return "bf16"
    if getattr(args, "fp16", False):
        return "fp16"
    return None


def _normalize_dtype_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, str):
        return value.replace("torch.", "")
    return value


def _plain_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
