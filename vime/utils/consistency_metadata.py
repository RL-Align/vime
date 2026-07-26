"""Compact consistency metadata for rollout-to-training audit paths.

The helpers in this module keep the rollout boundary observable without
shipping full token or mask payloads inside metadata records. Full tensors
continue to live in the existing training batch fields; metadata carries
stable fingerprints, active-token counts, and requested-vs-actual provenance
needed by audit/strict consistency checks.

This is intentionally vime runtime metadata. RL-Kernel-owned alignment schemas,
A0-A5 profiles, comparators, and tolerance contracts stay behind the
``vime.backends.rl_kernel_utils`` adapter boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

CONSISTENCY_METADATA_SCHEMA_VERSION = 1
CONSISTENCY_MODES = {"off", "audit", "strict"}

REQUIRED_COMPARISON_FIELDS = (
    ("sample.rollout_id", "rollout_id_missing"),
    ("tokens.response_token_ids_fingerprint", "token_ids_missing"),
    ("active_mask.mask_fingerprint", "active_mask_missing"),
    ("active_mask.active_token_count", "active_mask_missing"),
    ("tokenizer.fingerprint", "tokenizer_missing"),
    ("sampling.params_fingerprint", "sampling_config_missing"),
    ("padding.side", "padding_semantics_missing"),
    ("model.name", "model_name_missing"),
    ("weight.version", "weight_version_missing"),
    ("old_logp.source", "old_logp_source_missing"),
    ("old_logp.contract_id", "logprob_contract_id_missing"),
    ("provenance.actual_fingerprint", "actual_provenance_missing"),
)

_SAMPLING_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "max_tokens",
    "seed",
    "stop",
    "stop_token_ids",
    "skip_special_tokens",
)

_PARALLEL_ARG_KEYS = (
    "num_gpus",
    "num_gpus_per_node",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "megatron_tensor_parallel_size",
    "megatron_context_parallel_size",
    "megatron_expert_model_parallel_size",
    "tensor_model_parallel_size",
    "context_parallel_size",
)

_ROUTER_ARG_KEYS = (
    "router_policy",
    "vllm_router_ip",
    "vllm_router_port",
    "vllm_dp_size",
    "vllm_enable_prefix_caching",
    "vllm_enable_deterministic_inference",
    "rollout_data_transport",
)


@dataclass(frozen=True)
class ConsistencyMetadataIssue:
    code: str
    message: str
    severity: str
    sample_index: int | None = None
    rollout_id: int | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "sample_index": self.sample_index,
            "rollout_id": self.rollout_id,
            "field": self.field,
        }


@dataclass(frozen=True)
class ConsistencyMetadataValidation:
    mode: str
    active_token_count: int = 0
    zero_active_token_samples: list[dict[str, int | None]] = field(default_factory=list)
    warnings: list[ConsistencyMetadataIssue] = field(default_factory=list)
    failures: list[ConsistencyMetadataIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "active_token_count": self.active_token_count,
            "zero_active_token_samples": self.zero_active_token_samples,
            "warnings": [issue.to_dict() for issue in self.warnings],
            "failures": [issue.to_dict() for issue in self.failures],
        }


def _normalize_for_json(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _normalize_for_json(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu()
    if hasattr(value, "tolist") and callable(value.tolist) and not isinstance(value, (str, bytes)):
        return _normalize_for_json(value.tolist())
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(
        _normalize_for_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_consistency_mode(args: Any | None = None) -> str:
    value = None
    if args is not None:
        value = (
            getattr(args, "rlk_consistency", None)
            or getattr(args, "rl_kernel_consistency", None)
            or getattr(args, "rl_kernel_consistency_mode", None)
        )
    if value is None:
        value = os.environ.get("VIME_RLK_CONSISTENCY") or os.environ.get("VIME_RL_KERNEL_CONSISTENCY")
    mode = str(value or "off").lower()
    if mode not in CONSISTENCY_MODES:
        raise ValueError(
            f"Unsupported RL-Kernel consistency mode {value!r}; expected one of {sorted(CONSISTENCY_MODES)}"
        )
    return mode


def _to_list(value: Any | None) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu()
    if hasattr(value, "tolist") and callable(value.tolist) and not isinstance(value, (str, bytes)):
        value = value.tolist()
    return list(value)


def _to_int_mask(mask: Any | None, response_length: int) -> list[int]:
    if mask is None:
        return [1] * response_length
    values = [int(v) for v in _to_list(mask)]
    if len(values) != response_length:
        raise ValueError(f"loss_mask length {len(values)} != response_length {response_length}")
    return values


def count_active_tokens(loss_mask: Any | None, response_length: int) -> int:
    return sum(1 for value in _to_int_mask(loss_mask, response_length) if value)


def _metadata_dict(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _sample_status_value(sample: Any) -> str | None:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status)


def _compact_attrs(obj: Any | None, keys: tuple[str, ...]) -> dict[str, Any]:
    if obj is None:
        return {}
    return {key: getattr(obj, key) for key in keys if hasattr(obj, key) and getattr(obj, key) is not None}


def _compact_sampling_params(sampling_params: dict[str, Any] | None) -> dict[str, Any]:
    if not sampling_params:
        return {}
    return {key: sampling_params[key] for key in _SAMPLING_PARAM_KEYS if key in sampling_params}


def build_requested_actual_provenance(
    *,
    requested: dict[str, Any] | None = None,
    actual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested = _normalize_for_json(requested or {})
    actual = _normalize_for_json(actual or {})
    mismatches = {}
    for key in sorted(set(requested) | set(actual)):
        if key in requested and key in actual and requested[key] != actual[key]:
            mismatches[key] = {"requested": requested[key], "actual": actual[key]}

    fallback = actual.get("fallback")
    requested_fallback = requested.get("fallback")
    undeclared_fallback = bool(fallback) and requested_fallback is not True

    return {
        "requested": requested,
        "actual": actual,
        "requested_fingerprint": stable_fingerprint(requested) if requested else None,
        "actual_fingerprint": stable_fingerprint(actual) if actual else None,
        "mismatches": mismatches,
        "undeclared_fallback": undeclared_fallback,
    }


def _extract_requested_provenance(args: Any | None) -> dict[str, Any]:
    requested = _compact_attrs(args, _ROUTER_ARG_KEYS + _PARALLEL_ARG_KEYS)
    requested.update(_compact_attrs(args, ("hf_checkpoint", "load", "save")))
    return requested


def _extract_actual_provenance(args: Any | None, *, model_name: str | None) -> dict[str, Any]:
    actual = _compact_attrs(args, _ROUTER_ARG_KEYS + _PARALLEL_ARG_KEYS)
    if model_name is not None:
        actual["model_name"] = model_name
    return actual


def build_rollout_consistency_metadata(
    sample: Any,
    *,
    args: Any | None = None,
    sampling_params: dict[str, Any] | None = None,
    model_name: str | None = None,
    old_logp_source: str | None = None,
    logprob_contract_id: str | None = None,
    tokenizer_fingerprint: str | None = None,
    padding_side: str | None = None,
    requested_provenance: dict[str, Any] | None = None,
    actual_provenance: dict[str, Any] | None = None,
    batch_layout: dict[str, Any] | None = None,
    dynamic_sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _metadata_dict(sample)
    response_length = int(getattr(sample, "response_length", 0) or 0)
    tokens = [int(token) for token in _to_list(getattr(sample, "tokens", []))]
    response_tokens = tokens[-response_length:] if response_length else []
    active_mask = _to_int_mask(getattr(sample, "loss_mask", None), response_length)
    active_token_count = sum(1 for value in active_mask if value)
    rollout_log_probs = getattr(sample, "rollout_log_probs", None)
    weight_versions = list(getattr(sample, "weight_versions", None) or [])
    rollout_id = getattr(sample, "rollout_id", None)
    if rollout_id is None:
        rollout_id = getattr(sample, "index", None)

    sampling_summary = _compact_sampling_params(sampling_params) or metadata.get("sampling_params") or {}
    model_name = model_name or metadata.get("model_name") or getattr(args, "hf_checkpoint", None)
    old_logp_source = old_logp_source or metadata.get("old_logp_source")
    if old_logp_source is None and rollout_log_probs is not None:
        old_logp_source = "rollout_engine"
    logprob_contract_id = logprob_contract_id or metadata.get("logprob_contract_id")
    tokenizer_fingerprint = tokenizer_fingerprint or metadata.get("tokenizer_fingerprint")
    if tokenizer_fingerprint is None and getattr(args, "hf_checkpoint", None) is not None:
        tokenizer_fingerprint = stable_fingerprint({"hf_checkpoint": args.hf_checkpoint})
    padding_side = padding_side or metadata.get("padding_side") or getattr(args, "padding_side", None)

    position_cache = metadata.get("position_cache") or metadata.get("position_cache_metadata")
    quantization = metadata.get("quantization") or _compact_attrs(args, ("quantization", "vllm_quantization"))
    parallel_placement = metadata.get("parallel_placement") or _compact_attrs(args, _PARALLEL_ARG_KEYS)
    dynamic_sampling = dynamic_sampling or metadata.get("dynamic_sampling")

    requested = requested_provenance if requested_provenance is not None else _extract_requested_provenance(args)
    actual = (
        actual_provenance if actual_provenance is not None else _extract_actual_provenance(args, model_name=model_name)
    )

    record = {
        "schema_version": CONSISTENCY_METADATA_SCHEMA_VERSION,
        "sample": {
            "group_index": getattr(sample, "group_index", None),
            "index": getattr(sample, "index", None),
            "rollout_id": rollout_id,
            "session_id": getattr(sample, "session_id", None),
            "status": _sample_status_value(sample),
        },
        "tokens": {
            "total_token_count": len(tokens),
            "response_length": response_length,
            "token_ids_fingerprint": stable_fingerprint(tokens),
            "response_token_ids_fingerprint": stable_fingerprint(response_tokens),
        },
        "active_mask": {
            "response_length": response_length,
            "active_token_count": active_token_count,
            "mask_fingerprint": stable_fingerprint(active_mask),
            "zero_active_tokens": active_token_count == 0,
        },
        "tokenizer": {"fingerprint": tokenizer_fingerprint},
        "sampling": {
            "params_fingerprint": stable_fingerprint(sampling_summary) if sampling_summary else None,
            "summary": _normalize_for_json(sampling_summary),
        },
        "padding": {"side": padding_side},
        "position_cache": {
            "fingerprint": stable_fingerprint(position_cache) if position_cache else None,
        },
        "quantization": {
            "fingerprint": stable_fingerprint(quantization) if quantization else None,
            "summary": _normalize_for_json(quantization),
        },
        "parallel_placement": {
            "fingerprint": stable_fingerprint(parallel_placement) if parallel_placement else None,
            "summary": _normalize_for_json(parallel_placement),
        },
        "model": {"name": model_name},
        "old_logp": {
            "source": old_logp_source,
            "contract_id": logprob_contract_id,
            "num_values": len(rollout_log_probs) if rollout_log_probs is not None else None,
            "fingerprint": stable_fingerprint(rollout_log_probs) if rollout_log_probs is not None else None,
        },
        "weight": {
            "version": metadata.get("weight_version") or (weight_versions[-1] if weight_versions else None),
            "versions_fingerprint": stable_fingerprint(weight_versions) if weight_versions else None,
            "pre_update": metadata.get("pre_update"),
        },
        "provenance": build_requested_actual_provenance(requested=requested, actual=actual),
        "batch_layout": batch_layout,
        "dynamic_sampling": _normalize_for_json(dynamic_sampling),
    }
    record["fingerprint"] = stable_fingerprint(record)
    return record


def ensure_sample_consistency_metadata(
    sample: Any,
    *,
    args: Any | None = None,
    sampling_params: dict[str, Any] | None = None,
    overwrite: bool = False,
    **metadata_kwargs: Any,
) -> dict[str, Any]:
    existing = getattr(sample, "consistency_metadata", None)
    if existing is not None and not overwrite:
        return existing
    record = build_rollout_consistency_metadata(
        sample,
        args=args,
        sampling_params=sampling_params,
        **metadata_kwargs,
    )
    sample.consistency_metadata = record
    return record


def sample_consistency_metadata(sample: Any) -> dict[str, Any] | None:
    direct = getattr(sample, "consistency_metadata", None)
    if direct is not None:
        return direct
    metadata = _metadata_dict(sample)
    nested = metadata.get("consistency_metadata")
    return nested if isinstance(nested, dict) else None


def _get_path(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _sample_ids(sample: Any, metadata: dict[str, Any] | None) -> tuple[int | None, int | None]:
    sample_index = getattr(sample, "index", None)
    rollout_id = getattr(sample, "rollout_id", None)
    if metadata:
        sample_info = metadata.get("sample") or {}
        sample_index = sample_info.get("index", sample_index)
        rollout_id = sample_info.get("rollout_id", rollout_id)
    if rollout_id is None:
        rollout_id = sample_index
    return sample_index, rollout_id


def _issue(
    *,
    mode: str,
    code: str,
    message: str,
    sample_index: int | None,
    rollout_id: int | None,
    field: str | None = None,
    strict_failure: bool = True,
) -> tuple[ConsistencyMetadataIssue, bool]:
    is_failure = mode == "strict" and strict_failure
    severity = "error" if is_failure else "warning"
    return (
        ConsistencyMetadataIssue(
            code=code,
            message=message,
            severity=severity,
            sample_index=sample_index,
            rollout_id=rollout_id,
            field=field,
        ),
        is_failure,
    )


def validate_samples_consistency_metadata(
    samples: list[Any],
    *,
    mode: str,
    required_fields: tuple[tuple[str, str], ...] = REQUIRED_COMPARISON_FIELDS,
) -> ConsistencyMetadataValidation:
    mode = str(mode).lower()
    if mode not in CONSISTENCY_MODES:
        raise ValueError(f"Unsupported consistency mode {mode!r}")
    if mode == "off":
        return ConsistencyMetadataValidation(mode=mode)

    warnings: list[ConsistencyMetadataIssue] = []
    failures: list[ConsistencyMetadataIssue] = []
    zero_active_token_samples: list[dict[str, int | None]] = []
    active_token_count = 0

    def add_issue(issue: ConsistencyMetadataIssue, is_failure: bool) -> None:
        if is_failure:
            failures.append(issue)
        else:
            warnings.append(issue)

    for sample in samples:
        metadata = sample_consistency_metadata(sample)
        sample_index, rollout_id = _sample_ids(sample, metadata)
        if metadata is None:
            issue, is_failure = _issue(
                mode=mode,
                code="consistency_metadata_missing",
                message="Sample is missing consistency metadata required for audit/strict comparison.",
                sample_index=sample_index,
                rollout_id=rollout_id,
            )
            add_issue(issue, is_failure)
            continue

        for path, code in required_fields:
            if _get_path(metadata, path) in (None, ""):
                issue, is_failure = _issue(
                    mode=mode,
                    code=code,
                    message=f"Consistency metadata field {path!r} is missing.",
                    sample_index=sample_index,
                    rollout_id=rollout_id,
                    field=path,
                )
                add_issue(issue, is_failure)

        active = _get_path(metadata, "active_mask.active_token_count")
        if active is not None:
            active_token_count += int(active)
            if int(active) == 0:
                zero_active_token_samples.append({"sample_index": sample_index, "rollout_id": rollout_id})
                issue, is_failure = _issue(
                    mode=mode,
                    code="zero_active_tokens",
                    message="Sample has zero active response/action tokens for consistency aggregates.",
                    sample_index=sample_index,
                    rollout_id=rollout_id,
                    field="active_mask.active_token_count",
                    strict_failure=False,
                )
                add_issue(issue, is_failure)

        provenance = metadata.get("provenance") if isinstance(metadata, dict) else None
        if isinstance(provenance, dict):
            mismatches = provenance.get("mismatches") or {}
            if mismatches:
                issue, is_failure = _issue(
                    mode=mode,
                    code="requested_actual_provenance_mismatch",
                    message="Requested-vs-actual provenance differs for compared sample.",
                    sample_index=sample_index,
                    rollout_id=rollout_id,
                    field="provenance.mismatches",
                )
                add_issue(issue, is_failure)
            if provenance.get("undeclared_fallback"):
                issue, is_failure = _issue(
                    mode=mode,
                    code="undeclared_runtime_fallback",
                    message="Actual provenance reports fallback that was not declared in requested provenance.",
                    sample_index=sample_index,
                    rollout_id=rollout_id,
                    field="provenance.undeclared_fallback",
                )
                add_issue(issue, is_failure)

    return ConsistencyMetadataValidation(
        mode=mode,
        active_token_count=active_token_count,
        zero_active_token_samples=zero_active_token_samples,
        warnings=warnings,
        failures=failures,
    )


def raise_for_consistency_metadata_failures(validation: ConsistencyMetadataValidation) -> None:
    if not validation.failures:
        return
    preview = "; ".join(
        f"{issue.code}(sample_index={issue.sample_index}, rollout_id={issue.rollout_id}, field={issue.field})"
        for issue in validation.failures[:5]
    )
    remaining = len(validation.failures) - 5
    if remaining > 0:
        preview += f"; ... {remaining} more"
    raise ValueError(f"Strict consistency metadata validation failed: {preview}")


def build_batch_layout_fingerprints(
    train_data: dict[str, Any],
    *,
    partitions: list[list[int]],
    micro_batch_indices: list[list[list[int]]],
    num_microbatches: list[int],
    global_batch_sizes: list[int],
) -> list[dict[str, Any]]:
    sample_count = len(train_data["tokens"])
    total_lengths = train_data["total_lengths"]
    response_lengths = train_data["response_lengths"]
    loss_masks = train_data["loss_masks"]
    rollout_ids = train_data.get("rollout_ids", [None] * sample_count)
    sample_indices = train_data.get("sample_indices", [None] * sample_count)

    packed_order = [sample_index for partition in partitions for sample_index in partition]
    shared = {
        "dp_size": len(partitions),
        "num_microbatches": num_microbatches,
        "global_batch_sizes": global_batch_sizes,
        "sequence_lengths_fingerprint": stable_fingerprint(total_lengths),
        "packed_order_fingerprint": stable_fingerprint(packed_order),
    }

    layouts: list[dict[str, Any] | None] = [None] * sample_count
    for dp_rank, partition in enumerate(partitions):
        for microbatch_id, local_indices in enumerate(micro_batch_indices[dp_rank]):
            for microbatch_offset, local_index in enumerate(local_indices):
                global_index = partition[local_index]
                response_length = int(response_lengths[global_index])
                active_tokens = count_active_tokens(loss_masks[global_index], response_length)
                layout = {
                    "schema_version": CONSISTENCY_METADATA_SCHEMA_VERSION,
                    "sample_index": sample_indices[global_index],
                    "rollout_id": rollout_ids[global_index],
                    "total_length": int(total_lengths[global_index]),
                    "response_length": response_length,
                    "active_token_count": active_tokens,
                    "active_mask_density": active_tokens / response_length if response_length else 0.0,
                    "dp_rank": dp_rank,
                    "rank_local_index": local_index,
                    "microbatch_id": microbatch_id,
                    "microbatch_offset": microbatch_offset,
                    "microbatch_size": len(local_indices),
                    "global_batch_shape": shared,
                }
                layout["fingerprint"] = stable_fingerprint(layout)
                layouts[global_index] = layout

    for global_index, layout in enumerate(layouts):
        if layout is not None:
            continue
        response_length = int(response_lengths[global_index])
        active_tokens = count_active_tokens(loss_masks[global_index], response_length)
        dropped_layout = {
            "schema_version": CONSISTENCY_METADATA_SCHEMA_VERSION,
            "sample_index": sample_indices[global_index],
            "rollout_id": rollout_ids[global_index],
            "total_length": int(total_lengths[global_index]),
            "response_length": response_length,
            "active_token_count": active_tokens,
            "active_mask_density": active_tokens / response_length if response_length else 0.0,
            "dropped_by_schedule": True,
            "global_batch_shape": shared,
        }
        dropped_layout["fingerprint"] = stable_fingerprint(dropped_layout)
        layouts[global_index] = dropped_layout
    return [layout for layout in layouts if layout is not None]
