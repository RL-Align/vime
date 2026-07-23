from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

RLK_CONSISTENCY_MODES = ("off", "audit", "strict")
RLK_CONSISTENCY_MODE_ATTRS = (
    "rlk_consistency_mode",
    "rlk_consistency",
    "rl_kernel_consistency_mode",
    "rl_kernel_consistency",
)
RLK_CONSISTENCY_MODE_ENVS = (
    "VIME_RLK_CONSISTENCY_MODE",
    "VIME_RLK_CONSISTENCY",
    "VIME_RL_KERNEL_CONSISTENCY_MODE",
    "VIME_RL_KERNEL_CONSISTENCY",
)

MISSING_METADATA_FIELDS = (
    "model_name",
    "backend_id",
    "contract_id",
    "batch_layout_fingerprint",
    "provenance_fingerprint",
)


@dataclass(frozen=True)
class DlogpAuditWarning:
    code: str
    message: str
    sample_position: int | None = None
    sample_id: int | str | None = None
    field: str | None = None


@dataclass(frozen=True)
class DlogpAuditReport:
    metrics: dict[str, torch.Tensor]
    warnings: tuple[DlogpAuditWarning, ...]
    worst_token: dict[str, Any] | None


def get_rlk_consistency_mode(args: Any | None, environ: Mapping[str, str] | None = None) -> str:
    """Return the requested RL-Kernel consistency mode.

    The helper deliberately accepts several attribute/env names so this
    audit-only feature can coexist with older or stacked branches that used
    slightly different flag names.
    """

    raw_mode: Any | None = None
    if args is not None:
        for attr in RLK_CONSISTENCY_MODE_ATTRS:
            value = getattr(args, attr, None)
            if value is not None:
                raw_mode = value
                break

    if raw_mode is None:
        env = os.environ if environ is None else environ
        for env_name in RLK_CONSISTENCY_MODE_ENVS:
            value = env.get(env_name)
            if value is not None:
                raw_mode = value
                break

    if raw_mode is None:
        return "off"

    mode = str(raw_mode).strip().lower().replace("_", "-")
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "1": "audit",
        "true": "audit",
        "yes": "audit",
        "audit-only": "audit",
    }
    mode = aliases.get(mode, mode)
    if mode not in RLK_CONSISTENCY_MODES:
        raise ValueError(
            f"Unsupported RL-Kernel consistency mode {raw_mode!r}; "
            f"expected one of {', '.join(RLK_CONSISTENCY_MODES)}."
        )
    return mode


def is_dlogp_audit_enabled(args: Any | None, environ: Mapping[str, str] | None = None) -> bool:
    return get_rlk_consistency_mode(args, environ=environ) in {"audit", "strict"}


def compute_dlogp_diagnostics(
    train_log_probs: Sequence[torch.Tensor] | torch.Tensor,
    rollout_log_probs: Sequence[torch.Tensor] | torch.Tensor | None,
    loss_masks: Sequence[torch.Tensor] | torch.Tensor,
    *,
    sample_indices: Sequence[int | torch.Tensor] | torch.Tensor | None = None,
    rollout_ids: Sequence[int | torch.Tensor] | torch.Tensor | None = None,
    metadata: Sequence[Mapping[str, Any] | None] | None = None,
    rank: int | None = None,
    model_name: str | None = None,
    backend_id: str | None = None,
    contract_id: str | None = None,
    batch_layout_fingerprint: str | None = None,
    provenance_fingerprint: str | None = None,
    eps_clip: float = 0.2,
    prefix: str = "rlk_audit_",
) -> DlogpAuditReport:
    """Compute read-only rollout-training dlogp diagnostics over active tokens."""

    tensors = _as_tensor_list(train_log_probs)
    masks = _as_tensor_list(loss_masks)
    rollout_tensors = _as_tensor_list(rollout_log_probs) if rollout_log_probs is not None else []
    device = _first_device(tensors, masks, rollout_tensors)
    dtype = _first_floating_dtype(tensors, rollout_tensors)

    warnings: list[DlogpAuditWarning] = []
    if not rollout_tensors:
        warnings.append(
            DlogpAuditWarning(
                code="missing_rollout_log_probs",
                message="rollout_log_probs are required for dlogp diagnostics.",
                field="rollout_log_probs",
            )
        )

    if len(tensors) != len(masks):
        warnings.append(
            DlogpAuditWarning(
                code="sample_count_mismatch",
                message=f"train_log_probs has {len(tensors)} samples but loss_masks has {len(masks)}.",
            )
        )
    if rollout_tensors and len(tensors) != len(rollout_tensors):
        warnings.append(
            DlogpAuditWarning(
                code="sample_count_mismatch",
                message=f"train_log_probs has {len(tensors)} samples but rollout_log_probs has {len(rollout_tensors)}.",
                field="rollout_log_probs",
            )
        )

    context_metadata = {
        "model_name": model_name,
        "backend_id": backend_id,
        "contract_id": contract_id,
        "batch_layout_fingerprint": batch_layout_fingerprint,
        "provenance_fingerprint": provenance_fingerprint,
    }
    for field, value in context_metadata.items():
        if value is None and not _metadata_field_available(metadata, field):
            warnings.append(
                DlogpAuditWarning(
                    code="missing_metadata",
                    message=f"{field} is not available for dlogp diagnostics.",
                    field=field,
                )
            )

    dlogp_parts: list[torch.Tensor] = []
    abs_parts: list[torch.Tensor] = []
    ratio_parts: list[torch.Tensor] = []
    clip_parts: list[torch.Tensor] = []
    approx_kl_parts: list[torch.Tensor] = []
    total_token_count = 0
    active_token_count = 0
    zero_active_sample_count = 0
    worst_token: dict[str, Any] | None = None
    worst_abs_value: torch.Tensor | None = None

    sample_count = min(len(tensors), len(masks), len(rollout_tensors))
    with torch.no_grad():
        for sample_position in range(sample_count):
            train = tensors[sample_position].detach().flatten()
            rollout = rollout_tensors[sample_position].detach().flatten()
            mask = masks[sample_position].detach().flatten()
            sample_id = _value_at(sample_indices, sample_position)
            rollout_id = _value_at(rollout_ids, sample_position)
            sample_meta = (
                metadata[sample_position] if metadata is not None and sample_position < len(metadata) else None
            )

            if train.numel() != rollout.numel() or train.numel() != mask.numel():
                warnings.append(
                    DlogpAuditWarning(
                        code="shape_mismatch",
                        message=(
                            f"sample {sample_position} has train_log_probs={train.numel()}, "
                            f"rollout_log_probs={rollout.numel()}, loss_masks={mask.numel()}."
                        ),
                        sample_position=sample_position,
                        sample_id=sample_id,
                    )
                )
                continue

            total_token_count += int(mask.numel())
            active_mask = mask.to(dtype=torch.bool)
            sample_active_count = int(active_mask.sum().item())
            active_token_count += sample_active_count
            if sample_active_count == 0:
                zero_active_sample_count += 1
                warnings.append(
                    DlogpAuditWarning(
                        code="zero_active_tokens",
                        message=f"sample {sample_position} has no active response/action tokens.",
                        sample_position=sample_position,
                        sample_id=sample_id,
                    )
                )
                continue

            active_positions = active_mask.nonzero(as_tuple=False).flatten()
            sample_dlogp = (train.to(dtype=dtype) - rollout.to(dtype=dtype))[active_mask]
            finite_mask = torch.isfinite(sample_dlogp)
            if not bool(finite_mask.all().item()):
                dropped = int((~finite_mask).sum().item())
                warnings.append(
                    DlogpAuditWarning(
                        code="non_finite_dlogp",
                        message=f"sample {sample_position} has {dropped} non-finite active dlogp values.",
                        sample_position=sample_position,
                        sample_id=sample_id,
                    )
                )
                active_positions = active_positions[finite_mask]
                sample_dlogp = sample_dlogp[finite_mask]
                if sample_dlogp.numel() == 0:
                    continue

            sample_abs = sample_dlogp.abs()
            sample_ratio = sample_dlogp.exp()
            sample_clip = ((sample_ratio - 1.0).abs() > eps_clip).to(dtype=dtype)
            sample_approx_kl = sample_ratio - 1.0 - sample_dlogp

            dlogp_parts.append(sample_dlogp)
            abs_parts.append(sample_abs)
            ratio_parts.append(sample_ratio)
            clip_parts.append(sample_clip)
            approx_kl_parts.append(sample_approx_kl)

            sample_worst_abs, sample_worst_active_index = sample_abs.max(dim=0)
            if worst_abs_value is None or bool((sample_worst_abs > worst_abs_value).item()):
                token_position = int(active_positions[int(sample_worst_active_index.item())].item())
                worst_abs_value = sample_worst_abs
                worst_token = {
                    "abs_dlogp": float(sample_worst_abs.item()),
                    "dlogp": float(sample_dlogp[int(sample_worst_active_index.item())].item()),
                    "sample_position": sample_position,
                    "sample_id": sample_id,
                    "sample_index": sample_id,
                    "rollout_id": rollout_id,
                    "token_position": token_position,
                    "rank": rank,
                    "model_name": _metadata_value("model_name", sample_meta, model_name),
                    "backend_id": _metadata_value("backend_id", sample_meta, backend_id),
                    "contract_id": _metadata_value("contract_id", sample_meta, contract_id),
                    "batch_layout_fingerprint": _metadata_value(
                        "batch_layout_fingerprint",
                        sample_meta,
                        batch_layout_fingerprint,
                    ),
                    "provenance_fingerprint": _metadata_value(
                        "provenance_fingerprint",
                        sample_meta,
                        provenance_fingerprint,
                    ),
                }

        if dlogp_parts:
            abs_dlogp = torch.cat(abs_parts).to(device=device, dtype=dtype)
            ratio0 = torch.cat(ratio_parts).to(device=device, dtype=dtype)
            clipfrac0_values = torch.cat(clip_parts).to(device=device, dtype=dtype)
            approx_kl0_values = torch.cat(approx_kl_parts).to(device=device, dtype=dtype)
            quantiles = torch.quantile(abs_dlogp, torch.tensor([0.5, 0.9, 0.99], device=device, dtype=dtype))
            metrics = {
                f"{prefix}dlogp_abs_mean": abs_dlogp.mean(),
                f"{prefix}dlogp_abs_max": abs_dlogp.max(),
                f"{prefix}dlogp_abs_p50": quantiles[0],
                f"{prefix}dlogp_abs_p90": quantiles[1],
                f"{prefix}dlogp_abs_p99": quantiles[2],
                f"{prefix}ratio0_mean": ratio0.mean(),
                f"{prefix}clipfrac0": clipfrac0_values.mean(),
                f"{prefix}approx_kl0": approx_kl0_values.mean(),
            }
        else:
            metrics = {
                f"{prefix}dlogp_abs_mean": _zero(device, dtype),
                f"{prefix}dlogp_abs_max": _zero(device, dtype),
                f"{prefix}dlogp_abs_p50": _zero(device, dtype),
                f"{prefix}dlogp_abs_p90": _zero(device, dtype),
                f"{prefix}dlogp_abs_p99": _zero(device, dtype),
                f"{prefix}ratio0_mean": _zero(device, dtype),
                f"{prefix}clipfrac0": _zero(device, dtype),
                f"{prefix}approx_kl0": _zero(device, dtype),
            }

    metrics.update(
        {
            f"{prefix}active_token_count": torch.tensor(float(active_token_count), device=device, dtype=dtype),
            f"{prefix}mask_coverage": torch.tensor(
                float(active_token_count / total_token_count) if total_token_count else 0.0,
                device=device,
                dtype=dtype,
            ),
            f"{prefix}zero_active_sample_count": torch.tensor(
                float(zero_active_sample_count), device=device, dtype=dtype
            ),
            f"{prefix}warning_count": torch.tensor(float(len(warnings)), device=device, dtype=dtype),
        }
    )

    if worst_token is not None:
        metrics.update(
            {
                f"{prefix}worst_abs_dlogp": torch.tensor(worst_token["abs_dlogp"], device=device, dtype=dtype),
                f"{prefix}worst_dlogp": torch.tensor(worst_token["dlogp"], device=device, dtype=dtype),
                f"{prefix}worst_sample_position": torch.tensor(
                    float(worst_token["sample_position"]),
                    device=device,
                    dtype=dtype,
                ),
                f"{prefix}worst_sample_index": torch.tensor(
                    float(worst_token["sample_index"]) if _is_number(worst_token["sample_index"]) else -1.0,
                    device=device,
                    dtype=dtype,
                ),
                f"{prefix}worst_rollout_id": torch.tensor(
                    float(worst_token["rollout_id"]) if _is_number(worst_token["rollout_id"]) else -1.0,
                    device=device,
                    dtype=dtype,
                ),
                f"{prefix}worst_token_position": torch.tensor(
                    float(worst_token["token_position"]),
                    device=device,
                    dtype=dtype,
                ),
                f"{prefix}worst_rank": torch.tensor(
                    float(rank) if rank is not None else -1.0,
                    device=device,
                    dtype=dtype,
                ),
            }
        )
    else:
        metrics.update(
            {
                f"{prefix}worst_abs_dlogp": _zero(device, dtype),
                f"{prefix}worst_dlogp": _zero(device, dtype),
                f"{prefix}worst_sample_position": torch.tensor(-1.0, device=device, dtype=dtype),
                f"{prefix}worst_sample_index": torch.tensor(-1.0, device=device, dtype=dtype),
                f"{prefix}worst_rollout_id": torch.tensor(-1.0, device=device, dtype=dtype),
                f"{prefix}worst_token_position": torch.tensor(-1.0, device=device, dtype=dtype),
                f"{prefix}worst_rank": torch.tensor(
                    float(rank) if rank is not None else -1.0, device=device, dtype=dtype
                ),
            }
        )

    metrics = {key: value.clone().detach() for key, value in metrics.items()}
    return DlogpAuditReport(metrics=metrics, warnings=tuple(warnings), worst_token=worst_token)


def _as_tensor_list(value: Sequence[torch.Tensor] | torch.Tensor | None) -> list[torch.Tensor]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    return [torch.as_tensor(item) for item in value]


def _first_device(*groups: Sequence[torch.Tensor]) -> torch.device:
    for group in groups:
        for tensor in group:
            return tensor.device
    return torch.device("cpu")


def _first_floating_dtype(*groups: Sequence[torch.Tensor]) -> torch.dtype:
    for group in groups:
        for tensor in group:
            if tensor.is_floating_point():
                return tensor.dtype
    return torch.float32


def _zero(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(0.0, device=device, dtype=dtype)


def _value_at(values: Sequence[Any] | torch.Tensor | None, index: int) -> Any:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if index >= values.numel():
            return None
        return _python_scalar(values.flatten()[index])
    if index >= len(values):
        return None
    value = values[index]
    if isinstance(value, torch.Tensor):
        return _python_scalar(value)
    return value


def _python_scalar(value: torch.Tensor) -> Any:
    if value.numel() != 1:
        return value.detach().cpu().tolist()
    return value.detach().cpu().item()


def _metadata_value(field: str, sample_meta: Mapping[str, Any] | None, fallback: Any) -> Any:
    if sample_meta is not None and field in sample_meta:
        return sample_meta[field]
    return fallback


def _metadata_field_available(metadata: Sequence[Mapping[str, Any] | None] | None, field: str) -> bool:
    if metadata is None:
        return False
    return any(sample_meta is not None and sample_meta.get(field) is not None for sample_meta in metadata)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
