"""Optional provider boundary for Megatron ``linear_logp`` computation.

Vime owns token layout and loss composition. A provider owns the numerical
implementation, tensor-parallel reduction, and runtime provenance.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

ProviderMode = Literal["auto", "strict"]
TokenLayoutKind = Literal["single", "zigzag", "allgather"]


class LinearLogpProviderUnavailable(RuntimeError):
    """Ask Vime to use its native implementation when mode is ``auto``."""

    linear_logp_provider_unavailable = True


@dataclass(frozen=True)
class TokenLayout:
    world_size: int
    rank: int
    layout: TokenLayoutKind

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"invalid token topology: world_size={self.world_size}, rank={self.rank}")
        if (self.world_size == 1) != (self.layout == "single"):
            raise ValueError("single-rank token layout must be 'single'; multi-rank layout must not be 'single'")


@dataclass(frozen=True)
class LinearProjection:
    weight: torch.Tensor
    bias: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.weight, torch.Tensor) or self.weight.ndim != 2:
            raise ValueError("linear_logp projection weight must be rank 2")
        if self.bias is not None and (
            not isinstance(self.bias, torch.Tensor)
            or self.bias.shape != (self.weight.size(0),)
            or self.bias.device != self.weight.device
        ):
            raise ValueError("linear_logp projection bias must match the local vocabulary shard")


@dataclass(frozen=True)
class VocabPartition:
    local_start: int
    local_size: int
    real_size: int
    padded_size: int

    def __post_init__(self) -> None:
        if self.local_start < 0 or self.local_size <= 0:
            raise ValueError("linear_logp vocabulary partition is empty or negative")
        if self.real_size <= 0 or self.real_size > self.padded_size:
            raise ValueError("linear_logp global vocabulary sizes are invalid")
        if self.local_start + self.local_size > self.padded_size:
            raise ValueError("linear_logp local vocabulary shard exceeds the padded vocabulary")


@dataclass(frozen=True)
class LinearLogpContext:
    hidden: torch.Tensor
    projection: LinearProjection
    vocab_partition: VocabPartition

    def __post_init__(self) -> None:
        if not isinstance(self.hidden, torch.Tensor) or self.hidden.ndim != 2:
            raise ValueError("linear_logp hidden states must have shape [T, H]")
        if self.hidden.size(1) != self.projection.weight.size(1):
            raise ValueError("linear_logp hidden and projection widths do not match")
        if self.projection.weight.size(0) != self.vocab_partition.local_size:
            raise ValueError("linear_logp projection and vocabulary shard widths do not match")
        if self.hidden.device != self.projection.weight.device:
            raise ValueError("linear_logp context tensors must share a device")


@dataclass(frozen=True)
class LinearLogpRequest:
    logits: torch.Tensor
    target_ids: torch.Tensor
    tensor_parallel_group: Any
    token_layout: TokenLayout
    with_entropy: bool
    with_entropy_grad: bool
    chunk_size: int
    log_prob_keep_mask: torch.Tensor | None = None
    context: LinearLogpContext | None = None
    temperature: float | torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor) or self.logits.ndim != 2:
            raise ValueError("linear_logp logits must have shape [T, V_local]")
        if self.target_ids.shape != (self.logits.size(0),):
            raise ValueError("linear_logp target IDs must have shape [T]")
        if self.target_ids.device != self.logits.device:
            raise ValueError("linear_logp target IDs and logits must share a device")
        if self.target_ids.is_floating_point() or self.target_ids.is_complex():
            raise TypeError("linear_logp target IDs must use an integer dtype")
        if self.log_prob_keep_mask is not None and self.log_prob_keep_mask.shape != self.logits.shape:
            raise ValueError("linear_logp keep mask must match logits")
        if self.context is not None and (
            self.context.hidden.size(0) != self.logits.size(0)
            or self.context.projection.weight.size(0) != self.logits.size(1)
            or self.context.hidden.device != self.logits.device
        ):
            raise ValueError("linear_logp structural context does not match logits")


@dataclass(frozen=True)
class LinearLogpResult:
    logp: torch.Tensor
    entropy: torch.Tensor | None
    backend_id: str
    contract_id: str
    provenance: Mapping[str, Any] = field(default_factory=dict)


def linear_logp_provider_path(args: Any) -> str | None:
    path = str(getattr(args, "linear_logp_provider", "") or "").strip()
    return path or None


def linear_logp_provider_mode(args: Any) -> ProviderMode:
    mode = str(getattr(args, "linear_logp_provider_mode", "auto")).strip().lower()
    if mode not in {"auto", "strict"}:
        raise ValueError(f"linear_logp provider mode must be 'auto' or 'strict', got {mode!r}")
    return mode  # type: ignore[return-value]


def compute_linear_logp(
    *,
    args: Any,
    request: LinearLogpRequest,
    native: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    path = linear_logp_provider_path(args)
    if path is None:
        return _native(request, native)

    mode = linear_logp_provider_mode(args)
    try:
        provider = _load_provider(path)
        result = provider(request)
    except Exception as exc:
        unavailable = isinstance(exc, (ImportError, AttributeError)) or bool(
            getattr(exc, "linear_logp_provider_unavailable", False)
        )
        if not unavailable or mode == "strict":
            if unavailable and mode == "strict":
                raise RuntimeError(f"linear_logp provider {path!r} is unavailable: {exc}") from exc
            raise
        return _native(request, native)

    normalized = _normalize_result(result)
    _validate_result(normalized, request, strict=mode == "strict")
    return normalized.logp, normalized.entropy


def _load_provider(path: str) -> Callable[[LinearLogpRequest], Any]:
    module_path, separator, attribute = path.rpartition(".")
    if not separator:
        raise ImportError("linear_logp provider must be a fully qualified import path")
    provider = getattr(importlib.import_module(module_path), attribute)
    if not callable(provider):
        raise TypeError(f"linear_logp provider {path!r} is not callable")
    return provider


def _native(
    request: LinearLogpRequest,
    native: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return native(
        request.logits,
        request.target_ids,
        request.tensor_parallel_group,
        with_entropy=request.with_entropy,
        with_entropy_grad=request.with_entropy_grad,
        chunk_size=request.chunk_size,
        log_prob_keep_mask=request.log_prob_keep_mask,
    )


def _normalize_result(result: Any) -> LinearLogpResult:
    if isinstance(result, LinearLogpResult):
        return result
    getter = result.__getitem__ if isinstance(result, Mapping) else lambda name: getattr(result, name)
    try:
        return LinearLogpResult(
            logp=getter("logp"),
            entropy=getter("entropy"),
            backend_id=getter("backend_id"),
            contract_id=getter("contract_id"),
            provenance=getter("provenance"),
        )
    except (AttributeError, KeyError) as exc:
        raise TypeError("linear_logp provider returned an invalid result") from exc


def _validate_result(result: LinearLogpResult, request: LinearLogpRequest, *, strict: bool) -> None:
    if result.logp.shape != (request.logits.size(0), 1):
        raise ValueError("linear_logp provider must return logp with shape [T, 1]")
    if result.logp.device != request.logits.device or not result.logp.is_floating_point():
        raise ValueError("linear_logp provider returned logp with an invalid device or dtype")
    if request.with_entropy:
        if result.entropy is None or result.entropy.shape != (request.logits.size(0),):
            raise ValueError("linear_logp provider must return entropy with shape [T]")
    elif result.entropy is not None:
        raise ValueError("linear_logp provider returned entropy when it was not requested")
    if strict:
        if not result.backend_id or not result.contract_id:
            raise ValueError("strict linear_logp results require backend and contract IDs")
        if request.logits.requires_grad and not result.logp.requires_grad:
            raise ValueError("strict linear_logp result is detached from autograd")
        if request.with_entropy_grad and result.entropy is not None and not result.entropy.requires_grad:
            raise ValueError("strict linear_logp entropy is detached from autograd")


__all__ = [
    "LinearLogpContext",
    "LinearLogpProviderUnavailable",
    "LinearLogpRequest",
    "LinearLogpResult",
    "LinearProjection",
    "TokenLayout",
    "VocabPartition",
    "compute_linear_logp",
    "linear_logp_provider_mode",
    "linear_logp_provider_path",
]
