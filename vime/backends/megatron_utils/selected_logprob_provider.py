"""Optional selected-logprob provider boundary.

Vime owns response-token layout and loss composition.  An installed provider
owns the selected-logprob implementation, numeric contract, distributed
reduction, and backend provenance.  This module deliberately does not import
or name a particular provider package.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)
_LOGGED_PROVIDER_IDENTITIES: set[tuple[str, str]] = set()

ProviderMode = Literal["auto", "strict"]


class SelectedLogprobProviderUnavailable(RuntimeError):
    """A provider may raise this to request native execution in ``auto`` mode."""


@dataclass(frozen=True)
class ContextParallelLayout:
    """Ownership of the local logit rows handed to a provider."""

    world_size: int
    rank: int
    layout: Literal["single", "zigzag", "allgather"]

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"invalid context-parallel topology: world_size={self.world_size}, rank={self.rank}")
        if self.world_size == 1 and self.layout != "single":
            raise ValueError("single-rank context parallelism must use layout='single'")
        if self.world_size > 1 and self.layout == "single":
            raise ValueError("multi-rank context parallelism must use zigzag or allgather layout")


@dataclass(frozen=True)
class SelectedLogprobRequest:
    """Normalized selected-logprob inputs supplied by the Megatron backend.

    ``logits`` and ``target_ids`` are local tensors with matching first
    dimensions. ``logits`` have already been scaled by rollout temperature.
    The provider must perform vocabulary-parallel reduction through
    ``tensor_parallel_group`` when it is not ``None``. Context parallelism is
    represented only as row ownership; it must not be folded into a vocab LSE
    reduction.
    """

    logits: torch.Tensor
    target_ids: torch.Tensor
    tensor_parallel_group: Any
    context_parallel: ContextParallelLayout
    with_entropy: bool
    with_entropy_grad: bool
    chunk_size: int
    log_prob_keep_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hidden: torch.Tensor | None = None
    lm_head_weight: torch.Tensor | None = None
    lm_head_bias: torch.Tensor | None = None
    vocab_start_index: int = 0
    global_vocab_size: int | None = None
    real_vocab_size: int | None = None
    temperature: float | torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.logits.ndim != 2:
            raise ValueError(f"selected-logprob logits must be [T, V_local], got {tuple(self.logits.shape)}")
        if self.target_ids.shape != (self.logits.size(0),):
            raise ValueError(
                "selected-logprob target_ids must be [T] aligned with logits; "
                f"got {tuple(self.target_ids.shape)} for T={self.logits.size(0)}"
            )
        if self.target_ids.device != self.logits.device:
            raise ValueError("selected-logprob target_ids must share the logits device")
        if self.target_ids.is_floating_point() or self.target_ids.is_complex():
            raise TypeError("selected-logprob target_ids must use an integer dtype")
        if self.log_prob_keep_mask is not None and self.log_prob_keep_mask.shape != self.logits.shape:
            raise ValueError("selected-logprob log_prob_keep_mask must match logits shape")
        structural = (self.hidden, self.lm_head_weight, self.lm_head_bias)
        if self.hidden is not None or self.lm_head_weight is not None:
            if not isinstance(self.hidden, torch.Tensor):
                raise ValueError("linear_logp request.hidden must be a torch.Tensor")
            if not isinstance(self.lm_head_weight, torch.Tensor):
                raise ValueError("linear_logp request.lm_head_weight must be a torch.Tensor")
            if self.hidden.ndim != 2:
                raise ValueError("linear_logp request.hidden must be [T, hidden]")
            if self.lm_head_weight.ndim != 2:
                raise ValueError("linear_logp request.lm_head_weight must be [V_local, hidden]")
            if self.hidden.size(0) != self.logits.size(0):
                raise ValueError("linear_logp hidden rows must match local logits rows")
            if self.hidden.size(1) != self.lm_head_weight.size(1):
                raise ValueError("linear_logp hidden width must match LM-head width")
            if self.hidden.device != self.logits.device or self.lm_head_weight.device != self.logits.device:
                raise ValueError("linear_logp tensors must share the logits device")
            if self.lm_head_bias is not None:
                if not isinstance(self.lm_head_bias, torch.Tensor):
                    raise ValueError("linear_logp request.lm_head_bias must be a torch.Tensor")
                if self.lm_head_bias.ndim != 1 or self.lm_head_bias.size(0) != self.lm_head_weight.size(0):
                    raise ValueError("linear_logp LM-head bias must match local vocab width")
                if self.lm_head_bias.device != self.logits.device:
                    raise ValueError("linear_logp LM-head bias must share the logits device")
            for name, value in (
                ("global_vocab_size", self.global_vocab_size),
                ("real_vocab_size", self.real_vocab_size),
            ):
                if value is not None and (isinstance(value, bool) or int(value) <= 0):
                    raise ValueError(f"linear_logp {name} must be positive")
            if self.temperature is not None:
                if isinstance(self.temperature, torch.Tensor):
                    if self.temperature.numel() not in (1, self.logits.size(0)):
                        raise ValueError("linear_logp temperature must be scalar or [T]")
                    if bool((self.temperature <= 0).any().item()):
                        raise ValueError("linear_logp temperature must be positive")
                elif float(self.temperature) <= 0:
                    raise ValueError("linear_logp temperature must be positive")
        elif any(value is not None for value in structural):
            raise ValueError("linear_logp structural request fields must be supplied together")


@dataclass(frozen=True)
class SelectedLogprobResult:
    """Provider result and immutable identity used for runtime readback."""

    selected_logprobs: torch.Tensor
    entropy: torch.Tensor | None
    backend_id: str
    contract_id: str
    provenance: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SelectedLogprobProvider(Protocol):
    def __call__(self, request: SelectedLogprobRequest) -> SelectedLogprobResult: ...


def selected_logprob_provider_path(args: Any) -> str | None:
    path = getattr(args, "selected_logprob_provider", None)
    if path is None:
        return None
    normalized = str(path).strip()
    return normalized or None


def selected_logprob_provider_mode(args: Any) -> ProviderMode:
    mode = str(getattr(args, "selected_logprob_provider_mode", "auto")).strip().lower()
    if mode not in {"auto", "strict"}:
        raise ValueError(
            "selected_logprob_provider_mode must be 'auto' or 'strict', "
            f"got {getattr(args, 'selected_logprob_provider_mode', None)!r}."
        )
    return mode  # type: ignore[return-value]


def compute_selected_logprobs(
    *,
    args: Any,
    request: SelectedLogprobRequest,
    native: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use an external provider when configured, otherwise execute native code.

    ``auto`` falls back only when the provider cannot be loaded or explicitly
    raises :class:`SelectedLogprobProviderUnavailable`. Provider bugs and
    contract violations are never hidden by a native fallback.
    """

    path = selected_logprob_provider_path(args)
    if path is None:
        return _native(request, native)

    mode = selected_logprob_provider_mode(args)
    try:
        provider = _load_provider(path)
    except (ImportError, AttributeError) as exc:
        if mode == "strict":
            raise RuntimeError(f"selected-logprob provider {path!r} is unavailable: {exc}") from exc
        logger.warning("Selected-logprob provider %s is unavailable; using native path: %s", path, exc)
        return _native(request, native)
    try:
        result = provider(request)
    except Exception as exc:
        if not _is_provider_unavailable(exc):
            raise
        if mode == "strict":
            raise RuntimeError(f"selected-logprob provider {path!r} is unavailable: {exc}") from exc
        logger.warning("Selected-logprob provider %s is unavailable; using native path: %s", path, exc)
        return _native(request, native)

    normalized = _normalize_result(result)
    _validate_result(normalized, request, strict=mode == "strict")
    _log_provider_identity(normalized)
    return normalized.selected_logprobs, normalized.entropy


def _load_provider(path: str) -> SelectedLogprobProvider:
    module_path, separator, attribute = path.rpartition(".")
    if not separator or not module_path or not attribute:
        raise ImportError("provider must use the import path 'package.module.callable'")
    provider = getattr(importlib.import_module(module_path), attribute)
    if not callable(provider):
        raise TypeError(f"selected-logprob provider {path!r} is not callable")
    return provider


def _native(
    request: SelectedLogprobRequest,
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


def _is_provider_unavailable(exc: Exception) -> bool:
    """Keep provider packages independent from Vime's exception class."""

    return isinstance(exc, SelectedLogprobProviderUnavailable) or bool(
        getattr(exc, "selected_logprob_provider_unavailable", False)
    )


def _normalize_result(result: Any) -> SelectedLogprobResult:
    """Accept Vime's dataclass or a structural result from an external package."""

    if isinstance(result, SelectedLogprobResult):
        return result
    if isinstance(result, Mapping):

        def get(name):
            return result[name]

    else:

        def get(name):
            return getattr(result, name)

    try:
        provenance = get("provenance")
    except (AttributeError, KeyError) as exc:
        raise TypeError(
            "selected-logprob providers must return selected_logprobs, entropy, backend_id, "
            "contract_id, and provenance"
        ) from exc
    try:
        selected_logprobs = get("selected_logprobs")
        entropy = get("entropy")
        backend_id = get("backend_id")
        contract_id = get("contract_id")
    except (AttributeError, KeyError) as exc:
        raise TypeError(
            "selected-logprob providers must return selected_logprobs, entropy, backend_id, "
            "contract_id, and provenance"
        ) from exc
    if not isinstance(provenance, Mapping):
        raise TypeError("selected-logprob provider provenance must be a mapping")
    return SelectedLogprobResult(
        selected_logprobs=selected_logprobs,
        entropy=entropy,
        backend_id=backend_id,
        contract_id=contract_id,
        provenance=provenance,
    )


def _validate_result(result: SelectedLogprobResult, request: SelectedLogprobRequest, *, strict: bool) -> None:
    if result.selected_logprobs.shape != (request.logits.size(0), 1):
        raise ValueError(
            "selected-logprob provider returned invalid selected_logprobs shape "
            f"{tuple(result.selected_logprobs.shape)}; expected {(request.logits.size(0), 1)}."
        )
    if result.selected_logprobs.device != request.logits.device:
        raise ValueError("selected-logprob provider returned selected_logprobs on a different device")
    if not result.selected_logprobs.is_floating_point():
        raise TypeError("selected-logprob provider must return floating-point selected_logprobs")
    if request.with_entropy:
        if result.entropy is None or result.entropy.shape != (request.logits.size(0),):
            raise ValueError("selected-logprob provider must return [T] entropy when with_entropy=True")
        if result.entropy.device != request.logits.device:
            raise ValueError("selected-logprob provider returned entropy on a different device")
        if not result.entropy.is_floating_point():
            raise TypeError("selected-logprob provider must return floating-point entropy")
    elif result.entropy is not None:
        raise ValueError("selected-logprob provider returned entropy when with_entropy=False")
    if strict:
        if not result.backend_id or not result.contract_id:
            raise ValueError("strict selected-logprob provider runs require non-empty backend_id and contract_id")
        if request.logits.requires_grad and not result.selected_logprobs.requires_grad:
            raise ValueError("strict selected-logprob provider result is detached from autograd")
        if request.with_entropy_grad and result.entropy is not None and not result.entropy.requires_grad:
            raise ValueError("strict selected-logprob provider entropy is detached from autograd")


def _log_provider_identity(result: SelectedLogprobResult) -> None:
    identity = (result.backend_id, result.contract_id)
    if identity in _LOGGED_PROVIDER_IDENTITIES:
        return
    _LOGGED_PROVIDER_IDENTITIES.add(identity)
    logger.info(
        "Selected-logprob provider active: backend_id=%s contract_id=%s provenance=%s",
        result.backend_id,
        result.contract_id,
        dict(result.provenance),
    )


__all__ = [
    "ContextParallelLayout",
    "SelectedLogprobProvider",
    "SelectedLogprobProviderUnavailable",
    "SelectedLogprobRequest",
    "SelectedLogprobResult",
    "compute_selected_logprobs",
    "selected_logprob_provider_mode",
    "selected_logprob_provider_path",
]
