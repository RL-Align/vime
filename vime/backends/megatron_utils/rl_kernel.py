from __future__ import annotations

import hashlib
import importlib
import logging
import os
import re
import time
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from megatron.core import mpu

from vime.backends.rl_kernel_utils import (
    ExecutionDecision,
    FallbackReason,
    RlkOperatorUnavailable,
    build_rlk_operator_adapter,
    emit_execution_decision,
    linear_logp_inputs_from_vime,
)
from vime.utils.rl_kernel import is_rl_kernel_op_enabled, is_rl_kernel_requested, is_rl_kernel_strict

logger = logging.getLogger(__name__)

_LINEAR_LOGP_ADAPTER = None
_LINEAR_LOGP_ADAPTER_ERROR: Exception | None = None
_WARNED_FALLBACK_REASONS: set[str] = set()
_FALLBACK_COUNTS: dict[str, int] = {"linear_logp": 0}
_LINEAR_LOGP_SAVE_PROBS_CAST_LOGGED = False
_RUNTIME_COUNTER_KEYS = (
    "linear_logp_call_count",
    "linear_logp_token_count",
    "linear_logp_dispatch_elapsed_s",
    "linear_logp_fallback_count",
)
_RUNTIME_COUNTERS: dict[str, float] = dict.fromkeys(_RUNTIME_COUNTER_KEYS, 0.0)
_RUNTIME_COUNTER_LAST_SNAPSHOT: dict[str, float] = dict.fromkeys(_RUNTIME_COUNTER_KEYS, 0.0)
_NATIVE_LINEAR_LOGP_BACKEND = "vime.native.linear_logp"
_ZERO_TOKEN_LINEAR_LOGP_BACKEND = "vime.linear_logp.zero_tokens"
_RLK_LINEAR_LOGP_SUPPORT_PROVIDER = "rl_engine.kernels.support"
_RLK_LINEAR_LOGP_SUPPORT_UNAVAILABLE_ROW: dict[str, str] = {
    "source": "vime_adapter",
    "backend": "rl_kernel_unavailable",
    "implementation": "RL-Kernel support provider is unavailable",
    "dtype": "reported by RL-Kernel when installed",
    "hardware": "reported by RL-Kernel when installed",
    "tp": "reported by RL-Kernel when installed",
    "cp": "reported by RL-Kernel when installed; vime decides CP fallback before dispatch",
    "entropy": "linear_logp does not produce entropy; vime computes entropy on native fallback",
    "full_gradient": "reported by RL-Kernel when installed",
}
_VIME_LINEAR_LOGP_SUPPORT_ROWS: tuple[dict[str, str], ...] = (
    {
        "source": "vime_adapter",
        "backend": "registry",
        "implementation": "RlkRegistryOperatorAdapter.linear_logp -> kernel_registry.get_op('linear_logp')",
        "dtype": "reported by RL-Kernel; vime records fp32 selected logprobs",
        "hardware": "reported by the installed RL-Kernel backend",
        "tp": "vime passes tp_group, vocab_start_index, and global_vocab_size when available",
        "cp": "not supported; falls back before CP redistribution",
        "entropy": "not supported; falls back when entropy is requested",
        "full_gradient": "supported when the selected op returns an autograd-connected result",
    },
    {
        "source": "vime_adapter",
        "backend": "native",
        "implementation": "Megatron output layer + vime calculate_log_probs_and_entropy",
        "dtype": "vime native logits path, fp32 logprob computation",
        "hardware": "same as native vime/Megatron execution",
        "tp": "supported by the native vime/Megatron logprob path",
        "cp": "supported by the native vime/Megatron CP redistribution path",
        "entropy": "supported by the native vime/Megatron path",
        "full_gradient": "supported by native autograd over materialized logits",
    },
)


@dataclass
class LinearLogpRuntimeMetadata:
    operator: str = "linear_logp"
    requested_backend: str = "registry"
    actual_backend: str = "not_selected"
    backend_id: str | None = None
    contract_id: str | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    memory_probe_enabled: bool = False
    memory_alloc_delta_mb: float | None = None
    memory_peak_alloc_delta_mb: float | None = None
    memory_reserved_delta_mb: float | None = None
    memory_peak_reserved_delta_mb: float | None = None


_LINEAR_LOGP_RUNTIME_METADATA = LinearLogpRuntimeMetadata()


@dataclass(frozen=True)
class LinearLogpContext:
    lm_head_weight: torch.Tensor
    bias: torch.Tensor | None
    tp_group: Any
    vocab_start_index: int = 0
    global_vocab_size: int | None = None
    sequence_parallel: bool = False


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def _requested_linear_logp_backend() -> str:
    requested = os.getenv("VIME_RL_KERNEL_LINEAR_LOGP_BACKEND", "").strip().lower()
    aliases = {
        "": "registry",
        "auto": "registry",
        "registry": "registry",
        "rlk": "registry",
        "rl_kernel": "registry",
    }
    return aliases.get(requested, requested)


def _stable_descriptor_id(value: str | None) -> float:
    if not value:
        return 0.0
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return float(int(digest, 16))


def _fallback_code(reason: str) -> str:
    code = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")
    return code[:80] or "linear_logp_fallback"


def _requested_modes(args: Namespace) -> tuple[str, str]:
    config = getattr(args, "rlk_mode_config", None)
    if config is not None:
        return str(getattr(config, "fast", "off")), str(getattr(config, "consistency", "off"))
    if getattr(args, "rlk_fast", None) is not None:
        fast = str(args.rlk_fast)
    elif getattr(args, "rl_kernel_strict", False):
        fast = "strict"
    elif getattr(args, "enable_rl_kernel", False):
        fast = "auto"
    else:
        fast = "off"
    return fast, str(getattr(args, "rlk_consistency", "off") or "off")


def _parallel_context() -> dict[str, Any]:
    return {
        "tp_world_size": int(mpu.get_tensor_model_parallel_world_size()),
        "tp_rank": int(mpu.get_tensor_model_parallel_rank()),
        "cp_world_size": int(mpu.get_context_parallel_world_size()),
        "cp_rank": int(mpu.get_context_parallel_rank()),
    }


def _emit_linear_logp_decision(
    args: Namespace,
    *,
    decision: str,
    actual_backend: str | None,
    fallback: bool,
    fallback_reason: str | None = None,
    backend_id: str | None = None,
    contract_id: str | None = None,
    dtype: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    fast, consistency = _requested_modes(args)
    reason = None if fallback_reason is None else FallbackReason(code=_fallback_code(fallback_reason), message=fallback_reason)
    record = ExecutionDecision(
        operator="linear_logp",
        stage="train_logprob",
        requested_mode=f"fast={fast},consistency={consistency}",
        requested_backend=_requested_linear_logp_backend(),
        actual_backend=actual_backend,
        decision=decision,
        fallback=fallback,
        fallback_reason=reason,
        capability_backend_id=backend_id,
        contract_id=contract_id,
        dtype=dtype,
        parallel_context=_parallel_context(),
        strict_eligible=fast == "strict",
        details=details or {},
    )
    emit_execution_decision(record)


def _reset_linear_logp_runtime_metadata() -> None:
    global _LINEAR_LOGP_RUNTIME_METADATA
    _LINEAR_LOGP_RUNTIME_METADATA = LinearLogpRuntimeMetadata(
        requested_backend=_requested_linear_logp_backend(),
    )


def _clear_linear_logp_memory_metadata() -> None:
    _LINEAR_LOGP_RUNTIME_METADATA.memory_probe_enabled = False
    _LINEAR_LOGP_RUNTIME_METADATA.memory_alloc_delta_mb = None
    _LINEAR_LOGP_RUNTIME_METADATA.memory_peak_alloc_delta_mb = None
    _LINEAR_LOGP_RUNTIME_METADATA.memory_reserved_delta_mb = None
    _LINEAR_LOGP_RUNTIME_METADATA.memory_peak_reserved_delta_mb = None


def _set_linear_logp_fallback(args: Namespace, reason: str) -> None:
    _clear_linear_logp_memory_metadata()
    _LINEAR_LOGP_RUNTIME_METADATA.requested_backend = _requested_linear_logp_backend()
    _LINEAR_LOGP_RUNTIME_METADATA.actual_backend = _NATIVE_LINEAR_LOGP_BACKEND
    _LINEAR_LOGP_RUNTIME_METADATA.backend_id = _NATIVE_LINEAR_LOGP_BACKEND
    _LINEAR_LOGP_RUNTIME_METADATA.contract_id = "vime.native.linear_logp.selected_logprob"
    _LINEAR_LOGP_RUNTIME_METADATA.fallback = True
    _LINEAR_LOGP_RUNTIME_METADATA.fallback_reason = reason
    _emit_linear_logp_decision(
        args,
        decision="fallback-native",
        actual_backend=_NATIVE_LINEAR_LOGP_BACKEND,
        fallback=True,
        fallback_reason=reason,
        backend_id=_NATIVE_LINEAR_LOGP_BACKEND,
        contract_id=_LINEAR_LOGP_RUNTIME_METADATA.contract_id,
    )


def _set_linear_logp_selected_backend(args: Namespace, result: Any, dtype: torch.dtype) -> None:
    _clear_linear_logp_memory_metadata()
    decision = result.decision
    provenance = dict(getattr(decision, "provenance", {}) or {})
    backend_id = provenance.get("backend_id") or getattr(decision, "backend", None)
    contract_id = provenance.get("contract_id")
    _LINEAR_LOGP_RUNTIME_METADATA.requested_backend = _requested_linear_logp_backend()
    _LINEAR_LOGP_RUNTIME_METADATA.actual_backend = getattr(decision, "backend", None)
    _LINEAR_LOGP_RUNTIME_METADATA.backend_id = backend_id
    _LINEAR_LOGP_RUNTIME_METADATA.contract_id = contract_id
    _LINEAR_LOGP_RUNTIME_METADATA.fallback = False
    _LINEAR_LOGP_RUNTIME_METADATA.fallback_reason = None
    _emit_linear_logp_decision(
        args,
        decision="optimized",
        actual_backend=backend_id,
        fallback=False,
        backend_id=backend_id,
        contract_id=contract_id,
        dtype=str(dtype).replace("torch.", ""),
        details={"implementation": getattr(decision, "backend", None)},
    )


def _set_linear_logp_zero_token_decision(args: Namespace) -> None:
    _clear_linear_logp_memory_metadata()
    _LINEAR_LOGP_RUNTIME_METADATA.requested_backend = _requested_linear_logp_backend()
    _LINEAR_LOGP_RUNTIME_METADATA.actual_backend = _ZERO_TOKEN_LINEAR_LOGP_BACKEND
    _LINEAR_LOGP_RUNTIME_METADATA.backend_id = _ZERO_TOKEN_LINEAR_LOGP_BACKEND
    _LINEAR_LOGP_RUNTIME_METADATA.contract_id = None
    _LINEAR_LOGP_RUNTIME_METADATA.fallback = False
    _LINEAR_LOGP_RUNTIME_METADATA.fallback_reason = None
    _emit_linear_logp_decision(
        args,
        decision="optimized",
        actual_backend=_ZERO_TOKEN_LINEAR_LOGP_BACKEND,
        fallback=False,
        backend_id=_ZERO_TOKEN_LINEAR_LOGP_BACKEND,
        details={"zero_tokens": True},
    )


def _record_linear_logp_memory_probe(
    *,
    alloc_before: int,
    alloc_after: int,
    peak_alloc: int,
    reserved_before: int,
    reserved_after: int,
    peak_reserved: int,
) -> None:
    mb = float(1024**2)
    _LINEAR_LOGP_RUNTIME_METADATA.memory_probe_enabled = True
    _LINEAR_LOGP_RUNTIME_METADATA.memory_alloc_delta_mb = (alloc_after - alloc_before) / mb
    _LINEAR_LOGP_RUNTIME_METADATA.memory_peak_alloc_delta_mb = (peak_alloc - alloc_before) / mb
    _LINEAR_LOGP_RUNTIME_METADATA.memory_reserved_delta_mb = (reserved_after - reserved_before) / mb
    _LINEAR_LOGP_RUNTIME_METADATA.memory_peak_reserved_delta_mb = (peak_reserved - reserved_before) / mb


def get_linear_logp_support_matrix() -> tuple[dict[str, str], ...]:
    return (
        *_load_rl_kernel_linear_logp_support_rows(),
        *(dict(row) for row in _VIME_LINEAR_LOGP_SUPPORT_ROWS),
    )


def _load_rl_kernel_linear_logp_support_rows() -> tuple[dict[str, str], ...]:
    try:
        provider = importlib.import_module(_RLK_LINEAR_LOGP_SUPPORT_PROVIDER)
        return tuple(dict(row) for row in provider.get_linear_logp_support_matrix())
    except Exception:
        return (dict(_RLK_LINEAR_LOGP_SUPPORT_UNAVAILABLE_ROW),)


def get_linear_logp_runtime_metadata() -> dict[str, Any]:
    metadata = asdict(_LINEAR_LOGP_RUNTIME_METADATA)
    metadata["backend_descriptor_id"] = _stable_descriptor_id(metadata.get("backend_id"))
    metadata["contract_descriptor_id"] = _stable_descriptor_id(metadata.get("contract_id"))
    metadata["fallback_reason_descriptor_id"] = _stable_descriptor_id(metadata.get("fallback_reason"))
    return metadata


def get_linear_logp_runtime_log_metrics(prefix: str = "train/rl_kernel_linear_logp_") -> dict[str, float]:
    metadata = get_linear_logp_runtime_metadata()
    metrics = {
        f"{prefix}fallback": 1.0 if metadata.get("fallback") else 0.0,
        f"{prefix}backend_descriptor_id": float(metadata["backend_descriptor_id"]),
        f"{prefix}contract_descriptor_id": float(metadata["contract_descriptor_id"]),
        f"{prefix}fallback_reason_descriptor_id": float(metadata["fallback_reason_descriptor_id"]),
    }
    for key in (
        "memory_alloc_delta_mb",
        "memory_peak_alloc_delta_mb",
        "memory_reserved_delta_mb",
        "memory_peak_reserved_delta_mb",
    ):
        value = metadata.get(key)
        if value is not None:
            metrics[f"{prefix}{key}"] = float(value)
    return metrics


def get_rl_kernel_fallback_count(op: str | None = None) -> int:
    if op is not None:
        return _FALLBACK_COUNTS.get(op, 0)
    return sum(_FALLBACK_COUNTS.values())


def reset_rl_kernel_runtime_counters() -> None:
    for key in _RUNTIME_COUNTER_KEYS:
        _RUNTIME_COUNTERS[key] = 0.0
        _RUNTIME_COUNTER_LAST_SNAPSHOT[key] = 0.0
    _reset_linear_logp_runtime_metadata()


def get_rl_kernel_runtime_counters() -> dict[str, float]:
    return dict(_RUNTIME_COUNTERS)


def get_rl_kernel_runtime_counter_delta() -> dict[str, float]:
    current = get_rl_kernel_runtime_counters()
    delta = {key: current.get(key, 0.0) - _RUNTIME_COUNTER_LAST_SNAPSHOT.get(key, 0.0) for key in _RUNTIME_COUNTER_KEYS}
    _RUNTIME_COUNTER_LAST_SNAPSHOT.update(current)
    return delta


def _record_linear_logp_runtime(token_count: int, elapsed_s: float) -> None:
    _RUNTIME_COUNTERS["linear_logp_call_count"] += 1.0
    _RUNTIME_COUNTERS["linear_logp_token_count"] += float(token_count)
    _RUNTIME_COUNTERS["linear_logp_dispatch_elapsed_s"] += float(elapsed_s)


def _should_detach_linear_logp_hidden(args: Namespace) -> bool:
    override = _env_bool("VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN")
    if override is not None:
        return override
    patterns = tuple(getattr(args, "only_train_params_name_list", ()) or ())
    return bool(patterns) and all("output_layer" in str(pattern) for pattern in patterns)


def _linear_logp_needs_bf16_fast_path_cast() -> bool:
    return _env_flag("RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16") or _env_flag("RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL")


def _maybe_cast_hidden_for_bf16_fast_path(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    global _LINEAR_LOGP_SAVE_PROBS_CAST_LOGGED
    if not _linear_logp_needs_bf16_fast_path_cast():
        return hidden_states
    if not (hidden_states.is_cuda and weight.is_cuda and hidden_states.device == weight.device):
        return hidden_states
    if weight.dtype != torch.bfloat16 or hidden_states.dtype == torch.bfloat16:
        return hidden_states
    if not hidden_states.is_floating_point():
        return hidden_states

    if not _LINEAR_LOGP_SAVE_PROBS_CAST_LOGGED:
        logger.info(
            "Casting RL-Kernel linear_logp hidden states from %s to bf16 to enable bf16 fast path.",
            hidden_states.dtype,
        )
        _LINEAR_LOGP_SAVE_PROBS_CAST_LOGGED = True
    return hidden_states.to(dtype=torch.bfloat16)


def _warn_fallback(args: Namespace, reason: str) -> None:
    _FALLBACK_COUNTS["linear_logp"] = _FALLBACK_COUNTS.get("linear_logp", 0) + 1
    _RUNTIME_COUNTERS["linear_logp_fallback_count"] += 1.0
    _set_linear_logp_fallback(args, reason)
    if is_rl_kernel_strict(args):
        raise RuntimeError(f"RL-Kernel linear_logp is enabled but unavailable: {reason}")
    if reason not in _WARNED_FALLBACK_REASONS:
        logger.warning("Falling back to vime logprob path because RL-Kernel linear_logp is unavailable: %s", reason)
        _WARNED_FALLBACK_REASONS.add(reason)


def _get_linear_logp_adapter(args: Namespace):
    global _LINEAR_LOGP_ADAPTER, _LINEAR_LOGP_ADAPTER_ERROR
    if _LINEAR_LOGP_ADAPTER is not None:
        return _LINEAR_LOGP_ADAPTER
    if _LINEAR_LOGP_ADAPTER_ERROR is not None:
        _warn_fallback(args, str(_LINEAR_LOGP_ADAPTER_ERROR))
        return None

    try:
        _LINEAR_LOGP_ADAPTER = build_rlk_operator_adapter(args=args, backend="auto")
        logger.info("Using RL-Kernel operator adapter for linear_logp: %s", type(_LINEAR_LOGP_ADAPTER).__name__)
        return _LINEAR_LOGP_ADAPTER
    except Exception as exc:  # pragma: no cover - exercised when optional dependencies are unavailable
        _LINEAR_LOGP_ADAPTER_ERROR = exc
        _warn_fallback(args, str(exc))
        return None


def _unwrap_model_chunk(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _is_pipeline_last_stage_for_model(model) -> bool:
    module = _unwrap_model_chunk(model)
    vp_stage = getattr(module, "vp_stage", None)
    try:
        vp_world_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    except Exception:
        vp_world_size = None

    try:
        if vp_world_size is not None and vp_stage is not None:
            return bool(mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage))
        return bool(mpu.is_pipeline_last_stage(ignore_virtual=True))
    except Exception:
        return True


def _get_lm_head_weight(model, output_layer) -> torch.Tensor | None:
    if getattr(model, "share_embeddings_and_output_weights", False):
        shared_weight = getattr(model, "shared_embedding_or_output_weight", None)
        if callable(shared_weight):
            try:
                weight = shared_weight()
                if isinstance(weight, torch.Tensor):
                    return weight
            except Exception:
                logger.debug("Unable to read shared embedding/output weight for RL-Kernel linear_logp.", exc_info=True)

    weight = getattr(output_layer, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight

    shared_weight = getattr(model, "shared_embedding_or_output_weight", None)
    if callable(shared_weight):
        try:
            weight = shared_weight()
            if isinstance(weight, torch.Tensor):
                return weight
        except Exception:
            logger.debug("Unable to read shared embedding/output weight for RL-Kernel linear_logp.", exc_info=True)

    return None


def get_linear_logp_context_from_model(args: Namespace, model) -> LinearLogpContext | None:
    if not is_rl_kernel_op_enabled(args, "linear_logp"):
        return None

    if not _is_pipeline_last_stage_for_model(model):
        return None

    module = _unwrap_model_chunk(model)
    output_layer = getattr(module, "output_layer", None)
    if output_layer is None:
        _warn_fallback(args, "model output_layer is unavailable")
        return None

    weight = _get_lm_head_weight(module, output_layer)
    if weight is None:
        _warn_fallback(args, "LM-head weight is unavailable")
        return None

    bias = getattr(output_layer, "bias", None)
    if not isinstance(bias, torch.Tensor):
        bias = None

    tp_world_size = int(mpu.get_tensor_model_parallel_world_size())
    tp_group = mpu.get_tensor_model_parallel_group() if tp_world_size > 1 else None
    vocab_start_index = 0
    global_vocab_size = None
    if tp_world_size > 1:
        local_vocab_size = int(weight.size(0))
        vocab_start_index = int(mpu.get_tensor_model_parallel_rank()) * local_vocab_size
        global_vocab_size = getattr(args, "padded_vocab_size", None)
        if global_vocab_size is None:
            global_vocab_size = local_vocab_size * tp_world_size

    return LinearLogpContext(
        lm_head_weight=weight,
        bias=bias,
        tp_group=tp_group,
        vocab_start_index=vocab_start_index,
        global_vocab_size=None if global_vocab_size is None else int(global_vocab_size),
        sequence_parallel=bool(getattr(output_layer, "sequence_parallel", getattr(args, "sequence_parallel", False))),
    )


def _linear_logp_runtime_blocker(args: Namespace, *, with_entropy: bool) -> str | None:
    if with_entropy:
        return "entropy is requested"
    if getattr(args, "qkv_format", "thd") != "thd":
        return "only qkv_format=thd is supported by RL-Kernel linear_logp"
    if mpu.get_context_parallel_world_size() != 1 or getattr(args, "allgather_cp", False):
        return "context parallel logprob redistribution is not supported by RL-Kernel linear_logp"
    if getattr(args, "rollout_temperature", 1.0) <= 0:
        return "rollout_temperature must be positive"
    return None


def should_use_linear_logp_model_output(args: Namespace, *, with_entropy: bool) -> bool:
    if not is_rl_kernel_op_enabled(args, "linear_logp"):
        return False
    reason = _linear_logp_runtime_blocker(args, with_entropy=with_entropy)
    if reason is not None:
        _warn_fallback(args, reason)
        return False
    return True


def warn_linear_logp_fallback(args: Namespace, reason: str) -> None:
    if is_rl_kernel_requested(args):
        _warn_fallback(args, reason)


@contextmanager
def return_hidden_states_for_linear_logp(args: Namespace, model, context: LinearLogpContext | None):
    if context is None:
        yield False
        return

    module = _unwrap_model_chunk(model)
    if not hasattr(module, "post_process"):
        _warn_fallback(args, "model post_process flag is unavailable")
        yield False
        return

    old_post_process = module.post_process
    module.post_process = False
    try:
        yield True
    finally:
        module.post_process = old_post_process


def maybe_compute_linear_logp(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    context: LinearLogpContext | None,
    args: Namespace,
    with_entropy: bool,
) -> torch.Tensor | None:
    if not is_rl_kernel_op_enabled(args, "linear_logp"):
        return None

    reason = _linear_logp_runtime_blocker(args, with_entropy=with_entropy)
    if reason is not None:
        _warn_fallback(args, reason)
        return None

    if context is None:
        _warn_fallback(args, "hidden-state linear_logp context is unavailable")
        return None

    if target_ids.numel() == 0:
        _set_linear_logp_zero_token_decision(args)
        return hidden_states.new_zeros((0,), dtype=torch.float32)

    adapter = _get_linear_logp_adapter(args)
    if adapter is None:
        return None

    weight = context.lm_head_weight
    bias = context.bias
    rollout_temperature = float(getattr(args, "rollout_temperature", 1.0))
    if rollout_temperature != 1.0:
        weight = weight / rollout_temperature
        if bias is not None:
            bias = bias / rollout_temperature
    if _should_detach_linear_logp_hidden(args):
        hidden_states = hidden_states.detach()
    hidden_states = _maybe_cast_hidden_for_bf16_fast_path(hidden_states, weight)

    memory_probe = _env_flag("VIME_LINEAR_LOGP_MEMORY_PROBE") and hidden_states.is_cuda
    if memory_probe:
        probe_device = hidden_states.device
        torch.cuda.synchronize(probe_device)
        probe_before_alloc = torch.cuda.memory_allocated(probe_device)
        probe_before_reserved = torch.cuda.memory_reserved(probe_device)
        torch.cuda.reset_peak_memory_stats(probe_device)

    start_s = time.perf_counter()
    try:
        result = adapter.linear_logp(
            linear_logp_inputs_from_vime(
                hidden=hidden_states,
                lm_head_weight=weight,
                target_ids=target_ids.long(),
                bias=bias,
                tp_group=context.tp_group,
                vocab_start_index=context.vocab_start_index,
                global_vocab_size=context.global_vocab_size,
                metadata={"requested_backend": _requested_linear_logp_backend()},
            )
        )
    except RlkOperatorUnavailable:
        raise
    except Exception as exc:
        _warn_fallback(args, str(exc))
        return None

    elapsed_s = getattr(result.decision, "elapsed_s", 0.0) or (time.perf_counter() - start_s)
    if result.value is None:
        _warn_fallback(args, result.decision.reason or "adapter returned no linear_logp value")
        return None

    _set_linear_logp_selected_backend(args, result, hidden_states.dtype)
    _record_linear_logp_runtime(target_ids.numel(), elapsed_s)

    if memory_probe:
        torch.cuda.synchronize(probe_device)
        probe_after_alloc = torch.cuda.memory_allocated(probe_device)
        probe_after_reserved = torch.cuda.memory_reserved(probe_device)
        probe_peak_alloc = torch.cuda.max_memory_allocated(probe_device)
        probe_peak_reserved = torch.cuda.max_memory_reserved(probe_device)
        _record_linear_logp_memory_probe(
            alloc_before=probe_before_alloc,
            alloc_after=probe_after_alloc,
            peak_alloc=probe_peak_alloc,
            reserved_before=probe_before_reserved,
            reserved_after=probe_after_reserved,
            peak_reserved=probe_peak_reserved,
        )
        logger.info(
            "RL-Kernel linear_logp memory_probe: hidden_shape=%s weight_shape=%s tokens=%d alloc_delta_mb=%.2f peak_alloc_delta_mb=%.2f reserved_delta_mb=%.2f peak_reserved_delta_mb=%.2f",
            tuple(hidden_states.shape),
            tuple(weight.shape),
            int(target_ids.numel()),
            _LINEAR_LOGP_RUNTIME_METADATA.memory_alloc_delta_mb,
            _LINEAR_LOGP_RUNTIME_METADATA.memory_peak_alloc_delta_mb,
            _LINEAR_LOGP_RUNTIME_METADATA.memory_reserved_delta_mb,
            _LINEAR_LOGP_RUNTIME_METADATA.memory_peak_reserved_delta_mb,
        )

    return result.value.float().reshape(-1)
