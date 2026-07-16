from __future__ import annotations

import hashlib
import logging
import os
import time
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from megatron.core import mpu

from vime.utils.memory_utils import update_peak_memory_tracker
from vime.utils.rl_kernel import is_rl_kernel_op_enabled

from .cuda_event_timer import CudaEventTimerQueue

logger = logging.getLogger(__name__)

_LOGP_OP = None
_LOGP_OP_LOAD_ERROR: Exception | None = None
_LINEAR_LOGP_OP = None
_LINEAR_LOGP_OP_LOAD_ERROR: Exception | None = None
_LINEAR_LOGP_SELECTED_BACKEND: str | None = None
_WARNED_FALLBACK_REASONS: set[str] = set()
_FALLBACK_COUNTS: dict[str, int] = {"logp": 0, "linear_logp": 0}
_LINEAR_LOGP_SAVE_PROBS_CAST_LOGGED = False
_RUNTIME_COUNTER_KEYS = (
    "linear_logp_call_count",
    "linear_logp_token_count",
    "linear_logp_dispatch_elapsed_s",
    "linear_logp_fallback_count",
    "linear_logp_forward_cuda_event_count",
    "linear_logp_forward_cuda_event_elapsed_s",
    "linear_logp_forward_backward_cuda_event_count",
    "linear_logp_forward_backward_cuda_event_elapsed_s",
)
_RUNTIME_COUNTERS: dict[str, float] = dict.fromkeys(_RUNTIME_COUNTER_KEYS, 0.0)
_RUNTIME_COUNTER_LAST_SNAPSHOT: dict[str, float] = dict.fromkeys(_RUNTIME_COUNTER_KEYS, 0.0)
_CUDA_EVENT_TIMER_QUEUE = CudaEventTimerQueue()
_NATIVE_LINEAR_LOGP_BACKEND = "vime.native.linear_logp"
_ZERO_TOKEN_LINEAR_LOGP_BACKEND = "vime.linear_logp.zero_tokens"
_LINEAR_LOGP_SUPPORT_MATRIX: tuple[dict[str, str], ...] = (
    {
        "backend": "cuda_sm90",
        "implementation": "FusedLinearLogpSM90Op",
        "dtype": "bf16 inputs, fp32 selected logprob output",
        "hardware": "NVIDIA SM90/Hopper CUDA build with RL-Kernel extension",
        "tp": "supported through tp_group, vocab_start_index, global_vocab_size",
        "cp": "not supported; falls back before CP redistribution",
        "entropy": "not supported; falls back when entropy is requested",
        "full_gradient": "supported when the installed RL-Kernel op saves backward state",
    },
    {
        "backend": "triton",
        "implementation": "TritonLinearLogpOp",
        "dtype": "backend-defined floating input/output contract",
        "hardware": "CUDA devices supported by the installed Triton backend",
        "tp": "supported only when the op accepts TP metadata",
        "cp": "not supported; falls back before CP redistribution",
        "entropy": "not supported; falls back when entropy is requested",
        "full_gradient": "backend-defined; strict/full-gradient runs should validate saved-state support",
    },
    {
        "backend": "registry",
        "implementation": "kernel_registry.get_op('linear_logp')",
        "dtype": "reported by the selected RL-Kernel backend",
        "hardware": "reported by the selected RL-Kernel backend",
        "tp": "supported only when the selected op accepts TP metadata",
        "cp": "not supported; falls back before CP redistribution",
        "entropy": "not supported; falls back when entropy is requested",
        "full_gradient": "reported by the selected RL-Kernel backend",
    },
    {
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
    requested_backend: str = "auto"
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
        "": "auto",
        "auto": "auto",
        "registry": "registry",
        "triton": "triton",
        "triton_linear_logp": "triton",
        "cuda": "cuda_sm90",
        "sm90": "cuda_sm90",
        "cuda_sm90": "cuda_sm90",
        "fused_sm90": "cuda_sm90",
    }
    return aliases.get(requested, requested)


def _stable_descriptor_id(value: str | None) -> float:
    if not value:
        return 0.0
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return float(int(digest, 16))


def _op_text_attr(op: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(op, name, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if value is not None:
            return str(value)
    return None


def _op_backend_metadata(op: Any, selected_backend: str) -> tuple[str, str | None, str | None]:
    implementation = type(op).__name__
    backend_id = _op_text_attr(op, "backend_id", "backend_name", "name")
    contract_id = _op_text_attr(op, "contract_id", "numeric_contract_id")
    if backend_id is None:
        backend_id = f"rl_kernel.linear_logp.{selected_backend}.{implementation}"
    return implementation, backend_id, contract_id


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


def _set_linear_logp_fallback(reason: str) -> None:
    _clear_linear_logp_memory_metadata()
    _LINEAR_LOGP_RUNTIME_METADATA.requested_backend = _requested_linear_logp_backend()
    _LINEAR_LOGP_RUNTIME_METADATA.actual_backend = _NATIVE_LINEAR_LOGP_BACKEND
    _LINEAR_LOGP_RUNTIME_METADATA.backend_id = _NATIVE_LINEAR_LOGP_BACKEND
    _LINEAR_LOGP_RUNTIME_METADATA.contract_id = "vime.native.linear_logp.selected_logprob"
    _LINEAR_LOGP_RUNTIME_METADATA.fallback = True
    _LINEAR_LOGP_RUNTIME_METADATA.fallback_reason = reason


def _set_linear_logp_selected_backend(op: Any, selected_backend: str) -> None:
    _clear_linear_logp_memory_metadata()
    implementation, backend_id, contract_id = _op_backend_metadata(op, selected_backend)
    _LINEAR_LOGP_RUNTIME_METADATA.requested_backend = _requested_linear_logp_backend()
    _LINEAR_LOGP_RUNTIME_METADATA.actual_backend = implementation
    _LINEAR_LOGP_RUNTIME_METADATA.backend_id = backend_id
    _LINEAR_LOGP_RUNTIME_METADATA.contract_id = contract_id
    _LINEAR_LOGP_RUNTIME_METADATA.fallback = False
    _LINEAR_LOGP_RUNTIME_METADATA.fallback_reason = None


def _set_linear_logp_zero_token_decision() -> None:
    _clear_linear_logp_memory_metadata()
    _LINEAR_LOGP_RUNTIME_METADATA.requested_backend = _requested_linear_logp_backend()
    _LINEAR_LOGP_RUNTIME_METADATA.actual_backend = _ZERO_TOKEN_LINEAR_LOGP_BACKEND
    _LINEAR_LOGP_RUNTIME_METADATA.backend_id = _ZERO_TOKEN_LINEAR_LOGP_BACKEND
    _LINEAR_LOGP_RUNTIME_METADATA.contract_id = None
    _LINEAR_LOGP_RUNTIME_METADATA.fallback = False
    _LINEAR_LOGP_RUNTIME_METADATA.fallback_reason = None


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
    return tuple(dict(row) for row in _LINEAR_LOGP_SUPPORT_MATRIX)


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


@dataclass(frozen=True)
class LinearLogpContext:
    lm_head_weight: torch.Tensor
    bias: torch.Tensor | None
    tp_group: Any
    vocab_start_index: int = 0
    global_vocab_size: int | None = None
    sequence_parallel: bool = False


def get_rl_kernel_fallback_count(op: str | None = None) -> int:
    if op is not None:
        return _FALLBACK_COUNTS.get(op, 0)
    return sum(_FALLBACK_COUNTS.values())


def reset_rl_kernel_runtime_counters() -> None:
    _CUDA_EVENT_TIMER_QUEUE.clear()
    for key in _RUNTIME_COUNTER_KEYS:
        _RUNTIME_COUNTERS[key] = 0.0
        _RUNTIME_COUNTER_LAST_SNAPSHOT[key] = 0.0
    _reset_linear_logp_runtime_metadata()


def get_rl_kernel_runtime_counters() -> dict[str, float]:
    _CUDA_EVENT_TIMER_QUEUE.flush_ready()
    return dict(_RUNTIME_COUNTERS)


def get_rl_kernel_runtime_counter_delta() -> dict[str, float]:
    current = get_rl_kernel_runtime_counters()
    delta = {
        key: current.get(key, 0.0) - _RUNTIME_COUNTER_LAST_SNAPSHOT.get(key, 0.0) for key in _RUNTIME_COUNTER_KEYS
    }
    _RUNTIME_COUNTER_LAST_SNAPSHOT.update(current)
    return delta


def _record_linear_logp_runtime(token_count: int, elapsed_s: float) -> None:
    _RUNTIME_COUNTERS["linear_logp_call_count"] += 1.0
    _RUNTIME_COUNTERS["linear_logp_token_count"] += float(token_count)
    _RUNTIME_COUNTERS["linear_logp_dispatch_elapsed_s"] += float(elapsed_s)


def _record_linear_logp_forward_event_runtime(elapsed_s: float) -> None:
    _RUNTIME_COUNTERS["linear_logp_forward_cuda_event_count"] += 1.0
    _RUNTIME_COUNTERS["linear_logp_forward_cuda_event_elapsed_s"] += float(elapsed_s)


def _record_linear_logp_forward_backward_event_runtime(elapsed_s: float) -> None:
    _RUNTIME_COUNTERS["linear_logp_forward_backward_cuda_event_count"] += 1.0
    _RUNTIME_COUNTERS["linear_logp_forward_backward_cuda_event_elapsed_s"] += float(elapsed_s)


def _cuda_event_timer_enabled(tensor: torch.Tensor) -> bool:
    return _env_flag("VIME_RL_KERNEL_CUDA_EVENT_TIMER") and tensor.is_cuda


def _should_detach_linear_logp_hidden(args: Namespace) -> bool:
    override = _env_bool("VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN")
    if override is not None:
        return override

    patterns = tuple(getattr(args, "only_train_params_name_list", ()) or ())
    return bool(patterns) and all("output_layer" in str(pattern) for pattern in patterns)


def _linear_logp_needs_bf16_fast_path_cast() -> bool:
    return _env_flag("RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16") or _env_flag(
        "RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL"
    )


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
            "Casting RL-Kernel linear_logp hidden states from %s to bf16 "
            "to enable bf16 fast path; hidden_requires_grad=%s.",
            hidden_states.dtype,
            hidden_states.requires_grad,
        )
        _LINEAR_LOGP_SAVE_PROBS_CAST_LOGGED = True
    return hidden_states.to(dtype=torch.bfloat16)


def _register_linear_logp_backward_event_timer(
    *,
    start_event: torch.cuda.Event,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    watched_tensor = None
    # In full-gradient runs, shared/tied output weights can receive other
    # gradient contributions later in the model backward. Watch the op input
    # first so this timer captures the linear_logp backward boundary.
    for candidate in (hidden_states, weight, bias):
        if isinstance(candidate, torch.Tensor) and candidate.requires_grad:
            watched_tensor = candidate
            break
    if watched_tensor is None:
        return

    handle_box = {}

    def _hook(grad):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        _CUDA_EVENT_TIMER_QUEUE.enqueue(
            start_event,
            end_event,
            _record_linear_logp_forward_backward_event_runtime,
        )
        handle = handle_box.get("handle")
        if handle is not None:
            handle.remove()
        return grad

    handle_box["handle"] = watched_tensor.register_hook(_hook)


def _warn_fallback(args: Namespace, op: str, reason: str) -> None:
    _FALLBACK_COUNTS[op] = _FALLBACK_COUNTS.get(op, 0) + 1
    if op == "linear_logp":
        _RUNTIME_COUNTERS["linear_logp_fallback_count"] += 1.0
        _set_linear_logp_fallback(reason)
    if getattr(args, "rl_kernel_strict", False):
        raise RuntimeError(f"RL-Kernel {op} is enabled but unavailable: {reason}")
    warning_key = f"{op}: {reason}"
    if warning_key not in _WARNED_FALLBACK_REASONS:
        logger.warning("Falling back to vime logprob path because RL-Kernel %s is unavailable: %s", op, reason)
        _WARNED_FALLBACK_REASONS.add(warning_key)


def _get_logp_op(args: Namespace):
    global _LOGP_OP, _LOGP_OP_LOAD_ERROR
    if _LOGP_OP is not None:
        return _LOGP_OP
    if _LOGP_OP_LOAD_ERROR is not None:
        _warn_fallback(args, "logp", str(_LOGP_OP_LOAD_ERROR))
        return None

    try:
        from rl_engine.kernels.registry import kernel_registry

        _LOGP_OP = kernel_registry.get_op("logp")
        logger.info("Using RL-Kernel logp op: %s", type(_LOGP_OP).__name__)
        return _LOGP_OP
    except Exception as exc:  # pragma: no cover - exercised with missing optional package in integration envs
        _LOGP_OP_LOAD_ERROR = exc
        _warn_fallback(args, "logp", str(exc))
        return None


def _get_linear_logp_op(args: Namespace):
    global _LINEAR_LOGP_OP, _LINEAR_LOGP_OP_LOAD_ERROR, _LINEAR_LOGP_SELECTED_BACKEND
    if _LINEAR_LOGP_OP is not None:
        _set_linear_logp_selected_backend(_LINEAR_LOGP_OP, _LINEAR_LOGP_SELECTED_BACKEND or "registry")
        return _LINEAR_LOGP_OP
    if _LINEAR_LOGP_OP_LOAD_ERROR is not None:
        _warn_fallback(args, "linear_logp", str(_LINEAR_LOGP_OP_LOAD_ERROR))
        return None

    try:
        forced_backend = _requested_linear_logp_backend()
        if forced_backend == "triton":
            from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

            _LINEAR_LOGP_OP = TritonLinearLogpOp()
        elif forced_backend == "cuda_sm90":
            from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

            _LINEAR_LOGP_OP = FusedLinearLogpSM90Op()
        elif forced_backend in {"auto", "registry"}:
            from rl_engine.kernels.registry import kernel_registry

            _LINEAR_LOGP_OP = kernel_registry.get_op("linear_logp")
        else:
            raise ValueError(
                "unknown VIME_RL_KERNEL_LINEAR_LOGP_BACKEND="
                f"{forced_backend!r}; expected triton, cuda, sm90, auto, or registry"
            )
        _LINEAR_LOGP_SELECTED_BACKEND = forced_backend
        _set_linear_logp_selected_backend(_LINEAR_LOGP_OP, forced_backend)
        logger.info("Using RL-Kernel linear_logp op: %s", type(_LINEAR_LOGP_OP).__name__)
        return _LINEAR_LOGP_OP
    except Exception as exc:  # pragma: no cover - exercised with missing optional package in integration envs
        _LINEAR_LOGP_OP_LOAD_ERROR = exc
        _warn_fallback(args, "linear_logp", str(exc))
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
        _warn_fallback(args, "linear_logp", "model output_layer is unavailable")
        return None

    weight = _get_lm_head_weight(module, output_layer)
    if weight is None:
        _warn_fallback(args, "linear_logp", "LM-head weight is unavailable")
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
        _warn_fallback(args, "linear_logp", reason)
        return False
    return True


def warn_linear_logp_fallback(args: Namespace, reason: str) -> None:
    _warn_fallback(args, "linear_logp", reason)


@contextmanager
def return_hidden_states_for_linear_logp(args: Namespace, model, context: LinearLogpContext | None):
    if context is None:
        yield False
        return

    module = _unwrap_model_chunk(model)
    if not hasattr(module, "post_process"):
        _warn_fallback(args, "linear_logp", "model post_process flag is unavailable")
        yield False
        return

    old_post_process = module.post_process
    module.post_process = False
    try:
        yield True
    finally:
        module.post_process = old_post_process


def maybe_compute_logp(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    args: Namespace,
    with_entropy: bool,
) -> torch.Tensor | None:
    """Return selected log-probs from RL-Kernel when this runtime is safe.

    The first integration deliberately limits itself to forward-only logprob
    precompute paths: no autograd, no vocab tensor parallelism, no CP
    redistribution, and no entropy. Unsupported cases fall back to vime's
    Megatron-aware implementation.
    """
    if not is_rl_kernel_op_enabled(args, "logp"):
        return None

    if with_entropy:
        _warn_fallback(args, "logp", "entropy is requested")
        return None

    if logits.requires_grad or torch.is_grad_enabled():
        _warn_fallback(args, "logp", "autograd is enabled")
        return None

    if mpu.get_tensor_model_parallel_world_size() != 1:
        _warn_fallback(args, "logp", "tensor-parallel vocab shards are not supported by RL-Kernel logp")
        return None

    if mpu.get_context_parallel_world_size() != 1 or getattr(args, "allgather_cp", False):
        _warn_fallback(args, "logp", "context parallel logprob redistribution is not supported by RL-Kernel logp")
        return None

    if logits.size(0) == 0:
        return logits.new_zeros((0,), dtype=torch.float32)

    op = _get_logp_op(args)
    if op is None:
        return None

    try:
        if hasattr(op, "apply_fp32"):
            log_prob = op.apply_fp32(logits, tokens)
        else:
            log_prob = op(logits, tokens).float()
    except Exception as exc:
        _warn_fallback(args, "logp", str(exc))
        return None

    return log_prob.reshape(-1)


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
        _warn_fallback(args, "linear_logp", reason)
        return None

    if context is None:
        _warn_fallback(args, "linear_logp", "hidden-state linear_logp context is unavailable")
        return None

    if target_ids.numel() == 0:
        _set_linear_logp_zero_token_decision()
        return hidden_states.new_zeros((0,), dtype=torch.float32)

    op = _get_linear_logp_op(args)
    if op is None:
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

    start_s = time.perf_counter()
    event_timer = _cuda_event_timer_enabled(hidden_states)
    forward_start_event = None
    if event_timer:
        forward_start_event = torch.cuda.Event(enable_timing=True)
        forward_start_event.record()
    memory_probe = _env_flag("VIME_LINEAR_LOGP_MEMORY_PROBE") and hidden_states.is_cuda
    if memory_probe:
        probe_device = hidden_states.device
        torch.cuda.synchronize(probe_device)
        probe_before_alloc = torch.cuda.memory_allocated(probe_device)
        probe_before_reserved = torch.cuda.memory_reserved(probe_device)
        update_peak_memory_tracker("actor_train", device=probe_device)
        torch.cuda.reset_peak_memory_stats(probe_device)
    try:
        log_prob = op(
            hidden_states,
            weight,
            target_ids.long(),
            bias,
            tp_group=context.tp_group,
            vocab_start_index=context.vocab_start_index,
            global_vocab_size=context.global_vocab_size,
        )
    except Exception as exc:
        _warn_fallback(args, "linear_logp", str(exc))
        return None

    _set_linear_logp_selected_backend(op, _LINEAR_LOGP_SELECTED_BACKEND or _requested_linear_logp_backend())
    _record_linear_logp_runtime(target_ids.numel(), time.perf_counter() - start_s)

    if event_timer and forward_start_event is not None:
        forward_end_event = torch.cuda.Event(enable_timing=True)
        forward_end_event.record()
        _CUDA_EVENT_TIMER_QUEUE.enqueue(
            forward_start_event,
            forward_end_event,
            _record_linear_logp_forward_event_runtime,
        )
        if log_prob.requires_grad:
            _register_linear_logp_backward_event_timer(
                start_event=forward_start_event,
                hidden_states=hidden_states,
                weight=weight,
                bias=bias,
            )

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
        update_peak_memory_tracker(
            "actor_train",
            peak_alloc=probe_peak_alloc,
            peak_reserved=probe_peak_reserved,
            device=probe_device,
        )
        logger.info(
            "RL-Kernel linear_logp memory_probe: op=%s hidden_shape=%s weight_shape=%s "
            "tokens=%d alloc_before_mb=%.2f peak_alloc_mb=%.2f peak_delta_mb=%.2f "
            "alloc_after_mb=%.2f reserved_before_mb=%.2f reserved_after_mb=%.2f",
            type(op).__name__,
            tuple(hidden_states.shape),
            tuple(weight.shape),
            int(target_ids.numel()),
            probe_before_alloc / (1024**2),
            probe_peak_alloc / (1024**2),
            (probe_peak_alloc - probe_before_alloc) / (1024**2),
            probe_after_alloc / (1024**2),
            probe_before_reserved / (1024**2),
            probe_after_reserved / (1024**2),
        )

    return log_prob.float().reshape(-1)
