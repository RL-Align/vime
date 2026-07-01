from __future__ import annotations

import logging
import time
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core import mpu

from vime.utils.rl_kernel import is_rl_kernel_op_enabled

logger = logging.getLogger(__name__)

_LOGP_OP = None
_LOGP_OP_LOAD_ERROR: Exception | None = None
_LINEAR_LOGP_OP = None
_LINEAR_LOGP_OP_LOAD_ERROR: Exception | None = None
_WARNED_FALLBACK_REASONS: set[str] = set()
_FALLBACK_COUNTS: dict[str, int] = {"logp": 0, "linear_logp": 0}
_RUNTIME_COUNTER_KEYS = (
    "linear_logp_call_count",
    "linear_logp_token_count",
    "linear_logp_dispatch_elapsed_s",
)
_RUNTIME_COUNTERS: dict[str, float] = dict.fromkeys(_RUNTIME_COUNTER_KEYS, 0.0)
_RUNTIME_COUNTER_LAST_SNAPSHOT: dict[str, float] = dict.fromkeys(_RUNTIME_COUNTER_KEYS, 0.0)


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
    for key in _RUNTIME_COUNTER_KEYS:
        _RUNTIME_COUNTERS[key] = 0.0
        _RUNTIME_COUNTER_LAST_SNAPSHOT[key] = 0.0


def get_rl_kernel_runtime_counters() -> dict[str, float]:
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


def _warn_fallback(args: Namespace, op: str, reason: str) -> None:
    _FALLBACK_COUNTS[op] = _FALLBACK_COUNTS.get(op, 0) + 1
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
    global _LINEAR_LOGP_OP, _LINEAR_LOGP_OP_LOAD_ERROR
    if _LINEAR_LOGP_OP is not None:
        return _LINEAR_LOGP_OP
    if _LINEAR_LOGP_OP_LOAD_ERROR is not None:
        _warn_fallback(args, "linear_logp", str(_LINEAR_LOGP_OP_LOAD_ERROR))
        return None

    try:
        from rl_engine.kernels.registry import kernel_registry

        _LINEAR_LOGP_OP = kernel_registry.get_op("linear_logp")
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

    start_s = time.perf_counter()
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

    _record_linear_logp_runtime(target_ids.numel(), time.perf_counter() - start_s)
    return log_prob.float().reshape(-1)
