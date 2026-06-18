from __future__ import annotations

import logging
from argparse import Namespace

import torch
from megatron.core import mpu

from vime.utils.rl_kernel import is_rl_kernel_op_enabled

logger = logging.getLogger(__name__)

_LOGP_OP = None
_LOGP_OP_LOAD_ERROR: Exception | None = None
_WARNED_FALLBACK_REASONS: set[str] = set()


def _warn_fallback(args: Namespace, reason: str) -> None:
    if getattr(args, "rl_kernel_strict", False):
        raise RuntimeError(f"RL-Kernel logp is enabled but unavailable: {reason}")
    if reason not in _WARNED_FALLBACK_REASONS:
        logger.warning("Falling back to vime logprob path because RL-Kernel logp is unavailable: %s", reason)
        _WARNED_FALLBACK_REASONS.add(reason)


def _get_logp_op(args: Namespace):
    global _LOGP_OP, _LOGP_OP_LOAD_ERROR
    if _LOGP_OP is not None:
        return _LOGP_OP
    if _LOGP_OP_LOAD_ERROR is not None:
        _warn_fallback(args, str(_LOGP_OP_LOAD_ERROR))
        return None

    try:
        from rl_engine.kernels.registry import kernel_registry

        _LOGP_OP = kernel_registry.get_op("logp")
        logger.info("Using RL-Kernel logp op: %s", type(_LOGP_OP).__name__)
        return _LOGP_OP
    except Exception as exc:  # pragma: no cover - exercised with missing optional package in integration envs
        _LOGP_OP_LOAD_ERROR = exc
        _warn_fallback(args, str(exc))
        return None


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
        _warn_fallback(args, "entropy is requested")
        return None

    if logits.requires_grad or torch.is_grad_enabled():
        _warn_fallback(args, "autograd is enabled")
        return None

    if mpu.get_tensor_model_parallel_world_size() != 1:
        _warn_fallback(args, "tensor-parallel vocab shards are not supported by RL-Kernel logp")
        return None

    if mpu.get_context_parallel_world_size() != 1 or getattr(args, "allgather_cp", False):
        _warn_fallback(args, "context parallel logprob redistribution is not supported by RL-Kernel logp")
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
        _warn_fallback(args, str(exc))
        return None

    return log_prob.reshape(-1)
