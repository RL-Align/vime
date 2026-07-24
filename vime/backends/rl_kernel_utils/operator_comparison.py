"""Lazy re-export of RL-Kernel-owned operator comparison helpers."""

from __future__ import annotations

import importlib
from typing import Any

_RLK_OPERATOR_COMPARISON_MODULE = "rl_engine.alignment.cross_config.operator_comparison"

_OPERATOR_COMPARISON_EXPORTS = frozenset(
    {
        "OPERATOR_COMPARISON_SPECS",
        "PHASE4_TARGET_OPERATORS",
        "RLK_OP_ATTENTION",
        "RLK_OP_DPO_FRAGMENT",
        "RLK_OP_EMBEDDING",
        "RLK_OP_GRPO_FRAGMENT",
        "RLK_OP_LM_HEAD",
        "RLK_OP_LOGP",
        "RLK_OP_MATMUL_PROJECTION",
        "RLK_OP_PPO_FRAGMENT",
        "RLK_OP_RATIO_KL",
        "RLK_OP_RMSNORM",
        "RLK_OP_ROPE",
        "RLK_OP_SWIGLU",
        "BatchInvarianceCase",
        "ForwardChainComparisonResult",
        "ForwardChainStep",
        "OperatorComparisonResult",
        "OperatorComparisonSpec",
        "OperatorPair",
        "OperatorTolerance",
        "StrictBackendAdmissionReport",
        "build_single_card_batch_invariance_cases",
        "build_strict_backend_admission_report",
        "compare_batch_invariance",
        "compare_operator_outputs",
        "compare_operator_pair",
        "get_operator_comparison_spec",
        "iter_operator_comparison_specs",
        "reference_attention",
        "reference_embedding",
        "reference_linear_logp",
        "reference_lm_head",
        "reference_matmul_projection",
        "reference_ppo_fragment",
        "reference_ratio_kl",
        "reference_rmsnorm",
        "reference_rope",
        "reference_selected_logprobs",
        "reference_swiglu",
        "run_deterministic_repeatability_check",
        "run_forward_chain_comparison",
        "run_reference_operator",
    }
)


class RlkOperatorComparisonUnavailable(RuntimeError):
    """Raised when RL-Kernel's operator comparison standard cannot be imported."""


def load_operator_comparison_module() -> Any:
    try:
        return importlib.import_module(_RLK_OPERATOR_COMPARISON_MODULE)
    except Exception as exc:
        raise RlkOperatorComparisonUnavailable(f"RL-Kernel operator comparison standard is unavailable; install an RL-Kernel build that exposes {_RLK_OPERATOR_COMPARISON_MODULE!r}.") from exc


def __getattr__(name: str) -> Any:
    if name not in _OPERATOR_COMPARISON_EXPORTS:
        raise AttributeError(name)
    return getattr(load_operator_comparison_module(), name)


def __dir__() -> list[str]:
    return sorted((*globals(), *_OPERATOR_COMPARISON_EXPORTS))


__all__ = [
    "RlkOperatorComparisonUnavailable",
    "load_operator_comparison_module",
    *_OPERATOR_COMPARISON_EXPORTS,
]
