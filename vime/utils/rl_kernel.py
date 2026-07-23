from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable


def _normalize_ops(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).replace(",", " ").split())
    return tuple(dict.fromkeys(op.strip() for op in raw_items if op.strip()))


def _resolved_fast_mode(args: Namespace) -> str:
    config = getattr(args, "rlk_mode_config", None)
    if config is not None:
        return str(getattr(config, "fast", "off")).lower()
    if getattr(args, "rlk_fast", None) is not None:
        return str(args.rlk_fast).lower()
    if getattr(args, "rl_kernel_strict", False):
        return "strict"
    if getattr(args, "enable_rl_kernel", False):
        return "auto"
    return "off"


def _resolved_ops(args: Namespace) -> tuple[str, ...]:
    config = getattr(args, "rlk_mode_config", None)
    if config is not None:
        return _normalize_ops(getattr(config, "ops", ()))
    return _normalize_ops(getattr(args, "rl_kernel_ops", ()))


def is_rl_kernel_requested(args: Namespace) -> bool:
    return _resolved_fast_mode(args) != "off"


def is_rl_kernel_strict(args: Namespace) -> bool:
    return _resolved_fast_mode(args) == "strict"


def is_rl_kernel_op_enabled(args: Namespace, op: str) -> bool:
    if not is_rl_kernel_requested(args):
        return False
    ops = _resolved_ops(args)
    return "*" in ops or op in ops
