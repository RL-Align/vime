import os
from argparse import Namespace
from collections.abc import Iterable


RL_KERNEL_SUPPORTED_OPS = ("linear_logp",)
RL_KERNEL_INTEGRATED_OPS = ("linear_logp",)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def parse_rl_kernel_ops(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse a comma/space separated RL-Kernel op list."""
    if value is None:
        return ("linear_logp",)

    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).replace(",", " ").split())

    ops: list[str] = []
    for item in raw_items:
        op = item.strip().lower()
        if not op:
            continue
        if op not in RL_KERNEL_SUPPORTED_OPS:
            supported = ", ".join(RL_KERNEL_SUPPORTED_OPS)
            raise ValueError(f"Unsupported RL-Kernel op '{op}'. Supported ops: {supported}.")
        if op not in ops:
            ops.append(op)

    return tuple(ops) if ops else ("linear_logp",)


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of: 1/0, true/false, yes/no, on/off.")


def normalize_rl_kernel_args(args: Namespace) -> Namespace:
    """Apply environment overrides and validate RL-Kernel integration args."""
    env_enabled = _env_bool("VIME_RL_KERNEL")
    if env_enabled is not None:
        args.enable_rl_kernel = env_enabled

    env_strict = _env_bool("VIME_RL_KERNEL_STRICT")
    if env_strict is not None:
        args.rl_kernel_strict = env_strict

    env_ops = os.getenv("VIME_RL_KERNEL_OPS")
    if env_ops is not None:
        args.rl_kernel_ops = env_ops

    args.rl_kernel_ops = parse_rl_kernel_ops(getattr(args, "rl_kernel_ops", None))

    if getattr(args, "enable_rl_kernel", False):
        unsupported = [op for op in args.rl_kernel_ops if op not in RL_KERNEL_INTEGRATED_OPS]
        if unsupported:
            integrated = ", ".join(RL_KERNEL_INTEGRATED_OPS)
            raise ValueError(
                "This vime RL-Kernel integration currently supports only "
                f"{integrated}. Requested future ops: {', '.join(unsupported)}."
            )

    return args


def is_rl_kernel_op_enabled(args: Namespace, op: str) -> bool:
    return getattr(args, "enable_rl_kernel", False) and op in getattr(args, "rl_kernel_ops", ())
