from argparse import Namespace

import pytest

from vime.utils.rl_kernel import is_rl_kernel_op_enabled, normalize_rl_kernel_args, parse_rl_kernel_ops

NUM_GPUS = 0


@pytest.mark.unit
def test_parse_rl_kernel_ops_defaults_to_linear_logp():
    assert parse_rl_kernel_ops(None) == ("linear_logp",)
    assert parse_rl_kernel_ops("") == ("linear_logp",)


@pytest.mark.unit
def test_parse_rl_kernel_ops_deduplicates_comma_and_space_separated_values():
    assert parse_rl_kernel_ops("linear_logp, linear_logp") == ("linear_logp",)


@pytest.mark.unit
def test_parse_rl_kernel_ops_rejects_unknown_ops():
    with pytest.raises(ValueError, match="Unsupported RL-Kernel op"):
        parse_rl_kernel_ops("linear_logp,moe")


@pytest.mark.unit
def test_normalize_rl_kernel_args_keeps_default_disabled():
    args = Namespace(enable_rl_kernel=False, rl_kernel_ops="linear_logp", rl_kernel_strict=False)

    normalize_rl_kernel_args(args)

    assert args.enable_rl_kernel is False
    assert args.rl_kernel_ops == ("linear_logp",)
    assert is_rl_kernel_op_enabled(args, "linear_logp") is False


@pytest.mark.unit
def test_normalize_rl_kernel_args_accepts_env_enable(monkeypatch):
    monkeypatch.setenv("VIME_RL_KERNEL", "1")
    args = Namespace(enable_rl_kernel=False, rl_kernel_ops="linear_logp", rl_kernel_strict=False)

    normalize_rl_kernel_args(args)

    assert args.enable_rl_kernel is True
    assert args.rl_kernel_ops == ("linear_logp",)
    assert is_rl_kernel_op_enabled(args, "linear_logp") is True


@pytest.mark.unit
def test_normalize_rl_kernel_args_rejects_non_linear_logp_ops():
    args = Namespace(enable_rl_kernel=True, rl_kernel_ops="logp", rl_kernel_strict=False)

    with pytest.raises(ValueError, match="Unsupported RL-Kernel op"):
        normalize_rl_kernel_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("op", ["ratio_kl", "grpo_loss", "sampling"])
def test_normalize_rl_kernel_args_rejects_out_of_scope_ops(op):
    args = Namespace(enable_rl_kernel=True, rl_kernel_ops=op, rl_kernel_strict=False)

    with pytest.raises(ValueError, match="Unsupported RL-Kernel op"):
        normalize_rl_kernel_args(args)


@pytest.mark.unit
def test_normalize_rl_kernel_args_accepts_linear_logp():
    args = Namespace(enable_rl_kernel=True, rl_kernel_ops="linear_logp", rl_kernel_strict=False)

    normalize_rl_kernel_args(args)

    assert args.rl_kernel_ops == ("linear_logp",)
    assert is_rl_kernel_op_enabled(args, "linear_logp") is True


@pytest.mark.unit
def test_normalize_rl_kernel_args_rejects_bad_env_bool(monkeypatch):
    monkeypatch.setenv("VIME_RL_KERNEL", "maybe")
    args = Namespace(enable_rl_kernel=False, rl_kernel_ops="linear_logp", rl_kernel_strict=False)

    with pytest.raises(ValueError, match="VIME_RL_KERNEL"):
        normalize_rl_kernel_args(args)
