from argparse import Namespace

import pytest

from vime.utils.rl_kernel import is_rl_kernel_op_enabled, normalize_rl_kernel_args, parse_rl_kernel_ops

NUM_GPUS = 0


@pytest.mark.unit
def test_parse_rl_kernel_ops_defaults_to_logp():
    assert parse_rl_kernel_ops(None) == ("logp",)
    assert parse_rl_kernel_ops("") == ("logp",)


@pytest.mark.unit
def test_parse_rl_kernel_ops_deduplicates_comma_and_space_separated_values():
    assert parse_rl_kernel_ops("logp, ratio_kl logp") == ("logp", "ratio_kl")


@pytest.mark.unit
def test_parse_rl_kernel_ops_rejects_unknown_ops():
    with pytest.raises(ValueError, match="Unsupported RL-Kernel op"):
        parse_rl_kernel_ops("logp,moe")


@pytest.mark.unit
def test_normalize_rl_kernel_args_keeps_default_disabled():
    args = Namespace(enable_rl_kernel=False, rl_kernel_ops="logp", rl_kernel_strict=False)

    normalize_rl_kernel_args(args)

    assert args.enable_rl_kernel is False
    assert args.rl_kernel_ops == ("logp",)
    assert is_rl_kernel_op_enabled(args, "logp") is False


@pytest.mark.unit
def test_normalize_rl_kernel_args_accepts_env_enable(monkeypatch):
    monkeypatch.setenv("VIME_RL_KERNEL", "1")
    args = Namespace(enable_rl_kernel=False, rl_kernel_ops="logp", rl_kernel_strict=False)

    normalize_rl_kernel_args(args)

    assert args.enable_rl_kernel is True
    assert args.rl_kernel_ops == ("logp",)
    assert is_rl_kernel_op_enabled(args, "logp") is True


@pytest.mark.unit
def test_normalize_rl_kernel_args_rejects_future_ops_for_current_integration():
    args = Namespace(enable_rl_kernel=True, rl_kernel_ops="logp,ratio_kl", rl_kernel_strict=False)

    with pytest.raises(ValueError, match="currently supports only logp"):
        normalize_rl_kernel_args(args)


@pytest.mark.unit
def test_normalize_rl_kernel_args_rejects_bad_env_bool(monkeypatch):
    monkeypatch.setenv("VIME_RL_KERNEL", "maybe")
    args = Namespace(enable_rl_kernel=False, rl_kernel_ops="logp", rl_kernel_strict=False)

    with pytest.raises(ValueError, match="VIME_RL_KERNEL"):
        normalize_rl_kernel_args(args)
