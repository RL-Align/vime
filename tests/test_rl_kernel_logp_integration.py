from __future__ import annotations

import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

_unit_stubs.install_megatron_mpu_stub()
_unit_stubs.install_vime_distributed_utils_stub()

from megatron.core import mpu  # noqa: E402

from vime.backends.megatron_utils import loss as loss_mod  # noqa: E402
from vime.backends.megatron_utils import rl_kernel as rlk_mod  # noqa: E402


NUM_GPUS = 0


class _FakeLogpOp:
    calls = 0

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        type(self).calls += 1
        return torch.gather(torch.log_softmax(logits.float(), dim=-1), -1, token_ids.long().unsqueeze(-1)).squeeze(-1)


def _reset_rl_kernel_state():
    rlk_mod._LOGP_OP = None
    rlk_mod._LOGP_OP_LOAD_ERROR = None
    rlk_mod._WARNED_FALLBACK_REASONS.clear()
    _FakeLogpOp.calls = 0


def _install_fake_rl_engine(monkeypatch):
    rl_engine = types.ModuleType("rl_engine")
    kernels = types.ModuleType("rl_engine.kernels")
    registry = types.ModuleType("rl_engine.kernels.registry")
    registry.kernel_registry = types.SimpleNamespace(get_op=lambda name: _FakeLogpOp())
    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry)


def _make_args(**overrides) -> Namespace:
    values = {
        "enable_rl_kernel": True,
        "rl_kernel_ops": ("logp",),
        "rl_kernel_strict": False,
        "allgather_cp": False,
        "qkv_format": "thd",
        "rollout_temperature": 1.0,
        "log_probs_chunk_size": -1,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture(autouse=True)
def reset_parallelism():
    _reset_rl_kernel_state()
    mpu.get_tensor_model_parallel_world_size.return_value = 1
    mpu.get_tensor_model_parallel_group.return_value = None
    mpu.get_context_parallel_world_size.return_value = 1
    mpu.get_context_parallel_rank.return_value = 0
    yield
    _reset_rl_kernel_state()


@pytest.mark.unit
def test_rl_kernel_logp_matches_vime_response_slicing(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args()
    vocab_size = 257
    total_lengths = [9, 7, 11]
    response_lengths = [4, 3, 6]
    torch.manual_seed(17)
    unconcat_tokens = [torch.randint(0, vocab_size, (length,), dtype=torch.long) for length in total_lengths]
    logits = torch.randn(1, sum(total_lengths), vocab_size, dtype=torch.float32)

    with torch.no_grad():
        _, result = loss_mod.get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=False,
        )

    expected = []
    offset = 0
    log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths, strict=False):
        start = offset + total_length - response_length
        end = offset + total_length
        shifted_tokens = tokens[-response_length:]
        expected.append(torch.gather(log_probs[start - 1 : end - 1], -1, shifted_tokens.unsqueeze(-1)).squeeze(-1))
        offset += total_length

    assert _FakeLogpOp.calls == 1
    assert len(result["log_probs"]) == len(expected)
    for actual, expected_item in zip(result["log_probs"], expected, strict=True):
        torch.testing.assert_close(actual, expected_item, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_rl_kernel_logp_falls_back_when_entropy_requested(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args()
    logits = torch.randn(8, 16, dtype=torch.float32)
    tokens = torch.randint(0, 16, (8,), dtype=torch.long)

    with torch.no_grad():
        actual = rlk_mod.maybe_compute_logp(logits, tokens, args=args, with_entropy=True)

    assert actual is None
    assert _FakeLogpOp.calls == 0


@pytest.mark.unit
def test_rl_kernel_logp_falls_back_for_tensor_parallel_vocab(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    mpu.get_tensor_model_parallel_world_size.return_value = 2
    args = _make_args()
    logits = torch.randn(8, 16, dtype=torch.float32)
    tokens = torch.randint(0, 16, (8,), dtype=torch.long)

    with torch.no_grad():
        actual = rlk_mod.maybe_compute_logp(logits, tokens, args=args, with_entropy=False)

    assert actual is None
    assert _FakeLogpOp.calls == 0


@pytest.mark.unit
def test_rl_kernel_logp_strict_mode_raises_on_unsupported_parallelism(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    mpu.get_tensor_model_parallel_world_size.return_value = 2
    args = _make_args(rl_kernel_strict=True)
    logits = torch.randn(8, 16, dtype=torch.float32)
    tokens = torch.randint(0, 16, (8,), dtype=torch.long)

    with pytest.raises(RuntimeError, match="tensor-parallel vocab shards"):
        with torch.no_grad():
            rlk_mod.maybe_compute_logp(logits, tokens, args=args, with_entropy=False)


@pytest.mark.unit
def test_rl_kernel_logp_falls_back_when_optional_package_missing(monkeypatch):
    import builtins

    for name in list(sys.modules):
        if name == "rl_engine" or name.startswith("rl_engine."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    original_import = builtins.__import__

    def fail_rl_engine_import(name, *args, **kwargs):
        if name == "rl_engine" or name.startswith("rl_engine."):
            raise ModuleNotFoundError("No module named 'rl_engine'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_rl_engine_import)
    args = _make_args()
    logits = torch.randn(8, 16, dtype=torch.float32)
    tokens = torch.randint(0, 16, (8,), dtype=torch.long)

    with torch.no_grad():
        actual = rlk_mod.maybe_compute_logp(logits, tokens, args=args, with_entropy=False)

    assert actual is None
