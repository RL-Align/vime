from __future__ import annotations

import builtins
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

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


class _FakeLinearLogpOp:
    calls: list[dict] = []

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        type(self).calls.append(
            {
                "hidden_shape": tuple(hidden.shape),
                "weight_shape": tuple(weight.shape),
                "target_shape": tuple(target_ids.shape),
                "bias": bias is not None,
                "kwargs": kwargs,
            }
        )
        logits = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
        return torch.gather(torch.log_softmax(logits, dim=-1), -1, target_ids.long().unsqueeze(-1)).squeeze(-1)


class _FakeLegacyLinearLogpOp:
    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
        return torch.gather(torch.log_softmax(logits, dim=-1), -1, target_ids.long().unsqueeze(-1)).squeeze(-1)


def _reset_rl_kernel_state():
    rlk_mod._LOGP_OP = None
    rlk_mod._LOGP_OP_LOAD_ERROR = None
    rlk_mod._LINEAR_LOGP_OP = None
    rlk_mod._LINEAR_LOGP_OP_LOAD_ERROR = None
    rlk_mod._WARNED_FALLBACK_REASONS.clear()
    rlk_mod._FALLBACK_COUNTS.clear()
    rlk_mod._FALLBACK_COUNTS.update({"logp": 0, "linear_logp": 0})
    _FakeLinearLogpOp.calls.clear()


def _install_fake_rl_engine(monkeypatch):
    rl_engine = types.ModuleType("rl_engine")
    kernels = types.ModuleType("rl_engine.kernels")
    registry = types.ModuleType("rl_engine.kernels.registry")
    registry.kernel_registry = types.SimpleNamespace(get_op=lambda name: _FakeLinearLogpOp())
    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry)


def _install_fake_legacy_rl_engine(monkeypatch):
    rl_engine = types.ModuleType("rl_engine")
    kernels = types.ModuleType("rl_engine.kernels")
    registry = types.ModuleType("rl_engine.kernels.registry")
    registry.kernel_registry = types.SimpleNamespace(get_op=lambda name: _FakeLegacyLinearLogpOp())
    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry)


def _make_args(**overrides) -> Namespace:
    values = {
        "enable_rl_kernel": True,
        "rl_kernel_ops": ("linear_logp",),
        "rl_kernel_strict": False,
        "allgather_cp": False,
        "qkv_format": "thd",
        "rollout_temperature": 1.0,
        "log_probs_chunk_size": -1,
        "entropy_coef": 0.0,
        "sequence_parallel": False,
        "padded_vocab_size": None,
        "vocab_size": None,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture(autouse=True)
def reset_parallelism():
    _reset_rl_kernel_state()
    mpu.get_tensor_model_parallel_world_size.return_value = 1
    mpu.get_tensor_model_parallel_rank.return_value = 0
    mpu.get_tensor_model_parallel_group.return_value = None
    mpu.get_context_parallel_world_size.return_value = 1
    mpu.get_context_parallel_rank.return_value = 0
    mpu.get_virtual_pipeline_model_parallel_world_size.return_value = None
    mpu.is_pipeline_last_stage.return_value = True
    yield
    _reset_rl_kernel_state()


def _reference_logp(hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, bias: torch.Tensor | None):
    logits = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
    return torch.gather(torch.log_softmax(logits, dim=-1), -1, target.long().unsqueeze(-1)).squeeze(-1)


def _cpu_calculate_log_probs_and_entropy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group,
    *,
    with_entropy: bool,
    chunk_size: int,
):
    del tp_group, chunk_size
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    selected = torch.gather(log_probs, -1, tokens.long().unsqueeze(-1)).squeeze(-1)
    entropy = None
    if with_entropy:
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
    return selected, entropy


@pytest.mark.unit
def test_maybe_compute_linear_logp_passes_tensor_parallel_metadata(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args()
    torch.manual_seed(1)
    hidden = torch.randn(6, 5)
    weight = torch.randn(8, 5)
    bias = torch.randn(8)
    target = torch.randint(0, 8, (6,))
    context = rlk_mod.LinearLogpContext(
        lm_head_weight=weight,
        bias=bias,
        tp_group="tp",
        vocab_start_index=16,
        global_vocab_size=32,
    )

    actual = rlk_mod.maybe_compute_linear_logp(hidden, target, context=context, args=args, with_entropy=False)

    torch.testing.assert_close(actual, _reference_logp(hidden, weight, target, bias))
    assert _FakeLinearLogpOp.calls == [
        {
            "hidden_shape": (6, 5),
            "weight_shape": (8, 5),
            "target_shape": (6,),
            "bias": True,
            "kwargs": {
                "tp_group": "tp",
                "vocab_start_index": 16,
                "global_vocab_size": 32,
            },
        }
    ]


@pytest.mark.unit
def test_linear_logp_matches_vime_response_slicing_from_hidden_states(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args()
    vocab_size = 23
    hidden_size = 7
    total_lengths = [5, 4, 6]
    response_lengths = [2, 3, 4]
    torch.manual_seed(2)
    unconcat_tokens = [torch.randint(0, vocab_size, (length,), dtype=torch.long) for length in total_lengths]
    hidden = torch.randn(sum(total_lengths), 1, hidden_size)
    weight = torch.randn(vocab_size, hidden_size)
    bias = torch.randn(vocab_size)
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=bias, tp_group=None)

    _, result = loss_mod.get_log_probs_and_entropy(
        hidden,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        rl_kernel_linear_logp_context=context,
    )

    full_tokens = torch.zeros(sum(total_lengths), dtype=torch.long)
    offset = 0
    for tokens, total_length in zip(unconcat_tokens, total_lengths, strict=False):
        full_tokens[offset : offset + total_length - 1] = tokens[1:total_length]
        offset += total_length
    full_logp = _reference_logp(hidden.squeeze(1), weight, full_tokens, bias)

    expected = []
    offset = 0
    for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
        end = offset + total_length
        start = end - response_length
        expected.append(full_logp[start - 1 : end - 1])
        offset += total_length

    assert len(_FakeLinearLogpOp.calls) == 1
    for actual, expected_item in zip(result["log_probs"], expected, strict=True):
        torch.testing.assert_close(actual, expected_item, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_linear_logp_materializes_logits_fallback_when_optional_package_missing(monkeypatch):
    monkeypatch.setattr(loss_mod, "calculate_log_probs_and_entropy", _cpu_calculate_log_probs_and_entropy)
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
    vocab_size = 19
    hidden_size = 5
    total_lengths = [4, 5]
    response_lengths = [2, 3]
    torch.manual_seed(3)
    unconcat_tokens = [torch.randint(0, vocab_size, (length,), dtype=torch.long) for length in total_lengths]
    hidden = torch.randn(1, sum(total_lengths), hidden_size)
    weight = torch.randn(vocab_size, hidden_size)
    bias = torch.randn(vocab_size)
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=bias, tp_group=None)

    _, result = loss_mod.get_log_probs_and_entropy(
        hidden,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        rl_kernel_linear_logp_context=context,
    )

    logits = F.linear(hidden.squeeze(0).float(), weight.float(), bias.float())
    expected = []
    offset = 0
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths, strict=False):
        end = offset + total_length
        start = end - response_length
        target = tokens[-response_length:]
        expected.append(
            torch.gather(torch.log_softmax(logits[start - 1 : end - 1], dim=-1), -1, target.unsqueeze(-1)).squeeze(-1)
        )
        offset += total_length

    assert rlk_mod.get_rl_kernel_fallback_count("linear_logp") == 1
    for actual, expected_item in zip(result["log_probs"], expected, strict=True):
        torch.testing.assert_close(actual, expected_item, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_linear_logp_falls_back_when_op_lacks_tp_interface(monkeypatch):
    _install_fake_legacy_rl_engine(monkeypatch)
    args = _make_args()
    hidden = torch.randn(3, 4)
    weight = torch.randn(6, 4)
    target = torch.randint(0, 6, (3,))
    context = rlk_mod.LinearLogpContext(
        lm_head_weight=weight,
        bias=None,
        tp_group="tp_group",
        vocab_start_index=6,
        global_vocab_size=12,
    )

    actual = rlk_mod.maybe_compute_linear_logp(hidden, target, context=context, args=args, with_entropy=False)

    assert actual is None
    assert rlk_mod.get_rl_kernel_fallback_count("linear_logp") == 1


@pytest.mark.unit
def test_linear_logp_context_from_model_uses_tp_vocab_offsets():
    mpu.get_tensor_model_parallel_world_size.return_value = 4
    mpu.get_tensor_model_parallel_rank.return_value = 2
    mpu.get_tensor_model_parallel_group.return_value = "tp_group"

    output_layer = types.SimpleNamespace(
        weight=torch.empty(8, 4),
        bias=None,
        sequence_parallel=True,
    )
    model = types.SimpleNamespace(output_layer=output_layer, post_process=True)
    args = _make_args(padded_vocab_size=32, sequence_parallel=False)

    context = rlk_mod.get_linear_logp_context_from_model(args, model)

    assert context is not None
    assert context.lm_head_weight is output_layer.weight
    assert context.tp_group == "tp_group"
    assert context.vocab_start_index == 16
    assert context.global_vocab_size == 32
    assert context.sequence_parallel is True


@pytest.mark.unit
def test_linear_logp_context_uses_covered_padded_vocab_when_padded_vocab_size_missing():
    mpu.get_tensor_model_parallel_world_size.return_value = 4
    mpu.get_tensor_model_parallel_rank.return_value = 1
    mpu.get_tensor_model_parallel_group.return_value = "tp_group"

    output_layer = types.SimpleNamespace(weight=torch.empty(8, 4), bias=None)
    model = types.SimpleNamespace(output_layer=output_layer, post_process=True)
    args = _make_args(padded_vocab_size=None, vocab_size=30)

    context = rlk_mod.get_linear_logp_context_from_model(args, model)

    assert context is not None
    assert context.vocab_start_index == 8
    assert context.global_vocab_size == 32


@pytest.mark.unit
def test_return_hidden_states_for_linear_logp_restores_post_process_flag():
    args = _make_args()
    model = types.SimpleNamespace(post_process=True)
    context = rlk_mod.LinearLogpContext(
        lm_head_weight=torch.empty(4, 3),
        bias=None,
        tp_group=None,
    )

    with rlk_mod.return_hidden_states_for_linear_logp(args, model, context) as enabled:
        assert enabled is True
        assert model.post_process is False

    assert model.post_process is True


@pytest.mark.unit
def test_policy_loss_only_skips_entropy_when_linear_logp_context_is_active():
    args = _make_args(enable_rl_kernel=True, rl_kernel_ops=("linear_logp",), entropy_coef=0.0)
    context = rlk_mod.LinearLogpContext(
        lm_head_weight=torch.empty(4, 3),
        bias=None,
        tp_group=None,
    )

    assert loss_mod._policy_loss_needs_entropy(args, None) is True
    assert loss_mod._policy_loss_needs_entropy(args, context) is False

    args.entropy_coef = 0.01
    assert loss_mod._policy_loss_needs_entropy(args, context) is True
