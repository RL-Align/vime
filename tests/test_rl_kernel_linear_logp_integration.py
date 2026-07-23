from __future__ import annotations

import importlib
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

adapter_mod = importlib.import_module("vime.backends.rl_kernel_utils.adapter")


class _FakeLinearLogpOp:
    backend_id = "rlk.linear_logp.fake"
    contract_id = "rlk.linear_logp.fake.fp32"
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
                "hidden_requires_grad": hidden.requires_grad,
                "hidden_dtype": hidden.dtype,
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


class _FakeDetachedLinearLogpOp(_FakeLinearLogpOp):
    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return super().__call__(hidden, weight, target_ids, bias, **kwargs).detach()


def _drop_rl_engine_modules() -> None:
    for name in list(sys.modules):
        if name == "rl_engine" or name.startswith("rl_engine."):
            sys.modules.pop(name, None)


def _reset_rl_kernel_state() -> None:
    rlk_mod._LINEAR_LOGP_ADAPTER = None
    rlk_mod._LINEAR_LOGP_ADAPTER_ERROR = None
    rlk_mod._LINEAR_LOGP_SAVE_PROBS_CAST_LOGGED = False
    rlk_mod._WARNED_FALLBACK_REASONS.clear()
    rlk_mod._FALLBACK_COUNTS.clear()
    rlk_mod._FALLBACK_COUNTS.update({"linear_logp": 0})
    rlk_mod.reset_rl_kernel_runtime_counters()
    _FakeLinearLogpOp.calls.clear()
    _drop_rl_engine_modules()


def _install_fake_rl_engine(monkeypatch, op_factory=lambda: _FakeLinearLogpOp()) -> None:
    rl_engine = types.ModuleType("rl_engine")
    kernels = types.ModuleType("rl_engine.kernels")
    registry = types.ModuleType("rl_engine.kernels.registry")
    registry.kernel_registry = types.SimpleNamespace(get_op=lambda name: op_factory())
    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry)


def _make_args(**overrides) -> Namespace:
    values = {
        "rlk_fast": "auto",
        "rlk_consistency": "off",
        "rlk_mode_config": types.SimpleNamespace(fast="auto", consistency="off", ops=("linear_logp",)),
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
        "only_train_params_name_list": (),
    }
    values.update(overrides)
    if "rlk_mode_config" not in overrides:
        values["rlk_mode_config"] = types.SimpleNamespace(
            fast=values["rlk_fast"],
            consistency=values["rlk_consistency"],
            ops=values["rl_kernel_ops"],
        )
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


def _reference_logp(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    logits = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
    return torch.gather(torch.log_softmax(logits, dim=-1), -1, target.long().unsqueeze(-1)).squeeze(-1)


def _cpu_calculate_log_probs_and_entropy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group,
    *,
    with_entropy: bool,
    chunk_size: int,
    log_prob_keep_mask=None,
    with_entropy_grad: bool = True,
):
    del tp_group, chunk_size, with_entropy_grad
    masked_logits = logits.float()
    if log_prob_keep_mask is not None:
        masked_logits = masked_logits.masked_fill(~log_prob_keep_mask, float("-inf"))
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    selected = torch.gather(log_probs, -1, tokens.long().unsqueeze(-1)).squeeze(-1)
    entropy = None
    if with_entropy:
        native_log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * native_log_probs).sum(dim=-1)
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
            "hidden_requires_grad": False,
            "hidden_dtype": hidden.dtype,
        }
    ]
    counters = rlk_mod.get_rl_kernel_runtime_counters()
    assert counters["linear_logp_call_count"] == 1.0
    assert counters["linear_logp_token_count"] == 6.0
    assert counters["linear_logp_dispatch_elapsed_s"] >= 0.0
    assert counters["linear_logp_fallback_count"] == 0.0

    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["requested_backend"] == "registry"
    assert metadata["actual_backend"] == "_FakeLinearLogpOp"
    assert metadata["backend_id"] == "rlk.linear_logp.fake"
    assert metadata["contract_id"] == "rlk.linear_logp.fake.fp32"
    assert metadata["fallback"] is False
    assert metadata["fallback_reason"] is None

    log_metrics = rlk_mod.get_linear_logp_runtime_log_metrics(prefix="x/")
    assert log_metrics["x/fallback"] == 0.0
    assert log_metrics["x/backend_descriptor_id"] == metadata["backend_descriptor_id"]
    assert log_metrics["x/contract_descriptor_id"] == metadata["contract_descriptor_id"]
    assert log_metrics["x/fallback_reason_descriptor_id"] == metadata["fallback_reason_descriptor_id"]
    assert all("memory_" not in key for key in log_metrics)


@pytest.mark.unit
def test_linear_logp_full_gradient_path_matches_materialized_logits(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args()
    torch.manual_seed(11)
    hidden = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    bias = torch.randn(7, requires_grad=True)
    target = torch.randint(0, 7, (5,))
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=bias, tp_group=None)

    actual = rlk_mod.maybe_compute_linear_logp(hidden, target, context=context, args=args, with_entropy=False)
    actual.sum().backward()
    actual_grads = (hidden.grad.clone(), weight.grad.clone(), bias.grad.clone())

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    expected = _reference_logp(hidden_ref, weight_ref, target, bias_ref)
    expected.sum().backward()

    torch.testing.assert_close(actual.detach(), expected.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_grads[0], hidden_ref.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_grads[1], weight_ref.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_grads[2], bias_ref.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_linear_logp_strict_fast_success_records_no_fallback(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args(rlk_fast="strict")
    torch.manual_seed(12)
    hidden = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(6, 3, requires_grad=True)
    target = torch.randint(0, 6, (4,))
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=None, tp_group=None)

    actual = rlk_mod.maybe_compute_linear_logp(
        hidden,
        target,
        context=context,
        args=args,
        with_entropy=False,
        loss_masks=[torch.tensor([1, 1]), torch.tensor([1, 0])],
        response_lengths=[2, 2],
    )

    torch.testing.assert_close(actual, _reference_logp(hidden, weight, target, None))
    assert rlk_mod.get_rl_kernel_fallback_count("linear_logp") == 0
    counters = rlk_mod.get_rl_kernel_runtime_counters()
    assert counters["linear_logp_call_count"] == 1.0
    assert counters["linear_logp_fallback_count"] == 0.0
    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["actual_backend"] == "_FakeLinearLogpOp"
    assert metadata["fallback"] is False
    assert metadata["fallback_reason"] is None
    assert metadata["fallback_reason_code"] is None
    assert metadata["strict_failure"] is False
    assert rlk_mod.get_linear_logp_runtime_log_metrics(prefix="x/")["x/strict_failure"] == 0.0


@pytest.mark.unit
def test_linear_logp_auto_fallback_has_structured_temperature_reason(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args(rollout_temperature=0.7)
    hidden = torch.randn(3, 4)
    weight = torch.randn(6, 4)
    target = torch.randint(0, 6, (3,))
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=None, tp_group=None)

    actual = rlk_mod.maybe_compute_linear_logp(hidden, target, context=context, args=args, with_entropy=False)

    assert actual is None
    assert _FakeLinearLogpOp.calls == []
    assert rlk_mod.get_rl_kernel_fallback_count("linear_logp") == 1
    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["actual_backend"] == "vime.native.linear_logp"
    assert metadata["fallback"] is True
    assert metadata["fallback_reason_code"] == "unsupported_temperature"
    assert metadata["strict_failure"] is False


@pytest.mark.unit
def test_linear_logp_strict_fails_when_temperature_would_fallback(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args(rlk_fast="strict", rollout_temperature=0.7)
    hidden = torch.randn(3, 4)
    weight = torch.randn(6, 4)
    target = torch.randint(0, 6, (3,))
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=None, tp_group=None)

    with pytest.raises(RuntimeError, match="rollout_temperature=1.0"):
        rlk_mod.maybe_compute_linear_logp(hidden, target, context=context, args=args, with_entropy=False)

    assert _FakeLinearLogpOp.calls == []
    assert rlk_mod.get_rl_kernel_fallback_count("linear_logp") == 0
    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["actual_backend"] is None
    assert metadata["fallback"] is False
    assert metadata["fallback_reason_code"] == "unsupported_temperature"
    assert metadata["strict_failure"] is True


@pytest.mark.unit
def test_linear_logp_strict_requires_active_mask_metadata(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args(rlk_fast="strict")
    hidden = torch.randn(2, 4)
    weight = torch.randn(5, 4)
    target = torch.randint(0, 5, (2,))
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=None, tp_group=None)

    with pytest.raises(RuntimeError, match="active response loss masks"):
        rlk_mod.maybe_compute_linear_logp(hidden, target, context=context, args=args, with_entropy=False)

    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["fallback_reason_code"] == "active_mask_missing"
    assert metadata["strict_failure"] is True


@pytest.mark.unit
def test_linear_logp_strict_rejects_invalid_active_mask(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args(rlk_fast="strict")
    hidden = torch.randn(2, 4)
    weight = torch.randn(5, 4)
    target = torch.randint(0, 5, (2,))
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=None, tp_group=None)

    with pytest.raises(RuntimeError, match="0/1"):
        rlk_mod.maybe_compute_linear_logp(
            hidden,
            target,
            context=context,
            args=args,
            with_entropy=False,
            loss_masks=[torch.tensor([1, 2])],
            response_lengths=[2],
        )

    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["fallback_reason_code"] == "active_mask_not_binary"
    assert metadata["strict_failure"] is True


@pytest.mark.unit
def test_linear_logp_strict_requires_tensor_parallel_metadata(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    mpu.get_tensor_model_parallel_world_size.return_value = 2
    mpu.get_tensor_model_parallel_group.return_value = None
    args = _make_args(rlk_fast="strict")
    hidden = torch.randn(2, 4)
    weight = torch.randn(5, 4)
    target = torch.randint(0, 10, (2,))
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=None, tp_group=None, global_vocab_size=10)

    with pytest.raises(RuntimeError, match="tensor-parallel group metadata"):
        rlk_mod.maybe_compute_linear_logp(
            hidden,
            target,
            context=context,
            args=args,
            with_entropy=False,
            loss_masks=[torch.tensor([1, 1])],
            response_lengths=[2],
        )

    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["fallback_reason_code"] == "tp_group_missing"
    assert metadata["strict_failure"] is True


@pytest.mark.unit
def test_linear_logp_strict_rejects_missing_backward_saved_state(monkeypatch):
    _install_fake_rl_engine(monkeypatch, op_factory=lambda: _FakeDetachedLinearLogpOp())
    args = _make_args(rlk_fast="strict")
    hidden = torch.randn(2, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    target = torch.randint(0, 5, (2,))
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=None, tp_group=None)

    with pytest.raises(RuntimeError, match="autograd-connected"):
        rlk_mod.maybe_compute_linear_logp(
            hidden,
            target,
            context=context,
            args=args,
            with_entropy=False,
            loss_masks=[torch.tensor([1, 1])],
            response_lengths=[2],
        )

    assert len(_FakeLinearLogpOp.calls) == 1
    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["fallback_reason_code"] == "backward_saved_state_missing"
    assert metadata["fallback"] is False
    assert metadata["strict_failure"] is True


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
    for actual_item, expected_item in zip(result["log_probs"], expected, strict=True):
        torch.testing.assert_close(actual_item, expected_item, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_linear_logp_materializes_logits_fallback_when_optional_package_missing(monkeypatch):
    monkeypatch.setattr(loss_mod, "calculate_log_probs_and_entropy", _cpu_calculate_log_probs_and_entropy)
    original_import_module = adapter_mod.importlib.import_module

    def fail_rl_engine_import(name, *args, **kwargs):
        if name == "rl_engine.kernels.registry":
            raise ModuleNotFoundError("No module named 'rl_engine'")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(adapter_mod.importlib, "import_module", fail_rl_engine_import)
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
        expected.append(torch.gather(torch.log_softmax(logits[start - 1 : end - 1], dim=-1), -1, target.unsqueeze(-1)).squeeze(-1))
        offset += total_length

    assert rlk_mod.get_rl_kernel_fallback_count("linear_logp") == 1
    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["actual_backend"] == "vime.native.linear_logp"
    assert metadata["fallback"] is True
    assert "No module named 'rl_engine'" in metadata["fallback_reason"]
    log_metrics = rlk_mod.get_linear_logp_runtime_log_metrics(prefix="x/")
    assert log_metrics["x/fallback"] == 1.0
    assert log_metrics["x/backend_descriptor_id"] == metadata["backend_descriptor_id"]
    assert log_metrics["x/contract_descriptor_id"] == metadata["contract_descriptor_id"]
    assert log_metrics["x/fallback_reason_descriptor_id"] == metadata["fallback_reason_descriptor_id"]
    for actual_item, expected_item in zip(result["log_probs"], expected, strict=True):
        torch.testing.assert_close(actual_item, expected_item, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_linear_logp_falls_back_when_op_lacks_tp_interface(monkeypatch):
    _install_fake_rl_engine(monkeypatch, op_factory=lambda: _FakeLegacyLinearLogpOp())
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
    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["fallback"] is True
    assert metadata["actual_backend"] == "vime.native.linear_logp"
    assert "unexpected keyword argument" in metadata["fallback_reason"]


@pytest.mark.unit
def test_linear_logp_zero_tokens_reports_non_fallback_decision(monkeypatch):
    _install_fake_rl_engine(monkeypatch)
    args = _make_args()
    hidden = torch.randn(0, 3)
    weight = torch.randn(5, 3)
    target = torch.empty(0, dtype=torch.long)
    context = rlk_mod.LinearLogpContext(lm_head_weight=weight, bias=None, tp_group=None)

    actual = rlk_mod.maybe_compute_linear_logp(hidden, target, context=context, args=args, with_entropy=False)

    assert actual.shape == (0,)
    counters = rlk_mod.get_rl_kernel_runtime_counters()
    assert counters["linear_logp_call_count"] == 0.0
    assert counters["linear_logp_fallback_count"] == 0.0
    metadata = rlk_mod.get_linear_logp_runtime_metadata()
    assert metadata["actual_backend"] == "vime.linear_logp.zero_tokens"
    assert metadata["fallback"] is False
    assert metadata["fallback_reason"] is None


@pytest.mark.unit
def test_linear_logp_support_matrix_documents_issue20_fields():
    matrix = rlk_mod.get_linear_logp_support_matrix()
    assert matrix

    required_fields = {
        "backend",
        "implementation",
        "dtype",
        "hardware",
        "tp",
        "cp",
        "entropy",
        "full_gradient",
    }
    assert all(required_fields <= set(row) for row in matrix)
    assert {row["backend"] for row in matrix} >= {"registry", "native"}
    assert any(row["source"] == "vime_adapter" for row in matrix)
    assert all(row["source"] == "rl_kernel" for row in matrix if row["backend"] in {"cuda_sm90", "triton", "pytorch"})
    assert any("native" in row["backend"] and "Megatron" in row["implementation"] for row in matrix)


@pytest.mark.unit
def test_linear_logp_support_matrix_reports_unavailable_provider(monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name == "rl_engine.kernels.support":
            raise ModuleNotFoundError("No module named 'rl_engine'")
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(rlk_mod.importlib, "import_module", fail_import)

    matrix = rlk_mod.get_linear_logp_support_matrix()

    assert {row["backend"] for row in matrix} >= {"rl_kernel_unavailable", "registry", "native"}


@pytest.mark.unit
def test_linear_logp_support_matrix_imports_rl_kernel_rows(monkeypatch):
    _drop_rl_engine_modules()
    rl_engine = types.ModuleType("rl_engine")
    kernels = types.ModuleType("rl_engine.kernels")
    support = types.ModuleType("rl_engine.kernels.support")
    support.get_linear_logp_support_matrix = lambda: (
        {
            "source": "rl_kernel",
            "backend": "cuda_sm90",
            "implementation": "FusedLinearLogpSM90Op",
            "dtype": "backend-defined",
            "hardware": "SM90",
            "tp": "reported by RL-Kernel",
            "cp": "reported by RL-Kernel",
            "entropy": "reported by RL-Kernel",
            "full_gradient": "reported by RL-Kernel",
        },
    )
    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.support", support)

    matrix = rlk_mod.get_linear_logp_support_matrix()

    assert {row["backend"] for row in matrix} >= {"cuda_sm90", "registry", "native"}
    assert matrix[0]["source"] == "rl_kernel"


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
    args = _make_args(entropy_coef=0.0)
    context = rlk_mod.LinearLogpContext(
        lm_head_weight=torch.empty(4, 3),
        bias=None,
        tp_group=None,
    )

    assert loss_mod._policy_loss_needs_entropy(args, None) is True
    assert loss_mod._policy_loss_needs_entropy(args, context) is False

    args.entropy_coef = 0.01
    assert loss_mod._policy_loss_needs_entropy(args, None) is True
    assert loss_mod._policy_loss_needs_entropy(args, context) is True
