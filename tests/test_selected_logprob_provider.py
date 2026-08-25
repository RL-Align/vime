from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.selected_logprob_provider import (
    ContextParallelLayout,
    SelectedLogprobProviderUnavailable,
    SelectedLogprobRequest,
    SelectedLogprobResult,
    compute_selected_logprobs,
)


def _request(*, with_entropy: bool = False) -> SelectedLogprobRequest:
    logits = torch.randn(3, 5, requires_grad=True)
    return SelectedLogprobRequest(
        logits=logits,
        target_ids=torch.tensor([1, 2, 3]),
        tensor_parallel_group=None,
        context_parallel=ContextParallelLayout(world_size=2, rank=1, layout="zigzag"),
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy,
        chunk_size=64,
    )


def _native(*args, **kwargs):
    logits = args[0]
    entropy = logits.sum(dim=-1) if kwargs["with_entropy"] else None
    return logits[:, :1], entropy


def _structural_request(**overrides) -> SelectedLogprobRequest:
    values = dict(
        logits=torch.randn(3, 5),
        target_ids=torch.tensor([1, 2, 3]),
        tensor_parallel_group=None,
        context_parallel=ContextParallelLayout(world_size=1, rank=0, layout="single"),
        with_entropy=False,
        with_entropy_grad=False,
        chunk_size=64,
        hidden=torch.randn(3, 4),
        lm_head_weight=torch.randn(5, 4),
        lm_head_bias=torch.randn(5),
        vocab_start_index=0,
        global_vocab_size=5,
        real_vocab_size=4,
        temperature=torch.ones(3),
    )
    values.update(overrides)
    return SelectedLogprobRequest(**values)


def _install_provider(monkeypatch, provider):
    module_name = "selected_logprob_provider_fixture"
    module = types.ModuleType(module_name)
    module.provider = provider
    monkeypatch.setitem(sys.modules, module_name, module)
    return f"{module_name}.provider"


def test_unconfigured_provider_uses_native_path():
    request = _request()

    actual, entropy = compute_selected_logprobs(
        args=SimpleNamespace(selected_logprob_provider=None), request=request, native=_native
    )

    assert entropy is None
    torch.testing.assert_close(actual, request.logits[:, :1])


def test_provider_receives_normalized_request_and_returns_result(monkeypatch):
    request = _request(with_entropy=True)
    observed = {}

    def provider(actual_request):
        observed["request"] = actual_request
        return SelectedLogprobResult(
            selected_logprobs=actual_request.logits[:, :1],
            entropy=actual_request.logits.sum(dim=-1),
            backend_id="external.test",
            contract_id="external.test.v1",
            provenance={"tp_reduction": "provider_owned"},
        )

    path = _install_provider(monkeypatch, provider)
    actual, entropy = compute_selected_logprobs(
        args=SimpleNamespace(selected_logprob_provider=path, selected_logprob_provider_mode="strict"),
        request=request,
        native=_native,
    )

    assert observed["request"] is request
    assert entropy is not None
    torch.testing.assert_close(actual, request.logits[:, :1])
    torch.testing.assert_close(entropy, request.logits.sum(dim=-1))


def test_structural_request_accepts_aligned_hidden_and_lm_head():
    request = _structural_request()

    assert request.hidden is not None and request.hidden.shape == (3, 4)
    assert request.lm_head_weight is not None and request.lm_head_weight.shape == (5, 4)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"lm_head_weight": None}, "lm_head_weight"),
        ({"hidden": torch.randn(2, 4)}, "hidden rows"),
        ({"lm_head_weight": torch.randn(5, 6)}, "hidden width"),
        ({"temperature": torch.tensor([1.0, 0.0, 1.0])}, "temperature must be positive"),
    ],
)
def test_structural_request_rejects_incomplete_or_misaligned_inputs(overrides, match):
    with pytest.raises(ValueError, match=match):
        _structural_request(**overrides)


def test_provider_may_return_a_structural_result_from_an_external_package(monkeypatch):
    request = _request()

    def provider(actual_request):
        return SimpleNamespace(
            selected_logprobs=actual_request.logits[:, :1],
            entropy=None,
            backend_id="external.structural",
            contract_id="external.structural.v1",
            provenance={"tp_reduction": "provider_owned"},
        )

    path = _install_provider(monkeypatch, provider)
    actual, entropy = compute_selected_logprobs(
        args=SimpleNamespace(selected_logprob_provider=path, selected_logprob_provider_mode="strict"),
        request=request,
        native=_native,
    )

    assert entropy is None
    torch.testing.assert_close(actual, request.logits[:, :1])


def test_auto_mode_only_falls_back_for_explicit_unavailability(monkeypatch):
    request = _request()
    calls = {"native": 0}

    def provider(_request):
        raise SelectedLogprobProviderUnavailable("unsupported topology")

    def native(*args, **kwargs):
        calls["native"] += 1
        return _native(*args, **kwargs)

    path = _install_provider(monkeypatch, provider)
    actual, entropy = compute_selected_logprobs(
        args=SimpleNamespace(selected_logprob_provider=path, selected_logprob_provider_mode="auto"),
        request=request,
        native=native,
    )

    assert calls["native"] == 1
    assert entropy is None
    torch.testing.assert_close(actual, request.logits[:, :1])


def test_strict_mode_rejects_unavailable_provider(monkeypatch):
    def provider(_request):
        raise SelectedLogprobProviderUnavailable("unsupported topology")

    path = _install_provider(monkeypatch, provider)
    with pytest.raises(RuntimeError, match="unavailable"):
        compute_selected_logprobs(
            args=SimpleNamespace(selected_logprob_provider=path, selected_logprob_provider_mode="strict"),
            request=_request(),
            native=_native,
        )


def test_provider_contract_failure_never_silently_falls_back(monkeypatch):
    def provider(actual_request):
        return SelectedLogprobResult(
            selected_logprobs=actual_request.logits[:, 0],
            entropy=None,
            backend_id="external.test",
            contract_id="external.test.v1",
        )

    path = _install_provider(monkeypatch, provider)
    with pytest.raises(ValueError, match="invalid selected_logprobs shape"):
        compute_selected_logprobs(
            args=SimpleNamespace(selected_logprob_provider=path, selected_logprob_provider_mode="auto"),
            request=_request(),
            native=_native,
        )


def test_provider_error_never_silently_falls_back(monkeypatch):
    def provider(_request):
        raise AttributeError("provider bug")

    path = _install_provider(monkeypatch, provider)
    with pytest.raises(AttributeError, match="provider bug"):
        compute_selected_logprobs(
            args=SimpleNamespace(selected_logprob_provider=path, selected_logprob_provider_mode="auto"),
            request=_request(),
            native=_native,
        )
