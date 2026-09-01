from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import vime.backends.megatron_utils.linear_logp_provider as provider_module
from vime.backends.megatron_utils.linear_logp_provider import (
    LinearLogpContext,
    LinearLogpProviderUnavailable,
    LinearLogpRequest,
    LinearLogpResult,
    LinearProjection,
    TokenLayout,
    VocabPartition,
    compute_linear_logp,
)


def _request(*, with_entropy: bool = False) -> LinearLogpRequest:
    logits = torch.randn(3, 5, requires_grad=True)
    return LinearLogpRequest(
        logits=logits,
        target_ids=torch.tensor([1, 2, 3]),
        tensor_parallel_group=None,
        token_layout=TokenLayout(world_size=2, rank=1, layout="zigzag"),
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy,
        chunk_size=64,
    )


def _native(*args, **kwargs):
    logits = args[0]
    entropy = logits.sum(dim=-1) if kwargs["with_entropy"] else None
    return logits[:, :1], entropy


def _structural_request(**overrides) -> LinearLogpRequest:
    context = LinearLogpContext(
        hidden=torch.randn(3, 4),
        projection=LinearProjection(
            weight=torch.randn(5, 4),
            bias=torch.randn(5),
        ),
        vocab_partition=VocabPartition(local_start=0, local_size=5, real_size=4, padded_size=5),
    )
    values = dict(
        logits=torch.randn(3, 5),
        target_ids=torch.tensor([1, 2, 3]),
        tensor_parallel_group=None,
        token_layout=TokenLayout(world_size=1, rank=0, layout="single"),
        with_entropy=False,
        with_entropy_grad=False,
        chunk_size=64,
        context=context,
        temperature=torch.ones(3),
    )
    values.update(overrides)
    return LinearLogpRequest(**values)


def _install_provider(monkeypatch, provider):
    module_name = "linear_logp_provider_fixture"
    module = types.ModuleType(module_name)
    module.provider = provider
    monkeypatch.setitem(sys.modules, module_name, module)
    return f"{module_name}.provider"


def _args(path=None, mode="auto"):
    return SimpleNamespace(linear_logp_provider=path, linear_logp_provider_mode=mode)


def test_unconfigured_provider_uses_native_path_and_reports_actual_backend(caplog):
    request = _request()
    provider_module._LOGGED_NATIVE_IDENTITIES.clear()
    with caplog.at_level(logging.INFO, logger=provider_module.__name__):
        actual, entropy = compute_linear_logp(args=_args(), request=request, native=_native)

    assert entropy is None
    torch.testing.assert_close(actual, request.logits[:, :1])
    assert "linear_logp native active:" in caplog.text
    assert f"backend_id={__name__}._native" in caplog.text
    assert "contract_id=vime.native.linear_logp.v1" in caplog.text
    assert "route=unconfigured" in caplog.text
    assert "device=cpu" in caplog.text


def test_provider_receives_structured_request_and_returns_result(monkeypatch):
    request = _request(with_entropy=True)
    observed = {}

    def provider(actual_request):
        observed["request"] = actual_request
        return LinearLogpResult(
            logp=actual_request.logits[:, :1],
            entropy=actual_request.logits.sum(dim=-1),
            backend_id="external.test",
            contract_id="external.test.v1",
            provenance={"tp_reduction": "provider_owned"},
        )

    path = _install_provider(monkeypatch, provider)
    actual, entropy = compute_linear_logp(args=_args(path, "strict"), request=request, native=_native)

    assert observed["request"] is request
    assert entropy is not None
    torch.testing.assert_close(actual, request.logits[:, :1])
    torch.testing.assert_close(entropy, request.logits.sum(dim=-1))


def test_structural_request_contains_model_neutral_projection_contract():
    request = _structural_request()

    assert request.context is not None
    assert request.context.hidden.shape == (3, 4)
    assert request.context.projection.weight.shape == (5, 4)
    assert request.context.vocab_partition.local_start == 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "context": LinearLogpContext(
                    hidden=torch.randn(3, 4),
                    projection=LinearProjection(weight=torch.randn(4, 4)),
                    vocab_partition=VocabPartition(local_start=0, local_size=4, real_size=4, padded_size=5),
                )
            },
            "projection rows",
        ),
        ({"temperature": torch.tensor([1.0, 0.0, 1.0])}, "temperature must be positive"),
    ],
)
def test_structural_request_rejects_misaligned_inputs(overrides, match):
    with pytest.raises(ValueError, match=match):
        _structural_request(**overrides)


def test_provider_may_return_a_structural_result_from_an_external_package(monkeypatch):
    request = _request()

    def provider(actual_request):
        return SimpleNamespace(
            logp=actual_request.logits[:, :1],
            entropy=None,
            backend_id="external.structural",
            contract_id="external.structural.v1",
            provenance={"tp_reduction": "provider_owned"},
        )

    path = _install_provider(monkeypatch, provider)
    actual, entropy = compute_linear_logp(args=_args(path, "strict"), request=request, native=_native)

    assert entropy is None
    torch.testing.assert_close(actual, request.logits[:, :1])


def test_auto_mode_only_falls_back_for_explicit_unavailability(monkeypatch, caplog):
    request = _request()
    calls = {"native": 0}
    provider_module._LOGGED_NATIVE_IDENTITIES.clear()

    def provider(_request):
        raise LinearLogpProviderUnavailable("unsupported topology")

    def native(*args, **kwargs):
        calls["native"] += 1
        return _native(*args, **kwargs)

    path = _install_provider(monkeypatch, provider)
    with caplog.at_level(logging.INFO, logger=provider_module.__name__):
        actual, entropy = compute_linear_logp(args=_args(path), request=request, native=native)

    assert calls["native"] == 1
    assert entropy is None
    torch.testing.assert_close(actual, request.logits[:, :1])
    assert "route=provider_unavailable_fallback" in caplog.text


def test_strict_mode_rejects_unavailable_provider(monkeypatch):
    def provider(_request):
        raise LinearLogpProviderUnavailable("unsupported topology")

    path = _install_provider(monkeypatch, provider)
    with pytest.raises(RuntimeError, match="unavailable"):
        compute_linear_logp(args=_args(path, "strict"), request=_request(), native=_native)


def test_provider_contract_failure_never_silently_falls_back(monkeypatch):
    def provider(actual_request):
        return LinearLogpResult(
            logp=actual_request.logits[:, 0],
            entropy=None,
            backend_id="external.test",
            contract_id="external.test.v1",
        )

    path = _install_provider(monkeypatch, provider)
    with pytest.raises(ValueError, match="invalid logp shape"):
        compute_linear_logp(args=_args(path), request=_request(), native=_native)


def test_provider_error_never_silently_falls_back(monkeypatch):
    def provider(_request):
        raise AttributeError("provider bug")

    path = _install_provider(monkeypatch, provider)
    with pytest.raises(AttributeError, match="provider bug"):
        compute_linear_logp(args=_args(path), request=_request(), native=_native)
