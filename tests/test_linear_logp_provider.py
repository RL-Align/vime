from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

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


def _context() -> LinearLogpContext:
    return LinearLogpContext(
        hidden=torch.randn(3, 4),
        projection=LinearProjection(weight=torch.randn(5, 4), bias=torch.randn(5)),
        vocab_partition=VocabPartition(local_start=0, local_size=5, real_size=4, padded_size=5),
    )


def _request(*, with_entropy=False, context=None) -> LinearLogpRequest:
    logits = torch.randn(3, 5, requires_grad=True)
    return LinearLogpRequest(
        logits=logits,
        target_ids=torch.tensor([1, 2, 3]),
        tensor_parallel_group=None,
        token_layout=TokenLayout(world_size=2, rank=1, layout="zigzag"),
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy,
        chunk_size=64,
        context=context,
    )


def _native(logits, *_args, with_entropy, **_kwargs):
    return logits[:, :1], logits.sum(dim=-1) if with_entropy else None


def _args(path=None, mode="auto"):
    return SimpleNamespace(linear_logp_provider=path, linear_logp_provider_mode=mode)


def _install_provider(monkeypatch, provider):
    module = types.ModuleType("linear_logp_provider_fixture")
    module.provider = provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}.provider"


@pytest.mark.unit
def test_unconfigured_provider_uses_native_path():
    request = _request()

    logp, entropy = compute_linear_logp(args=_args(), request=request, native=_native)

    torch.testing.assert_close(logp, request.logits[:, :1])
    assert entropy is None


@pytest.mark.unit
def test_provider_receives_structural_context(monkeypatch):
    context = _context()
    request = _request(with_entropy=True, context=context)
    observed = {}

    def provider(actual_request):
        observed["context"] = actual_request.context
        return LinearLogpResult(
            logp=actual_request.logits[:, :1],
            entropy=actual_request.logits.sum(dim=-1),
            backend_id="fixture",
            contract_id="fixture.v1",
        )

    path = _install_provider(monkeypatch, provider)
    logp, entropy = compute_linear_logp(args=_args(path, "strict"), request=request, native=_native)

    assert observed["context"] is context
    torch.testing.assert_close(logp, request.logits[:, :1])
    torch.testing.assert_close(entropy, request.logits.sum(dim=-1))


@pytest.mark.unit
def test_auto_falls_back_only_for_explicit_unavailability(monkeypatch):
    request = _request()

    def unavailable(_request):
        raise LinearLogpProviderUnavailable("unsupported")

    path = _install_provider(monkeypatch, unavailable)
    logp, _ = compute_linear_logp(args=_args(path), request=request, native=_native)
    torch.testing.assert_close(logp, request.logits[:, :1])

    def broken(_request):
        raise RuntimeError("provider bug")

    path = _install_provider(monkeypatch, broken)
    with pytest.raises(RuntimeError, match="provider bug"):
        compute_linear_logp(args=_args(path), request=request, native=_native)


@pytest.mark.unit
def test_strict_mode_fails_when_provider_is_unavailable(monkeypatch):
    def provider(_request):
        raise LinearLogpProviderUnavailable("unsupported")

    path = _install_provider(monkeypatch, provider)
    with pytest.raises(RuntimeError, match="is unavailable"):
        compute_linear_logp(args=_args(path, "strict"), request=_request(), native=_native)


@pytest.mark.unit
def test_strict_mode_validates_identity_and_autograd(monkeypatch):
    request = _request()

    def provider(actual_request):
        return {
            "logp": actual_request.logits[:, :1].detach(),
            "entropy": None,
            "backend_id": "fixture",
            "contract_id": "fixture.v1",
            "provenance": {},
        }

    path = _install_provider(monkeypatch, provider)
    with pytest.raises(ValueError, match="detached"):
        compute_linear_logp(args=_args(path, "strict"), request=request, native=_native)


@pytest.mark.unit
def test_structural_context_rejects_misaligned_projection():
    with pytest.raises(ValueError, match="projection and vocabulary"):
        LinearLogpContext(
            hidden=torch.randn(3, 4),
            projection=LinearProjection(weight=torch.randn(5, 4)),
            vocab_partition=VocabPartition(local_start=0, local_size=4, real_size=5, padded_size=5),
        )
