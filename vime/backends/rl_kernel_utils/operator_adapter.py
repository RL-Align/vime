"""vime-owned boundary for optional RL-Kernel operator calls.

This module is the only production vime surface that should know how to call
RL-Kernel-backed operators.  Training, rollout, and audit code pass already
resolved policy/config context into an :class:`RlkOperatorAdapter`; the adapter
then returns a structured decision/result without taking ownership of the
surrounding lifecycle.

RL-Kernel is intentionally imported lazily inside the concrete registry backend
so importing vime, the protocol, or test mocks does not require RL-Kernel to be
installed.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from vime.backends.rl_kernel_utils.execution import (
    BackendCapability,
    CapabilityQueryResult,
    ExecutionDecision,
    FallbackReason,
    LogprobContractMetadata,
    RlKernelCapabilities,
    emit_execution_decision,
    query_rl_kernel_capabilities,
    select_execution_decision,
)

logger = logging.getLogger(__name__)

RlkOperatorName = Literal["selected_logprob", "reference_score", "linear_logp"]
_PUBLIC_TO_BACKEND_OPERATOR = {
    "selected_logprob": "logp",
    "reference_score": "reference_score",
    "linear_logp": "linear_logp",
}


@dataclass(frozen=True)
class RlkPolicyContext:
    """Resolved vime policy/config passed into an RL-Kernel adapter.

    The adapter does not parse CLI flags or environment variables.  Callers must
    resolve those into this object before constructing or invoking an adapter.
    """

    fast_path: Literal["off", "auto", "strict"] = "off"
    consistency_path: Literal["off", "audit", "strict"] = "off"
    enabled_ops: frozenset[str] = field(default_factory=frozenset)
    requested_backend: str | None = None
    dtype: str | None = None
    stage: str = "operator"
    parallel_context: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    telemetry_enabled: bool = True
    telemetry_sample_rate: float = 1.0
    telemetry_sample_seed: int | str = 0

    def __post_init__(self) -> None:
        if self.fast_path not in {"off", "auto", "strict"}:
            raise ValueError(f"unsupported RL-Kernel fast path mode {self.fast_path!r}")
        if self.consistency_path not in {"off", "audit", "strict"}:
            raise ValueError(f"unsupported RL-Kernel consistency path mode {self.consistency_path!r}")
        object.__setattr__(self, "enabled_ops", frozenset(self.enabled_ops))
        object.__setattr__(self, "parallel_context", dict(self.parallel_context))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def disabled(self) -> bool:
        return self.fast_path == "off" and self.consistency_path == "off"

    def operator_enabled(self, public_operator: RlkOperatorName) -> bool:
        backend_operator = _backend_operator_name(public_operator)
        return "*" in self.enabled_ops or public_operator in self.enabled_ops or backend_operator in self.enabled_ops

    def to_log_record(self) -> dict[str, Any]:
        return {
            "fast_path": self.fast_path,
            "consistency_path": self.consistency_path,
            "enabled_ops": sorted(self.enabled_ops),
            "requested_backend": self.requested_backend,
            "dtype": self.dtype,
            "stage": self.stage,
            "parallel_context": dict(self.parallel_context),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class RlkSelectedLogprobRequest:
    logits: Any
    token_ids: Any
    stage: str | None = None
    dtype: str | None = None
    requested_backend: str | None = None
    parallel_context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fp32_output: bool = False

    def __post_init__(self) -> None:
        _freeze_request_mappings(self)


@dataclass(frozen=True)
class RlkReferenceScoreRequest:
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    stage: str | None = None
    dtype: str | None = None
    requested_backend: str | None = None
    parallel_context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        _freeze_request_mappings(self)


@dataclass(frozen=True)
class RlkLinearLogpRequest:
    hidden: Any
    lm_head_weight: Any
    target_ids: Any
    bias: Any = None
    tp_group: Any = None
    vocab_start_index: int = 0
    global_vocab_size: int | None = None
    stage: str | None = None
    dtype: str | None = None
    requested_backend: str | None = None
    parallel_context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_request_mappings(self)


@dataclass(frozen=True)
class RlkOperatorProvenance:
    adapter_id: str
    adapter_kind: Literal["noop", "mock", "rl_kernel"]
    requested_backend: str | None
    actual_backend: str | None
    capability_backend_id: str | None
    contract_id: str | None
    runtime_fingerprint: str | None = None
    build_fingerprint: str | None = None
    policy_context: Mapping[str, Any] = field(default_factory=dict)
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_context", dict(self.policy_context))
        object.__setattr__(self, "backend_metadata", dict(self.backend_metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RlkOperatorResult:
    handled: bool
    value: Any
    decision: ExecutionDecision
    contract: LogprobContractMetadata | None = None
    provenance: RlkOperatorProvenance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def supported(self) -> bool:
        return self.handled


@runtime_checkable
class RlkOperatorAdapter(Protocol):
    """Protocol consumed by production code and tests.

    Implementations return ``handled=False`` when the caller should continue with
    vime's native implementation.  They must not own rollout scheduling,
    training loops, weight synchronization, or artifact layout.
    """

    policy_context: RlkPolicyContext

    def query_capabilities(self, operator: RlkOperatorName | str | None = None) -> CapabilityQueryResult: ...

    def selected_logprob(self, request: RlkSelectedLogprobRequest) -> RlkOperatorResult: ...

    def reference_score(self, request: RlkReferenceScoreRequest) -> RlkOperatorResult: ...

    def linear_logp(self, request: RlkLinearLogpRequest) -> RlkOperatorResult: ...


class RlKernelRegistryBackend:
    """Lazy facade over RL-Kernel's operator registry."""

    def __init__(self, registry_module: str = "rl_engine.kernels.registry") -> None:
        self._registry_module = registry_module
        self._registry: Any = None

    def get_op(self, operator: str) -> Any:
        if self._registry is None:
            module = importlib.import_module(self._registry_module)
            self._registry = module.kernel_registry
        return self._registry.get_op(operator)


class NoopRlkOperatorAdapter:
    """Disabled adapter used when no RL-Kernel mode is active."""

    def __init__(self, policy_context: RlkPolicyContext | None = None) -> None:
        self.policy_context = policy_context or RlkPolicyContext()
        self._capabilities = RlKernelCapabilities(available=False, reason="rl_kernel_disabled")

    def query_capabilities(
        self, operator: RlkOperatorName | str | None = None
    ) -> CapabilityQueryResult:  # noqa: ARG002
        reason = FallbackReason(
            code="rl_kernel_disabled",
            message="RL-Kernel operator adapter is disabled.",
        )
        return CapabilityQueryResult(capabilities=self._capabilities, fallback_reason=reason)

    def selected_logprob(self, request: RlkSelectedLogprobRequest) -> RlkOperatorResult:
        return self._unhandled("selected_logprob", request)

    def reference_score(self, request: RlkReferenceScoreRequest) -> RlkOperatorResult:
        return self._unhandled("reference_score", request)

    def linear_logp(self, request: RlkLinearLogpRequest) -> RlkOperatorResult:
        return self._unhandled("linear_logp", request)

    def _unhandled(self, public_operator: RlkOperatorName, request: Any) -> RlkOperatorResult:
        reason = FallbackReason(
            code="rl_kernel_disabled",
            message="RL-Kernel operator adapter is disabled.",
        )
        decision = _select_decision(
            self.policy_context,
            public_operator,
            request,
            capabilities=self._capabilities,
            force_fast_path="off",
        )
        return _result(
            handled=False,
            value=None,
            decision=decision,
            policy_context=self.policy_context,
            adapter_kind="noop",
            adapter_id=type(self).__name__,
            fallback_reason=reason,
            capabilities=self._capabilities,
            request_metadata=_request_metadata(request),
        )


class RlKernelOperatorAdapter:
    """Adapter that routes supported calls to RL-Kernel operators."""

    def __init__(
        self,
        policy_context: RlkPolicyContext,
        *,
        capability_result: CapabilityQueryResult | None = None,
        backend: RlKernelRegistryBackend | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.policy_context = policy_context
        self._capability_result = capability_result or query_rl_kernel_capabilities()
        self._backend = backend or RlKernelRegistryBackend()
        self._log = log or logger

    def query_capabilities(self, operator: RlkOperatorName | str | None = None) -> CapabilityQueryResult:
        if operator is None:
            return self._capability_result
        backend_operator = _backend_operator_name(operator)
        capabilities = self._capability_result.capabilities
        filtered = RlKernelCapabilities(
            available=capabilities.available,
            backends=capabilities.matching_backends(backend_operator),
            reason=capabilities.reason,
            runtime_fingerprint=capabilities.runtime_fingerprint,
            build_fingerprint=capabilities.build_fingerprint,
        )
        return CapabilityQueryResult(capabilities=filtered, fallback_reason=self._capability_result.fallback_reason)

    def selected_logprob(self, request: RlkSelectedLogprobRequest) -> RlkOperatorResult:
        def invoke(op: Any) -> Any:
            if request.fp32_output:
                for method_name in ("forward_fp32", "apply_fp32", "online_fp32"):
                    method = getattr(op, method_name, None)
                    if method is not None:
                        return method(request.logits, request.token_ids)
            return op(request.logits, request.token_ids)

        return self._call("selected_logprob", request, invoke)

    def reference_score(self, request: RlkReferenceScoreRequest) -> RlkOperatorResult:
        return self._call("reference_score", request, lambda op: op(*request.args, **dict(request.kwargs)))

    def linear_logp(self, request: RlkLinearLogpRequest) -> RlkOperatorResult:
        def invoke(op: Any) -> Any:
            return op(
                request.hidden,
                request.lm_head_weight,
                request.target_ids,
                request.bias,
                tp_group=request.tp_group,
                vocab_start_index=request.vocab_start_index,
                global_vocab_size=request.global_vocab_size,
            )

        return self._call("linear_logp", request, invoke)

    def _call(self, public_operator: RlkOperatorName, request: Any, invoke: Callable[[Any], Any]) -> RlkOperatorResult:
        if not self.policy_context.operator_enabled(public_operator):
            return _unsupported_result(
                self.policy_context,
                public_operator,
                request,
                code="operator_disabled",
                message="RL-Kernel operator is not enabled by vime policy context.",
                capabilities=self._capability_result.capabilities,
                adapter_kind="rl_kernel",
                adapter_id=type(self).__name__,
            )

        decision = _select_decision(
            self.policy_context,
            public_operator,
            request,
            capabilities=self._capability_result.capabilities,
        )
        emit_execution_decision(
            decision,
            log=self._log,
            enabled=self.policy_context.telemetry_enabled,
            sample_rate=self.policy_context.telemetry_sample_rate,
            sample_seed=self.policy_context.telemetry_sample_seed,
        )
        if decision.decision in {"native", "audit-only", "fallback-native", "strict-failure"}:
            return _result(
                handled=False,
                value=None,
                decision=decision,
                policy_context=self.policy_context,
                adapter_kind="rl_kernel",
                adapter_id=type(self).__name__,
                fallback_reason=decision.fallback_reason,
                capabilities=self._capability_result.capabilities,
                request_metadata=_request_metadata(request),
            )

        op = self._backend.get_op(_backend_operator_name(public_operator))
        value = invoke(op)
        return _result(
            handled=True,
            value=value,
            decision=decision,
            policy_context=self.policy_context,
            adapter_kind="rl_kernel",
            adapter_id=type(self).__name__,
            backend_metadata={"operator_impl": type(op).__name__},
            capabilities=self._capability_result.capabilities,
            request_metadata=_request_metadata(request),
        )


class MockRlkOperatorAdapter:
    """Small boundary test double for Phase 1 and later operator integration tests."""

    def __init__(
        self,
        policy_context: RlkPolicyContext | None = None,
        *,
        handlers: Mapping[str, Callable[[Any], Any] | Any] | None = None,
        capabilities: RlKernelCapabilities | None = None,
    ) -> None:
        self.policy_context = policy_context or RlkPolicyContext(fast_path="auto", enabled_ops=frozenset({"*"}))
        self._handlers = dict(handlers or {})
        self._capabilities = capabilities or _mock_capabilities(self._handlers)

    def query_capabilities(self, operator: RlkOperatorName | str | None = None) -> CapabilityQueryResult:
        if operator is None:
            return CapabilityQueryResult(capabilities=self._capabilities)
        backend_operator = _backend_operator_name(operator)
        return CapabilityQueryResult(
            capabilities=RlKernelCapabilities(
                available=self._capabilities.available,
                backends=self._capabilities.matching_backends(backend_operator),
                reason=self._capabilities.reason,
                runtime_fingerprint=self._capabilities.runtime_fingerprint,
                build_fingerprint=self._capabilities.build_fingerprint,
            )
        )

    def selected_logprob(self, request: RlkSelectedLogprobRequest) -> RlkOperatorResult:
        return self._dispatch("selected_logprob", request)

    def reference_score(self, request: RlkReferenceScoreRequest) -> RlkOperatorResult:
        return self._dispatch("reference_score", request)

    def linear_logp(self, request: RlkLinearLogpRequest) -> RlkOperatorResult:
        return self._dispatch("linear_logp", request)

    def _dispatch(self, public_operator: RlkOperatorName, request: Any) -> RlkOperatorResult:
        if not self.policy_context.operator_enabled(public_operator):
            return _unsupported_result(
                self.policy_context,
                public_operator,
                request,
                code="operator_disabled",
                message="Mock RL-Kernel operator is not enabled by vime policy context.",
                capabilities=self._capabilities,
                adapter_kind="mock",
                adapter_id=type(self).__name__,
            )

        handler = self._handlers.get(public_operator, self._handlers.get(_backend_operator_name(public_operator)))
        if handler is None:
            return _unsupported_result(
                self.policy_context,
                public_operator,
                request,
                code="operator_unsupported",
                message="Mock RL-Kernel operator has no handler for this call.",
                capabilities=self._capabilities,
                adapter_kind="mock",
                adapter_id=type(self).__name__,
            )

        value = handler(request) if callable(handler) else handler
        if isinstance(value, RlkOperatorResult):
            return value

        decision = _select_decision(self.policy_context, public_operator, request, capabilities=self._capabilities)
        return _result(
            handled=True,
            value=value,
            decision=decision,
            policy_context=self.policy_context,
            adapter_kind="mock",
            adapter_id=type(self).__name__,
            capabilities=self._capabilities,
            request_metadata=_request_metadata(request),
        )


def build_rlk_operator_adapter(
    policy_context: RlkPolicyContext,
    *,
    capability_provider: Any = None,
    backend: RlKernelRegistryBackend | None = None,
    log: logging.Logger | None = None,
) -> RlkOperatorAdapter:
    """Select the concrete adapter implementation for already-resolved context."""

    if policy_context.disabled:
        return NoopRlkOperatorAdapter(policy_context)
    capability_result = query_rl_kernel_capabilities(capability_provider)
    return RlKernelOperatorAdapter(
        policy_context,
        capability_result=capability_result,
        backend=backend,
        log=log,
    )


def _freeze_request_mappings(request: Any) -> None:
    for name in ("parallel_context", "metadata", "kwargs"):
        if hasattr(request, name):
            object.__setattr__(request, name, dict(getattr(request, name)))


def _backend_operator_name(operator: RlkOperatorName | str) -> str:
    return _PUBLIC_TO_BACKEND_OPERATOR.get(operator, operator)


def _stage(policy_context: RlkPolicyContext, request: Any) -> str:
    return getattr(request, "stage", None) or policy_context.stage


def _dtype(policy_context: RlkPolicyContext, request: Any) -> str | None:
    return getattr(request, "dtype", None) or policy_context.dtype


def _requested_backend(policy_context: RlkPolicyContext, request: Any) -> str | None:
    return getattr(request, "requested_backend", None) or policy_context.requested_backend


def _parallel_context(policy_context: RlkPolicyContext, request: Any) -> dict[str, Any]:
    context = dict(policy_context.parallel_context)
    context.update(getattr(request, "parallel_context", {}) or {})
    return context


def _request_metadata(request: Any) -> dict[str, Any]:
    return dict(getattr(request, "metadata", {}) or {})


def _select_decision(
    policy_context: RlkPolicyContext,
    public_operator: RlkOperatorName,
    request: Any,
    *,
    capabilities: RlKernelCapabilities | None,
    force_fast_path: Literal["off", "auto", "strict"] | None = None,
) -> ExecutionDecision:
    return select_execution_decision(
        operator=_backend_operator_name(public_operator),
        stage=_stage(policy_context, request),
        requested_fast=force_fast_path or policy_context.fast_path,
        requested_consistency=policy_context.consistency_path,
        capabilities=capabilities,
        requested_backend=_requested_backend(policy_context, request),
        dtype=_dtype(policy_context, request),
        parallel_context=_parallel_context(policy_context, request),
    )


def _result(
    *,
    handled: bool,
    value: Any,
    decision: ExecutionDecision,
    policy_context: RlkPolicyContext,
    adapter_kind: Literal["noop", "mock", "rl_kernel"],
    adapter_id: str,
    fallback_reason: FallbackReason | None = None,
    backend_metadata: Mapping[str, Any] | None = None,
    capabilities: RlKernelCapabilities | None = None,
    request_metadata: Mapping[str, Any] | None = None,
) -> RlkOperatorResult:
    backend_metadata = dict(backend_metadata or {})
    if request_metadata:
        backend_metadata["request_metadata"] = dict(request_metadata)
    if fallback_reason is not None:
        backend_metadata["fallback_reason"] = asdict(fallback_reason)
    contract = None
    if decision.contract_id is not None:
        contract = LogprobContractMetadata(
            source=decision.stage,
            contract_id=decision.contract_id,
            dtype=decision.dtype,
            backend_id=decision.actual_backend,
        )
    runtime_fingerprint, build_fingerprint = _fingerprints_for(decision, capabilities)
    provenance = RlkOperatorProvenance(
        adapter_id=adapter_id,
        adapter_kind=adapter_kind,
        requested_backend=decision.requested_backend,
        actual_backend=decision.actual_backend,
        capability_backend_id=decision.capability_backend_id,
        contract_id=decision.contract_id,
        runtime_fingerprint=runtime_fingerprint,
        build_fingerprint=build_fingerprint,
        policy_context=policy_context.to_log_record(),
        backend_metadata=backend_metadata,
    )
    return RlkOperatorResult(
        handled=handled,
        value=value,
        decision=decision,
        contract=contract,
        provenance=provenance,
        metadata=backend_metadata,
    )


def _unsupported_result(
    policy_context: RlkPolicyContext,
    public_operator: RlkOperatorName,
    request: Any,
    *,
    code: str,
    message: str,
    capabilities: RlKernelCapabilities | None,
    adapter_kind: Literal["mock", "rl_kernel"],
    adapter_id: str,
) -> RlkOperatorResult:
    reason = FallbackReason(
        code=code,
        message=message,
        details={"operator": _backend_operator_name(public_operator)},
    )
    if code == "operator_disabled":
        decision = _operator_disabled_decision(policy_context, public_operator, request, reason)
    else:
        decision = _select_decision(policy_context, public_operator, request, capabilities=capabilities)
    details = {**decision.details, **reason.details}
    if decision.fallback_reason is not None and decision.fallback_reason != reason:
        details["selection_fallback_reason"] = asdict(decision.fallback_reason)
    decision = ExecutionDecision(
        operator=decision.operator,
        stage=decision.stage,
        requested_mode=decision.requested_mode,
        requested_backend=decision.requested_backend,
        actual_backend=decision.actual_backend,
        decision=decision.decision,
        fallback=decision.fallback,
        fallback_reason=reason,
        capability_backend_id=decision.capability_backend_id,
        contract_id=decision.contract_id,
        dtype=decision.dtype,
        parallel_context=decision.parallel_context,
        strict_eligible=decision.strict_eligible,
        details=details,
    )
    return _result(
        handled=False,
        value=None,
        decision=decision,
        policy_context=policy_context,
        adapter_kind=adapter_kind,
        adapter_id=adapter_id,
        fallback_reason=reason,
        capabilities=capabilities,
        request_metadata=_request_metadata(request),
    )


def _operator_disabled_decision(
    policy_context: RlkPolicyContext,
    public_operator: RlkOperatorName,
    request: Any,
    reason: FallbackReason,
) -> ExecutionDecision:
    backend_operator = _backend_operator_name(public_operator)
    decision = "audit-only" if policy_context.consistency_path in {"audit", "strict"} else "native"
    return ExecutionDecision(
        operator=backend_operator,
        stage=_stage(policy_context, request),
        requested_mode=f"fast={policy_context.fast_path},consistency={policy_context.consistency_path}",
        requested_backend=_requested_backend(policy_context, request),
        actual_backend="vime.native",
        decision=decision,
        fallback=False,
        fallback_reason=reason,
        contract_id=f"vime.native.{backend_operator}",
        dtype=_dtype(policy_context, request),
        parallel_context=_parallel_context(policy_context, request),
        details=reason.details,
    )


def _mock_capabilities(handlers: Mapping[str, Any]) -> RlKernelCapabilities:
    backends = []
    for raw_operator in sorted(handlers):
        operator = _backend_operator_name(raw_operator)
        backends.append(
            BackendCapability(
                operator=operator,
                backend_id=f"mock.{operator}",
                implementation_kind="optimized",
                dtypes=(),
                deterministic=True,
                batch_invariant=True,
                strict_fast_eligible=True,
                production=False,
                runtime_fingerprint="mock-runtime",
                build_fingerprint="mock-build",
            )
        )
    return RlKernelCapabilities(
        available=True,
        backends=tuple(backends),
        runtime_fingerprint="mock-runtime",
        build_fingerprint="mock-build",
    )


def _fingerprints_for(
    decision: ExecutionDecision,
    capabilities: RlKernelCapabilities | None,
) -> tuple[str | None, str | None]:
    if capabilities is None:
        return None, None
    runtime_fingerprint = capabilities.runtime_fingerprint
    build_fingerprint = capabilities.build_fingerprint
    for backend in capabilities.backends:
        if backend.backend_id == decision.capability_backend_id:
            runtime_fingerprint = backend.runtime_fingerprint or runtime_fingerprint
            build_fingerprint = backend.build_fingerprint or build_fingerprint
            break
    return runtime_fingerprint, build_fingerprint
