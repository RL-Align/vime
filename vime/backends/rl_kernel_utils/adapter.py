"""vime-owned boundary for optional RL-Kernel operator calls.

This module is the only production vime surface that should know how to ask
RL-Kernel for an operator implementation. Importing it is deliberately cheap:
it does not import ``rl_engine`` and it does not initialize CUDA, Triton, Ray,
Megatron, or any cross-config runner machinery.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

RLK_OP_LINEAR_LOGP = "linear_logp"
RLK_OP_SELECTED_LOGPROBS = "logp"
RLK_OP_REFERENCE_LOGPROBS = "reference_logp"
RLK_ALL_OPERATORS = "*"

_FAST_CHOICES = {"off", "auto", "strict"}
_CONSISTENCY_CHOICES = {"off", "audit", "strict"}


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _normalize_ops(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(op.strip() for op in value.split(",") if op.strip())
    return tuple(str(op).strip() for op in value if str(op).strip())


@dataclass(frozen=True)
class RlkPolicyContext:
    """Resolved vime policy passed into an RL-Kernel adapter.

    The adapter consumes this object; it never parses CLI flags or environment
    variables itself. ``enabled_ops`` is an allowlist. Use ``("*",)`` only for
    tests or explicitly opted-in future configs that want every known hook.
    """

    fast: str = "off"
    consistency: str = "off"
    enabled_ops: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fast = str(self.fast).strip().lower()
        consistency = str(self.consistency).strip().lower()
        if fast not in _FAST_CHOICES:
            raise ValueError(f"fast must be one of {sorted(_FAST_CHOICES)}, got {self.fast!r}.")
        if consistency not in _CONSISTENCY_CHOICES:
            raise ValueError(f"consistency must be one of {sorted(_CONSISTENCY_CHOICES)}, got {self.consistency!r}.")
        object.__setattr__(self, "fast", fast)
        object.__setattr__(self, "consistency", consistency)
        object.__setattr__(self, "enabled_ops", _normalize_ops(self.enabled_ops))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @property
    def native_only(self) -> bool:
        return self.fast == "off" and self.consistency == "off"

    @property
    def strict_fast(self) -> bool:
        return self.fast == "strict"

    def operator_enabled(self, op_name: str) -> bool:
        if self.fast == "off":
            return False
        if RLK_ALL_OPERATORS in self.enabled_ops:
            return True
        if op_name in self.enabled_ops:
            return True
        # Reference scoring reuses selected-logprob implementations unless a
        # future backend advertises a distinct reference scorer.
        return op_name == RLK_OP_REFERENCE_LOGPROBS and RLK_OP_SELECTED_LOGPROBS in self.enabled_ops


def rlk_policy_context_from_args(args: Any) -> RlkPolicyContext:
    """Build a policy context from already-resolved vime args.

    ``vime.utils.arguments.resolve_rlk_mode_config`` owns parsing and alias
    resolution. This helper only reads the resolved attributes so callers can
    pass a compact, immutable policy into adapter construction.
    """

    config = getattr(args, "rlk_mode_config", None)
    if config is not None:
        fast = config.fast
        consistency = config.consistency
        enabled_ops = config.ops
    else:
        fast = getattr(args, "rlk_fast", None) or "off"
        consistency = getattr(args, "rlk_consistency", None) or "off"
        enabled_ops = getattr(args, "rl_kernel_ops", None) or ()
    return RlkPolicyContext(
        fast=fast,
        consistency=consistency,
        enabled_ops=enabled_ops,
        metadata={"source": "vime_args"},
    )


@dataclass(frozen=True)
class RlkRuntimeBatchMetadata:
    """Borrowed rollout/training metadata for operator calls.

    Tensor/list fields are stored by reference. The adapter does not own rollout
    samples, data iterators, training lifecycle, or cross-process artifacts.
    """

    total_lengths: tuple[int, ...] = ()
    response_lengths: tuple[int, ...] = ()
    loss_masks: Any | None = None
    rollout_log_probs: Any | None = None
    rollout_top_p_token_ids: Any | None = None
    rollout_top_p_token_offsets: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_lengths", tuple(int(x) for x in self.total_lengths))
        object.__setattr__(self, "response_lengths", tuple(int(x) for x in self.response_lengths))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


def runtime_batch_metadata_from_vime_batch(batch: Mapping[str, Any] | None) -> RlkRuntimeBatchMetadata:
    """Map a vime rollout/training batch to borrowed RL-Kernel call metadata."""

    if not batch:
        return RlkRuntimeBatchMetadata()
    return RlkRuntimeBatchMetadata(
        total_lengths=tuple(batch.get("total_lengths", ())),
        response_lengths=tuple(batch.get("response_lengths", ())),
        loss_masks=batch.get("loss_masks"),
        rollout_log_probs=batch.get("rollout_log_probs"),
        rollout_top_p_token_ids=batch.get("rollout_top_p_token_ids"),
        rollout_top_p_token_offsets=batch.get("rollout_top_p_token_offsets"),
        metadata=batch.get("metadata") or {},
    )


@dataclass(frozen=True)
class SelectedLogprobInputs:
    logits: Any
    target_ids: Any
    mask: Any | None = None
    temperature: float = 1.0
    runtime: RlkRuntimeBatchMetadata = field(default_factory=RlkRuntimeBatchMetadata)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ReferenceScoreInputs:
    logits: Any
    target_ids: Any
    mask: Any | None = None
    runtime: RlkRuntimeBatchMetadata = field(default_factory=RlkRuntimeBatchMetadata)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class LinearLogpInputs:
    hidden: Any
    lm_head_weight: Any
    target_ids: Any
    bias: Any | None = None
    tp_group: Any | None = None
    vocab_start_index: int = 0
    global_vocab_size: int | None = None
    runtime: RlkRuntimeBatchMetadata = field(default_factory=RlkRuntimeBatchMetadata)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vocab_start_index", int(self.vocab_start_index))
        if self.global_vocab_size is not None:
            object.__setattr__(self, "global_vocab_size", int(self.global_vocab_size))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


def linear_logp_inputs_from_vime(
    *,
    hidden: Any,
    lm_head_weight: Any,
    target_ids: Any,
    bias: Any | None = None,
    batch: Mapping[str, Any] | None = None,
    tp_group: Any | None = None,
    vocab_start_index: int = 0,
    global_vocab_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LinearLogpInputs:
    """Create a ``linear_logp`` call payload from vime-owned runtime objects."""

    return LinearLogpInputs(
        hidden=hidden,
        lm_head_weight=lm_head_weight,
        target_ids=target_ids,
        bias=bias,
        tp_group=tp_group,
        vocab_start_index=vocab_start_index,
        global_vocab_size=global_vocab_size,
        runtime=runtime_batch_metadata_from_vime_batch(batch),
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class RlkCapability:
    op_name: str
    available: bool
    backend: str = "none"
    reason: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _immutable_mapping(self.provenance))


@dataclass(frozen=True)
class RlkOperatorContract:
    op_name: str
    policy: Mapping[str, Any]
    runtime: RlkRuntimeBatchMetadata | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _immutable_mapping(self.policy))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        object.__setattr__(self, "provenance", _immutable_mapping(self.provenance))


@dataclass(frozen=True)
class RlkOperatorDecision:
    op_name: str
    path: str
    backend: str = "none"
    reason: str | None = None
    elapsed_s: float = 0.0
    token_count: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _immutable_mapping(self.provenance))


@dataclass(frozen=True)
class RlkOperatorResult:
    value: Any
    decision: RlkOperatorDecision


@dataclass
class RlkOperatorTelemetry:
    capability_queries: dict[str, int] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    fallback_counts: dict[str, int] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)
    dispatch_elapsed_s: dict[str, float] = field(default_factory=dict)
    last_decision: RlkOperatorDecision | None = None

    def record_capability_query(self, op_name: str) -> None:
        self.capability_queries[op_name] = self.capability_queries.get(op_name, 0) + 1

    def record_decision(self, decision: RlkOperatorDecision) -> None:
        op_name = decision.op_name
        self.last_decision = decision
        if decision.path in {"fast", "reference", "mock"}:
            self.call_counts[op_name] = self.call_counts.get(op_name, 0) + 1
        if decision.path in {"fallback", "disabled", "unsupported"}:
            self.fallback_counts[op_name] = self.fallback_counts.get(op_name, 0) + 1
        if decision.token_count is not None:
            self.token_counts[op_name] = self.token_counts.get(op_name, 0) + int(decision.token_count)
        self.dispatch_elapsed_s[op_name] = self.dispatch_elapsed_s.get(op_name, 0.0) + float(decision.elapsed_s)

    def snapshot(self) -> dict[str, Any]:
        return {
            "capability_queries": dict(self.capability_queries),
            "call_counts": dict(self.call_counts),
            "fallback_counts": dict(self.fallback_counts),
            "token_counts": dict(self.token_counts),
            "dispatch_elapsed_s": dict(self.dispatch_elapsed_s),
            "last_decision": None if self.last_decision is None else self.last_decision,
        }


class RlkOperatorUnavailable(RuntimeError):
    """Raised when strict vime policy requires an unavailable RL-Kernel op."""


@runtime_checkable
class RlkOperatorAdapter(Protocol):
    policy: RlkPolicyContext
    telemetry: RlkOperatorTelemetry

    def capability(self, op_name: str, *, metadata: Mapping[str, Any] | None = None) -> RlkCapability: ...

    def contract(
        self,
        op_name: str,
        *,
        runtime: RlkRuntimeBatchMetadata | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RlkOperatorContract: ...

    def selected_logprobs(self, inputs: SelectedLogprobInputs) -> RlkOperatorResult: ...

    def reference_logprobs(self, inputs: ReferenceScoreInputs) -> RlkOperatorResult: ...

    def linear_logp(self, inputs: LinearLogpInputs) -> RlkOperatorResult: ...

    def provenance(self) -> Mapping[str, Any]: ...


class NoOpRlkOperatorAdapter:
    """Disabled adapter that keeps native vime execution unchanged."""

    name = "noop"

    def __init__(self, policy: RlkPolicyContext | None = None) -> None:
        self.policy = policy or RlkPolicyContext()
        self.telemetry = RlkOperatorTelemetry()

    def capability(self, op_name: str, *, metadata: Mapping[str, Any] | None = None) -> RlkCapability:
        self.telemetry.record_capability_query(op_name)
        reason = "RL-Kernel operator path is disabled by vime policy."
        return RlkCapability(op_name=op_name, available=False, reason=reason, provenance=metadata or {})

    def contract(
        self,
        op_name: str,
        *,
        runtime: RlkRuntimeBatchMetadata | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RlkOperatorContract:
        return RlkOperatorContract(
            op_name=op_name,
            policy=_policy_payload(self.policy),
            runtime=runtime,
            metadata=metadata or {},
            provenance=self.provenance(),
        )

    def _disabled_result(self, op_name: str, token_count: int | None = None) -> RlkOperatorResult:
        decision = RlkOperatorDecision(
            op_name=op_name,
            path="disabled",
            reason="RL-Kernel operator path is disabled by vime policy.",
            token_count=token_count,
            provenance=self.provenance(),
        )
        self.telemetry.record_decision(decision)
        return RlkOperatorResult(value=None, decision=decision)

    def selected_logprobs(self, inputs: SelectedLogprobInputs) -> RlkOperatorResult:
        return self._disabled_result(RLK_OP_SELECTED_LOGPROBS, _num_tokens_from_targets(inputs.target_ids))

    def reference_logprobs(self, inputs: ReferenceScoreInputs) -> RlkOperatorResult:
        return self._disabled_result(RLK_OP_REFERENCE_LOGPROBS, _num_tokens_from_targets(inputs.target_ids))

    def linear_logp(self, inputs: LinearLogpInputs) -> RlkOperatorResult:
        return self._disabled_result(RLK_OP_LINEAR_LOGP, _num_tokens_from_targets(inputs.target_ids))

    def provenance(self) -> Mapping[str, Any]:
        return _immutable_mapping(
            {
                "adapter": self.name,
                "fast": self.policy.fast,
                "consistency": self.policy.consistency,
                "enabled_ops": self.policy.enabled_ops,
            }
        )


class RlkRegistryOperatorAdapter:
    """Optional adapter backed by RL-Kernel's public kernel registry."""

    name = "registry"

    def __init__(self, policy: RlkPolicyContext) -> None:
        self.policy = policy
        self.telemetry = RlkOperatorTelemetry()
        self._ops: dict[str, Any] = {}

    def _registry(self) -> Any:
        module = importlib.import_module("rl_engine.kernels.registry")
        return module.kernel_registry

    def _get_op(self, op_name: str) -> Any:
        if op_name not in self._ops:
            registry_op_name = RLK_OP_SELECTED_LOGPROBS if op_name == RLK_OP_REFERENCE_LOGPROBS else op_name
            self._ops[op_name] = self._registry().get_op(registry_op_name)
        return self._ops[op_name]

    def capability(self, op_name: str, *, metadata: Mapping[str, Any] | None = None) -> RlkCapability:
        self.telemetry.record_capability_query(op_name)
        if not self.policy.operator_enabled(op_name):
            return RlkCapability(
                op_name=op_name,
                available=False,
                reason=f"{op_name!r} is not enabled by vime RL-Kernel policy.",
                provenance=metadata or {},
            )
        try:
            op = self._get_op(op_name)
        except Exception as exc:
            return RlkCapability(
                op_name=op_name,
                available=False,
                reason=f"RL-Kernel registry could not provide {op_name!r}: {exc}",
                provenance=metadata or {},
            )
        return RlkCapability(
            op_name=op_name,
            available=True,
            backend=type(op).__name__,
            provenance={"adapter": self.name, **dict(metadata or {})},
        )

    def contract(
        self,
        op_name: str,
        *,
        runtime: RlkRuntimeBatchMetadata | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RlkOperatorContract:
        return RlkOperatorContract(
            op_name=op_name,
            policy=_policy_payload(self.policy),
            runtime=runtime,
            metadata=metadata or {},
            provenance=self.provenance(),
        )

    def _unsupported_result_or_raise(
        self,
        op_name: str,
        capability: RlkCapability,
        *,
        token_count: int | None = None,
    ) -> RlkOperatorResult:
        path = "unsupported" if self.policy.strict_fast else "fallback"
        decision = RlkOperatorDecision(
            op_name=op_name,
            path=path,
            backend=capability.backend,
            reason=capability.reason,
            token_count=token_count,
            provenance=self.provenance(),
        )
        self.telemetry.record_decision(decision)
        if self.policy.strict_fast:
            raise RlkOperatorUnavailable(capability.reason or f"RL-Kernel op {op_name!r} is unavailable.")
        return RlkOperatorResult(value=None, decision=decision)

    def _call(self, op_name: str, token_count: int | None, fn: Callable[[Any], Any]) -> RlkOperatorResult:
        capability = self.capability(op_name)
        if not capability.available:
            return self._unsupported_result_or_raise(op_name, capability, token_count=token_count)
        op = self._get_op(op_name)
        start = time.perf_counter()
        try:
            value = fn(op)
        except Exception as exc:
            capability = RlkCapability(
                op_name=op_name,
                available=False,
                backend=type(op).__name__,
                reason=f"RL-Kernel op {op_name!r} failed during execution: {exc}",
            )
            return self._unsupported_result_or_raise(op_name, capability, token_count=token_count)
        elapsed_s = time.perf_counter() - start
        backend_id = _op_text_attr(op, "backend_id", "backend_name", "name")
        contract_id = _op_text_attr(op, "contract_id", "numeric_contract_id")
        provenance = dict(self.provenance())
        if backend_id is not None:
            provenance["backend_id"] = backend_id
        if contract_id is not None:
            provenance["contract_id"] = contract_id
        decision = RlkOperatorDecision(
            op_name=op_name,
            path="fast",
            backend=type(op).__name__,
            elapsed_s=elapsed_s,
            token_count=token_count,
            provenance=provenance,
        )
        self.telemetry.record_decision(decision)
        return RlkOperatorResult(value=value, decision=decision)

    def selected_logprobs(self, inputs: SelectedLogprobInputs) -> RlkOperatorResult:
        if inputs.temperature != 1.0:
            capability = RlkCapability(
                op_name=RLK_OP_SELECTED_LOGPROBS,
                available=False,
                reason="selected_logprobs adapter path does not own temperature replay yet.",
            )
            return self._unsupported_result_or_raise(
                RLK_OP_SELECTED_LOGPROBS,
                capability,
                token_count=_num_tokens_from_targets(inputs.target_ids),
            )

        def invoke(op: Any) -> Any:
            value = op(inputs.logits, inputs.target_ids)
            return _mask_inactive_values(value, inputs.mask)

        return self._call(RLK_OP_SELECTED_LOGPROBS, _num_tokens_from_targets(inputs.target_ids), invoke)

    def reference_logprobs(self, inputs: ReferenceScoreInputs) -> RlkOperatorResult:
        def invoke(op: Any) -> Any:
            value = op(inputs.logits, inputs.target_ids)
            return _mask_inactive_values(value, inputs.mask)

        return self._call(RLK_OP_REFERENCE_LOGPROBS, _num_tokens_from_targets(inputs.target_ids), invoke)

    def linear_logp(self, inputs: LinearLogpInputs) -> RlkOperatorResult:
        def invoke(op: Any) -> Any:
            return op(
                inputs.hidden,
                inputs.lm_head_weight,
                inputs.target_ids,
                inputs.bias,
                tp_group=inputs.tp_group,
                vocab_start_index=inputs.vocab_start_index,
                global_vocab_size=inputs.global_vocab_size,
            )

        return self._call(RLK_OP_LINEAR_LOGP, _num_tokens_from_targets(inputs.target_ids), invoke)

    def provenance(self) -> Mapping[str, Any]:
        return _immutable_mapping(
            {
                "adapter": self.name,
                "fast": self.policy.fast,
                "consistency": self.policy.consistency,
                "enabled_ops": self.policy.enabled_ops,
                "boundary": "vime.backends.rl_kernel_utils",
            }
        )


class MockRlkOperatorAdapter:
    """Test adapter that satisfies ``RlkOperatorAdapter`` without RL-Kernel."""

    name = "mock"

    def __init__(
        self,
        policy: RlkPolicyContext | None = None,
        *,
        available_ops: tuple[str, ...] = (RLK_ALL_OPERATORS,),
        handlers: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        self.policy = policy or RlkPolicyContext(fast="auto", enabled_ops=available_ops)
        self.available_ops = set(available_ops)
        self.handlers = dict(handlers or {})
        self.telemetry = RlkOperatorTelemetry()

    def capability(self, op_name: str, *, metadata: Mapping[str, Any] | None = None) -> RlkCapability:
        self.telemetry.record_capability_query(op_name)
        enabled = self.policy.operator_enabled(op_name)
        available = enabled and self._mock_op_available(op_name)
        return RlkCapability(
            op_name=op_name,
            available=available,
            backend=self.name if available else "none",
            reason=None if available else f"{op_name!r} is unsupported by the mock adapter.",
            provenance=metadata or {},
        )

    def _mock_op_available(self, op_name: str) -> bool:
        return (
            RLK_ALL_OPERATORS in self.available_ops
            or op_name in self.available_ops
            or (op_name == RLK_OP_REFERENCE_LOGPROBS and RLK_OP_SELECTED_LOGPROBS in self.available_ops)
        )

    def contract(
        self,
        op_name: str,
        *,
        runtime: RlkRuntimeBatchMetadata | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RlkOperatorContract:
        return RlkOperatorContract(
            op_name=op_name,
            policy=_policy_payload(self.policy),
            runtime=runtime,
            metadata=metadata or {},
            provenance=self.provenance(),
        )

    def _call(self, op_name: str, payload: Any, token_count: int | None) -> RlkOperatorResult:
        capability = self.capability(op_name)
        if not capability.available:
            decision = RlkOperatorDecision(
                op_name=op_name,
                path="unsupported",
                backend="none",
                reason=capability.reason,
                token_count=token_count,
                provenance=self.provenance(),
            )
            self.telemetry.record_decision(decision)
            return RlkOperatorResult(value=None, decision=decision)
        handler = self.handlers.get(op_name) or self.handlers.get(_registry_op_name(op_name), lambda value: value)
        value = handler(payload)
        decision = RlkOperatorDecision(
            op_name=op_name,
            path="mock",
            backend=self.name,
            token_count=token_count,
            provenance=self.provenance(),
        )
        self.telemetry.record_decision(decision)
        return RlkOperatorResult(value=value, decision=decision)

    def selected_logprobs(self, inputs: SelectedLogprobInputs) -> RlkOperatorResult:
        return self._call(RLK_OP_SELECTED_LOGPROBS, inputs, _num_tokens_from_targets(inputs.target_ids))

    def reference_logprobs(self, inputs: ReferenceScoreInputs) -> RlkOperatorResult:
        return self._call(RLK_OP_REFERENCE_LOGPROBS, inputs, _num_tokens_from_targets(inputs.target_ids))

    def linear_logp(self, inputs: LinearLogpInputs) -> RlkOperatorResult:
        return self._call(RLK_OP_LINEAR_LOGP, inputs, _num_tokens_from_targets(inputs.target_ids))

    def provenance(self) -> Mapping[str, Any]:
        return _immutable_mapping(
            {
                "adapter": self.name,
                "fast": self.policy.fast,
                "consistency": self.policy.consistency,
                "enabled_ops": self.policy.enabled_ops,
                "available_ops": tuple(sorted(self.available_ops)),
            }
        )


def build_rlk_operator_adapter(
    policy: RlkPolicyContext | None = None,
    *,
    args: Any | None = None,
    backend: str = "auto",
) -> RlkOperatorAdapter:
    """Small construction point for selecting a concrete RL-Kernel adapter."""

    if policy is None:
        policy = rlk_policy_context_from_args(args) if args is not None else RlkPolicyContext()
    if policy.fast == "off" or not policy.enabled_ops:
        return NoOpRlkOperatorAdapter(policy)

    normalized_backend = backend.strip().lower().replace("-", "_")
    if normalized_backend in {"auto", "registry", "rl_kernel", "rlk"}:
        return RlkRegistryOperatorAdapter(policy)
    if normalized_backend in {"none", "noop", "no_op", "disabled"}:
        return NoOpRlkOperatorAdapter(policy)
    raise ValueError(f"Unknown RL-Kernel operator adapter backend: {backend!r}.")


def _num_tokens_from_targets(target_ids: Any) -> int | None:
    if target_ids is None:
        return None
    if hasattr(target_ids, "numel"):
        return int(target_ids.numel())
    try:
        return len(target_ids)
    except TypeError:
        return None


def _policy_payload(policy: RlkPolicyContext) -> dict[str, Any]:
    return {
        "fast": policy.fast,
        "consistency": policy.consistency,
        "enabled_ops": policy.enabled_ops,
        "metadata": dict(policy.metadata),
    }


def _registry_op_name(op_name: str) -> str:
    return RLK_OP_SELECTED_LOGPROBS if op_name == RLK_OP_REFERENCE_LOGPROBS else op_name


def _op_text_attr(op: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(op, name, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if value is not None:
            return str(value)
    return None


def _mask_inactive_values(value: Any, mask: Any | None) -> Any:
    if mask is None:
        return value
    bool_mask = mask.bool() if hasattr(mask, "bool") else mask
    return value.masked_fill(~bool_mask, 0.0)
