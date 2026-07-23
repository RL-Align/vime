import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

logger = logging.getLogger(__name__)

RLK_DECISION_EVENT = "rl_kernel.execution_decision"
_NATIVE_BACKEND_ID = "vime.native"

ExecutionDecisionKind = Literal[
    "native",
    "audit-only",
    "optimized",
    "strict-fast",
    "strict-reference",
    "fallback-native",
    "strict-failure",
    "contract-match",
    "audit-warning",
]


@dataclass(frozen=True)
class FallbackReason:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NumericContract:
    contract_id: str
    accumulation_dtype: str | None = None
    reduction_order: str | None = None
    downcast_point: str | None = None
    sharding_rule: str | None = None
    merge_semantics: str | None = None
    quantization_policy: str | None = None
    backend_id: str | None = None
    tolerance_by_dtype: dict[str, Any] = field(default_factory=dict)

    def tolerance_for_dtype(self, dtype: str) -> Any:
        try:
            return self.tolerance_by_dtype[dtype]
        except KeyError as exc:
            raise KeyError(f"contract {self.contract_id!r} has no tolerance for dtype {dtype!r}") from exc


@dataclass(frozen=True)
class LogprobContractMetadata:
    source: str
    contract_id: str | None
    dtype: str | None = None
    backend_id: str | None = None


@dataclass(frozen=True)
class BackendCapability:
    operator: str
    backend_id: str
    implementation_kind: str
    dtypes: tuple[str, ...] = ()
    hardware_targets: tuple[str, ...] = ()
    autograd_modes: tuple[str, ...] = ()
    parallel_modes: tuple[str, ...] = ()
    shape_ranges: dict[str, Any] = field(default_factory=dict)
    deterministic: bool = False
    batch_invariant: bool = False
    strict_fast_eligible: bool = False
    production: bool = False
    fallback_behavior: str | None = None
    runtime_fingerprint: str | None = None
    build_fingerprint: str | None = None
    config_lifecycle: str | None = None
    numeric_contract: NumericContract | None = None

    @property
    def contract_id(self) -> str | None:
        return None if self.numeric_contract is None else self.numeric_contract.contract_id

    @property
    def is_reference(self) -> bool:
        return self.implementation_kind == "reference"

    def supports_dtype(self, dtype: str | None) -> bool:
        return dtype is None or not self.dtypes or dtype in self.dtypes


@dataclass(frozen=True)
class RlKernelCapabilities:
    available: bool
    backends: tuple[BackendCapability, ...] = ()
    reason: str | None = None
    runtime_fingerprint: str | None = None
    build_fingerprint: str | None = None

    def matching_backends(self, operator: str, dtype: str | None = None) -> tuple[BackendCapability, ...]:
        return tuple(
            backend for backend in self.backends if backend.operator == operator and backend.supports_dtype(dtype)
        )

    def tolerance_for_contract(self, contract_id: str, dtype: str) -> Any:
        for backend in self.backends:
            contract = backend.numeric_contract
            if contract is not None and contract.contract_id == contract_id:
                return contract.tolerance_for_dtype(dtype)
        raise KeyError(f"unknown RL-Kernel numeric contract {contract_id!r}")


@dataclass(frozen=True)
class CapabilityQueryResult:
    capabilities: RlKernelCapabilities
    fallback_reason: FallbackReason | None = None


@dataclass(frozen=True)
class ExecutionDecision:
    operator: str
    stage: str
    requested_mode: str
    requested_backend: str | None
    actual_backend: str | None
    decision: ExecutionDecisionKind
    fallback: bool = False
    fallback_reason: FallbackReason | None = None
    capability_backend_id: str | None = None
    contract_id: str | None = None
    dtype: str | None = None
    parallel_context: dict[str, Any] = field(default_factory=dict)
    strict_eligible: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_log_record(self) -> dict[str, Any]:
        return {
            "event": RLK_DECISION_EVENT,
            "operator": self.operator,
            "stage": self.stage,
            "requested_mode": self.requested_mode,
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "decision": self.decision,
            "fallback": self.fallback,
            "fallback_reason": None if self.fallback_reason is None else asdict(self.fallback_reason),
            "capability_backend_id": self.capability_backend_id,
            "contract_id": self.contract_id,
            "dtype": self.dtype,
            "parallel_context": dict(self.parallel_context),
            "strict_eligible": self.strict_eligible,
            "details": dict(self.details),
        }


def _native_contract_id(operator: str) -> str:
    return f"{_NATIVE_BACKEND_ID}.{operator}"


def _normalize_caps(value: Any) -> RlKernelCapabilities:
    if isinstance(value, RlKernelCapabilities):
        return value
    if isinstance(value, dict):
        backends = tuple(_normalize_backend(backend) for backend in value.get("backends", ()))
        return RlKernelCapabilities(
            available=bool(value.get("available", True)),
            backends=backends,
            reason=value.get("reason"),
            runtime_fingerprint=value.get("runtime_fingerprint"),
            build_fingerprint=value.get("build_fingerprint"),
        )
    raise TypeError(f"RL-Kernel capability provider returned unsupported value {type(value)!r}")


def _normalize_backend(value: Any) -> BackendCapability:
    if isinstance(value, BackendCapability):
        return value
    if isinstance(value, dict):
        data = _filter_dataclass_fields(value, BackendCapability)
        contract = data.get("numeric_contract")
        if isinstance(contract, dict):
            data["numeric_contract"] = NumericContract(**_filter_dataclass_fields(contract, NumericContract))
        for key in ("dtypes", "hardware_targets", "autograd_modes", "parallel_modes"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return BackendCapability(**data)
    raise TypeError(f"unsupported RL-Kernel backend descriptor {type(value)!r}")


def _filter_dataclass_fields(data: dict[str, Any], target: type[Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(target)}
    return {key: value for key, value in data.items() if key in allowed}


def query_rl_kernel_capabilities(provider: Any = None) -> CapabilityQueryResult:
    if provider is None:
        reason = FallbackReason(
            code="capability_provider_missing",
            message="RL-Kernel capability provider was not configured.",
        )
        return CapabilityQueryResult(
            capabilities=RlKernelCapabilities(available=False, reason=reason.code),
            fallback_reason=reason,
        )

    try:
        if hasattr(provider, "get_vime_capabilities"):
            raw_capabilities = provider.get_vime_capabilities()
        elif callable(provider):
            raw_capabilities = provider()
        else:
            raw_capabilities = provider
        capabilities = _normalize_caps(raw_capabilities)
    except Exception as exc:
        logger.debug("Provider query failed", exc_info=True)
        reason = FallbackReason(
            code="capability_query_failed",
            message="Failed to query RL-Kernel capabilities.",
            details={"error": repr(exc)},
        )
        return CapabilityQueryResult(
            capabilities=RlKernelCapabilities(available=False, reason=reason.code),
            fallback_reason=reason,
        )

    fallback_reason = None
    if not capabilities.available:
        fallback_reason = FallbackReason(
            code=capabilities.reason or "rl_kernel_unavailable",
            message="RL-Kernel reported that it is unavailable.",
        )
    return CapabilityQueryResult(capabilities=capabilities, fallback_reason=fallback_reason)


def select_execution_decision(
    *,
    operator: str,
    stage: str,
    requested_fast: str = "off",
    requested_consistency: str = "off",
    capabilities: RlKernelCapabilities | None = None,
    requested_backend: str | None = None,
    dtype: str | None = None,
    parallel_context: dict[str, Any] | None = None,
) -> ExecutionDecision:
    requested_mode = f"fast={requested_fast},consistency={requested_consistency}"
    parallel_context = parallel_context or {}

    if requested_fast == "off":
        decision = "audit-only" if requested_consistency in {"audit", "strict"} else "native"
        return ExecutionDecision(
            operator=operator,
            stage=stage,
            requested_mode=requested_mode,
            requested_backend=requested_backend,
            actual_backend=_NATIVE_BACKEND_ID,
            decision=decision,
            contract_id=_native_contract_id(operator),
            dtype=dtype,
            parallel_context=parallel_context,
        )

    if capabilities is None or not capabilities.available:
        reason = FallbackReason(
            code="capability_data_missing",
            message="RL-Kernel capability data is unavailable.",
        )
        if requested_fast == "strict":
            return _strict_failure_decision(
                operator=operator,
                stage=stage,
                requested_mode=requested_mode,
                requested_backend=requested_backend,
                dtype=dtype,
                parallel_context=parallel_context,
                reason=reason,
            )
        return _native_fallback_decision(
            operator=operator,
            stage=stage,
            requested_mode=requested_mode,
            requested_backend=requested_backend,
            dtype=dtype,
            parallel_context=parallel_context,
            reason=reason,
        )

    candidates = _filter_requested_backend(capabilities.matching_backends(operator, dtype), requested_backend)
    if requested_consistency == "strict":
        strict_backend = _first_strict_fast_backend(candidates)
        if strict_backend is not None:
            return _backend_decision(
                operator,
                stage,
                requested_mode,
                requested_backend,
                dtype,
                parallel_context,
                strict_backend,
                "strict-fast",
            )

        reference_backend = _first_reference_backend(candidates)
        if requested_fast == "auto" and reference_backend is not None:
            return _backend_decision(
                operator,
                stage,
                requested_mode,
                requested_backend,
                dtype,
                parallel_context,
                reference_backend,
                "strict-reference",
            )

        reason = FallbackReason(
            code="strict_backend_unavailable",
            message="No RL-Kernel backend satisfies the requested strict consistency policy.",
            details={"operator": operator, "requested_backend": requested_backend, "dtype": dtype},
        )
        return _strict_failure_decision(
            operator=operator,
            stage=stage,
            requested_mode=requested_mode,
            requested_backend=requested_backend,
            dtype=dtype,
            parallel_context=parallel_context,
            reason=reason,
        )

    backend = _first_non_reference_backend(candidates)
    if backend is not None:
        return _backend_decision(
            operator, stage, requested_mode, requested_backend, dtype, parallel_context, backend, "optimized"
        )

    reason = FallbackReason(
        code="backend_unavailable",
        message="No RL-Kernel backend matches the requested operator, backend, and dtype.",
        details={"operator": operator, "requested_backend": requested_backend, "dtype": dtype},
    )
    if requested_fast == "strict":
        return _strict_failure_decision(
            operator=operator,
            stage=stage,
            requested_mode=requested_mode,
            requested_backend=requested_backend,
            dtype=dtype,
            parallel_context=parallel_context,
            reason=reason,
        )
    return _native_fallback_decision(
        operator=operator,
        stage=stage,
        requested_mode=requested_mode,
        requested_backend=requested_backend,
        dtype=dtype,
        parallel_context=parallel_context,
        reason=reason,
    )


def _filter_requested_backend(
    candidates: tuple[BackendCapability, ...],
    requested_backend: str | None,
) -> tuple[BackendCapability, ...]:
    if requested_backend is None:
        return candidates
    return tuple(backend for backend in candidates if backend.backend_id == requested_backend)


def _first_strict_fast_backend(candidates: tuple[BackendCapability, ...]) -> BackendCapability | None:
    return next((backend for backend in candidates if backend.strict_fast_eligible and not backend.is_reference), None)


def _first_reference_backend(candidates: tuple[BackendCapability, ...]) -> BackendCapability | None:
    return next((backend for backend in candidates if backend.is_reference), None)


def _first_non_reference_backend(candidates: tuple[BackendCapability, ...]) -> BackendCapability | None:
    return next((backend for backend in candidates if not backend.is_reference), None)


def _backend_decision(
    operator: str,
    stage: str,
    requested_mode: str,
    requested_backend: str | None,
    dtype: str | None,
    parallel_context: dict[str, Any],
    backend: BackendCapability,
    decision: ExecutionDecisionKind,
) -> ExecutionDecision:
    return ExecutionDecision(
        operator=operator,
        stage=stage,
        requested_mode=requested_mode,
        requested_backend=requested_backend,
        actual_backend=backend.backend_id,
        decision=decision,
        capability_backend_id=backend.backend_id,
        contract_id=backend.contract_id,
        dtype=dtype,
        parallel_context=parallel_context,
        strict_eligible=backend.strict_fast_eligible or backend.is_reference,
    )


def _native_fallback_decision(
    *,
    operator: str,
    stage: str,
    requested_mode: str,
    requested_backend: str | None,
    dtype: str | None,
    parallel_context: dict[str, Any],
    reason: FallbackReason,
) -> ExecutionDecision:
    return ExecutionDecision(
        operator=operator,
        stage=stage,
        requested_mode=requested_mode,
        requested_backend=requested_backend,
        actual_backend=_NATIVE_BACKEND_ID,
        decision="fallback-native",
        fallback=True,
        fallback_reason=reason,
        contract_id=_native_contract_id(operator),
        dtype=dtype,
        parallel_context=parallel_context,
    )


def _strict_failure_decision(
    *,
    operator: str,
    stage: str,
    requested_mode: str,
    requested_backend: str | None,
    dtype: str | None,
    parallel_context: dict[str, Any],
    reason: FallbackReason,
) -> ExecutionDecision:
    return ExecutionDecision(
        operator=operator,
        stage=stage,
        requested_mode=requested_mode,
        requested_backend=requested_backend,
        actual_backend=None,
        decision="strict-failure",
        fallback=False,
        fallback_reason=reason,
        dtype=dtype,
        parallel_context=parallel_context,
    )


def build_logprob_contract_decision(
    *,
    rollout_contract: LogprobContractMetadata | None,
    recomputed_contract: LogprobContractMetadata | None,
    strict: bool,
    operator: str = "logp",
    stage: str = "audit",
) -> ExecutionDecision:
    if rollout_contract is None or recomputed_contract is None:
        reason = FallbackReason(
            code="contract_id_missing",
            message="Missing rollout or recomputed logprob contract ID.",
            details={
                "rollout_contract_id": None if rollout_contract is None else rollout_contract.contract_id,
                "recomputed_contract_id": None if recomputed_contract is None else recomputed_contract.contract_id,
            },
        )
        return _contract_problem_decision(operator, stage, strict, reason)

    if rollout_contract.contract_id is None or recomputed_contract.contract_id is None:
        reason = FallbackReason(
            code="contract_id_missing",
            message="Missing rollout or recomputed logprob contract ID.",
            details={
                "rollout_contract_id": rollout_contract.contract_id,
                "recomputed_contract_id": recomputed_contract.contract_id,
            },
        )
        return _contract_problem_decision(operator, stage, strict, reason)

    if rollout_contract.contract_id != recomputed_contract.contract_id:
        reason = FallbackReason(
            code="contract_id_mismatch",
            message="Rollout and recomputed logprob contract IDs do not match.",
            details={
                "rollout_contract_id": rollout_contract.contract_id,
                "recomputed_contract_id": recomputed_contract.contract_id,
            },
        )
        return _contract_problem_decision(operator, stage, strict, reason)

    return ExecutionDecision(
        operator=operator,
        stage=stage,
        requested_mode="contract-check",
        requested_backend=rollout_contract.backend_id,
        actual_backend=recomputed_contract.backend_id,
        decision="contract-match",
        contract_id=rollout_contract.contract_id,
        dtype=recomputed_contract.dtype or rollout_contract.dtype,
        strict_eligible=True,
        details={
            "rollout_contract_id": rollout_contract.contract_id,
            "recomputed_contract_id": recomputed_contract.contract_id,
        },
    )


def _contract_problem_decision(
    operator: str,
    stage: str,
    strict: bool,
    reason: FallbackReason,
) -> ExecutionDecision:
    decision: ExecutionDecisionKind = "strict-failure" if strict else "audit-warning"
    return ExecutionDecision(
        operator=operator,
        stage=stage,
        requested_mode="contract-check",
        requested_backend=None,
        actual_backend=None if strict else _NATIVE_BACKEND_ID,
        decision=decision,
        fallback=False,
        fallback_reason=reason,
        details=reason.details,
    )


def execution_decision_sample_value(
    *,
    seed: int | str = 0,
    rank: int | None = None,
    key: str = "",
) -> float:
    """Return a deterministic sample value in [0, 1) that is partitioned by rank."""
    rank = _current_process_rank() if rank is None else rank
    payload = f"{seed}\0{rank}\0{key}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _current_process_rank() -> int:
    for env_name in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            logger.debug("Ignoring non-integer %s=%r while sampling RL-Kernel decision logs.", env_name, value)
    return 0


def emit_execution_decision(
    decision: ExecutionDecision,
    *,
    log: logging.Logger | None = None,
    enabled: bool = True,
    sample_rate: float = 1.0,
    random_value: float | None = None,
    sample_seed: int | str = 0,
    sample_rank: int | None = None,
    sample_key: str | None = None,
) -> dict[str, Any]:
    record = decision.to_log_record()
    if random_value is None:
        random_value = execution_decision_sample_value(
            seed=sample_seed,
            rank=sample_rank,
            key=sample_key or json.dumps(record, sort_keys=True, default=str),
        )
    if not enabled or sample_rate <= 0 or random_value >= sample_rate:
        return record

    (log or logger).info("%s %s", RLK_DECISION_EVENT, json.dumps(record, sort_keys=True))
    return record
