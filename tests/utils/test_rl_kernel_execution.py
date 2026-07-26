import json
import logging
import sys

import pytest

from vime.backends.rl_kernel_utils import (
    BackendCapability,
    FallbackReason,
    LogprobContractMetadata,
    NumericContract,
    RlKernelCapabilities,
    build_logprob_contract_decision,
    emit_execution_decision,
    execution_decision_sample_value,
    query_rl_kernel_capabilities,
    select_execution_decision,
)
from vime.backends.rl_kernel_utils.execution import RLK_DECISION_EVENT


def _drop_rl_engine_modules() -> None:
    for name in list(sys.modules):
        if name == "rl_engine" or name.startswith("rl_engine."):
            sys.modules.pop(name)


def _contract(contract_id: str = "rlk.linear_logp.fp32") -> NumericContract:
    return NumericContract(
        contract_id=contract_id,
        accumulation_dtype="fp32",
        reduction_order="vocab-shard-local-then-global",
        downcast_point="after-selected-logprob",
        sharding_rule="tp-vocab",
        merge_semantics="selected-token",
        quantization_policy="none",
        backend_id="rlk.linear_logp.fast",
        tolerance_by_dtype={"bf16": {"source": "rl-kernel", "dtype": "bf16"}},
    )


def _backend(
    backend_id: str = "rlk.linear_logp.fast",
    *,
    operator: str = "linear_logp",
    implementation_kind: str = "optimized",
    strict_fast_eligible: bool = True,
    dtypes: tuple[str, ...] = ("bf16",),
    contract_id: str = "rlk.linear_logp.fp32",
) -> BackendCapability:
    return BackendCapability(
        operator=operator,
        backend_id=backend_id,
        implementation_kind=implementation_kind,
        dtypes=dtypes,
        hardware_targets=("cuda",),
        autograd_modes=("full-gradient",),
        parallel_modes=("tp",),
        deterministic=True,
        batch_invariant=True,
        strict_fast_eligible=strict_fast_eligible,
        runtime_fingerprint="runtime-1",
        build_fingerprint="build-1",
        config_lifecycle="process-start",
        numeric_contract=_contract(contract_id),
    )


def _caps(*backends: BackendCapability) -> RlKernelCapabilities:
    return RlKernelCapabilities(available=True, backends=tuple(backends))


@pytest.mark.unit
def test_execution_helpers_do_not_import_rl_engine():
    _drop_rl_engine_modules()

    result = query_rl_kernel_capabilities()

    assert result.capabilities.available is False
    assert not any(name == "rl_engine" or name.startswith("rl_engine.") for name in sys.modules)


@pytest.mark.unit
def test_off_off_selects_native_without_fallback():
    decision = select_execution_decision(operator="linear_logp", stage="train_logprob", dtype="bf16")

    assert decision.decision == "native"
    assert decision.actual_backend == "vime.native"
    assert decision.fallback is False
    assert decision.contract_id == "vime.native.linear_logp"


@pytest.mark.unit
@pytest.mark.parametrize("consistency_mode", ["audit", "strict"])
def test_fast_off_consistency_modes_are_audit_only(consistency_mode):
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="off",
        requested_consistency=consistency_mode,
    )

    assert decision.decision == "audit-only"
    assert decision.actual_backend == "vime.native"
    assert decision.fallback is False


@pytest.mark.unit
def test_auto_missing_capabilities_falls_back_with_structured_reason():
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="auto",
        requested_consistency="off",
    )

    assert decision.decision == "fallback-native"
    assert decision.fallback is True
    assert decision.actual_backend == "vime.native"
    assert decision.fallback_reason == FallbackReason(
        code="capability_data_missing",
        message="RL-Kernel capability data is unavailable.",
    )


@pytest.mark.unit
def test_strict_missing_capabilities_returns_strict_failure():
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="strict",
        requested_consistency="off",
    )

    assert decision.decision == "strict-failure"
    assert decision.actual_backend is None
    assert decision.fallback is False
    assert decision.fallback_reason.code == "capability_data_missing"


@pytest.mark.unit
def test_auto_selects_matching_optimized_backend():
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="auto",
        capabilities=_caps(_backend()),
        requested_backend="rlk.linear_logp.fast",
        dtype="bf16",
        parallel_context={"tp": 2},
    )

    assert decision.decision == "optimized"
    assert decision.actual_backend == "rlk.linear_logp.fast"
    assert decision.capability_backend_id == "rlk.linear_logp.fast"
    assert decision.contract_id == "rlk.linear_logp.fp32"
    assert decision.parallel_context == {"tp": 2}


@pytest.mark.unit
def test_auto_strict_consistency_selects_strict_fast_backend():
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="auto",
        requested_consistency="strict",
        capabilities=_caps(_backend(strict_fast_eligible=True)),
        dtype="bf16",
    )

    assert decision.decision == "strict-fast"
    assert decision.actual_backend == "rlk.linear_logp.fast"
    assert decision.strict_eligible is True


@pytest.mark.unit
def test_auto_strict_consistency_uses_reference_when_no_strict_fast_backend():
    opportunistic = _backend(strict_fast_eligible=False)
    reference = _backend(
        "rlk.linear_logp.reference",
        implementation_kind="reference",
        strict_fast_eligible=False,
        contract_id="rlk.linear_logp.reference.fp32",
    )

    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="auto",
        requested_consistency="strict",
        capabilities=_caps(opportunistic, reference),
        dtype="bf16",
    )

    assert decision.decision == "strict-reference"
    assert decision.actual_backend == "rlk.linear_logp.reference"
    assert decision.strict_eligible is True


@pytest.mark.unit
def test_strict_fast_strict_consistency_does_not_silently_use_reference():
    reference = _backend(
        "rlk.linear_logp.reference",
        implementation_kind="reference",
        strict_fast_eligible=False,
        contract_id="rlk.linear_logp.reference.fp32",
    )

    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="strict",
        requested_consistency="strict",
        capabilities=_caps(reference),
        dtype="bf16",
    )

    assert decision.decision == "strict-failure"
    assert decision.actual_backend is None
    assert decision.fallback_reason.code == "strict_backend_unavailable"


@pytest.mark.unit
def test_requested_backend_mismatch_falls_back_in_auto_mode():
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="auto",
        capabilities=_caps(_backend("rlk.other")),
        requested_backend="rlk.missing",
        dtype="bf16",
    )

    assert decision.decision == "fallback-native"
    assert decision.fallback_reason.code == "backend_unavailable"
    assert decision.fallback_reason.details["requested_backend"] == "rlk.missing"


@pytest.mark.unit
def test_dtype_mismatch_returns_structured_backend_unavailable_reason():
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="auto",
        capabilities=_caps(_backend(dtypes=("fp16",))),
        dtype="bf16",
    )

    assert decision.decision == "fallback-native"
    assert decision.fallback_reason.code == "backend_unavailable"
    assert decision.fallback_reason.details["dtype"] == "bf16"


@pytest.mark.unit
def test_query_capabilities_without_provider_is_explicitly_unavailable():
    result = query_rl_kernel_capabilities()

    assert result.capabilities.available is False
    assert result.fallback_reason.code == "capability_provider_missing"


@pytest.mark.unit
def test_query_capabilities_normalizes_dict_provider():
    result = query_rl_kernel_capabilities(
        lambda: {
            "available": True,
            "runtime_fingerprint": "runtime-1",
            "backends": [
                {
                    "operator": "linear_logp",
                    "backend_id": "rlk.linear_logp.fast",
                    "implementation_kind": "optimized",
                    "dtypes": ["bf16"],
                    "strict_fast_eligible": True,
                    "numeric_contract": {
                        "contract_id": "rlk.linear_logp.fp32",
                        "tolerance_by_dtype": {"bf16": {"source": "provider"}},
                    },
                }
            ],
        }
    )

    assert result.capabilities.available is True
    assert result.capabilities.runtime_fingerprint == "runtime-1"
    assert result.capabilities.backends[0].backend_id == "rlk.linear_logp.fast"
    assert result.capabilities.backends[0].numeric_contract.tolerance_for_dtype("bf16") == {"source": "provider"}


@pytest.mark.unit
def test_query_capabilities_filters_unknown_backend_fields():
    result = query_rl_kernel_capabilities(
        lambda: {
            "available": True,
            "backends": [
                {
                    "operator": "linear_logp",
                    "backend_id": "rlk.linear_logp.fast",
                    "implementation_kind": "optimized",
                    "dtypes": ["bf16"],
                    "unknown_backend_key": "ignored",
                    "numeric_contract": {
                        "contract_id": "rlk.linear_logp.fp32",
                        "tolerance_by_dtype": {"bf16": {"source": "provider"}},
                        "unknown_contract_key": "ignored",
                    },
                }
            ],
        }
    )

    assert result.fallback_reason is None
    assert result.capabilities.available is True
    assert result.capabilities.backends[0].backend_id == "rlk.linear_logp.fast"
    assert result.capabilities.backends[0].dtypes == ("bf16",)
    assert result.capabilities.backends[0].numeric_contract.contract_id == "rlk.linear_logp.fp32"


@pytest.mark.unit
def test_query_capabilities_provider_failure_is_structured_and_debug_logged(caplog):
    def provider():
        raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="vime.backends.rl_kernel_utils.execution"):
        result = query_rl_kernel_capabilities(provider)

    assert result.capabilities.available is False
    assert result.fallback_reason.code == "capability_query_failed"
    assert "boom" in result.fallback_reason.details["error"]
    assert any(record.message == "Provider query failed" and record.exc_info is not None for record in caplog.records)


@pytest.mark.unit
def test_tolerance_lookup_uses_contract_metadata_without_hardcoded_thresholds():
    capabilities = _caps(_backend())

    assert capabilities.tolerance_for_contract("rlk.linear_logp.fp32", "bf16") == {
        "source": "rl-kernel",
        "dtype": "bf16",
    }


@pytest.mark.unit
def test_tolerance_lookup_rejects_unknown_contract():
    capabilities = _caps(_backend())

    with pytest.raises(KeyError, match="unknown RL-Kernel numeric contract"):
        capabilities.tolerance_for_contract("missing", "bf16")


@pytest.mark.unit
def test_logprob_contract_match_returns_reportable_contract_identity():
    decision = build_logprob_contract_decision(
        rollout_contract=LogprobContractMetadata(
            source="rollout",
            contract_id="rlk.linear_logp.fp32",
            dtype="bf16",
            backend_id="rlk.linear_logp.fast",
        ),
        recomputed_contract=LogprobContractMetadata(
            source="train",
            contract_id="rlk.linear_logp.fp32",
            dtype="bf16",
            backend_id="rlk.linear_logp.fast",
        ),
        strict=True,
    )

    assert decision.decision == "contract-match"
    assert decision.contract_id == "rlk.linear_logp.fp32"
    assert decision.strict_eligible is True
    assert decision.details == {
        "rollout_contract_id": "rlk.linear_logp.fp32",
        "recomputed_contract_id": "rlk.linear_logp.fp32",
    }


@pytest.mark.unit
@pytest.mark.parametrize("strict,expected_decision", [(False, "audit-warning"), (True, "strict-failure")])
def test_logprob_contract_mismatch_is_structured(strict, expected_decision):
    decision = build_logprob_contract_decision(
        rollout_contract=LogprobContractMetadata(source="rollout", contract_id="rollout.contract"),
        recomputed_contract=LogprobContractMetadata(source="train", contract_id="train.contract"),
        strict=strict,
    )

    assert decision.decision == expected_decision
    assert decision.fallback_reason.code == "contract_id_mismatch"
    assert decision.fallback_reason.details == {
        "rollout_contract_id": "rollout.contract",
        "recomputed_contract_id": "train.contract",
    }


@pytest.mark.unit
def test_logprob_contract_missing_fails_closed_in_strict_mode():
    decision = build_logprob_contract_decision(
        rollout_contract=LogprobContractMetadata(source="rollout", contract_id=None),
        recomputed_contract=LogprobContractMetadata(source="train", contract_id="train.contract"),
        strict=True,
    )

    assert decision.decision == "strict-failure"
    assert decision.fallback_reason.code == "contract_id_missing"


@pytest.mark.unit
def test_execution_decision_log_record_is_stable_and_json_serializable():
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="auto",
        capabilities=_caps(_backend()),
        dtype="bf16",
    )

    record = decision.to_log_record()
    encoded = json.dumps(record, sort_keys=True)

    assert json.loads(encoded)["event"] == RLK_DECISION_EVENT
    assert sorted(record) == [
        "actual_backend",
        "capability_backend_id",
        "contract_id",
        "decision",
        "details",
        "dtype",
        "event",
        "fallback",
        "fallback_reason",
        "operator",
        "parallel_context",
        "requested_backend",
        "requested_mode",
        "stage",
        "strict_eligible",
    ]


@pytest.mark.unit
def test_emit_execution_decision_logs_json_payload(caplog):
    decision = select_execution_decision(
        operator="linear_logp",
        stage="train_logprob",
        requested_fast="auto",
        capabilities=_caps(_backend()),
        dtype="bf16",
    )

    with caplog.at_level(logging.INFO, logger="vime.backends.rl_kernel_utils.execution"):
        emit_execution_decision(decision)

    assert RLK_DECISION_EVENT in caplog.text
    payload = caplog.records[0].message.split(RLK_DECISION_EVENT, 1)[1].strip()
    assert json.loads(payload)["actual_backend"] == "rlk.linear_logp.fast"


@pytest.mark.unit
def test_emit_execution_decision_can_be_disabled_or_sampled_out(caplog):
    decision = select_execution_decision(operator="linear_logp", stage="train_logprob")

    with caplog.at_level(logging.INFO, logger="vime.backends.rl_kernel_utils.execution"):
        record = emit_execution_decision(decision, enabled=False)
        sampled = emit_execution_decision(decision, sample_rate=0.5, random_value=0.75)

    assert record["decision"] == "native"
    assert sampled["decision"] == "native"
    assert caplog.records == []


@pytest.mark.unit
def test_execution_decision_sample_value_is_rank_specific_and_stable():
    rank0 = execution_decision_sample_value(seed="run-1", rank=0, key="linear_logp")
    rank1 = execution_decision_sample_value(seed="run-1", rank=1, key="linear_logp")

    assert 0 <= rank0 < 1
    assert 0 <= rank1 < 1
    assert rank0 == execution_decision_sample_value(seed="run-1", rank=0, key="linear_logp")
    assert rank0 != rank1


@pytest.mark.unit
def test_emit_execution_decision_can_sample_with_rank_specific_values(caplog):
    decision = select_execution_decision(operator="linear_logp", stage="train_logprob")
    sample_key = "same-event"
    rank0 = execution_decision_sample_value(seed="run-1", rank=0, key=sample_key)
    rank1 = execution_decision_sample_value(seed="run-1", rank=1, key=sample_key)
    sample_rate = (rank0 + rank1) / 2

    with caplog.at_level(logging.INFO, logger="vime.backends.rl_kernel_utils.execution"):
        emit_execution_decision(
            decision, sample_rate=sample_rate, sample_seed="run-1", sample_rank=0, sample_key=sample_key
        )
        emit_execution_decision(
            decision, sample_rate=sample_rate, sample_seed="run-1", sample_rank=1, sample_key=sample_key
        )

    assert len(caplog.records) == 1
