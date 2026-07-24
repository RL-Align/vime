from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs  # noqa: E402

_unit_stubs.install_rollout_optional_stubs()

from vime.rollout.data_source import RolloutDataSourceWithBuffer  # noqa: E402
from vime.utils.consistency_metadata import (  # noqa: E402
    build_batch_layout_fingerprints,
    build_requested_actual_provenance,
    build_rollout_consistency_metadata,
    ensure_sample_consistency_metadata,
    get_consistency_mode,
    raise_for_consistency_metadata_failures,
    stable_fingerprint,
    validate_samples_consistency_metadata,
)
from vime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


def _args(**overrides):
    values = dict(
        hf_checkpoint="unit/model",
        padding_side="right",
        num_gpus=2,
        num_gpus_per_node=2,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
        router_policy="consistent_hash",
        vllm_router_ip="127.0.0.1",
        vllm_router_port=8000,
        rlk_consistency="audit",
        buffer_filter_path=None,
        rollout_global_dataset=False,
        n_samples_per_prompt=2,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _sample(**overrides) -> Sample:
    values = dict(
        index=5,
        group_index=2,
        rollout_id=11,
        session_id="session-1",
        tokens=[101, 201, 202, 203],
        response_length=3,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.1, -0.2, -0.3],
        weight_versions=["weights-v1"],
        status=Sample.Status.COMPLETED,
        metadata={},
    )
    values.update(overrides)
    return Sample(**values)


def _complete_sample(**overrides) -> Sample:
    sample = _sample(**overrides)
    sample.consistency_metadata = build_rollout_consistency_metadata(
        sample,
        args=_args(),
        sampling_params={"temperature": 0.7, "top_p": 0.95, "top_k": 50, "max_new_tokens": 128},
        logprob_contract_id="rlk.logp.fp32.v1",
        requested_provenance={"backend": "native", "fallback": False},
        actual_provenance={"backend": "native", "fallback": False},
    )
    return sample


@pytest.mark.unit
def test_stable_fingerprint_is_order_insensitive_for_dicts():
    assert stable_fingerprint({"b": 2, "a": [1, 2]}) == stable_fingerprint({"a": [1, 2], "b": 2})
    assert stable_fingerprint({"a": [1, 2]}) != stable_fingerprint({"a": [2, 1]})


@pytest.mark.unit
def test_rollout_metadata_records_compact_token_and_active_mask_fingerprints():
    sample = _complete_sample()
    metadata = sample.consistency_metadata

    assert metadata["tokens"]["total_token_count"] == 4
    assert metadata["tokens"]["response_length"] == 3
    assert metadata["tokens"]["response_token_ids_fingerprint"] == stable_fingerprint([201, 202, 203])
    assert metadata["active_mask"]["active_token_count"] == 2
    assert metadata["active_mask"]["mask_fingerprint"] == stable_fingerprint([1, 0, 1])
    assert metadata["old_logp"]["source"] == "rollout_engine"
    assert metadata["old_logp"]["contract_id"] == "rlk.logp.fp32.v1"


@pytest.mark.unit
def test_response_token_fingerprint_excludes_prompt_tokens():
    first = _complete_sample(tokens=[1, 10, 11], response_length=2, loss_mask=[1, 1])
    second = _complete_sample(tokens=[999, 10, 11], response_length=2, loss_mask=[1, 1])

    assert (
        first.consistency_metadata["tokens"]["response_token_ids_fingerprint"]
        == second.consistency_metadata["tokens"]["response_token_ids_fingerprint"]
    )
    assert (
        first.consistency_metadata["tokens"]["token_ids_fingerprint"]
        != second.consistency_metadata["tokens"]["token_ids_fingerprint"]
    )


@pytest.mark.unit
def test_metadata_uses_sample_index_as_default_rollout_identifier():
    sample = _complete_sample(rollout_id=None, index=42)

    assert sample.consistency_metadata["sample"]["rollout_id"] == 42
    validation = validate_samples_consistency_metadata([sample], mode="strict")
    assert validation.ok


@pytest.mark.unit
def test_loss_mask_length_mismatch_is_rejected_before_audit_claims():
    sample = _sample(response_length=3, loss_mask=[1, 0])
    with pytest.raises(ValueError, match="loss_mask length"):
        build_rollout_consistency_metadata(
            sample,
            args=_args(),
            sampling_params={"temperature": 1.0},
            logprob_contract_id="contract",
        )


@pytest.mark.unit
def test_audit_mode_reports_missing_custom_metadata_as_structured_warning():
    validation = validate_samples_consistency_metadata([_sample(consistency_metadata=None)], mode="audit")

    assert validation.ok
    assert [issue.code for issue in validation.warnings] == ["consistency_metadata_missing"]
    assert validation.warnings[0].sample_index == 5
    assert validation.warnings[0].rollout_id == 11
    assert validation.failures == []


@pytest.mark.unit
def test_strict_mode_fails_closed_when_required_metadata_is_missing():
    validation = validate_samples_consistency_metadata([_sample(consistency_metadata=None)], mode="strict")

    assert not validation.ok
    assert [issue.code for issue in validation.failures] == ["consistency_metadata_missing"]
    with pytest.raises(ValueError, match="Strict consistency metadata validation failed"):
        raise_for_consistency_metadata_failures(validation)


@pytest.mark.unit
def test_complete_metadata_passes_strict_validation_and_counts_active_tokens():
    validation = validate_samples_consistency_metadata([_complete_sample()], mode="strict")

    assert validation.ok
    assert validation.active_token_count == 2
    assert validation.zero_active_token_samples == []
    assert validation.warnings == []
    assert validation.failures == []


@pytest.mark.unit
def test_zero_active_token_sample_is_identified_without_becoming_strict_failure():
    validation = validate_samples_consistency_metadata(
        [_complete_sample(tokens=[1, 2], response_length=2, loss_mask=[0, 0])],
        mode="strict",
    )

    assert validation.ok
    assert validation.active_token_count == 0
    assert validation.zero_active_token_samples == [{"sample_index": 5, "rollout_id": 11}]
    assert [issue.code for issue in validation.warnings] == ["zero_active_tokens"]


@pytest.mark.unit
def test_requested_actual_provenance_mismatch_is_audit_warning_and_strict_failure():
    sample = _complete_sample()
    sample.consistency_metadata["provenance"] = build_requested_actual_provenance(
        requested={"backend": "native", "fallback": False},
        actual={"backend": "fallback-native", "fallback": True},
    )

    audit = validate_samples_consistency_metadata([sample], mode="audit")
    assert {issue.code for issue in audit.warnings} == {
        "requested_actual_provenance_mismatch",
        "undeclared_runtime_fallback",
    }

    strict = validate_samples_consistency_metadata([sample], mode="strict")
    assert {issue.code for issue in strict.failures} == {
        "requested_actual_provenance_mismatch",
        "undeclared_runtime_fallback",
    }


@pytest.mark.unit
def test_ensure_sample_consistency_metadata_does_not_overwrite_existing_record_by_default():
    sample = _sample(consistency_metadata={"schema_version": 1, "fingerprint": "existing"})

    record = ensure_sample_consistency_metadata(
        sample,
        args=_args(),
        sampling_params={"temperature": 1.0},
        logprob_contract_id="contract",
    )

    assert record == {"schema_version": 1, "fingerprint": "existing"}
    assert sample.consistency_metadata == {"schema_version": 1, "fingerprint": "existing"}


@pytest.mark.unit
def test_batch_layout_fingerprints_map_samples_to_rank_and_microbatch():
    train_data = {
        "tokens": [[1, 2], [3, 4, 5], [6, 7], [8, 9]],
        "total_lengths": [2, 3, 2, 2],
        "response_lengths": [2, 2, 2, 2],
        "loss_masks": [[1, 1], [1, 0], [0, 1], [0, 0]],
        "rollout_ids": [10, 11, 12, 13],
        "sample_indices": [0, 1, 2, 3],
    }

    layouts = build_batch_layout_fingerprints(
        train_data,
        partitions=[[0, 2], [1, 3]],
        micro_batch_indices=[[[0], [1]], [[0, 1]]],
        num_microbatches=[2],
        global_batch_sizes=[2],
    )

    assert layouts[2]["dp_rank"] == 0
    assert layouts[2]["microbatch_id"] == 1
    assert layouts[2]["active_token_count"] == 1
    assert layouts[3]["dp_rank"] == 1
    assert layouts[3]["microbatch_size"] == 2
    assert layouts[3]["active_mask_density"] == 0.0
    assert layouts[0]["global_batch_shape"]["packed_order_fingerprint"] == stable_fingerprint([0, 2, 1, 3])


@pytest.mark.unit
def test_batch_layout_fingerprints_mark_samples_dropped_by_existing_schedule_trim():
    train_data = {
        "tokens": [[1, 2], [3, 4]],
        "total_lengths": [2, 2],
        "response_lengths": [2, 2],
        "loss_masks": [[1, 1], [1, 0]],
        "rollout_ids": [10, 11],
        "sample_indices": [0, 1],
    }

    layouts = build_batch_layout_fingerprints(
        train_data,
        partitions=[[0]],
        micro_batch_indices=[[[0]]],
        num_microbatches=[1],
        global_batch_sizes=[1],
    )

    assert "dropped_by_schedule" not in layouts[0]
    assert layouts[1]["dropped_by_schedule"] is True
    assert layouts[1]["active_token_count"] == 1


@pytest.mark.unit
def test_rollout_data_source_buffer_preserves_consistency_metadata():
    source = RolloutDataSourceWithBuffer(_args())
    sample = _complete_sample()

    source.add_samples([[sample, _complete_sample(index=6, rollout_id=12)]])
    returned = source.get_samples(1)

    assert returned[0][0].consistency_metadata == sample.consistency_metadata


@pytest.mark.unit
def test_get_consistency_mode_accepts_args_and_env_alias(monkeypatch):
    assert get_consistency_mode(_args(rlk_consistency="strict")) == "strict"

    monkeypatch.setenv("VIME_RLK_CONSISTENCY", "audit")
    assert get_consistency_mode(argparse.Namespace()) == "audit"

    with pytest.raises(ValueError, match="Unsupported RL-Kernel consistency mode"):
        get_consistency_mode(_args(rlk_consistency="maybe"))
