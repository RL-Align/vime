from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from vime.utils.consistency_audit import (
    build_batch_invariance_replay_cases,
    build_consistency_diagnostic_metadata,
    build_consistency_replay_manifest,
    run_consistency_audit,
    validate_consistency_audit_batch,
)
from vime.utils.consistency_metadata import build_rollout_consistency_metadata, stable_fingerprint
from vime.utils.types import Sample

NUM_GPUS = 0


def _args(mode: str = "audit", **overrides) -> Namespace:
    values = dict(
        rlk_consistency=mode,
        model_name="unit/model",
        train_backend="megatron",
        hf_checkpoint="unit/model",
        padding_side="right",
        quantization="none",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        router_policy="consistent_hash",
        vllm_enable_prefix_caching=False,
        vllm_enable_deterministic_inference=True,
        params_dtype="bf16",
        eps_clip=0.2,
    )
    values.update(overrides)
    return Namespace(**values)


def _sample(index: int = 5, **overrides) -> Sample:
    values = dict(
        index=index,
        group_index=2,
        rollout_id=index + 10,
        session_id=f"session-{index}",
        tokens=[101, 201, 202],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[-0.1, -0.2],
        weight_versions=["weights-v1"],
        status=Sample.Status.COMPLETED,
        metadata={
            "pre_update": True,
            "position_cache": {"position_ids": [0, 1, 2], "cache_policy": "none"},
            "quantization": {"policy": "none"},
        },
    )
    values.update(overrides)
    return Sample(**values)


def _metadata(sample: Sample | None = None, *, args: Namespace | None = None, **overrides):
    sample = sample or _sample()
    record = build_rollout_consistency_metadata(
        sample,
        args=args or _args(),
        sampling_params={"temperature": 1.0, "top_p": 1.0, "top_k": 0},
        logprob_contract_id="rlk.logp.native.fp32",
        requested_provenance={"backend": "native", "fallback": False},
        actual_provenance={"backend": "native", "fallback": False},
    )
    record.update(overrides)
    return record


def _batch(*, metadata=None, layout=None):
    metadata = [_metadata()] if metadata is None else metadata
    layout = (
        [
            {
                "fingerprint": "layout-1",
                "active_mask_density": 1.0,
                "dp_rank": 0,
                "microbatch_id": 0,
                "microbatch_offset": 0,
            }
        ]
        if layout is None
        else layout
    )
    return {
        "tokens": [torch.tensor([101, 201, 202])],
        "unconcat_tokens": [torch.tensor([101, 201, 202])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1])],
        "rollout_log_probs": [torch.tensor([-0.1, -0.2])],
        "sample_indices": [5],
        "rollout_ids": [15],
        "consistency_metadata": metadata,
        "consistency_batch_layout_fingerprints": layout,
    }


@pytest.mark.unit
def test_diagnostic_metadata_prefers_consistency_records_and_batch_layouts():
    metadata = build_consistency_diagnostic_metadata(_batch())

    assert metadata[0]["model_name"] == "unit/model"
    assert metadata[0]["backend_id"] == "native"
    assert metadata[0]["contract_id"] == "rlk.logp.native.fp32"
    assert metadata[0]["batch_layout_fingerprint"] == "layout-1"
    assert metadata[0]["provenance_fingerprint"].startswith("sha256:")


@pytest.mark.unit
def test_run_consistency_audit_builds_metrics_manifest_and_result_cube():
    batch = _batch()

    result = run_consistency_audit(
        [torch.tensor([-0.1, -0.1])],
        batch["rollout_log_probs"],
        batch["loss_masks"],
        args=_args("audit"),
        batch=batch,
        rank=3,
    )

    assert result.metadata_validation.ok
    assert result.metrics["rlk_audit_active_token_count"].item() == pytest.approx(2.0)
    assert result.metrics["rlk_audit_dlogp_abs_max"].item() == pytest.approx(0.1)
    assert result.metrics["rlk_audit_metadata_warning_count"].item() == pytest.approx(0.0)
    assert result.metrics["rlk_audit_replay_case_count"].item() == pytest.approx(5.0)
    assert result.replay_manifest["rank"] == 3
    assert result.replay_manifest["samples"][0]["batch_layout_fingerprint"] == "layout-1"
    assert result.result_cube["axes"]["batch_layout"] == "layout-1"
    assert result.result_cube["axes"]["dtype"] == "bf16"
    assert result.result_cube["axes"]["tp"] == 1
    assert result.result_cube["axes"]["logp_backend"] == "native"
    assert result.result_cube["metrics"]["max_abs_dlogp"] == pytest.approx(0.1)


@pytest.mark.unit
def test_run_consistency_audit_prefers_runtime_provenance_for_result_cube():
    runtime_provenance = {
        "operator": "linear_logp",
        "requested_backend": "registry",
        "actual_backend": "rl_engine.linear_logp",
        "backend_id": "rlk.linear_logp.fast",
        "contract_id": "rlk.linear_logp.fp32",
        "fallback": False,
        "strict_failure": False,
    }

    result = run_consistency_audit(
        [torch.tensor([-0.1, -0.2])],
        [torch.tensor([-0.1, -0.2])],
        [torch.tensor([1, 1])],
        args=_args("audit", params_dtype=torch.bfloat16),
        batch=_batch(),
        runtime_provenance=runtime_provenance,
    )

    assert result.metrics["rlk_audit_runtime_fallback"].item() == pytest.approx(0.0)
    assert result.replay_manifest["runtime_provenance"] == runtime_provenance
    assert result.replay_manifest["runtime_provenance_fingerprint"].startswith("sha256:")
    assert result.result_cube["axes"]["dtype"] == "bfloat16"
    assert result.result_cube["axes"]["logp_backend"] == "rlk.linear_logp.fast"
    assert result.result_cube["runtime_provenance"] == runtime_provenance


@pytest.mark.unit
def test_strict_audit_rejects_missing_position_cache_before_drift_attribution():
    record = _metadata()
    record["position_cache"] = {"fingerprint": None}
    batch = _batch(metadata=[record])

    with pytest.raises(ValueError, match="position_cache_metadata_missing"):
        run_consistency_audit(
            [torch.tensor([-0.1, -0.2])],
            batch["rollout_log_probs"],
            batch["loss_masks"],
            args=_args("strict"),
            batch=batch,
        )


@pytest.mark.unit
def test_strict_audit_requires_batch_layout_before_drift_attribution():
    batch = _batch(layout=[])

    with pytest.raises(ValueError, match="batch_layout_missing"):
        run_consistency_audit(
            [torch.tensor([-0.1, -0.2])],
            batch["rollout_log_probs"],
            batch["loss_masks"],
            args=_args("strict"),
            batch=batch,
        )


@pytest.mark.unit
def test_audit_mode_reports_missing_quantization_as_metadata_warning():
    record = _metadata()
    record["quantization"] = {"fingerprint": None}
    batch = _batch(metadata=[record])

    validation = validate_consistency_audit_batch(batch, mode="audit")

    assert validation.ok
    assert [issue.code for issue in validation.warnings] == ["quantization_metadata_missing"]


@pytest.mark.unit
def test_runtime_fallback_is_warning_in_audit_and_failure_in_strict():
    runtime_provenance = {
        "operator": "linear_logp",
        "requested_backend": "registry",
        "actual_backend": "vime.native.linear_logp",
        "fallback": True,
        "fallback_reason": "unit fallback",
    }

    validation = validate_consistency_audit_batch(
        _batch(),
        mode="audit",
        runtime_provenance=runtime_provenance,
    )

    assert validation.ok
    assert [issue.code for issue in validation.warnings] == ["undeclared_linear_logp_runtime_fallback"]

    with pytest.raises(ValueError, match="undeclared_linear_logp_runtime_fallback"):
        run_consistency_audit(
            [torch.tensor([-0.1, -0.2])],
            [torch.tensor([-0.1, -0.2])],
            [torch.tensor([1, 1])],
            args=_args("strict"),
            batch=_batch(),
            runtime_provenance=runtime_provenance,
        )


@pytest.mark.unit
def test_audit_batch_missing_metadata_warning_is_separate_from_dlogp_warning():
    batch = {
        "rollout_log_probs": [torch.tensor([0.0])],
        "loss_masks": [torch.tensor([1])],
        "sample_indices": [1],
        "rollout_ids": [2],
    }

    result = run_consistency_audit(
        [torch.tensor([0.25])],
        batch["rollout_log_probs"],
        batch["loss_masks"],
        args=_args("audit", rlk_contract_id="contract", rlk_batch_layout_fingerprint="layout"),
        batch=batch,
        model_name="unit/model",
        backend_id="megatron",
        provenance_fingerprint="prov",
    )

    assert result.metrics["rlk_audit_warning_count"].item() == pytest.approx(0.0)
    assert result.metrics["rlk_audit_metadata_warning_count"].item() == pytest.approx(1.0)
    assert result.metadata_validation.warnings[0].code == "consistency_metadata_missing"


@pytest.mark.unit
def test_replay_manifest_treats_1d_tensor_fields_as_one_sample():
    batch = {
        "response_lengths": torch.tensor([2]),
        "total_lengths": torch.tensor([3]),
        "loss_masks": torch.tensor([1, 0]),
        "rollout_log_probs": torch.tensor([-0.1, -0.2]),
    }

    manifest = build_consistency_replay_manifest(batch, mode="audit", rank=0)

    assert manifest["sample_count"] == 1
    assert manifest["samples"][0]["response_length"] == 2
    assert manifest["samples"][0]["active_token_count"] == 1
    assert manifest["samples"][0]["has_rollout_log_probs"] is True
    assert len(manifest["batch_invariance_cases"]) == 5


@pytest.mark.unit
def test_replay_manifest_contains_batch_invariance_cases_without_tensor_payloads():
    batch = _batch()
    batch["consistency_metadata"][0]["dynamic_sampling"] = {"keep": True, "reason": "unit"}
    batch["consistency_metadata"][0]["fingerprint"] = stable_fingerprint(batch["consistency_metadata"][0])

    manifest = build_consistency_replay_manifest(batch, mode="audit", rank=0)
    cases = build_batch_invariance_replay_cases(batch)

    assert manifest["sample_count"] == 1
    assert manifest["samples"][0]["has_rollout_log_probs"] is True
    assert manifest["samples"][0]["dynamic_sampling"] == {"keep": True, "reason": "unit"}
    assert {case["case"] for case in cases} == {
        "same_sample_alone",
        "same_sample_mixed_batch",
        "padding_packing_variant",
        "active_token_density_variant",
        "dynamic_sampling_keep_drop_variant",
    }
    assert all("tokens" not in sample for sample in manifest["samples"])
    assert manifest["fingerprint"].startswith("sha256:")


@pytest.mark.unit
def test_debug_train_dump_preserves_consistency_replay_manifest(tmp_path, monkeypatch):
    from vime.utils import train_dump_utils

    manifest = build_consistency_replay_manifest(_batch(), mode="audit", rank=0)
    path_template = str(tmp_path / "train_{rollout_id}_{rank}.pt")
    args = Namespace(save_debug_train_data=path_template)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    train_dump_utils.save_debug_train_data(
        args,
        rollout_id=12,
        rollout_data={"consistency_replay_manifest": manifest},
    )

    payload = torch.load(tmp_path / "train_12_0.pt", weights_only=False)
    assert payload["rollout_id"] == 12
    assert payload["rank"] == 0
    assert payload["rollout_data"]["consistency_replay_manifest"]["fingerprint"] == manifest["fingerprint"]
