from __future__ import annotations

import sys
from pathlib import Path

import pytest

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

_unit_stubs.install_rollout_optional_stubs()
_unit_stubs.install_vllm_cli_stubs()

import vime.ray.rollout as rollout_mod  # noqa: E402
from vime.ray.rollout import RolloutManager  # noqa: E402
from vime.utils.consistency_metadata import build_rollout_consistency_metadata  # noqa: E402
from vime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


class Args:
    reward_key = None
    advantage_estimator = "grpo"
    rewards_normalization = False
    grpo_std_normalization = False
    rollout_top_p = 1.0
    rlk_consistency = "audit"
    hf_checkpoint = "unit/model"
    padding_side = "right"
    num_gpus = 1
    rollout_num_gpus = 1
    rollout_num_gpus_per_engine = 1
    router_policy = "round_robin"
    global_batch_size = 1
    micro_batch_size = 1
    use_dynamic_batch_size = False
    max_tokens_per_gpu = None
    balance_data = False
    balance_by_flops = False
    rollout_data_transport = "object-store"


def _manager(mode: str):
    cls = RolloutManager.__ray_actor_class__
    manager = cls.__new__(cls)
    manager.args = Args()
    manager.args.rlk_consistency = mode
    manager.custom_reward_post_process_func = None
    manager.custom_convert_samples_to_train_data_func = None
    return manager


def _sample(index: int = 0) -> Sample:
    return Sample(
        index=index,
        rollout_id=index,
        tokens=[101, 201, 202],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[-0.1, -0.2],
        weight_versions=["w1"],
        reward=1.0,
        status=Sample.Status.COMPLETED,
        metadata={},
    )


@pytest.mark.unit
def test_convert_samples_to_train_data_leaves_consistency_fields_out_when_off():
    train_data = _manager("off")._convert_samples_to_train_data([_sample()])

    assert "consistency_metadata" not in train_data
    assert "consistency_metadata_validation" not in train_data


@pytest.mark.unit
def test_convert_samples_to_train_data_reports_audit_warning_for_missing_metadata():
    train_data = _manager("audit")._convert_samples_to_train_data([_sample()])

    assert train_data["consistency_metadata"] == [None]
    assert train_data["consistency_metadata_validation"]["ok"] is True
    assert [issue["code"] for issue in train_data["consistency_metadata_validation"]["warnings"]] == [
        "consistency_metadata_missing"
    ]


@pytest.mark.unit
def test_convert_samples_to_train_data_fails_closed_in_strict_mode():
    with pytest.raises(ValueError, match="Strict consistency metadata validation failed"):
        _manager("strict")._convert_samples_to_train_data([_sample()])


@pytest.mark.unit
def test_convert_samples_to_train_data_carries_sample_metadata_when_present():
    sample = _sample()
    sample.consistency_metadata = build_rollout_consistency_metadata(
        sample,
        args=Args(),
        sampling_params={"temperature": 1.0, "top_p": 1.0},
        logprob_contract_id="contract-v1",
        requested_provenance={"backend": "native", "fallback": False},
        actual_provenance={"backend": "native", "fallback": False},
    )

    train_data = _manager("strict")._convert_samples_to_train_data([sample])

    assert train_data["consistency_metadata"] == [sample.consistency_metadata]
    assert train_data["consistency_metadata_validation"]["ok"] is True
    assert train_data["consistency_metadata_validation"]["active_token_count"] == 2


@pytest.mark.unit
def test_split_train_data_attaches_consistency_replay_manifest(monkeypatch):
    sample = _sample()
    sample.metadata["position_cache"] = {"position_ids": [0, 1, 2], "cache_policy": "none"}
    sample.metadata["quantization"] = {"policy": "none"}
    sample.consistency_metadata = build_rollout_consistency_metadata(
        sample,
        args=Args(),
        sampling_params={"temperature": 1.0, "top_p": 1.0},
        logprob_contract_id="contract-v1",
        requested_provenance={"backend": "native", "fallback": False},
        actual_provenance={"backend": "native", "fallback": False},
    )
    manager = _manager("audit")
    manager.train_parallel_config = {
        "dp_size": 1,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    monkeypatch.setattr(rollout_mod.ray, "put", lambda value, **kwargs: value)

    train_data = manager._convert_samples_to_train_data([sample])
    refs = manager._split_train_data_by_dp(train_data)
    rollout_data = refs[0].inner
    manifest = rollout_data["consistency_replay_manifest"]

    assert manifest["mode"] == "audit"
    assert manifest["rank"] == 0
    assert manifest["sample_count"] == 1
    assert manifest["samples"][0]["sample_index"] == 0
    assert manifest["samples"][0]["rollout_id"] == 0
    assert manifest["samples"][0]["has_rollout_log_probs"] is True
    assert {case["case"] for case in manifest["batch_invariance_cases"]} == {
        "same_sample_alone",
        "same_sample_mixed_batch",
        "padding_packing_variant",
        "active_token_density_variant",
        "dynamic_sampling_keep_drop_variant",
    }
