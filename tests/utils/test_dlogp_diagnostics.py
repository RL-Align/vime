from __future__ import annotations

import importlib
import math
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

from vime.utils.consistency_metadata import stable_fingerprint
from vime.utils.dlogp_diagnostics import compute_dlogp_diagnostics, get_rlk_consistency_mode, is_dlogp_audit_enabled

NUM_GPUS = 0


FULL_METADATA = {
    "model_name": "unit-model",
    "backend_id": "megatron",
    "contract_id": "contract-v1",
    "batch_layout_fingerprint": "layout-abc",
    "provenance_fingerprint": "prov-def",
}

LOSS_MODULE_PATH = "vime.backends.megatron_utils.loss"
MEGATRON_STUB_MODULES = (
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
    "megatron.core.transformer",
    "megatron.core.transformer.transformer_layer",
    LOSS_MODULE_PATH,
)


def _report_with_full_metadata(**kwargs):
    return compute_dlogp_diagnostics(
        model_name=FULL_METADATA["model_name"],
        backend_id=FULL_METADATA["backend_id"],
        contract_id=FULL_METADATA["contract_id"],
        batch_layout_fingerprint=FULL_METADATA["batch_layout_fingerprint"],
        provenance_fingerprint=FULL_METADATA["provenance_fingerprint"],
        **kwargs,
    )


@pytest.fixture()
def megatron_loss_module():
    saved = _unit_stubs.save_sys_modules(MEGATRON_STUB_MODULES)
    for module_name in MEGATRON_STUB_MODULES:
        sys.modules.pop(module_name, None)
    _unit_stubs.install_megatron_mpu_stub()
    try:
        yield importlib.import_module(LOSS_MODULE_PATH)
    finally:
        _unit_stubs.restore_sys_modules(saved)


def _policy_args(mode: str) -> Namespace:
    return Namespace(
        use_rollout_logprobs=False,
        rollout_top_p=1.0,
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
        rlk_consistency_mode=mode,
        model_name=FULL_METADATA["model_name"],
        train_backend=FULL_METADATA["backend_id"],
        rlk_contract_id=FULL_METADATA["contract_id"],
        rlk_batch_layout_fingerprint=FULL_METADATA["batch_layout_fingerprint"],
        rlk_provenance_fingerprint=FULL_METADATA["provenance_fingerprint"],
    )


def _policy_batch() -> dict:
    return {
        "advantages": [torch.ones(2)],
        "log_probs": [torch.zeros(2)],
        "rollout_log_probs": [torch.zeros(2)],
        "response_lengths": [2],
        "total_lengths": [2],
        "loss_masks": [torch.ones(2, dtype=torch.int32)],
        "unconcat_tokens": [torch.tensor([1, 2])],
        "sample_indices": [42],
        "rollout_ids": [7],
    }


def _complete_consistency_record() -> dict:
    record = {
        "schema_version": 1,
        "sample": {"index": 42, "rollout_id": 7, "session_id": "session-42"},
        "tokens": {"response_token_ids_fingerprint": stable_fingerprint([1, 2])},
        "active_mask": {"mask_fingerprint": stable_fingerprint([1, 1]), "active_token_count": 2},
        "tokenizer": {"fingerprint": "tokenizer"},
        "sampling": {"params_fingerprint": "sampling"},
        "padding": {"side": "right"},
        "position_cache": {"fingerprint": "position-cache"},
        "quantization": {"fingerprint": "quantization"},
        "model": {"name": "record-model"},
        "weight": {"version": "weights-v1", "pre_update": True},
        "old_logp": {"source": "rollout_engine", "contract_id": "record-contract"},
        "provenance": {
            "actual": {"backend": "record-backend", "fallback": False},
            "actual_fingerprint": "record-provenance",
            "mismatches": {},
            "undeclared_fallback": False,
        },
    }
    record["fingerprint"] = stable_fingerprint(record)
    return record


@pytest.mark.unit
def test_dlogp_metrics_use_active_tokens_only_and_identify_worst_token():
    train_log_probs = [
        torch.tensor([-1.0, -2.0, -3.0]),
        torch.tensor([-0.1, -0.2, 10.0]),
    ]
    rollout_log_probs = [
        torch.tensor([-1.0, -1.5, -4.0]),
        torch.tensor([-0.6, -0.2, -10.0]),
    ]
    loss_masks = [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 0])]

    report = _report_with_full_metadata(
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        sample_indices=torch.tensor([10, 11]),
        rollout_ids=torch.tensor([3, 4]),
        rank=7,
        eps_clip=0.2,
    )

    # Active dlogp values are [0.0, 1.0, 0.5, 0.0]. The masked 20.0 delta must not win.
    metrics = report.metrics
    assert metrics["rlk_audit_active_token_count"].item() == pytest.approx(4.0)
    assert metrics["rlk_audit_mask_coverage"].item() == pytest.approx(4.0 / 6.0)
    assert metrics["rlk_audit_dlogp_abs_mean"].item() == pytest.approx(0.375)
    assert metrics["rlk_audit_dlogp_abs_max"].item() == pytest.approx(1.0)
    assert metrics["rlk_audit_dlogp_abs_p50"].item() == pytest.approx(0.25)
    assert metrics["rlk_audit_dlogp_abs_p90"].item() == pytest.approx(0.85)
    assert metrics["rlk_audit_dlogp_abs_p99"].item() == pytest.approx(0.985)
    assert metrics["rlk_audit_warning_count"].item() == pytest.approx(0.0)

    assert report.worst_token == {
        "abs_dlogp": pytest.approx(1.0),
        "dlogp": pytest.approx(1.0),
        "sample_position": 0,
        "sample_id": 10,
        "sample_index": 10,
        "rollout_id": 3,
        "token_position": 2,
        "rank": 7,
        **FULL_METADATA,
    }
    assert metrics["rlk_audit_worst_sample_index"].item() == pytest.approx(10.0)
    assert metrics["rlk_audit_worst_rollout_id"].item() == pytest.approx(3.0)
    assert metrics["rlk_audit_worst_token_position"].item() == pytest.approx(2.0)
    assert metrics["rlk_audit_worst_rank"].item() == pytest.approx(7.0)


@pytest.mark.unit
def test_dlogp_ratio_clipfrac_and_approx_kl_follow_issue_formulas():
    dlogp = torch.tensor([0.0, math.log(1.5), math.log(0.75), math.log(1.1)])
    report = _report_with_full_metadata(
        train_log_probs=[dlogp],
        rollout_log_probs=[torch.zeros_like(dlogp)],
        loss_masks=[torch.ones_like(dlogp)],
        eps_clip=0.2,
    )

    ratio0 = dlogp.exp()
    expected_clipfrac = ((ratio0 - 1.0).abs() > 0.2).float().mean()
    expected_approx_kl = (ratio0 - 1.0 - dlogp).mean()

    assert report.metrics["rlk_audit_ratio0_mean"].item() == pytest.approx(ratio0.mean().item())
    assert report.metrics["rlk_audit_clipfrac0"].item() == pytest.approx(expected_clipfrac.item())
    assert report.metrics["rlk_audit_approx_kl0"].item() == pytest.approx(expected_approx_kl.item())


@pytest.mark.unit
def test_dlogp_zero_active_sample_is_reported_without_worst_token():
    report = _report_with_full_metadata(
        train_log_probs=[torch.tensor([1.0, 2.0])],
        rollout_log_probs=[torch.tensor([1.0, 2.0])],
        loss_masks=[torch.tensor([0, 0])],
    )

    assert report.metrics["rlk_audit_active_token_count"].item() == pytest.approx(0.0)
    assert report.metrics["rlk_audit_mask_coverage"].item() == pytest.approx(0.0)
    assert report.metrics["rlk_audit_zero_active_sample_count"].item() == pytest.approx(1.0)
    assert report.metrics["rlk_audit_worst_token_position"].item() == pytest.approx(-1.0)
    assert report.worst_token is None
    assert [warning.code for warning in report.warnings] == ["zero_active_tokens"]


@pytest.mark.unit
def test_dlogp_shape_mismatch_is_a_structured_warning_and_skips_sample():
    report = _report_with_full_metadata(
        train_log_probs=[torch.tensor([1.0, 2.0])],
        rollout_log_probs=[torch.tensor([1.0])],
        loss_masks=[torch.tensor([1, 1])],
        sample_indices=[123],
    )

    assert report.metrics["rlk_audit_active_token_count"].item() == pytest.approx(0.0)
    assert report.metrics["rlk_audit_warning_count"].item() == pytest.approx(1.0)
    assert report.warnings[0].code == "shape_mismatch"
    assert report.warnings[0].sample_id == 123


@pytest.mark.unit
def test_dlogp_missing_rollout_log_probs_is_a_structured_warning():
    report = _report_with_full_metadata(
        train_log_probs=[torch.tensor([1.0, 2.0])],
        rollout_log_probs=None,
        loss_masks=[torch.tensor([1, 1])],
    )

    assert report.metrics["rlk_audit_active_token_count"].item() == pytest.approx(0.0)
    assert report.warnings[0].code == "missing_rollout_log_probs"
    assert report.warnings[0].field == "rollout_log_probs"


@pytest.mark.unit
def test_dlogp_missing_metadata_produces_structured_warnings():
    report = compute_dlogp_diagnostics(
        train_log_probs=[torch.tensor([1.0])],
        rollout_log_probs=[torch.tensor([0.0])],
        loss_masks=[torch.tensor([1])],
    )

    warning_fields = {warning.field for warning in report.warnings if warning.code == "missing_metadata"}
    assert warning_fields == {
        "model_name",
        "backend_id",
        "contract_id",
        "batch_layout_fingerprint",
        "provenance_fingerprint",
    }
    assert report.metrics["rlk_audit_warning_count"].item() == pytest.approx(5.0)


@pytest.mark.unit
def test_dlogp_sample_metadata_can_supply_context_fields():
    metadata = [
        {
            "model_name": "sample-model",
            "backend_id": "sample-backend",
            "contract_id": "sample-contract",
            "batch_layout_fingerprint": "sample-layout",
            "provenance_fingerprint": "sample-provenance",
        }
    ]

    report = compute_dlogp_diagnostics(
        train_log_probs=[torch.tensor([2.0])],
        rollout_log_probs=[torch.tensor([0.0])],
        loss_masks=[torch.tensor([1])],
        metadata=metadata,
    )

    assert not report.warnings
    assert report.worst_token is not None
    assert report.worst_token["model_name"] == "sample-model"
    assert report.worst_token["backend_id"] == "sample-backend"
    assert report.worst_token["contract_id"] == "sample-contract"
    assert report.worst_token["batch_layout_fingerprint"] == "sample-layout"
    assert report.worst_token["provenance_fingerprint"] == "sample-provenance"


@pytest.mark.unit
def test_dlogp_diagnostics_are_read_only_and_detached():
    train = torch.tensor([1.0, 2.0], requires_grad=True)
    rollout = torch.tensor([0.5, 1.5])
    mask = torch.tensor([1, 0])
    train_before = train.detach().clone()
    rollout_before = rollout.clone()
    mask_before = mask.clone()

    report = _report_with_full_metadata(
        train_log_probs=[train],
        rollout_log_probs=[rollout],
        loss_masks=[mask],
    )

    torch.testing.assert_close(train.detach(), train_before)
    torch.testing.assert_close(rollout, rollout_before)
    torch.testing.assert_close(mask, mask_before)
    assert all(not metric.requires_grad for metric in report.metrics.values())


@pytest.mark.unit
def test_rlk_consistency_mode_defaults_to_off_and_supports_args_env_and_aliases():
    assert get_rlk_consistency_mode(None, environ={}) == "off"
    assert not is_dlogp_audit_enabled(Namespace(), environ={})

    assert get_rlk_consistency_mode(Namespace(rlk_consistency_mode="audit"), environ={}) == "audit"
    assert is_dlogp_audit_enabled(Namespace(rlk_consistency_mode="strict"), environ={})
    assert is_dlogp_audit_enabled(Namespace(rlk_consistency_mode="audit"), environ={"VIME_RLK_CONSISTENCY": "off"})
    assert get_rlk_consistency_mode(Namespace(), environ={"VIME_RLK_CONSISTENCY": "audit-only"}) == "audit"
    assert get_rlk_consistency_mode(Namespace(), environ={"VIME_RL_KERNEL_CONSISTENCY": "true"}) == "audit"

    with pytest.raises(ValueError, match="Unsupported RL-Kernel consistency mode"):
        get_rlk_consistency_mode(Namespace(rlk_consistency_mode="fast"), environ={})


@pytest.mark.unit
def test_policy_loss_adds_audit_metrics_only_when_enabled_without_changing_loss(monkeypatch, megatron_loss_module):
    train_log_probs = [torch.tensor([0.1, 0.3])]

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {"log_probs": train_log_probs, "entropy": [torch.zeros(2)]}

    def fake_compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high):
        del advantages, eps_clip, eps_clip_high
        return torch.ones_like(ppo_kl), torch.zeros_like(ppo_kl)

    monkeypatch.setattr(megatron_loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr(megatron_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    def reducer(tensor):
        return tensor.mean()

    logits = torch.zeros(1, 2, 4)
    off_loss, off_metrics = megatron_loss_module.policy_loss_function(
        _policy_args("off"),
        _policy_batch(),
        logits,
        reducer,
    )
    audit_loss, audit_metrics = megatron_loss_module.policy_loss_function(
        _policy_args("audit"),
        _policy_batch(),
        logits,
        reducer,
    )

    torch.testing.assert_close(audit_loss, off_loss)
    assert not any(key.startswith("rlk_audit_") for key in off_metrics)
    assert audit_metrics["rlk_audit_active_token_count"].item() == pytest.approx(2.0)
    assert audit_metrics["rlk_audit_dlogp_abs_mean"].item() == pytest.approx(0.2)
    assert audit_metrics["rlk_audit_worst_sample_index"].item() == pytest.approx(42.0)
    assert audit_metrics["rlk_audit_worst_rollout_id"].item() == pytest.approx(7.0)


@pytest.mark.unit
def test_policy_loss_uses_consistency_metadata_for_audit_context(monkeypatch, megatron_loss_module):
    train_log_probs = [torch.tensor([0.1, 0.3])]

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {"log_probs": train_log_probs, "entropy": [torch.zeros(2)]}

    def fake_compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high):
        del advantages, eps_clip, eps_clip_high
        return torch.ones_like(ppo_kl), torch.zeros_like(ppo_kl)

    monkeypatch.setattr(megatron_loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr(megatron_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    def reducer(tensor):
        return tensor.mean()

    args = _policy_args("audit")
    args.model_name = None
    args.train_backend = None
    args.rlk_contract_id = None
    args.rlk_batch_layout_fingerprint = None
    args.rlk_provenance_fingerprint = None
    batch = _policy_batch()
    batch["consistency_metadata"] = [_complete_consistency_record()]
    batch["consistency_batch_layout_fingerprints"] = [{"fingerprint": "record-layout"}]

    _, metrics = megatron_loss_module.policy_loss_function(
        args,
        batch,
        torch.zeros(1, 2, 4),
        reducer,
    )

    assert metrics["rlk_audit_warning_count"].item() == pytest.approx(0.0)
    assert metrics["rlk_audit_metadata_warning_count"].item() == pytest.approx(0.0)
    assert metrics["rlk_audit_replay_case_count"].item() == pytest.approx(5.0)
    assert metrics["rlk_audit_worst_sample_index"].item() == pytest.approx(42.0)


@pytest.mark.unit
def test_policy_loss_adds_linear_logp_runtime_provenance(monkeypatch, megatron_loss_module):
    train_log_probs = [torch.tensor([0.1, 0.3])]

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {"log_probs": train_log_probs, "entropy": [torch.zeros(2)]}

    def fake_compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high):
        del advantages, eps_clip, eps_clip_high
        return torch.ones_like(ppo_kl), torch.zeros_like(ppo_kl)

    runtime_provenance = {
        "operator": "linear_logp",
        "requested_backend": "registry",
        "actual_backend": "vime.native.linear_logp",
        "fallback": True,
        "fallback_reason": "unit fallback",
    }

    monkeypatch.setattr(megatron_loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr(megatron_loss_module, "compute_policy_loss", fake_compute_policy_loss)
    monkeypatch.setattr(megatron_loss_module, "get_linear_logp_runtime_metadata", lambda: runtime_provenance)

    def reducer(tensor):
        return tensor.mean()

    batch = _policy_batch()
    batch["consistency_metadata"] = [_complete_consistency_record()]
    batch["consistency_batch_layout_fingerprints"] = [{"fingerprint": "record-layout"}]
    _, metrics = megatron_loss_module.policy_loss_function(
        _policy_args("audit"),
        batch,
        torch.zeros(1, 2, 4),
        reducer,
        rl_kernel_linear_logp_context=object(),
    )

    assert metrics["rlk_audit_runtime_fallback"].item() == pytest.approx(1.0)
    assert metrics["rlk_audit_metadata_warning_count"].item() == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
