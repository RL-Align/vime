from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vime.utils.consistency_drift_report import (
    build_consistency_drift_report,
    render_consistency_drift_report,
    write_consistency_drift_report,
)


def _artifacts(*, with_timestamp: bool = False):
    sample = {"sample_position": 0, "sample_index": 17, "rollout_id": 23, "batch_layout_fingerprint": "layout-1"}
    if with_timestamp:
        sample.update(start_ts=10.0, end_ts=10.5)
    manifest = {
        "mode": "audit",
        "samples": [sample],
        "batch_invariance_cases": [{"case": "same_sample_alone"}],
        "validation": {"warnings": [], "failures": []},
        "runtime_provenance": {"operator": "linear_logp"},
    }
    cube = {
        "mode": "audit",
        "rank": 0,
        "axes": {"dtype": "bf16", "cp": 1, "logp_backend": "rlk.linear_logp.fast"},
        "metrics": {
            "active_token_count": 2,
            "max_abs_dlogp": 0.125,
            "warning_count": 1,
            "metadata_warning_count": 0,
            "metadata_failure_count": 0,
        },
        "worst_token": {"abs_dlogp": 0.125, "sample_position": 0, "token_position": 4},
        "metadata_validation": {"warnings": [], "failures": []},
        "runtime_provenance": {"actual_backend": "rl_engine.linear_logp", "fallback": False},
    }
    return manifest, cube


@pytest.mark.unit
def test_report_uses_ordinal_timeline_without_fabricating_timestamps():
    manifest, cube = _artifacts()

    report = build_consistency_drift_report(replay_manifest=manifest, result_cube=cube)

    assert report["timeline_mode"] == "ordinal_diagnostic"
    assert "not elapsed time" in report["timeline_note"]
    assert {event["lane"] for event in report["events"]} == {
        "Training audit",
        "Rollout samples",
        "Operator / backend",
        "Drift markers",
    }
    assert report["status"] == "warning"


@pytest.mark.unit
def test_report_prefers_actual_backend_and_timestamp_mode():
    manifest, cube = _artifacts(with_timestamp=True)
    cube["runtime_provenance"] = {
        "requested_backend": "registry",
        "actual_backend": "vime.native.linear_logp",
        "fallback": True,
    }

    report = build_consistency_drift_report(replay_manifest=manifest, result_cube=cube)
    operator = next(event for event in report["events"] if event["id"] == "operator-backend")

    assert report["timeline_mode"] == "timestamp"
    assert operator["label"] == "vime.native.linear_logp"
    assert operator["status"] == "warning"


@pytest.mark.unit
def test_rendered_report_is_self_contained_and_escapes_details(tmp_path: Path):
    manifest, cube = _artifacts()
    manifest["validation"]["warnings"] = [{"code": "bad<&", "message": "value </script>"}]
    cube["metadata_validation"] = manifest["validation"]

    report = build_consistency_drift_report(
        replay_manifest=manifest,
        result_cube=cube,
        title="<diagnostic>",
    )
    html = render_consistency_drift_report(report)
    output = write_consistency_drift_report(report, tmp_path / "drift.html")

    assert output.exists()
    assert "<diagnostic>" not in html
    assert "value </script>" not in html
    assert "ordinal_diagnostic" in html
    assert "Operator / backend" in html
    assert "http://" not in html
    assert "https://" not in html
    assert "detail.innerHTML" not in html
    assert "detail.replaceChildren" in html


@pytest.mark.unit
def test_debug_dump_loader_derives_drift_when_cube_was_not_serialized(tmp_path: Path):
    from tools.consistency_drift_report import load_consistency_artifacts

    path = tmp_path / "train.pt"
    torch.save(
        {
            "rollout_data": {
                "consistency_replay_manifest": {"mode": "audit", "samples": [{}], "validation": {}},
                "log_probs": [torch.tensor([0.1, 0.4])],
                "rollout_log_probs": [torch.tensor([0.1, 0.1])],
                "loss_masks": [torch.tensor([1, 1])],
            }
        },
        path,
    )

    manifest, cube, _ = load_consistency_artifacts([path])

    assert manifest["sample_count"] == 1
    assert cube["metrics"]["active_token_count"] == 2.0
    assert cube["metrics"]["max_abs_dlogp"] == pytest.approx(0.3)
    assert cube["worst_token"]["token_position"] == 1
