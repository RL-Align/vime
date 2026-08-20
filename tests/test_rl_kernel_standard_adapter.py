import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any

import pytest
import torch

from vime.backends.rl_kernel_utils import (
    RlkAlignmentStandard,
    RlkAlignmentStandardUnavailable,
    alignment_standard_metadata,
    build_standard_score_artifact,
    compare_standard_score_records,
    iter_operator_ablation_cases,
    load_alignment_standard,
    load_operator_ablation_matrix,
    select_operator_ablation_case,
)


def _matrix() -> dict[str, object]:
    return {
        "schema_version": "rlkernel.debug_matrix.v1",
        "modules": {
            "attention": {"axes": [{"id": "position_rope"}]},
            "ffn": {"axes": [{"id": "gemm_reduction"}]},
            "logp": {"axes": [{"id": "vocab_lse_reduction"}]},
        },
    }


@dataclass(frozen=True)
class _FakeStandard:
    module_debug_matrix: Mapping[str, Any] = field(default_factory=_matrix)
    compare_score_artifacts: Callable[..., Any] | None = None
    schema_types: Mapping[str, Any] = field(default_factory=dict)
    source: str = "rl_kernel.fake"
    standard_id: str = "rl_kernel.cross_config.module_debug_matrix"
    tolerance_fingerprint: str = "tolerance-sha"


@dataclass(frozen=True)
class _FakeIdentity:
    checkpoint_id: str
    model_version: str


class _FakeScoreSide(str, Enum):
    ROLLOUT = "rollout"
    TRAINING = "training"


@dataclass(frozen=True)
class _FakeScorer:
    side: _FakeScoreSide
    backend_id: str
    dtype: str


@dataclass(frozen=True)
class _FakeProvenance:
    requested: dict[str, object]
    actual: dict[str, object]


@dataclass(frozen=True)
class _FakeScoreArtifact:
    case_id: str
    attempt_id: str
    side: _FakeScoreSide
    identity: _FakeIdentity
    scorer: _FakeScorer
    selected_logprobs: torch.Tensor
    active_mask: torch.Tensor
    provenance: _FakeProvenance


@pytest.mark.unit
def test_module_matrix_is_consumed_without_a_vime_copy():
    standard = _FakeStandard()

    assert load_operator_ablation_matrix(standard) is standard.module_debug_matrix
    assert alignment_standard_metadata(standard)["alignment_matrix_schema_version"] == "rlkernel.debug_matrix.v1"

    cases = iter_operator_ablation_cases("attention", provider=standard)

    assert [(case.case_id, case.training_implementation, case.rollout_implementation) for case in cases] == [
        ("P/P", "production", "production"),
        ("R/R", "rl_kernel", "rl_kernel"),
        ("P/R", "production", "rl_kernel"),
        ("R/P", "rl_kernel", "production"),
    ]
    assert all(case.diagnostic_axes == ("position_rope",) for case in cases)
    assert select_operator_ablation_case("attention", "p/r", provider=standard).purpose == "rollout-only mismatch"


@pytest.mark.unit
def test_load_alignment_standard_reads_rl_kernel_public_contracts(monkeypatch):
    compared = {}

    def compare(rollout, training):
        compared["rollout"] = rollout
        compared["training"] = training
        return "compared-through-rl-kernel"

    rl_engine = ModuleType("rl_engine")
    alignment = ModuleType("rl_engine.alignment")
    kernels = ModuleType("rl_engine.kernels")
    gtest = ModuleType("rl_engine.kernels.gtest")
    cross_config = ModuleType("rl_engine.alignment.cross_config")
    debug_matrix = ModuleType("rl_engine.alignment.cross_config.debug_matrix")
    schema = ModuleType("rl_engine.alignment.cross_config.schema")
    tolerance = ModuleType("rl_engine.kernels.gtest.tolerance")
    cross_config.compare_score_artifacts = compare
    debug_matrix.module_debug_matrix = _matrix
    schema.RuntimeProvenance = _FakeProvenance
    schema.ScoreArtifact = _FakeScoreArtifact
    schema.ScorerSpec = _FakeScorer
    schema.ScoreSide = _FakeScoreSide
    schema.SemanticIdentitySpec = _FakeIdentity
    tolerance.tolerance_contract_fingerprint = lambda: "tolerance-fingerprint"

    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment", alignment)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.gtest", gtest)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment.cross_config", cross_config)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment.cross_config.debug_matrix", debug_matrix)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment.cross_config.schema", schema)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.gtest.tolerance", tolerance)

    standard = load_alignment_standard()

    assert isinstance(standard, RlkAlignmentStandard)
    assert standard.compare_score_artifacts is compare
    assert standard.schema_types["ScoreArtifact"] is _FakeScoreArtifact
    assert alignment_standard_metadata(standard) == {
        "alignment_standard_source": "rl_kernel",
        "alignment_standard_id": "rl_kernel.cross_config.module_debug_matrix",
        "alignment_matrix_schema_version": "rlkernel.debug_matrix.v1",
        "alignment_standard_fingerprint": standard.fingerprint,
        "alignment_tolerance_fingerprint": "tolerance-fingerprint",
    }


@pytest.mark.unit
def test_invalid_or_unknown_matrix_cases_are_rejected():
    with pytest.raises(ValueError, match="unknown RL-Kernel ablation module"):
        iter_operator_ablation_cases("unknown", provider=_FakeStandard())
    with pytest.raises(ValueError, match="unknown RL-Kernel ablation case"):
        select_operator_ablation_case("logp", "P/R/R", provider=_FakeStandard())
    with pytest.raises(RlkAlignmentStandardUnavailable, match="module debug matrix"):
        load_alignment_standard({"module_debug_matrix": {}})


@pytest.mark.unit
def test_framework_score_records_compare_through_standard_comparator():
    compared = {}

    def compare(rollout, training):
        compared["rollout"] = rollout
        compared["training"] = training
        return {
            "case_id": rollout.case_id,
            "mismatch_count": int(
                (training.selected_logprobs[training.active_mask] != rollout.selected_logprobs[rollout.active_mask])
                .sum()
                .item()
            ),
        }

    standard = _FakeStandard(
        compare_score_artifacts=compare,
        schema_types={
            "RuntimeProvenance": _FakeProvenance,
            "ScoreArtifact": _FakeScoreArtifact,
            "ScorerSpec": _FakeScorer,
            "ScoreSide": _FakeScoreSide,
            "SemanticIdentitySpec": _FakeIdentity,
        },
    )
    base_record = {
        "case_id": "case-1",
        "attempt_id": "attempt-1",
        "identity": {"checkpoint_id": "ckpt", "model_version": "weights"},
        "scorer": {"backend_id": "vime.native", "dtype": "float32"},
        "selected_logprobs": torch.tensor([0.0, -1.0, -2.0]),
        "active_mask": torch.tensor([1, 0, 1], dtype=torch.bool),
        "provenance": {"requested": {"backend": "rlk"}, "actual": {"backend": "rlk"}},
    }

    result = compare_standard_score_records(
        {**base_record, "side": "rollout"},
        {**base_record, "side": "training", "selected_logprobs": torch.tensor([0.0, -9.0, -2.5])},
        standard=standard,
    )

    assert result == {"case_id": "case-1", "mismatch_count": 1}
    assert compared["rollout"].side is _FakeScoreSide.ROLLOUT
    assert compared["training"].scorer.side is _FakeScoreSide.TRAINING
    assert compared["training"].active_mask.dtype is torch.bool


@pytest.mark.unit
def test_standard_score_artifact_export_requires_rl_kernel_schema():
    with pytest.raises(RlkAlignmentStandardUnavailable, match="score-artifact schema"):
        build_standard_score_artifact({}, standard=_FakeStandard())
