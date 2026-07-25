import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any

import pytest
import torch

from vime.backends.rl_kernel_utils import (
    RlkAlignmentStandardUnavailable,
    alignment_standard_metadata,
    build_standard_score_artifact,
    compare_standard_score_records,
    iter_alignment_profiles,
    load_alignment_standard,
    select_least_restrictive_passing_profile,
)

PROFILE_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5")


def _profiles(source: str = "rl_kernel.fake") -> dict[str, "_FakeProfile"]:
    return {
        name: _FakeProfile(
            name=name,
            description=f"{name} from standard provider",
            aligned_axes=("metadata",),
            source=source,
            production_like=name == "A5",
        )
        for name in PROFILE_ORDER
    }


@dataclass(frozen=True)
class _FakeProfile:
    name: str
    description: str
    aligned_axes: tuple[str, ...]
    mismatched_axes: tuple[str, ...] = ()
    production_like: bool = False
    source: str = "rl_kernel.fake"


@dataclass(frozen=True)
class _FakeStandard:
    profiles: Mapping[str, _FakeProfile] = field(default_factory=_profiles)
    source: str = "rl_kernel.fake"
    compare_score_artifacts: Callable[..., Any] | None = None
    resolve_logprob_threshold: Callable[[str], float] | None = None
    schema_types: Mapping[str, Any] = field(default_factory=dict)
    standard_id: str = "rl_kernel.cross_config.alignment_standard"
    profile_version: str = "profiles.v1"
    fingerprint: str = "standard-sha"
    tolerance_fingerprint: str = "tolerance-sha"

    def iter_profiles(self) -> tuple[_FakeProfile, ...]:
        return tuple(self.profiles[name] for name in PROFILE_ORDER)

    def profile(self, name: str) -> _FakeProfile:
        return self.profiles[name]

    def to_metadata(self) -> dict[str, object]:
        return {
            "alignment_standard_source": self.source,
            "alignment_standard_id": self.standard_id,
            "alignment_profile_version": self.profile_version,
            "alignment_standard_fingerprint": self.fingerprint,
            "alignment_tolerance_fingerprint": self.tolerance_fingerprint,
        }


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
def test_alignment_standard_provider_is_consumed_as_source_of_truth():
    standard = load_alignment_standard(_FakeStandard(resolve_logprob_threshold=lambda dtype: 0.25 if dtype == "float32" else 0.5))

    assert standard.source == "rl_kernel.fake"
    assert [profile.name for profile in standard.iter_profiles()] == list(PROFILE_ORDER)
    assert standard.profile("A5").production_like
    assert standard.resolve_logprob_threshold("float32") == pytest.approx(0.25)
    assert alignment_standard_metadata(standard) == {
        "alignment_standard_source": "rl_kernel.fake",
        "alignment_standard_id": "rl_kernel.cross_config.alignment_standard",
        "alignment_profile_version": "profiles.v1",
        "alignment_standard_fingerprint": "standard-sha",
        "alignment_tolerance_fingerprint": "tolerance-sha",
    }
    assert select_least_restrictive_passing_profile({"A0", "A3"}, provider=standard).name == "A3"


@pytest.mark.unit
def test_load_alignment_standard_imports_rl_kernel_public_provider(monkeypatch):
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
    schema = ModuleType("rl_engine.alignment.cross_config.schema")
    tolerance = ModuleType("rl_engine.kernels.gtest.tolerance")

    def get_alignment_standard():
        return _FakeStandard(
            source="rl_kernel",
            profiles=_profiles("rl_kernel"),
            compare_score_artifacts=compare,
            resolve_logprob_threshold=lambda dtype: 0.125 if dtype == "float32" else 0.25,
            schema_types={
                "RuntimeProvenance": _FakeProvenance,
                "ScoreArtifact": _FakeScoreArtifact,
                "ScorerSpec": _FakeScorer,
                "ScoreSide": _FakeScoreSide,
                "SemanticIdentitySpec": _FakeIdentity,
            },
            profile_version="profiles.v2",
            fingerprint="profile-fingerprint",
            tolerance_fingerprint="tolerance-fingerprint",
        )

    cross_config.get_alignment_standard = get_alignment_standard
    schema.RuntimeProvenance = _FakeProvenance
    schema.ScoreArtifact = _FakeScoreArtifact
    schema.ScorerSpec = _FakeScorer
    schema.ScoreSide = _FakeScoreSide
    schema.SemanticIdentitySpec = _FakeIdentity
    tolerance.resolve_logprob_threshold = lambda dtype: 0.125 if dtype == "float32" else 0.25
    tolerance.tolerance_contract_fingerprint = lambda: "tolerance-fingerprint"

    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment", alignment)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment.cross_config", cross_config)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment.cross_config.schema", schema)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.gtest", gtest)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.gtest.tolerance", tolerance)

    standard = load_alignment_standard()

    assert [profile.name for profile in iter_alignment_profiles()] == list(PROFILE_ORDER)
    assert standard.compare_score_artifacts is compare
    assert standard.schema_types["ScoreArtifact"] is _FakeScoreArtifact
    assert standard.resolve_logprob_threshold("float32") == pytest.approx(0.125)
    assert alignment_standard_metadata(standard) == {
        "alignment_standard_source": "rl_kernel",
        "alignment_standard_id": "rl_kernel.cross_config.alignment_standard",
        "alignment_profile_version": "profiles.v2",
        "alignment_standard_fingerprint": "profile-fingerprint",
        "alignment_tolerance_fingerprint": "tolerance-fingerprint",
    }


@pytest.mark.unit
def test_load_alignment_standard_requires_rl_kernel_standard_provider(monkeypatch):
    rl_engine = ModuleType("rl_engine")
    alignment = ModuleType("rl_engine.alignment")
    cross_config = ModuleType("rl_engine.alignment.cross_config")

    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment", alignment)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment.cross_config", cross_config)

    with pytest.raises(RlkAlignmentStandardUnavailable, match="does not expose get_alignment_standard"):
        load_alignment_standard()


@pytest.mark.unit
def test_standard_score_artifact_export_requires_rl_kernel_schema():
    standard = _FakeStandard(schema_types={})

    with pytest.raises(RlkAlignmentStandardUnavailable, match="score-artifact schema"):
        build_standard_score_artifact({}, standard=standard)


@pytest.mark.unit
def test_framework_score_records_compare_through_standard_comparator():
    compared = {}

    def compare(rollout, training):
        compared["rollout"] = rollout
        compared["training"] = training
        return {
            "case_id": rollout.case_id,
            "mismatch_count": int((training.selected_logprobs[training.active_mask] != rollout.selected_logprobs[rollout.active_mask]).sum().item()),
        }

    standard = _FakeStandard(
        source="rl_kernel.fake",
        profiles=_profiles(),
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
        {
            **base_record,
            "side": "training",
            "selected_logprobs": torch.tensor([0.0, -9.0, -2.5]),
        },
        standard=standard,
    )

    assert result == {"case_id": "case-1", "mismatch_count": 1}
    assert compared["rollout"].side is _FakeScoreSide.ROLLOUT
    assert compared["training"].scorer.side is _FakeScoreSide.TRAINING
    assert compared["training"].active_mask.dtype is torch.bool
