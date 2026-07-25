"""Thin access layer for RL-Kernel-owned alignment standards.

vime should not be the source of truth for cross-framework alignment profiles,
score-artifact schemas, comparators, or logprob tolerance rules. This module
keeps the RL-Kernel touchpoint narrow: import the public standard provider,
translate vime-owned score records into RL-Kernel's schema, and call
RL-Kernel's comparator.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import torch


class RlkAlignmentStandardUnavailable(RuntimeError):
    """Raised when RL-Kernel's public alignment standard cannot be used."""


def load_alignment_standard(provider: Any = None) -> Any:
    """Return RL-Kernel's public alignment standard object."""

    if provider is None:
        try:
            cross_config = importlib.import_module("rl_engine.alignment.cross_config")
        except Exception as exc:
            raise RlkAlignmentStandardUnavailable("RL-Kernel alignment standard is unavailable; install RL-Kernel or pass an explicit test provider.") from exc
        get_standard = getattr(cross_config, "get_alignment_standard", None)
        if get_standard is None:
            raise RlkAlignmentStandardUnavailable("RL-Kernel does not expose get_alignment_standard(); vime does not carry a local A0-A5 matrix.")
        standard = get_standard()
    else:
        get_standard = getattr(provider, "get_alignment_standard", None)
        standard = get_standard() if get_standard is not None else provider
    _require_profiles(standard)
    return standard


def iter_alignment_profiles(provider: Any = None) -> tuple[Any, ...]:
    standard = load_alignment_standard(provider)
    iter_profiles = getattr(standard, "iter_profiles", None)
    if iter_profiles is not None:
        return tuple(iter_profiles())
    return tuple(_profiles_by_name(standard).values())


def select_least_restrictive_passing_profile(
    profile_names: set[str],
    *,
    provider: Any = None,
) -> Any | None:
    for profile in reversed(iter_alignment_profiles(provider)):
        if _profile_name(profile) in profile_names:
            return profile
    return None


def alignment_standard_metadata(standard: Any | None = None) -> dict[str, Any]:
    """Return stable report metadata that identifies the RL-Kernel standard."""

    standard = load_alignment_standard() if standard is None else standard
    to_metadata = getattr(standard, "to_metadata", None)
    if to_metadata is not None:
        return dict(to_metadata())

    metadata = dict(_standard_value(standard, "metadata", {}) or {})
    source = _standard_value(standard, "source", "rl_kernel")
    values = {
        "alignment_standard_source": source,
        "alignment_standard_id": _standard_value(standard, "standard_id", ""),
        "alignment_profile_version": _standard_value(standard, "profile_version", ""),
        "alignment_standard_fingerprint": _standard_value(
            standard,
            "fingerprint",
            _standard_value(standard, "standard_fingerprint", ""),
        ),
        "alignment_tolerance_fingerprint": _standard_value(
            standard,
            "tolerance_fingerprint",
            _standard_value(standard, "tolerance_contract_fingerprint", ""),
        ),
    }
    metadata.update({key: value for key, value in values.items() if value})
    issues = _standard_value(standard, "issues", ())
    if issues:
        metadata["alignment_standard_issues"] = tuple(issues)
    return metadata


def build_standard_score_artifact(
    record: Mapping[str, Any],
    *,
    standard: Any | None = None,
) -> Any:
    """Convert a framework-owned score record into RL-Kernel's ScoreArtifact."""

    standard = load_alignment_standard() if standard is None else standard
    schema_types = _schema_types(standard)
    if not _has_score_schema(schema_types):
        raise RlkAlignmentStandardUnavailable("RL-Kernel cross_config score-artifact schema is unavailable.")

    score_side_type = schema_types["ScoreSide"]
    side = record["side"]
    side_value = side if isinstance(side, score_side_type) else score_side_type(str(side))
    identity = _coerce_schema_value(
        schema_types["SemanticIdentitySpec"],
        record["identity"],
    )
    scorer_type = schema_types["ScorerSpec"]
    scorer_value = record["scorer"]
    if isinstance(scorer_value, scorer_type):
        scorer = scorer_value
    else:
        scorer_record = dict(scorer_value)
        scorer_record.setdefault("side", side_value)
        scorer = _coerce_schema_value(scorer_type, scorer_record)
    provenance = _coerce_schema_value(
        schema_types["RuntimeProvenance"],
        record["provenance"],
    )
    return schema_types["ScoreArtifact"](
        case_id=record["case_id"],
        attempt_id=record["attempt_id"],
        side=side_value,
        identity=identity,
        scorer=scorer,
        selected_logprobs=_tensor(record["selected_logprobs"]),
        active_mask=_tensor(record["active_mask"]).to(dtype=torch.bool),
        provenance=provenance,
    )


def compare_standard_score_records(
    rollout: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    standard: Any | None = None,
) -> Any:
    """Compare framework-owned rollout/training score records through RL-Kernel."""

    standard = load_alignment_standard() if standard is None else standard
    compare_score_artifacts = _standard_value(standard, "compare_score_artifacts")
    if compare_score_artifacts is None:
        raise RlkAlignmentStandardUnavailable("RL-Kernel score comparator is unavailable.")
    return compare_score_artifacts(
        build_standard_score_artifact(rollout, standard=standard),
        build_standard_score_artifact(training, standard=standard),
    )


def _require_profiles(standard: Any) -> None:
    iter_profiles = getattr(standard, "iter_profiles", None)
    profiles = tuple(iter_profiles()) if iter_profiles is not None else tuple(_profiles_by_name(standard).values())
    names = {_profile_name(profile) for profile in profiles}
    missing = [name for name in ("A0", "A1", "A2", "A3", "A4", "A5") if name not in names]
    if missing:
        raise RlkAlignmentStandardUnavailable(f"RL-Kernel alignment standard is missing profiles: {missing!r}.")


def _profiles_by_name(standard: Any) -> Mapping[str, Any]:
    profiles = _standard_value(standard, "profiles")
    if not isinstance(profiles, Mapping):
        raise RlkAlignmentStandardUnavailable("RL-Kernel alignment standard did not provide A0-A5 profiles.")
    return profiles


def _profile_name(profile: Any) -> str:
    if isinstance(profile, Mapping):
        return str(profile["name"])
    return str(profile.name)


def _schema_types(standard: Any) -> Mapping[str, Any]:
    schema_types = _standard_value(standard, "schema_types", {})
    return schema_types if isinstance(schema_types, Mapping) else {}


def _has_score_schema(schema_types: Mapping[str, Any]) -> bool:
    required = {
        "RuntimeProvenance",
        "ScoreArtifact",
        "ScorerSpec",
        "ScoreSide",
        "SemanticIdentitySpec",
    }
    return required.issubset({key for key, value in schema_types.items() if value is not None})


def _coerce_schema_value(schema_type: Any, value: Any) -> Any:
    if isinstance(value, schema_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping for {schema_type!r}, got {type(value)!r}.")
    return schema_type(**dict(value))


def _standard_value(standard: Any, name: str, default: Any = None) -> Any:
    if isinstance(standard, Mapping):
        return standard.get(name, default)
    return getattr(standard, name, default)


def _tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


__all__ = [
    "RlkAlignmentStandardUnavailable",
    "alignment_standard_metadata",
    "build_standard_score_artifact",
    "compare_standard_score_records",
    "iter_alignment_profiles",
    "load_alignment_standard",
    "select_least_restrictive_passing_profile",
]
