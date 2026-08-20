"""Thin access layer for RL-Kernel-owned alignment contracts.

vime consumes the public score comparator and module mismatch matrix, but does
not redefine either contract. The matrix remains a fixed-replay diagnostic
manifest; this module only makes its module-level P/R cases available to vime
callers and external runners.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


class RlkAlignmentStandardUnavailable(RuntimeError):
    """Raised when RL-Kernel's public alignment contracts cannot be used."""


@dataclass(frozen=True)
class RlkAlignmentStandard:
    """Normalized view of the public RL-Kernel alignment contracts."""

    module_debug_matrix: Mapping[str, Any]
    compare_score_artifacts: Any
    schema_types: Mapping[str, Any]
    source: str = "rl_kernel"
    standard_id: str = "rl_kernel.cross_config.module_debug_matrix"
    tolerance_fingerprint: str = ""

    @property
    def profile_version(self) -> str:
        return str(self.module_debug_matrix["schema_version"])

    @property
    def fingerprint(self) -> str:
        return _stable_fingerprint(self.module_debug_matrix)

    def to_metadata(self) -> dict[str, str]:
        return {
            "alignment_standard_source": self.source,
            "alignment_standard_id": self.standard_id,
            "alignment_matrix_schema_version": self.profile_version,
            "alignment_standard_fingerprint": self.fingerprint,
            "alignment_tolerance_fingerprint": self.tolerance_fingerprint,
        }


@dataclass(frozen=True)
class OperatorAblationCase:
    """One module-local production/RL-Kernel comparison case."""

    module: str
    case_id: str
    training_implementation: str
    rollout_implementation: str
    purpose: str
    diagnostic_axes: tuple[str, ...]
    matrix_schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "case_id": self.case_id,
            "training_implementation": self.training_implementation,
            "rollout_implementation": self.rollout_implementation,
            "purpose": self.purpose,
            "diagnostic_axes": self.diagnostic_axes,
            "matrix_schema_version": self.matrix_schema_version,
        }


_CASE_DEFINITIONS = (
    ("P/P", "production", "production", "native baseline"),
    ("R/R", "rl_kernel", "rl_kernel", "RL-Kernel control"),
    ("P/R", "production", "rl_kernel", "rollout-only mismatch"),
    ("R/P", "rl_kernel", "production", "training-only mismatch"),
)


def load_alignment_standard(provider: Any = None) -> Any:
    """Return the normalized RL-Kernel public alignment contracts."""

    if provider is not None:
        get_standard = getattr(provider, "get_alignment_standard", None)
        standard = get_standard() if get_standard is not None else provider
        if _standard_value(standard, "module_debug_matrix") is not None:
            _require_module_debug_matrix(_standard_value(standard, "module_debug_matrix"))
        elif not _has_legacy_profiles(standard):
            raise RlkAlignmentStandardUnavailable("RL-Kernel did not provide a module debug matrix.")
        return standard

    try:
        cross_config = importlib.import_module("rl_engine.alignment.cross_config")
        debug_matrix = importlib.import_module("rl_engine.alignment.cross_config.debug_matrix")
        schema = importlib.import_module("rl_engine.alignment.cross_config.schema")
        tolerance = importlib.import_module("rl_engine.kernels.gtest.tolerance")
    except Exception as exc:
        raise RlkAlignmentStandardUnavailable(
            "RL-Kernel alignment contracts are unavailable; install RL-Kernel or pass an explicit test provider."
        ) from exc

    module_debug_matrix = getattr(debug_matrix, "module_debug_matrix", None)
    if module_debug_matrix is None:
        raise RlkAlignmentStandardUnavailable("RL-Kernel does not expose module_debug_matrix().")
    compare_score_artifacts = getattr(cross_config, "compare_score_artifacts", None)
    if compare_score_artifacts is None:
        raise RlkAlignmentStandardUnavailable("RL-Kernel does not expose compare_score_artifacts().")

    matrix = module_debug_matrix()
    _require_module_debug_matrix(matrix)
    schema_types = {
        name: getattr(schema, name, None)
        for name in ("RuntimeProvenance", "ScoreArtifact", "ScorerSpec", "ScoreSide", "SemanticIdentitySpec")
    }
    if not _has_score_schema(schema_types):
        raise RlkAlignmentStandardUnavailable("RL-Kernel cross_config score-artifact schema is unavailable.")
    tolerance_fingerprint = getattr(tolerance, "tolerance_contract_fingerprint", lambda: "")()
    return RlkAlignmentStandard(
        module_debug_matrix=matrix,
        compare_score_artifacts=compare_score_artifacts,
        schema_types=schema_types,
        tolerance_fingerprint=str(tolerance_fingerprint),
    )


def load_operator_ablation_matrix(provider: Any = None) -> Mapping[str, Any]:
    """Return RL-Kernel's module mismatch manifest without redefining it."""

    standard = load_alignment_standard(provider)
    matrix = _standard_value(standard, "module_debug_matrix")
    _require_module_debug_matrix(matrix)
    return matrix


def iter_operator_ablation_cases(
    module: str,
    *,
    provider: Any = None,
) -> tuple[OperatorAblationCase, ...]:
    """Build the four one-module P/R cases for a manifest module."""

    matrix = load_operator_ablation_matrix(provider)
    try:
        module_manifest = matrix["modules"][module]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown RL-Kernel ablation module {module!r}") from exc
    axes = tuple(str(axis["id"]) for axis in module_manifest.get("axes", ()))
    return tuple(
        OperatorAblationCase(
            module=module,
            case_id=case_id,
            training_implementation=training_implementation,
            rollout_implementation=rollout_implementation,
            purpose=purpose,
            diagnostic_axes=axes,
            matrix_schema_version=str(matrix["schema_version"]),
        )
        for case_id, training_implementation, rollout_implementation, purpose in _CASE_DEFINITIONS
    )


def select_operator_ablation_case(
    module: str,
    case_id: str,
    *,
    provider: Any = None,
) -> OperatorAblationCase:
    """Select one fixed P/R case by its stable table label."""

    normalized_case_id = case_id.strip().upper()
    for case in iter_operator_ablation_cases(module, provider=provider):
        if case.case_id == normalized_case_id:
            return case
    raise ValueError(f"unknown RL-Kernel ablation case {case_id!r}")


def iter_alignment_profiles(provider: Any = None) -> tuple[Any, ...]:
    """Compatibility reader for an explicitly supplied legacy provider.

    The default RL-Kernel path has no profile catalog; callers should use
    ``iter_operator_ablation_cases`` for the current module-level contract.
    """

    standard = load_alignment_standard(provider)
    iterator = getattr(standard, "iter_profiles", None)
    if iterator is not None:
        return tuple(iterator())
    profiles = _standard_value(standard, "profiles")
    if isinstance(profiles, Mapping):
        return tuple(profiles.values())
    raise RlkAlignmentStandardUnavailable("RL-Kernel alignment profiles are not part of the current public contract.")


def select_least_restrictive_passing_profile(
    profile_names: set[str],
    *,
    provider: Any = None,
) -> Any | None:
    """Compatibility selector for an explicitly supplied legacy provider."""

    for profile in reversed(iter_alignment_profiles(provider)):
        name = profile.get("name") if isinstance(profile, Mapping) else getattr(profile, "name", None)
        if str(name) in profile_names:
            return profile
    return None


def alignment_standard_metadata(standard: Any | None = None) -> dict[str, Any]:
    """Return stable metadata that identifies the consumed RL-Kernel contract."""

    standard = load_alignment_standard() if standard is None else standard
    to_metadata = getattr(standard, "to_metadata", None)
    if to_metadata is not None:
        return {key: value for key, value in dict(to_metadata()).items() if value != ""}

    matrix = _standard_value(standard, "module_debug_matrix")
    _require_module_debug_matrix(matrix)
    return {
        "alignment_standard_source": _standard_value(standard, "source", "rl_kernel"),
        "alignment_standard_id": _standard_value(standard, "standard_id", "rl_kernel.cross_config.module_debug_matrix"),
        "alignment_matrix_schema_version": str(matrix["schema_version"]),
        "alignment_standard_fingerprint": _standard_value(standard, "fingerprint", _stable_fingerprint(matrix)),
        "alignment_tolerance_fingerprint": _standard_value(standard, "tolerance_fingerprint", ""),
    }


def build_standard_score_artifact(
    record: Mapping[str, Any],
    *,
    standard: Any | None = None,
) -> Any:
    """Convert a vime-owned score record into RL-Kernel's ScoreArtifact."""

    standard = load_alignment_standard() if standard is None else standard
    schema_types = _schema_types(standard)
    if not _has_score_schema(schema_types):
        raise RlkAlignmentStandardUnavailable("RL-Kernel cross_config score-artifact schema is unavailable.")

    score_side_type = schema_types["ScoreSide"]
    side = record["side"]
    side_value = side if isinstance(side, score_side_type) else score_side_type(str(side))
    identity = _coerce_schema_value(schema_types["SemanticIdentitySpec"], record["identity"])
    scorer_type = schema_types["ScorerSpec"]
    scorer_value = record["scorer"]
    if isinstance(scorer_value, scorer_type):
        scorer = scorer_value
    else:
        scorer_record = dict(scorer_value)
        scorer_record.setdefault("side", side_value)
        scorer = _coerce_schema_value(scorer_type, scorer_record)
    provenance = _coerce_schema_value(schema_types["RuntimeProvenance"], record["provenance"])
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
    """Compare vime-owned rollout/training score records through RL-Kernel."""

    standard = load_alignment_standard() if standard is None else standard
    compare_score_artifacts = _standard_value(standard, "compare_score_artifacts")
    if compare_score_artifacts is None:
        raise RlkAlignmentStandardUnavailable("RL-Kernel score comparator is unavailable.")
    return compare_score_artifacts(
        build_standard_score_artifact(rollout, standard=standard),
        build_standard_score_artifact(training, standard=standard),
    )


def _require_module_debug_matrix(matrix: Any) -> None:
    if not isinstance(matrix, Mapping) or not isinstance(matrix.get("schema_version"), str):
        raise RlkAlignmentStandardUnavailable("RL-Kernel did not provide a valid module debug matrix.")
    modules = matrix.get("modules")
    if not isinstance(modules, Mapping) or not {"attention", "ffn", "logp"}.issubset(modules):
        raise RlkAlignmentStandardUnavailable("RL-Kernel module debug matrix is missing attention, ffn, or logp.")


def _has_legacy_profiles(standard: Any) -> bool:
    iterator = getattr(standard, "iter_profiles", None)
    if iterator is not None:
        return True
    return isinstance(_standard_value(standard, "profiles"), Mapping)


def _schema_types(standard: Any) -> Mapping[str, Any]:
    schema_types = _standard_value(standard, "schema_types", {})
    return schema_types if isinstance(schema_types, Mapping) else {}


def _has_score_schema(schema_types: Mapping[str, Any]) -> bool:
    required = {"RuntimeProvenance", "ScoreArtifact", "ScorerSpec", "ScoreSide", "SemanticIdentitySpec"}
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


def _stable_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


__all__ = [
    "OperatorAblationCase",
    "RlkAlignmentStandard",
    "RlkAlignmentStandardUnavailable",
    "alignment_standard_metadata",
    "build_standard_score_artifact",
    "compare_standard_score_records",
    "iter_alignment_profiles",
    "iter_operator_ablation_cases",
    "load_alignment_standard",
    "load_operator_ablation_matrix",
    "select_least_restrictive_passing_profile",
    "select_operator_ablation_case",
]
