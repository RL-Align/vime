import ast
import sys
import types
from pathlib import Path

import pytest


def _drop_rl_engine_modules(monkeypatch):
    for name in list(sys.modules):
        if name == "rl_engine" or name.startswith("rl_engine."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _install_fake_rl_kernel_registry(monkeypatch, registry):
    rl_engine = types.ModuleType("rl_engine")
    rl_engine.__path__ = []
    kernels = types.ModuleType("rl_engine.kernels")
    kernels.__path__ = []
    registry_mod = types.ModuleType("rl_engine.kernels.registry")
    registry_mod.kernel_registry = registry

    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry_mod)


def _linear_logp_capabilities():
    from vime.backends.rl_kernel_utils import BackendCapability, NumericContract, RlKernelCapabilities

    return RlKernelCapabilities(
        available=True,
        runtime_fingerprint="runtime-1",
        build_fingerprint="build-1",
        backends=(
            BackendCapability(
                operator="linear_logp",
                backend_id="rlk.linear_logp.fast",
                implementation_kind="optimized",
                dtypes=("bf16",),
                deterministic=True,
                batch_invariant=True,
                strict_fast_eligible=True,
                runtime_fingerprint="runtime-1",
                build_fingerprint="build-1",
                numeric_contract=NumericContract(
                    contract_id="rlk.linear_logp.fp32",
                    accumulation_dtype="fp32",
                    reduction_order="canonical",
                    backend_id="rlk.linear_logp.fast",
                    tolerance_by_dtype={"bf16": {"atol": 1e-3}},
                ),
            ),
        ),
    )


@pytest.mark.unit
def test_disabled_factory_returns_noop_without_querying_or_importing_rl_kernel(monkeypatch):
    _drop_rl_engine_modules(monkeypatch)

    from vime.backends.rl_kernel_utils import RlkLinearLogpRequest, RlkPolicyContext, build_rlk_operator_adapter
    from vime.backends.rl_kernel_utils.operator_adapter import NoopRlkOperatorAdapter

    queried = False

    def provider():
        nonlocal queried
        queried = True
        raise AssertionError("disabled adapter must not query capabilities")

    adapter = build_rlk_operator_adapter(
        RlkPolicyContext(fast_path="off", consistency_path="off", enabled_ops=frozenset({"linear_logp"})),
        capability_provider=provider,
    )
    result = adapter.linear_logp(RlkLinearLogpRequest(hidden="h", lm_head_weight="w", target_ids="t"))

    assert isinstance(adapter, NoopRlkOperatorAdapter)
    assert queried is False
    assert result.handled is False
    assert result.supported is False
    assert result.value is None
    assert result.decision.decision == "native"
    assert result.decision.actual_backend == "vime.native"
    assert result.contract.contract_id == "vime.native.linear_logp"
    assert result.metadata["fallback_reason"]["code"] == "rl_kernel_disabled"
    assert not any(name == "rl_engine" or name.startswith("rl_engine.") for name in sys.modules)


@pytest.mark.unit
def test_noop_adapter_reports_audit_only_when_consistency_audit_is_resolved():
    from vime.backends.rl_kernel_utils import NoopRlkOperatorAdapter, RlkPolicyContext, RlkSelectedLogprobRequest

    adapter = NoopRlkOperatorAdapter(
        RlkPolicyContext(fast_path="off", consistency_path="audit", enabled_ops=frozenset({"selected_logprob"}))
    )
    result = adapter.selected_logprob(RlkSelectedLogprobRequest(logits="logits", token_ids="ids", stage="audit"))

    assert result.handled is False
    assert result.decision.decision == "audit-only"
    assert result.decision.operator == "logp"
    assert result.provenance.adapter_kind == "noop"


@pytest.mark.unit
def test_mock_adapter_successful_linear_logp_boundary_result():
    from vime.backends.rl_kernel_utils import MockRlkOperatorAdapter, RlkLinearLogpRequest, RlkPolicyContext

    seen = {}

    def handler(request):
        seen["request"] = request
        return {"logp": [1.0, 2.0]}

    policy = RlkPolicyContext(
        fast_path="auto",
        consistency_path="strict",
        enabled_ops=frozenset({"linear_logp"}),
        dtype="bf16",
        stage="train_logprob",
        parallel_context={"tp": 2},
        provenance={"weight_version": "policy-resolved-by-vime"},
    )
    adapter = MockRlkOperatorAdapter(policy, handlers={"linear_logp": handler})

    result = adapter.linear_logp(
        RlkLinearLogpRequest(
            hidden="hidden",
            lm_head_weight="weight",
            target_ids="target",
            parallel_context={"rank": 1},
            metadata={"tokens": 2},
        )
    )

    assert seen["request"].metadata == {"tokens": 2}
    assert result.handled is True
    assert result.value == {"logp": [1.0, 2.0]}
    assert result.decision.decision == "strict-fast"
    assert result.decision.actual_backend == "mock.linear_logp"
    assert result.decision.parallel_context == {"tp": 2, "rank": 1}
    assert result.metadata["request_metadata"] == {"tokens": 2}
    assert result.provenance.adapter_kind == "mock"
    assert result.provenance.runtime_fingerprint == "mock-runtime"
    assert result.provenance.build_fingerprint == "mock-build"
    assert result.provenance.policy_context["provenance"] == {"weight_version": "policy-resolved-by-vime"}


@pytest.mark.unit
def test_mock_adapter_supports_selected_logprob_and_reference_score_hooks():
    from vime.backends.rl_kernel_utils import (
        MockRlkOperatorAdapter,
        RlkPolicyContext,
        RlkReferenceScoreRequest,
        RlkSelectedLogprobRequest,
    )

    adapter = MockRlkOperatorAdapter(
        RlkPolicyContext(fast_path="auto", enabled_ops=frozenset({"selected_logprob", "reference_score"})),
        handlers={
            "selected_logprob": lambda request: ("selected", request.logits, request.token_ids),
            "reference_score": lambda request: ("reference", request.args, dict(request.kwargs)),
        },
    )

    selected = adapter.selected_logprob(RlkSelectedLogprobRequest(logits="logits", token_ids="tokens"))
    reference = adapter.reference_score(RlkReferenceScoreRequest(args=("sample",), kwargs={"model": "ref"}))

    assert selected.handled is True
    assert selected.value == ("selected", "logits", "tokens")
    assert selected.decision.actual_backend == "mock.logp"
    assert reference.handled is True
    assert reference.value == ("reference", ("sample",), {"model": "ref"})
    assert reference.decision.actual_backend == "mock.reference_score"


@pytest.mark.unit
def test_mock_adapter_unsupported_operator_is_boundary_level_fallback():
    from vime.backends.rl_kernel_utils import MockRlkOperatorAdapter, RlkPolicyContext, RlkSelectedLogprobRequest

    adapter = MockRlkOperatorAdapter(
        RlkPolicyContext(fast_path="auto", enabled_ops=frozenset({"selected_logprob"})),
        handlers={},
    )
    result = adapter.selected_logprob(RlkSelectedLogprobRequest(logits="logits", token_ids="tokens"))

    assert result.handled is False
    assert result.decision.decision == "fallback-native"
    assert result.decision.fallback_reason.code == "operator_unsupported"
    assert result.decision.fallback_reason.details == {"operator": "logp"}
    assert result.metadata["fallback_reason"]["code"] == "operator_unsupported"


@pytest.mark.unit
def test_policy_disabled_operator_does_not_select_available_backend():
    from vime.backends.rl_kernel_utils import MockRlkOperatorAdapter, RlkLinearLogpRequest, RlkPolicyContext

    adapter = MockRlkOperatorAdapter(
        RlkPolicyContext(fast_path="auto", enabled_ops=frozenset({"selected_logprob"}), dtype="bf16"),
        handlers={"linear_logp": lambda request: "would-be-fast"},
    )

    result = adapter.linear_logp(RlkLinearLogpRequest(hidden="h", lm_head_weight="w", target_ids="t"))

    assert result.handled is False
    assert result.decision.decision == "native"
    assert result.decision.actual_backend == "vime.native"
    assert result.decision.fallback_reason.code == "operator_disabled"
    assert result.contract.contract_id == "vime.native.linear_logp"


@pytest.mark.unit
def test_registry_adapter_lazily_invokes_rl_kernel_linear_logp_with_vime_context(monkeypatch):
    _drop_rl_engine_modules(monkeypatch)

    from vime.backends.rl_kernel_utils import RlkLinearLogpRequest, RlkPolicyContext, build_rlk_operator_adapter

    class FakeLinearLogpOp:
        def __init__(self):
            self.calls = []

        def __call__(self, hidden, lm_head_weight, target_ids, bias=None, **kwargs):
            self.calls.append(
                {
                    "hidden": hidden,
                    "lm_head_weight": lm_head_weight,
                    "target_ids": target_ids,
                    "bias": bias,
                    "kwargs": kwargs,
                }
            )
            return "linear-logp-value"

    class FakeRegistry:
        def __init__(self):
            self.requested_ops = []
            self.op = FakeLinearLogpOp()

        def get_op(self, operator):
            self.requested_ops.append(operator)
            return self.op

    registry = FakeRegistry()
    _install_fake_rl_kernel_registry(monkeypatch, registry)

    policy = RlkPolicyContext(
        fast_path="auto",
        enabled_ops=frozenset({"linear_logp"}),
        requested_backend="rlk.linear_logp.fast",
        dtype="bf16",
        stage="train_logprob",
        parallel_context={"tp": 2},
        provenance={"source": "resolved-vime-config"},
        telemetry_enabled=False,
    )
    adapter = build_rlk_operator_adapter(policy, capability_provider=_linear_logp_capabilities())

    result = adapter.linear_logp(
        RlkLinearLogpRequest(
            hidden="hidden",
            lm_head_weight="weight",
            target_ids="target",
            bias="bias",
            tp_group="tp-group",
            vocab_start_index=16,
            global_vocab_size=128,
            parallel_context={"rank": 0},
        )
    )

    assert registry.requested_ops == ["linear_logp"]
    assert registry.op.calls == [
        {
            "hidden": "hidden",
            "lm_head_weight": "weight",
            "target_ids": "target",
            "bias": "bias",
            "kwargs": {
                "tp_group": "tp-group",
                "vocab_start_index": 16,
                "global_vocab_size": 128,
            },
        }
    ]
    assert result.handled is True
    assert result.value == "linear-logp-value"
    assert result.decision.decision == "optimized"
    assert result.decision.actual_backend == "rlk.linear_logp.fast"
    assert result.contract.contract_id == "rlk.linear_logp.fp32"
    assert result.contract.dtype == "bf16"
    assert result.metadata["operator_impl"] == "FakeLinearLogpOp"
    assert result.provenance.adapter_kind == "rl_kernel"
    assert result.provenance.runtime_fingerprint == "runtime-1"
    assert result.provenance.build_fingerprint == "build-1"
    assert result.provenance.policy_context["provenance"] == {"source": "resolved-vime-config"}


@pytest.mark.unit
def test_registry_adapter_does_not_import_rl_kernel_until_supported_call(monkeypatch):
    _drop_rl_engine_modules(monkeypatch)

    from vime.backends.rl_kernel_utils import RlkPolicyContext, build_rlk_operator_adapter

    adapter = build_rlk_operator_adapter(
        RlkPolicyContext(fast_path="auto", enabled_ops=frozenset({"linear_logp"})),
        capability_provider=_linear_logp_capabilities(),
    )

    assert adapter.query_capabilities("linear_logp").capabilities.backends[0].backend_id == "rlk.linear_logp.fast"
    assert not any(name == "rl_engine" or name.startswith("rl_engine.") for name in sys.modules)


@pytest.mark.unit
def test_production_vime_imports_rl_kernel_only_through_operator_adapter_boundary():
    repo_root = Path(__file__).resolve().parents[2]
    production_root = repo_root / "vime"
    adapter_path = production_root / "backends" / "rl_kernel_utils" / "operator_adapter.py"
    forbidden_runtime_prefixes = (
        "rl_engine.executors",
        "rl_engine.testing",
        "rl_engine.tests",
        "benchmarks",
        "scripts",
    )
    forbidden_source_markers = (
        "rl_engine.executors",
        "rl_engine.testing",
        "rl_engine.tests",
        "benchmark_rl_kernels",
        "benchmark_linear_logp",
        "child_process",
        "attempt_artifact",
        "attempt_layout",
        "cross_config",
    )

    violations = []
    for path in production_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        for module in imported_modules:
            if module == "rl_engine" or module.startswith("rl_engine."):
                if path != adapter_path:
                    violations.append(f"{path.relative_to(repo_root)} imports {module}")
                if module.startswith(forbidden_runtime_prefixes):
                    violations.append(f"{path.relative_to(repo_root)} imports forbidden RL-Kernel runtime {module}")
            if module.startswith(forbidden_runtime_prefixes):
                violations.append(f"{path.relative_to(repo_root)} imports forbidden runner/artifact module {module}")

        if path != adapter_path:
            for marker in forbidden_source_markers:
                if marker in source:
                    violations.append(f"{path.relative_to(repo_root)} references forbidden marker {marker}")

    assert violations == []
