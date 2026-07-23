import ast
import importlib
import sys
import types
import warnings
from pathlib import Path

import pytest


def _fresh_adapter_module():
    sys.modules.pop("vime.backends.rl_kernel_utils", None)
    sys.modules.pop("vime.backends.rl_kernel_utils.adapter", None)
    return importlib.import_module("vime.backends.rl_kernel_utils.adapter")


def _drop_rl_engine_modules():
    for name in list(sys.modules):
        if name == "rl_engine" or name.startswith("rl_engine."):
            sys.modules.pop(name, None)


@pytest.mark.unit
def test_adapter_import_does_not_import_rl_engine():
    _drop_rl_engine_modules()

    _fresh_adapter_module()

    assert not any(name == "rl_engine" or name.startswith("rl_engine.") for name in sys.modules)


@pytest.mark.unit
def test_policy_context_is_built_from_resolved_vime_args():
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")
    args = types.SimpleNamespace(
        rlk_mode_config=types.SimpleNamespace(fast="auto", consistency="audit", ops=("linear_logp", "logp")),
    )

    policy = module.rlk_policy_context_from_args(args)

    assert policy.fast == "auto"
    assert policy.consistency == "audit"
    assert policy.enabled_ops == ("linear_logp", "logp")
    assert policy.operator_enabled("linear_logp")
    assert policy.operator_enabled("reference_logp")


@pytest.mark.unit
def test_policy_context_from_args_tolerates_unresolved_none_values():
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")
    args = types.SimpleNamespace(rlk_fast=None, rlk_consistency=None, rl_kernel_ops=None)

    policy = module.rlk_policy_context_from_args(args)

    assert policy.fast == "off"
    assert policy.consistency == "off"
    assert policy.enabled_ops == ()


@pytest.mark.unit
def test_noop_adapter_preserves_native_behavior_when_disabled():
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")
    policy = module.RlkPolicyContext(fast="off", consistency="off", enabled_ops=("linear_logp",))

    adapter = module.build_rlk_operator_adapter(policy)
    result = adapter.linear_logp(
        module.LinearLogpInputs(hidden="hidden", lm_head_weight="weight", target_ids=[1, 2, 3])
    )

    assert isinstance(adapter, module.NoOpRlkOperatorAdapter)
    assert result.value is None
    assert result.decision.path == "disabled"
    assert result.decision.token_count == 3
    assert adapter.telemetry.fallback_counts == {"linear_logp": 1}


@pytest.mark.unit
def test_consistency_audit_without_fast_path_keeps_operator_calls_native_only():
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")
    policy = module.RlkPolicyContext(fast="off", consistency="audit", enabled_ops=("linear_logp", "logp"))

    adapter = module.build_rlk_operator_adapter(policy)
    capability = adapter.capability("linear_logp")
    result = adapter.linear_logp(module.LinearLogpInputs(hidden="hidden", lm_head_weight="weight", target_ids=[1, 2]))

    assert isinstance(adapter, module.NoOpRlkOperatorAdapter)
    assert not policy.operator_enabled("linear_logp")
    assert not capability.available
    assert result.value is None
    assert result.decision.path == "disabled"


@pytest.mark.unit
def test_mock_adapter_captures_linear_logp_inputs_and_runtime_metadata():
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")
    loss_masks = object()
    batch = {
        "total_lengths": [5, 7],
        "response_lengths": [2, 3],
        "loss_masks": loss_masks,
        "rollout_log_probs": ["old-logp"],
        "rollout_top_p_token_ids": ["ids"],
        "rollout_top_p_token_offsets": ["offsets"],
    }
    policy = module.RlkPolicyContext(fast="auto", consistency="audit", enabled_ops=("linear_logp",))
    adapter = module.MockRlkOperatorAdapter(
        policy,
        available_ops=("linear_logp",),
        handlers={"linear_logp": lambda payload: (payload.hidden, payload.runtime.total_lengths, payload.metadata)},
    )

    inputs = module.linear_logp_inputs_from_vime(
        hidden="hidden-ref",
        lm_head_weight="weight-ref",
        target_ids=[4, 5],
        batch=batch,
        tp_group="tp-group",
        vocab_start_index=8,
        global_vocab_size=16,
        metadata={"weight_version": "actor@1"},
    )
    result = adapter.linear_logp(inputs)

    assert inputs.runtime.total_lengths == (5, 7)
    assert inputs.runtime.response_lengths == (2, 3)
    assert inputs.runtime.loss_masks is loss_masks
    assert inputs.tp_group == "tp-group"
    assert inputs.vocab_start_index == 8
    assert inputs.global_vocab_size == 16
    assert result.value == ("hidden-ref", (5, 7), {"weight_version": "actor@1"})
    assert result.decision.path == "mock"
    assert adapter.telemetry.call_counts == {"linear_logp": 1}
    assert adapter.telemetry.token_counts == {"linear_logp": 2}

    contract = adapter.contract("linear_logp", runtime=inputs.runtime, metadata={"source": "unit"})
    assert contract.op_name == "linear_logp"
    assert contract.runtime is inputs.runtime
    assert contract.metadata == {"source": "unit"}
    assert contract.policy["fast"] == "auto"


@pytest.mark.unit
def test_mock_adapter_reports_unsupported_operator_without_rl_kernel():
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")
    policy = module.RlkPolicyContext(fast="auto", consistency="off", enabled_ops=("linear_logp",))
    adapter = module.MockRlkOperatorAdapter(policy, available_ops=("logp",))

    result = adapter.linear_logp(module.LinearLogpInputs(hidden="hidden", lm_head_weight="weight", target_ids=[1, 2]))

    assert result.value is None
    assert result.decision.path == "unsupported"
    assert result.decision.token_count == 2
    assert "unsupported" in result.decision.reason
    assert adapter.telemetry.fallback_counts == {"linear_logp": 1}


@pytest.mark.unit
def test_mock_reference_logprobs_reuses_logp_handler_when_reference_handler_is_absent():
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")
    policy = module.RlkPolicyContext(fast="auto", consistency="audit", enabled_ops=("logp",))
    adapter = module.MockRlkOperatorAdapter(
        policy,
        available_ops=("logp",),
        handlers={"logp": lambda payload: ("logp", payload.logits, payload.target_ids)},
    )

    result = adapter.reference_logprobs(module.ReferenceScoreInputs(logits="ref-logits", target_ids=[3]))

    assert result.value == ("logp", "ref-logits", [3])
    assert result.decision.path == "mock"
    assert adapter.telemetry.call_counts == {"reference_logp": 1}


@pytest.mark.unit
def test_registry_adapter_defers_rl_engine_import_until_capability_query(monkeypatch):
    module = _fresh_adapter_module()
    _drop_rl_engine_modules()

    adapter = module.build_rlk_operator_adapter(
        module.RlkPolicyContext(fast="auto", consistency="off", enabled_ops=("linear_logp",))
    )

    assert isinstance(adapter, module.RlkRegistryOperatorAdapter)
    assert not any(name == "rl_engine" or name.startswith("rl_engine.") for name in sys.modules)

    class FakeLinearLogpOp:
        def __init__(self):
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "rlk-output"

    fake_op = FakeLinearLogpOp()

    class FakeRegistry:
        def __init__(self):
            self.requested = []

        def get_op(self, op_name):
            self.requested.append(op_name)
            assert op_name == "linear_logp"
            return fake_op

    fake_registry = FakeRegistry()
    rl_engine_mod = types.ModuleType("rl_engine")
    kernels_mod = types.ModuleType("rl_engine.kernels")
    registry_mod = types.ModuleType("rl_engine.kernels.registry")
    registry_mod.kernel_registry = fake_registry
    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine_mod)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", kernels_mod)
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry_mod)

    result = adapter.linear_logp(
        module.LinearLogpInputs(
            hidden="hidden",
            lm_head_weight="weight",
            target_ids=[9],
            bias="bias",
            tp_group="tp",
            vocab_start_index=4,
            global_vocab_size=12,
        )
    )

    assert fake_registry.requested == ["linear_logp"]
    assert result.value == "rlk-output"
    assert result.decision.path == "fast"
    assert result.decision.backend == "FakeLinearLogpOp"
    assert fake_op.calls == [
        (
            ("hidden", "weight", [9], "bias"),
            {"tp_group": "tp", "vocab_start_index": 4, "global_vocab_size": 12},
        )
    ]


@pytest.mark.unit
def test_registry_logprob_hooks_accept_int_masks(monkeypatch):
    torch = pytest.importorskip("torch")
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")

    class FakeLogpOp:
        def __call__(self, logits, target_ids):
            return torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32)

    class FakeRegistry:
        def get_op(self, op_name):
            assert op_name == "logp"
            return FakeLogpOp()

    registry_mod = types.ModuleType("rl_engine.kernels.registry")
    registry_mod.kernel_registry = FakeRegistry()
    monkeypatch.setitem(sys.modules, "rl_engine", types.ModuleType("rl_engine"))
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", types.ModuleType("rl_engine.kernels"))
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry_mod)

    adapter = module.RlkRegistryOperatorAdapter(
        module.RlkPolicyContext(fast="auto", consistency="audit", enabled_ops=("logp",))
    )
    inputs = module.SelectedLogprobInputs(
        logits="unused",
        target_ids=torch.tensor([1, 2, 3]),
        mask=torch.tensor([1, 0, 1], dtype=torch.int32),
    )

    result = adapter.selected_logprobs(inputs)

    torch.testing.assert_close(result.value, torch.tensor([-0.1, 0.0, -0.3], dtype=torch.float32))


@pytest.mark.unit
def test_registry_adapter_strict_mode_raises_when_enabled_op_is_unavailable(monkeypatch):
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")

    class MissingRegistry:
        def get_op(self, op_name):
            raise RuntimeError(f"{op_name} missing")

    registry_mod = types.ModuleType("rl_engine.kernels.registry")
    registry_mod.kernel_registry = MissingRegistry()
    monkeypatch.setitem(sys.modules, "rl_engine", types.ModuleType("rl_engine"))
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", types.ModuleType("rl_engine.kernels"))
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry_mod)

    adapter = module.RlkRegistryOperatorAdapter(
        module.RlkPolicyContext(fast="strict", consistency="off", enabled_ops=("linear_logp",))
    )

    with pytest.raises(module.RlkOperatorUnavailable, match="linear_logp"):
        adapter.linear_logp(module.LinearLogpInputs(hidden="h", lm_head_weight="w", target_ids=[1]))


@pytest.mark.unit
def test_registry_adapter_fallback_records_token_count_when_op_is_unavailable(monkeypatch):
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")

    class MissingRegistry:
        def get_op(self, op_name):
            raise RuntimeError(f"{op_name} missing")

    registry_mod = types.ModuleType("rl_engine.kernels.registry")
    registry_mod.kernel_registry = MissingRegistry()
    monkeypatch.setitem(sys.modules, "rl_engine", types.ModuleType("rl_engine"))
    monkeypatch.setitem(sys.modules, "rl_engine.kernels", types.ModuleType("rl_engine.kernels"))
    monkeypatch.setitem(sys.modules, "rl_engine.kernels.registry", registry_mod)

    adapter = module.RlkRegistryOperatorAdapter(
        module.RlkPolicyContext(fast="auto", consistency="off", enabled_ops=("linear_logp",))
    )
    result = adapter.linear_logp(module.LinearLogpInputs(hidden="h", lm_head_weight="w", target_ids=[1, 2, 3]))

    assert result.value is None
    assert result.decision.path == "fallback"
    assert result.decision.token_count == 3
    assert adapter.telemetry.fallback_counts == {"linear_logp": 1}
    assert adapter.telemetry.token_counts == {"linear_logp": 3}


@pytest.mark.unit
def test_selected_and_reference_logprob_hooks_share_boundary_contract():
    module = importlib.import_module("vime.backends.rl_kernel_utils.adapter")
    policy = module.RlkPolicyContext(fast="auto", consistency="audit", enabled_ops=("logp",))
    adapter = module.MockRlkOperatorAdapter(
        policy,
        available_ops=("logp",),
        handlers={
            "logp": lambda payload: ("selected", payload.logits, payload.target_ids),
            "reference_logp": lambda payload: ("reference", payload.logits, payload.target_ids),
        },
    )

    selected = adapter.selected_logprobs(module.SelectedLogprobInputs(logits="logits", target_ids=[1, 2]))
    reference = adapter.reference_logprobs(module.ReferenceScoreInputs(logits="ref-logits", target_ids=[3]))

    assert selected.value == ("selected", "logits", [1, 2])
    assert reference.value == ("reference", "ref-logits", [3])
    assert adapter.telemetry.call_counts == {"logp": 1, "reference_logp": 1}


@pytest.mark.unit
def test_production_vime_code_does_not_import_rl_engine_outside_adapter_boundary():
    repo = Path(__file__).resolve().parents[1]
    allowed = Path("vime/backends/rl_kernel_utils/adapter.py")
    bad_imports = []

    for package_root in (repo / "vime", repo / "vime_plugins"):
        for path in package_root.rglob("*.py"):
            rel = path.relative_to(repo).as_posix()
            if rel == allowed.as_posix() or "__pycache__" in path.parts:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message="invalid escape sequence.*")
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "rl_engine" or alias.name.startswith("rl_engine."):
                            bad_imports.append((rel, node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if module_name == "rl_engine" or module_name.startswith("rl_engine."):
                        bad_imports.append((rel, node.lineno, module_name))

    assert bad_imports == []


@pytest.mark.unit
def test_adapter_boundary_only_references_rl_kernel_public_registry():
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "vime" / "backends" / "rl_kernel_utils" / "adapter.py").read_text(encoding="utf-8")

    assert '"rl_engine.kernels.registry"' in source
    forbidden_fragments = [
        "PairedRunner",
        "RuntimeTools",
        "cross_config",
        "child_process",
        "artifact_layout",
        "attempt_artifact",
    ]
    assert [fragment for fragment in forbidden_fragments if fragment in source] == []


@pytest.mark.unit
def test_rl_kernel_adapter_boundary_is_documented():
    repo = Path(__file__).resolve().parents[1]
    doc = repo / "docs" / "en" / "advanced" / "rl-kernel-operator-adapter.md"
    text = doc.read_text(encoding="utf-8")

    assert "RlkOperatorAdapter" in text
    assert "vime.backends.rl_kernel_utils" in text
    assert "rl_engine.kernels.registry" in text
    assert "contract(op_name" in text
