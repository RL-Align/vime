import importlib
import sys
from types import ModuleType

import pytest


def _drop_rl_engine_modules() -> None:
    for name in list(sys.modules):
        if name == "rl_engine" or name.startswith("rl_engine."):
            sys.modules.pop(name, None)


def _install_fake_rl_kernel_operator_comparison(monkeypatch) -> ModuleType:
    rl_engine = ModuleType("rl_engine")
    alignment = ModuleType("rl_engine.alignment")
    cross_config = ModuleType("rl_engine.alignment.cross_config")
    operator_comparison = ModuleType("rl_engine.alignment.cross_config.operator_comparison")

    class FakeOperatorTolerance:
        def __init__(self, *, atol=1e-6, rtol=1e-6):
            self.atol = atol
            self.rtol = rtol

    operator_comparison.RLK_OP_LOGP = "logp"
    operator_comparison.PHASE4_TARGET_OPERATORS = ("logp",)
    operator_comparison.OPERATOR_COMPARISON_SPECS = {"logp": "spec"}
    operator_comparison.OperatorTolerance = FakeOperatorTolerance
    operator_comparison.iter_operator_comparison_specs = lambda: ("spec",)
    operator_comparison.compare_operator_outputs = lambda op_name, train, infer, **kwargs: {
        "op_name": op_name,
        "train": train,
        "infer": infer,
        "kwargs": kwargs,
    }

    monkeypatch.setitem(sys.modules, "rl_engine", rl_engine)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment", alignment)
    monkeypatch.setitem(sys.modules, "rl_engine.alignment.cross_config", cross_config)
    monkeypatch.setitem(
        sys.modules,
        "rl_engine.alignment.cross_config.operator_comparison",
        operator_comparison,
    )
    return operator_comparison


@pytest.mark.unit
def test_operator_comparison_adapter_import_does_not_import_rl_engine():
    _drop_rl_engine_modules()
    sys.modules.pop("vime.backends.rl_kernel_utils.operator_comparison", None)

    importlib.import_module("vime.backends.rl_kernel_utils.operator_comparison")

    assert not any(name == "rl_engine" or name.startswith("rl_engine.") for name in sys.modules)


@pytest.mark.unit
def test_operator_comparison_adapter_delegates_to_rl_kernel(monkeypatch):
    _drop_rl_engine_modules()
    sys.modules.pop("vime.backends.rl_kernel_utils.operator_comparison", None)
    fake = _install_fake_rl_kernel_operator_comparison(monkeypatch)
    adapter = importlib.import_module("vime.backends.rl_kernel_utils.operator_comparison")

    assert adapter.RLK_OP_LOGP == "logp"
    assert adapter.PHASE4_TARGET_OPERATORS == ("logp",)
    assert adapter.OPERATOR_COMPARISON_SPECS == {"logp": "spec"}
    assert adapter.iter_operator_comparison_specs() == ("spec",)
    assert adapter.OperatorTolerance(atol=0.5).atol == pytest.approx(0.5)
    assert adapter.compare_operator_outputs("logp", "train", "infer") == {
        "op_name": "logp",
        "train": "train",
        "infer": "infer",
        "kwargs": {},
    }
    assert adapter.load_operator_comparison_module() is fake


@pytest.mark.unit
def test_package_reexports_operator_comparison_lazily(monkeypatch):
    _drop_rl_engine_modules()
    sys.modules.pop("vime.backends.rl_kernel_utils", None)
    sys.modules.pop("vime.backends.rl_kernel_utils.operator_comparison", None)

    package = importlib.import_module("vime.backends.rl_kernel_utils")

    assert not any(name == "rl_engine" or name.startswith("rl_engine.") for name in sys.modules)

    _install_fake_rl_kernel_operator_comparison(monkeypatch)

    assert package.RLK_OP_LOGP == "logp"
    assert package.iter_operator_comparison_specs() == ("spec",)


@pytest.mark.unit
def test_unavailable_operator_comparison_error_is_clear(monkeypatch):
    _drop_rl_engine_modules()
    sys.modules.pop("vime.backends.rl_kernel_utils.operator_comparison", None)
    adapter = importlib.import_module("vime.backends.rl_kernel_utils.operator_comparison")

    def fail_import(name, *args, **kwargs):
        if name == "rl_engine.alignment.cross_config.operator_comparison":
            raise ModuleNotFoundError("No module named 'rl_engine'")
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(adapter.importlib, "import_module", fail_import)

    with pytest.raises(adapter.RlkOperatorComparisonUnavailable, match="operator comparison standard"):
        adapter.load_operator_comparison_module()


@pytest.mark.unit
def test_vime_operator_comparison_does_not_carry_reference_implementations():
    module = importlib.import_module("vime.backends.rl_kernel_utils.operator_comparison")
    source = module.__loader__.get_source(module.__name__)

    assert "rl_engine.alignment.cross_config.operator_comparison" in source
    assert "def reference_rmsnorm" not in source
    assert "torch.nn.functional" not in source
