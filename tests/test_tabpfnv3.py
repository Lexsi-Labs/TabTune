"""
TabPFN v3 — full integration + fine-tuning test suite
=====================================================
Run (fast structural/wiring tests only — most need no torch/weights):
    pytest tests/test_tabpfnv3.py -v -m "not slow"

Run everything (adds weight-loading inference + real fine-tuning; needs torch,
network access to Prior-Labs/tabpfn_3, and ideally a GPU):
    pytest tests/test_tabpfnv3.py -v

Test ladder
-----------
1. Imports & vendored-package wiring            (fast)
2. The V3-pin bug fix (native FT uses v3)        (fast — class/AST level)
3. LoRA / PEFT target config                     (fast — pure-Python)
4. DataProcessor wiring                          (fast)
5. Pipeline construction (clf + reg)             (needs torch; no weights)
6. End-to-end inference                          (slow — weights)
7. End-to-end fine-tuning: native/sft/meta/peft  (slow — weights + training)
8. End-to-end regression FT: native/turn-by-turn (slow)
"""
import ast
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PIN_PATH = _REPO_ROOT / "tabtune/models/tabpfnv3/finetuning/_tabtune_v3_pin.py"


# ───────────────────────── fixtures ─────────────────────────
@pytest.fixture
def cls_data():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=300, n_features=8, n_informative=5,
                               n_classes=3, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    cols = [f"f{i}" for i in range(8)]
    return (pd.DataFrame(Xtr, columns=cols), pd.DataFrame(Xte, columns=cols),
            pd.Series(ytr, name="target"), pd.Series(yte, name="target"))


@pytest.fixture
def reg_data():
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=300, n_features=8, noise=0.1, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    cols = [f"f{i}" for i in range(8)]
    return (pd.DataFrame(Xtr, columns=cols), pd.DataFrame(Xte, columns=cols),
            pd.Series(ytr, name="target"), pd.Series(yte, name="target"))


@pytest.fixture
def fast_ft_params():
    import torch
    return {
        "epochs": 1,
        "batch_size": 64,
        "learning_rate": 1e-5,
        "show_progress": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@pytest.fixture
def peft_config():
    return {"r": 4, "lora_alpha": 8, "lora_dropout": 0.05}


# ───────────── 1. Imports & vendored-package wiring ─────────────
class TestImport:
    def test_import_via_init(self):
        from tabtune.models.tabpfnv3 import TabPFNv3Classifier, TabPFNv3Regressor
        assert TabPFNv3Classifier is not None and TabPFNv3Regressor is not None

    def test_base_names_reexported(self):
        from tabtune.models.tabpfnv3 import TabPFNClassifier, TabPFNRegressor
        assert TabPFNClassifier is not None and TabPFNRegressor is not None

    def test_import_regression_wrapper(self):
        from tabtune.models.regression.tabpfnv3.regressor import TabPFNv3RegressorWrapper
        assert TabPFNv3RegressorWrapper is not None

    def test_finetuning_module_imports(self):
        from tabtune.models.tabpfnv3.finetuning import (
            FinetunedTabPFNClassifier, FinetunedTabPFNRegressor,
        )
        assert FinetunedTabPFNClassifier is not None
        assert FinetunedTabPFNRegressor is not None

    def test_pin_module_imports(self):
        from tabtune.models.tabpfnv3.finetuning._tabtune_v3_pin import (
            V3PinnedFinetunedClassifier, V3PinnedFinetunedRegressor,
        )
        assert V3PinnedFinetunedClassifier is not None
        assert V3PinnedFinetunedRegressor is not None

    def test_classifier_sklearn_compatible(self):
        from sklearn.base import ClassifierMixin
        from tabtune.models.tabpfnv3 import TabPFNv3Classifier
        assert issubclass(TabPFNv3Classifier, ClassifierMixin)

    def test_regressor_sklearn_compatible(self):
        from sklearn.base import RegressorMixin
        from tabtune.models.tabpfnv3 import TabPFNv3Regressor
        assert issubclass(TabPFNv3Regressor, RegressorMixin)

    def test_model_version_v3(self):
        from tabtune.models.tabpfnv3.constants import ModelVersion
        assert ModelVersion.V3.value == "v3"

    def test_v3_default_checkpoints(self):
        from tabtune.models.tabpfnv3.model_loading import ModelSource
        assert "v3" in ModelSource.get_classifier_v3().default_filename
        assert "v3" in ModelSource.get_regressor_v3().default_filename

    def test_telemetry_shim_hermetic(self):
        from tabtune.models.tabpfnv3._compat.telemetry import track_model_call
        from tabtune.models.tabpfnv3._compat.telemetry.interactive import (
            capture_session, ping,
        )

        @track_model_call
        def f(x):
            return x + 1

        assert f(1) == 2
        assert capture_session() is None and ping() is None


# ───────────── 2. The V3-pin bug fix (critical) ─────────────
class TestV3PinBugFix:
    """Upstream FinetunedTabPFN hardcodes ModelVersion.V2_5 in _create_estimator;
    TabTune's pin subclasses must restore V3 so native FT trains v3 weights."""

    def test_pin_defaults_to_v3(self):
        from tabtune.models.tabpfnv3.constants import ModelVersion
        from tabtune.models.tabpfnv3.finetuning._tabtune_v3_pin import (
            V3PinnedFinetunedClassifier, V3PinnedFinetunedRegressor,
        )
        clf = V3PinnedFinetunedClassifier(epochs=1)
        reg = V3PinnedFinetunedRegressor(epochs=1)
        assert clf._pinned_model_version == ModelVersion.V3
        assert reg._pinned_model_version == ModelVersion.V3

    def test_pin_version_overridable(self):
        from tabtune.models.tabpfnv3.constants import ModelVersion
        from tabtune.models.tabpfnv3.finetuning._tabtune_v3_pin import (
            V3PinnedFinetunedClassifier,
        )
        clf = V3PinnedFinetunedClassifier(epochs=1, model_version=ModelVersion.V2_6)
        assert clf._pinned_model_version == ModelVersion.V2_6

    def test_pin_overrides_use_pinned_version(self):
        """AST-level: both override bodies use self._pinned_model_version and
        neither hardcodes V2_5."""
        tree = ast.parse(_PIN_PATH.read_text())
        override_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_create_estimator":
                override_count += 1
                body_src = ast.dump(node)
                assert "_pinned_model_version" in body_src
                assert "V2_5" not in body_src
        assert override_count == 2


# ───────────── 3. LoRA / PEFT target config ─────────────
class TestPEFTConfig:
    def test_v3_lora_target_config_present(self):
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS
        assert "TabPFNv3" in MODEL_LORA_TARGETS
        targets = MODEL_LORA_TARGETS["TabPFNv3"].target_substrings
        for needle in ("q_projection", "k_projection", "v_projection",
                       "out_projection", "x_embed"):
            assert needle in targets

    @requires_torch
    def test_v3_lora_injection_on_toy_module(self):
        import torch.nn as nn
        from tabtune.TuningManager.peft_utils import apply_tabular_lora, LoRALinear

        class ToyV3(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_projection = nn.Linear(8, 8)
                self.out_projection = nn.Linear(8, 8)
                self.x_embed = nn.Linear(8, 8)
                self.col_y_encoder = nn.Linear(1, 8)  # must be EXCLUDED
                self.unrelated = nn.Linear(8, 8)      # not a target

        m = ToyV3()
        apply_tabular_lora("TabPFNv3", m, peft_config={"r": 4, "lora_alpha": 8})
        assert isinstance(m.q_projection, LoRALinear)
        assert isinstance(m.out_projection, LoRALinear)
        assert isinstance(m.x_embed, LoRALinear)
        assert not isinstance(m.col_y_encoder, LoRALinear)
        assert m.q_projection.base.weight.requires_grad is False
        assert m.q_projection.lora_A.weight.requires_grad is True


# ───────────── 4. DataProcessor wiring ─────────────
class TestDataProcessor:
    def test_categorical_encoding_default(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        dp = DataProcessor(model_name="TabPFNv3", task_type="classification")
        assert dp.categorical_encoding == "tabpfn_special"

    def test_regression_processor_built(self, reg_data):
        from tabtune.Dataprocess.data_processor import DataProcessor
        Xtr, _, ytr, _ = reg_data
        dp = DataProcessor(model_name="TabPFNv3", task_type="regression")
        dp.fit(Xtr, ytr)
        assert dp.regression_processor_ is not None


# ───────────── 5. Pipeline construction (torch, no weights) ─────────────
@requires_torch
class TestPipelineConstruction:
    def test_classifier_builds(self):
        from tabtune import TabularPipeline
        from tabtune.models.tabpfnv3 import TabPFNv3Classifier
        p = TabularPipeline(model_name="TabPFNv3", task_type="classification",
                            tuning_strategy="inference", model_params={"device": "cpu"})
        assert isinstance(p.model, TabPFNv3Classifier)

    def test_regressor_builds(self):
        from tabtune import TabularPipeline
        from tabtune.models.regression.tabpfnv3.regressor import TabPFNv3RegressorWrapper
        p = TabularPipeline(model_name="TabPFNv3", task_type="regression",
                            tuning_strategy="inference", model_params={"device": "cpu"})
        assert isinstance(p.model, TabPFNv3RegressorWrapper)

    def test_classification_finetune_builds(self):
        from tabtune import TabularPipeline
        p = TabularPipeline(model_name="TabPFNv3", task_type="classification",
                            tuning_strategy="finetune",
                            tuning_params={"finetune_mode": "meta-learning"},
                            model_params={"device": "cpu"})
        assert p.model is not None

    def test_regression_finetune_allowed(self):
        from tabtune import TabularPipeline
        p = TabularPipeline(model_name="TabPFNv3", task_type="regression",
                            tuning_strategy="finetune", model_params={"device": "cpu"})
        assert p.model is not None


# ───────────── 6. End-to-end inference (slow) ─────────────
@pytest.mark.slow
@pytest.mark.finetuning
@requires_torch
class TestEndToEndInference:
    def test_classification_inference(self, cls_data):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = cls_data
        p = TabularPipeline(model_name="TabPFNv3", task_type="classification",
                            tuning_strategy="inference", model_params={"device": "cpu"})
        p.fit(Xtr, ytr)
        preds = p.predict(Xte)
        proba = p.predict_proba(Xte)
        assert preds.shape[0] == len(yte)
        assert proba.shape == (len(yte), len(np.unique(ytr)))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)

    def test_regression_inference(self, reg_data):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = reg_data
        p = TabularPipeline(model_name="TabPFNv3", task_type="regression",
                            tuning_strategy="inference", model_params={"device": "cpu"})
        p.fit(Xtr, ytr)
        preds = p.predict(Xte)
        assert preds.shape[0] == len(yte) and np.isfinite(preds).all()


# ───────────── 7. End-to-end classification fine-tuning (slow) ─────────────
@pytest.mark.slow
@pytest.mark.finetuning
@requires_torch
class TestClassificationFineTuning:
    @pytest.mark.parametrize("mode", ["meta-learning", "sft", "native"])
    def test_finetune_modes_fit_and_predict(self, cls_data, fast_ft_params, mode):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = cls_data
        params = dict(fast_ft_params)
        params["finetune_mode"] = mode
        p = TabularPipeline(model_name="TabPFNv3", task_type="classification",
                            tuning_strategy="finetune", tuning_params=params,
                            model_params={"device": params["device"]})
        p.fit(Xtr, ytr)
        assert p._is_fitted
        assert p.predict(Xte).shape[0] == len(yte)

    def test_peft_lora_finetune(self, cls_data, fast_ft_params, peft_config):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = cls_data
        params = dict(fast_ft_params)
        params["finetune_mode"] = "meta-learning"
        params["peft_config"] = peft_config
        p = TabularPipeline(model_name="TabPFNv3", task_type="classification",
                            tuning_strategy="peft", tuning_params=params,
                            model_params={"device": params["device"]})
        p.fit(Xtr, ytr)
        assert p._is_fitted
        assert p.predict(Xte).shape[0] == len(yte)


# ───────────── 8. End-to-end regression fine-tuning (slow) ─────────────
@pytest.mark.slow
@pytest.mark.finetuning
@requires_torch
class TestRegressionFineTuning:
    @pytest.mark.parametrize("mode", ["native", "turn_by_turn"])
    def test_regression_finetune_modes(self, reg_data, fast_ft_params, mode):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = reg_data
        params = dict(fast_ft_params)
        params["finetune_mode"] = mode
        p = TabularPipeline(model_name="TabPFNv3", task_type="regression",
                            tuning_strategy="finetune", tuning_params=params,
                            model_params={"device": params["device"]})
        p.fit(Xtr, ytr)
        assert p._is_fitted
        preds = p.predict(Xte)
        assert preds.shape[0] == len(yte) and np.isfinite(preds).all()
