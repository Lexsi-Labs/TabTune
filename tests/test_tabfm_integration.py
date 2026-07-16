"""
Integration / wiring tests for the TabFM model in TabTune.

The `unit` tests here verify TabFM is registered across every subsystem
(DataProcessor, peft_utils, pipeline dispatch, package exports) and that
`import`ing the TabFM package does not eagerly pull the heavy vendored engine.
They need the full TabTune install (torch etc.) but NOT TabFM weights or a GPU.

The `TestTabFMEndToEnd` class runs the REAL model (downloads
`google/tabfm-1.0.0-pytorch`) and is skipped unless you opt in with:
    TABFM_RUN_WEIGHTS=1 pytest tests/test_tabfm_integration.py -v -m integration

Run (fast wiring only):  pytest tests/test_tabfm_integration.py -v -m "not integration"
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _require_full_tabtune():
    """Skip if the full TabTune package (the whole model zoo) cannot import --
    e.g. a pre-existing sklearn-version incompatibility in another model. This
    keeps TabFM wiring tests from failing for reasons unrelated to TabFM."""
    try:
        import tabtune  # noqa: F401
        from tabtune.TuningManager.tuning import TuningManager  # noqa: F401
    except Exception as e:
        pytest.skip(f"full TabTune package not importable (unrelated to TabFM): {e}")


@pytest.mark.unit
@pytest.mark.model_tabfm
class TestTabFMWiring:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classifier_and_regressor_importable(self):
        from tabtune.models.tabfm import TabFMClassifier
        from tabtune.models.regression.tabfm.regressor import TabFMRegressorWrapper
        assert TabFMClassifier is not None and TabFMRegressorWrapper is not None

    def test_dataprocessor_registers_tabfm(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        dp = DataProcessor(model_name="TabFM", task_type="classification")
        assert dp.categorical_encoding == "tabfm_special"
        from tabtune.Dataprocess.tabfm_preprocessor import TabFMPreprocessor
        assert isinstance(dp._get_custom_preprocessor(), TabFMPreprocessor)

    def test_regression_processor_registered(self):
        from tabtune.Dataprocess.data_processor import DataProcessor
        from tabtune.Dataprocess.regression.tabfm_processor import TabFMRegressionProcessor
        dp = DataProcessor(model_name="TabFM", task_type="regression")
        assert isinstance(dp._get_regression_processor(), TabFMRegressionProcessor)

    def test_peft_targets_registered(self):
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS
        assert "TabFM" in MODEL_LORA_TARGETS
        subs = MODEL_LORA_TARGETS["TabFM"].target_substrings
        assert "q_proj" in subs and "tf_icl" in subs

    def test_tuning_manager_imports_tabfm(self):
        from tabtune.TuningManager import tuning
        assert hasattr(tuning, "TabFMClassifier")
        assert hasattr(tuning.TuningManager, "_finetune_tabfm")
        assert hasattr(tuning.TuningManager, "_tabfm_episode_tensors")

    def test_preprocessor_fit_transform(self):
        from tabtune.Dataprocess.tabfm_preprocessor import TabFMPreprocessor
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
        y = pd.Series([0, 1, 0, 1])
        pre = TabFMPreprocessor(task_type="classification").fit(X, y)
        Xt, yt = pre.transform(X, y)
        assert Xt.shape == (4, 2) and Xt.dtype == np.float32
        assert set(np.unique(yt)).issubset({0, 1})
        assert pre.label_encoder_ is not None


@pytest.mark.unit
@pytest.mark.model_tabfm
class TestPipelineConstruction:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_classification_pipeline_selects_tabfm(self):
        from tabtune import TabularPipeline
        from tabtune.models.tabfm import TabFMClassifier
        # inference strategy does NOT download weights at construction time
        pipe = TabularPipeline(model_name="TabFM", task_type="classification",
                               tuning_strategy="inference", model_params={"device": "cpu"})
        assert isinstance(pipe.model, TabFMClassifier)

    def test_regression_pipeline_selects_tabfm(self):
        from tabtune import TabularPipeline
        from tabtune.models.regression.tabfm.regressor import TabFMRegressorWrapper
        pipe = TabularPipeline(model_name="TabFM", task_type="regression",
                               tuning_strategy="inference", model_params={"device": "cpu"})
        assert isinstance(pipe.model, TabFMRegressorWrapper)

    def test_regression_finetune_allowed(self, monkeypatch):
        from tabtune import TabularPipeline
        from tabtune.models.regression.tabfm.regressor import TabFMRegressorWrapper
        # No-op the backbone load so this stays a fast wiring test (no HF download).
        monkeypatch.setattr(TabFMRegressorWrapper, "_load_model", lambda self: None)
        monkeypatch.setattr(TabFMRegressorWrapper, "_initialize_model_variables", lambda self: None)
        # should NOT raise the "regression finetuning not enabled" ValueError
        TabularPipeline(model_name="TabFM", task_type="regression",
                        tuning_strategy="finetune", model_params={"device": "cpu"},
                        tuning_params={"finetune_mode": "turn_by_turn"})


@pytest.mark.unit
@pytest.mark.model_tabfm
class TestImportSafety:
    def test_importing_package_does_not_load_heavy_engine(self):
        try:
            import tabtune  # noqa: F401  (base package; needs colorlog etc.)
        except Exception as e:
            pytest.skip(f"base tabtune not importable (unrelated to TabFM): {e}")
        # Drop any previously-imported heavy submodules, import the package fresh,
        # and confirm the vendored engine / loader are NOT eagerly imported.
        for m in list(sys.modules):
            if m.startswith("tabtune.models.tabfm.classifier_and_regressor") or \
               m.startswith("tabtune.models.tabfm.model_loading") or \
               m == "tabtune.models.tabfm.model.model":
                del sys.modules[m]
        import tabtune.models.tabfm  # noqa: F401
        assert "tabtune.models.tabfm.classifier_and_regressor" not in sys.modules
        assert "tabtune.models.tabfm.model_loading" not in sys.modules


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.model_tabfm
class TestTabFMEndToEnd:
    """Real weights. Opt in with TABFM_RUN_WEIGHTS=1 (downloads ~HF checkpoint)."""

    @pytest.fixture(autouse=True)
    def _gate(self):
        if os.environ.get("TABFM_RUN_WEIGHTS") != "1":
            pytest.skip("set TABFM_RUN_WEIGHTS=1 to run the real TabFM weights end-to-end")

    def _data(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        d = load_breast_cancer(as_frame=True)
        return train_test_split(d.data, d.target, test_size=0.25, random_state=42, stratify=d.target)

    def test_inference_classification(self):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = self._data()
        pipe = TabularPipeline(model_name="TabFM", task_type="classification", tuning_strategy="inference")
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)
        assert proba.shape[0] == len(Xte)
        metrics = pipe.evaluate(Xte, yte)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_peft_classification(self):
        from tabtune import TabularPipeline
        Xtr, Xte, ytr, yte = self._data()
        pipe = TabularPipeline(
            model_name="TabFM", task_type="classification", tuning_strategy="peft",
            tuning_params={"finetune_mode": "meta-learning", "epochs": 1, "steps_per_epoch": 5,
                           "learning_rate": 2e-6, "show_progress": False,
                           "peft_config": {"r": 8, "lora_alpha": 16}},
        )
        pipe.fit(Xtr, ytr)
        assert pipe.predict(Xte).shape[0] == len(Xte)
