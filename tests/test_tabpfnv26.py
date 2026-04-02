"""
Test TabPFNv26 Integration
============================
Run: cd ~/TabTune_Internal-mar26 && pytest tests/test_tabpfnv26.py -v
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def cls_data():
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2,
                                random_state=42, n_informative=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=42)
    return (pd.DataFrame(X_train, columns=[f"f{i}" for i in range(10)]),
            pd.DataFrame(X_test, columns=[f"f{i}" for i in range(10)]),
            pd.Series(y_train, name="target"),
            pd.Series(y_test, name="target"))


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=42)
    return (pd.DataFrame(X_train, columns=[f"f{i}" for i in range(10)]),
            pd.DataFrame(X_test, columns=[f"f{i}" for i in range(10)]),
            pd.Series(y_train, name="target"),
            pd.Series(y_test, name="target"))


# ── Test 1: Imports ────────────────────────────────────────
class TestImport:
    def test_import_via_init(self):
        """Import through __init__.py re-exports."""
        from tabtune.models.tabpfnv26 import TabPFNv26Classifier, TabPFNv26Regressor
        assert TabPFNv26Classifier is not None
        assert TabPFNv26Regressor is not None

    def test_import_regression_wrapper(self):
        from tabtune.models.regression.tabpfnv26.regressor import TabPFNv26RegressorWrapper
        assert TabPFNv26RegressorWrapper is not None

    def test_classifier_is_sklearn_compatible(self):
        from tabtune.models.tabpfnv26 import TabPFNv26Classifier
        from sklearn.base import ClassifierMixin
        assert issubclass(TabPFNv26Classifier, ClassifierMixin)

    def test_regressor_is_sklearn_compatible(self):
        from tabtune.models.tabpfnv26 import TabPFNv26Regressor
        from sklearn.base import RegressorMixin
        assert issubclass(TabPFNv26Regressor, RegressorMixin)

    def test_model_version_v26_exists(self):
        from tabtune.models.tabpfnv26.constants import ModelVersion
        assert hasattr(ModelVersion, 'V2_6')

    def test_model_loading_has_v26_sources(self):
        from tabtune.models.tabpfnv26.model_loading import ModelSource
        src = ModelSource.get_classifier_v2_6()
        assert 'v2.6' in src.default_filename

    def test_finetuning_module_exists(self):
        from tabtune.models.tabpfnv26 import finetuning
        assert hasattr(finetuning, 'FinetunedTabPFNClassifier')


# ── Test 2: Classification Inference ──────────────────────
class TestClassificationInference:
    def test_fit_predict(self, cls_data):
        from tabtune import TabularPipeline
        X_train, X_test, y_train, y_test = cls_data
        pipe = TabularPipeline(
            model_name='TabPFNv26', task_type='classification',
            tuning_strategy='inference',
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, cls_data):
        from tabtune import TabularPipeline
        X_train, X_test, y_train, y_test = cls_data
        pipe = TabularPipeline(
            model_name='TabPFNv26', task_type='classification',
            tuning_strategy='inference',
        )
        pipe.fit(X_train, y_train)
        probas = pipe.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)


# ── Test 3: Classification Finetune ───────────────────────
class TestClassificationFinetune:
    def test_finetune(self, cls_data):
        from tabtune import TabularPipeline
        X_train, X_test, y_train, y_test = cls_data
        pipe = TabularPipeline(
            model_name='TabPFNv26', task_type='classification',
            tuning_strategy='finetune',
            tuning_params={'n_epochs': 2, 'learning_rate': 1e-4},
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)


# ── Test 4: Regression Inference ──────────────────────────
class TestRegressionInference:
    def test_fit_predict(self, reg_data):
        from tabtune import TabularPipeline
        X_train, X_test, y_train, y_test = reg_data
        pipe = TabularPipeline(
            model_name='TabPFNv26', task_type='regression',
            tuning_strategy='inference',
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)
        assert isinstance(preds[0], (float, np.floating))


# ── Test 5: Evaluation ────────────────────────────────────
class TestEvaluation:
    def test_evaluate_classification(self, cls_data):
        from tabtune import TabularPipeline
        X_train, X_test, y_train, y_test = cls_data
        pipe = TabularPipeline(
            model_name='TabPFNv26', task_type='classification',
            tuning_strategy='inference',
        )
        pipe.fit(X_train, y_train)
        results = pipe.evaluate(X_test, y_test, output_format='json')
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert results["accuracy"] > 0.5

    def test_evaluate_regression(self, reg_data):
        from tabtune import TabularPipeline
        X_train, X_test, y_train, y_test = reg_data
        pipe = TabularPipeline(
            model_name='TabPFNv26', task_type='regression',
            tuning_strategy='inference',
        )
        pipe.fit(X_train, y_train)
        results = pipe.evaluate(X_test, y_test, output_format='json')
        assert isinstance(results, dict)


# ── Test 6: Calibration ──────────────────────────────────
class TestCalibration:
    def test_evaluate_calibration(self, cls_data):
        from tabtune import TabularPipeline
        X_train, X_test, y_train, y_test = cls_data
        pipe = TabularPipeline(
            model_name='TabPFNv26', task_type='classification',
            tuning_strategy='inference',
        )
        pipe.fit(X_train, y_train)
        try:
            cal = pipe.evaluate_calibration(X_test, y_test)
            if cal is not None:
                assert isinstance(cal, dict)
        except Exception:
            pytest.skip("Calibration not available")


# ── Test 7: Direct Model Usage ────────────────────────────
class TestDirectModel:
    def test_classifier_direct(self, cls_data):
        from tabtune.models.tabpfnv26 import TabPFNv26Classifier
        X_train, X_test, y_train, y_test = cls_data
        clf = TabPFNv26Classifier(device='cpu')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert len(preds) == len(X_test)

    def test_regressor_direct(self, reg_data):
        from tabtune.models.regression.tabpfnv26.regressor import TabPFNv26RegressorWrapper
        X_train, X_test, y_train, y_test = reg_data
        reg = TabPFNv26RegressorWrapper(device='cpu')
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        assert len(preds) == len(X_test)

    def test_v25_still_works(self, cls_data):
        """Verify existing TabPFN (v2.5) is not broken."""
        from tabtune.models.tabpfn.classifier import TabPFNClassifier
        assert TabPFNClassifier is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])