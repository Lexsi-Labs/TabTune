"""
Tests for TabularPipeline - the main user-facing API.

These tests cover initialization, fit, predict, and evaluate methods.
"""
import pytest
import pandas as pd
import numpy as np
from tabtune import TabularPipeline


class TestTabularPipelineInitialization:
    """Test TabularPipeline initialization."""
    
    def test_init_with_valid_model(self, minimal_data):
        """Test initialization with a valid model name."""
        X_train, _, _, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        assert pipeline.model_name == 'TabPFN'
        assert pipeline.tuning_strategy == 'inference'
        assert pipeline._is_fitted == False
    
    def test_init_with_tuning_params(self):
        """Test initialization with tuning parameters."""
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference',
            tuning_params={'device': 'cpu'}
        )
        
        assert 'device' in pipeline.tuning_params
        assert pipeline.tuning_params['device'] == 'cpu'
    
    def test_init_with_model_params(self):
        """Test initialization with model parameters."""
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference',
            model_params={'ignore_pretraining_limits': True}
        )
        
        assert 'ignore_pretraining_limits' in pipeline.model_params
    
    def test_init_invalid_model(self):
        """Test initialization with invalid model raises error."""
        with pytest.raises(ValueError, match="not supported"):
            TabularPipeline(
                model_name='InvalidModel',
                tuning_strategy='inference'
            )
    
    @pytest.mark.parametrize("model_name", ['TabPFN', 'TabICL', 'Mitra'])
    def test_init_multiple_models(self, model_name):
        """Test initialization of multiple supported models."""
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        assert pipeline.model_name == model_name


class TestTabularPipelineFit:
    """Test TabularPipeline fit method."""
    
    @pytest.mark.slow
    def test_fit_tabpfn_inference(self, minimal_data):
        """Test fitting TabPFN in inference mode."""
        X_train, _, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        assert pipeline._is_fitted == True
        assert hasattr(pipeline, 'X_train_processed_')
    
    def test_fit_without_y_raises_error(self):
        """Test that fit requires target variable."""
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        # The pipeline tries to call y.copy() first, which raises AttributeError
        # Then later validation should catch it, but the first error is AttributeError
        with pytest.raises((TypeError, ValueError, RuntimeError, AttributeError)):
            pipeline.fit(X, None)


class TestTabularPipelinePredict:
    """Test TabularPipeline predict method."""
    
    @pytest.mark.slow
    def test_predict_after_fit(self, minimal_data):
        """Test predict works after fitting."""
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_before_fit_raises_error(self, minimal_data):
        """Test predict raises error if not fitted."""
        _, X_test, _, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        with pytest.raises(RuntimeError, match="must call fit"):
            pipeline.predict(X_test)
    
    @pytest.mark.slow
    def test_predict_proba_shape(self, minimal_data):
        """Test predict_proba returns correct shape."""
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)
        
        assert probabilities is not None
        assert len(probabilities) == len(X_test)
        assert probabilities.shape[1] >= 2  # At least 2 classes for binary
        # Probabilities should sum to ~1.0
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.1)


class TestTabularPipelineEvaluate:
    """Test TabularPipeline evaluate method."""
    
    @pytest.mark.slow
    def test_evaluate_returns_metrics(self, minimal_data):
        """Test evaluate returns expected metrics."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        metrics = pipeline.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc_score' in metrics
        assert isinstance(metrics['accuracy'], (int, float))
    
    @pytest.mark.slow
    def test_evaluate_json_format(self, minimal_data):
        """Test evaluate with JSON output format."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        metrics = pipeline.evaluate(X_test, y_test, output_format='json')
        
        assert isinstance(metrics, dict)


class TestTabularPipelineSaveLoad:
    """Test TabularPipeline save and load methods."""
    
    @pytest.mark.slow
    def test_save_load_pipeline(self, minimal_data, tmp_path):
        """Test saving and loading a pipeline."""
        import joblib
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        # Save
        save_path = tmp_path / "test_pipeline.joblib"
        pipeline.save(str(save_path))
        assert save_path.exists()
        
        # Load
        loaded_pipeline = TabularPipeline.load(str(save_path))
        assert loaded_pipeline._is_fitted == True
        assert loaded_pipeline.model_name == pipeline.model_name
        
        # Test loaded pipeline can predict
        predictions = loaded_pipeline.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_save_before_fit_raises_error(self, tmp_path):
        """Test saving before fitting raises error."""
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        save_path = tmp_path / "test_pipeline.joblib"
        with pytest.raises(RuntimeError, match="only save a pipeline after"):
            pipeline.save(str(save_path))

