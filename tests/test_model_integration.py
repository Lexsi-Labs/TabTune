"""
Tests for Model Integration - All models with inference mode.

This module tests all 7 supported models (TabPFN, TabICL, OrionMSP, OrionBix, 
Mitra, ContextTab, TabDPT) with inference strategy to ensure basic functionality
works across all models.
"""
import pytest
import pandas as pd
import numpy as np
from tabtune import TabularPipeline

# All supported models
ALL_MODELS = [
    'TabPFN',
    'TabICL',
    'OrionMSP',
    'OrionBix',
    'Mitra',
    'ContextTab',
    'TabDPT',
]


class TestModelIntegrationFit:
    """Test fitting all models in inference mode."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_fit_inference_all_models(self, minimal_data, model_name):
        """Test fitting each model in inference mode."""
        X_train, _, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        assert pipeline._is_fitted == True
        assert hasattr(pipeline, 'X_train_processed_') or hasattr(pipeline, 'model')


class TestModelIntegrationPredict:
    """Test prediction for all models in inference mode."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_predict_after_fit_all_models(self, minimal_data, model_name):
        """Test predict works after fitting for each model."""
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray) or isinstance(predictions, pd.Series) or isinstance(predictions, list)
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_predict_proba_all_models(self, minimal_data, model_name):
        """Test predict_proba returns correct shape for each model."""
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        # Some models might not support predict_proba, skip if not available
        try:
            probabilities = pipeline.predict_proba(X_test)
            
            assert probabilities is not None
            assert len(probabilities) == len(X_test)
            assert probabilities.shape[1] >= 2  # At least 2 classes for binary
            # Probabilities should sum to ~1.0 (within tolerance)
            assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.15)  # Slightly more tolerance
        except NotImplementedError:
            pytest.skip(f"{model_name} does not support predict_proba")
        except Exception as e:
            # If predict_proba fails for other reasons, that's a test failure
            pytest.fail(f"{model_name} predict_proba failed: {e}")


class TestModelIntegrationEvaluate:
    """Test evaluation for all models in inference mode."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_evaluate_all_models(self, minimal_data, model_name):
        """Test evaluate returns expected metrics for each model."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        metrics = pipeline.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc_score' in metrics
        assert isinstance(metrics['accuracy'], (int, float))
        # Accuracy should be between 0 and 1 (or 0 and 100 if percentage)
        assert 0 <= metrics['accuracy'] <= 1 or 0 <= metrics['accuracy'] <= 100
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_evaluate_json_format_all_models(self, minimal_data, model_name):
        """Test evaluate with JSON output format for each model."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        metrics = pipeline.evaluate(X_test, y_test, output_format='json')
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics


class TestModelIntegrationSaveLoad:
    """Test save and load functionality for all models."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_save_load_all_models(self, minimal_data, model_name, tmp_path):
        """Test saving and loading pipeline for each model."""
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        # Save
        save_path = tmp_path / f"test_{model_name}_pipeline.joblib"
        pipeline.save(str(save_path))
        assert save_path.exists()
        
        # Load
        loaded_pipeline = TabularPipeline.load(str(save_path))
        assert loaded_pipeline._is_fitted == True
        assert loaded_pipeline.model_name == pipeline.model_name
        
        # Test loaded pipeline can predict
        predictions = loaded_pipeline.predict(X_test)
        assert len(predictions) == len(X_test)

