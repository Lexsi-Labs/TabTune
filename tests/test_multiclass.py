"""
Tests for Multiclass Classification.

This module tests all models with multiclass classification (3 classes).
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


class TestMulticlassClassification:
    """Test multiclass classification for all models."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_multiclass_fit_predict(self, multiclass_data, model_name):
        """Test fitting and predicting with multiclass data for each model."""
        X_train, X_test, y_train, y_test = multiclass_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
        
        # Predictions should be in range [0, 2] for 3 classes
        unique_predictions = np.unique(predictions)
        assert all(0 <= p <= 2 for p in unique_predictions), \
            f"Predictions out of range: {unique_predictions}"
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_multiclass_predict_proba(self, multiclass_data, model_name):
        """Test predict_proba with multiclass data for each model."""
        X_train, X_test, y_train, _ = multiclass_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        try:
            probabilities = pipeline.predict_proba(X_test)
            
            assert probabilities is not None
            assert len(probabilities) == len(X_test)
            # Should have 3 columns for 3 classes
            assert probabilities.shape[1] == 3, \
                f"Expected 3 probability columns, got {probabilities.shape[1]}"
            
            # Probabilities should sum to ~1.0
            assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.15)
            
            # All probabilities should be in [0, 1]
            assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        except NotImplementedError:
            pytest.skip(f"{model_name} does not support predict_proba")
        except Exception as e:
            pytest.fail(f"{model_name} predict_proba failed: {e}")
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    def test_multiclass_evaluate(self, multiclass_data, model_name):
        """Test evaluation with multiclass data for each model."""
        X_train, X_test, y_train, y_test = multiclass_data
        
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
        
        # For multiclass, accuracy should be between 0 and 1
        assert 0 <= metrics['accuracy'] <= 1

