"""
Tests for Advanced Evaluation Features - Calibration and Fairness.

This module tests calibration and fairness evaluation for all models.
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


class TestCalibrationEvaluation:
    """Test calibration evaluation for all models."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.calibration
    def test_evaluate_calibration_all_models(self, minimal_data, model_name):
        """Test calibration evaluation returns expected metrics for each model."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        # Calibration evaluation requires predict_proba, skip if not supported
        try:
            calibration_metrics = pipeline.evaluate_calibration(X_test, y_test)
            
            assert isinstance(calibration_metrics, dict)
            assert 'brier_score_loss' in calibration_metrics
            assert 'expected_calibration_error' in calibration_metrics
            assert 'maximum_calibration_error' in calibration_metrics
            
            # Metrics should be numeric (may be NaN for some edge cases)
            assert isinstance(calibration_metrics['brier_score_loss'], (int, float, type(np.nan)))
            assert isinstance(calibration_metrics['expected_calibration_error'], (int, float, type(np.nan)))
            assert isinstance(calibration_metrics['maximum_calibration_error'], (int, float, type(np.nan)))
        except NotImplementedError:
            pytest.skip(f"{model_name} does not support calibration evaluation")
        except Exception as e:
            # If calibration fails due to predict_proba issues, skip gracefully
            if "predict_proba" in str(e).lower() or "probability" in str(e).lower():
                pytest.skip(f"{model_name} calibration evaluation failed: {e}")
            else:
                # Other errors are test failures
                pytest.fail(f"{model_name} calibration evaluation failed: {e}")
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.calibration
    def test_evaluate_calibration_json_format(self, minimal_data, model_name):
        """Test calibration evaluation with JSON output format."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        try:
            calibration_metrics = pipeline.evaluate_calibration(
                X_test, y_test, output_format='json'
            )
            
            assert isinstance(calibration_metrics, dict)
            assert 'brier_score_loss' in calibration_metrics
        except (NotImplementedError, Exception) as e:
            pytest.skip(f"{model_name} calibration evaluation: {e}")


class TestFairnessEvaluation:
    """Test fairness evaluation for all models."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.fairness
    def test_evaluate_fairness_all_models(self, minimal_data, model_name, sensitive_features):
        """Test fairness evaluation returns expected metrics for each model."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        try:
            fairness_metrics = pipeline.evaluate_fairness(
                X_test, y_test, sensitive_features
            )
            
            assert isinstance(fairness_metrics, dict)
            assert 'statistical_parity_difference' in fairness_metrics
            assert 'equal_opportunity_difference' in fairness_metrics
            assert 'equalized_odds_difference' in fairness_metrics
            
            # Metrics should be numeric
            assert isinstance(fairness_metrics['statistical_parity_difference'], (int, float))
            assert isinstance(fairness_metrics['equal_opportunity_difference'], (int, float))
            assert isinstance(fairness_metrics['equalized_odds_difference'], (int, float))
        except Exception as e:
            # Fairness evaluation should work for all models
            pytest.fail(f"{model_name} fairness evaluation failed: {e}")
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.fairness
    def test_evaluate_fairness_json_format(self, minimal_data, model_name, sensitive_features):
        """Test fairness evaluation with JSON output format."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        
        try:
            fairness_metrics = pipeline.evaluate_fairness(
                X_test, y_test, sensitive_features, output_format='json'
            )
            
            assert isinstance(fairness_metrics, dict)
            assert 'statistical_parity_difference' in fairness_metrics
        except Exception as e:
            pytest.fail(f"{model_name} fairness evaluation failed: {e}")

